import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from time import perf_counter
from itertools import product
from typing import Optional
from tqdm import tqdm

from mr_sim.phantoms import shepp_logan

from mr_recon.imperfections.field import alpha_segementation
from mr_recon.utils import gen_grd, cvplot
from mr_recon.recons import CG_SENSE_recon
from mr_recon.linops import encoding_matrix, sense_linop
from mr_recon.fourier import sigpy_nufft

from hofft.matvec import matvec_cur, matvec_rnd
from hofft.forward_model import hofft_linop, hofft_compressed_linop
from hofft.decomp import hofft_params
from hofft.sparse_fit import sparse_params
from hofft.utils import normalize
from hofft.pipelines import (
    time_seg_decomp_linop,
    als_hofft,
    als_hofft_sparse_lstsq,
    als_hofft_sparse_smooth
)
from hofft.phase_coeffs import (
    rescale_phis_alphas,
    whiten_phis_alphas,
    trj_dev_to_phis_alphas,
    apply_phase_midpoints
)

from einops import rearrange, einsum


class GPUTimer:
    """
    Context manager for timing a block of (possibly GPU-async) work.

    On a CUDA device, uses ``torch.cuda.Event`` so kernel launches that
    haven't actually finished yet don't make the block look artificially
    fast; on CPU it falls back to a plain ``perf_counter``. Elapsed time
    (in seconds) is available as ``.elapsed`` once the ``with`` block exits.
    """
    def __init__(self, torch_dev: torch.device):
        self.is_cuda = torch_dev.type == 'cuda'

    def __enter__(self):
        if self.is_cuda:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
            self._start_evt.record()
        else:
            self._t0 = perf_counter()
        return self

    def __exit__(self, *exc_info):
        if self.is_cuda:
            self._end_evt.record()
            torch.cuda.synchronize()
            self.elapsed = self._start_evt.elapsed_time(self._end_evt) / 1000  # ms -> s
        else:
            self.elapsed = perf_counter() - self._t0
        return False


def time_repeated(fn, torch_dev: torch.device, n_reps: int = 1, reduction: str = 'median'):
    """
    Run ``fn()`` ``n_reps`` times under :class:`GPUTimer`, reducing the elapsed
    times to a single scalar to smooth out GPU timing noise. Returns
    ``(elapsed, result)`` where ``result`` is ``fn()``'s return value from the
    last repeat.
    """
    times = []
    result = None
    for _ in range(n_reps):
        with GPUTimer(torch_dev) as t:
            result = fn()
        times.append(t.elapsed)
    times = torch.tensor(times)
    elapsed = {'mean': times.mean, 'median': times.median, 'min': times.min}[reduction]().item()
    return elapsed, result


def nrmse(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Relative Frobenius error of ``img`` against ``ref``, optionally restricted to ``mask``."""
    if mask is not None:
        img = img * mask
        ref = ref * mask
    return (torch.linalg.norm(img.abs() - ref.abs()) / torch.linalg.norm(ref.abs())).item()


# Params
os = 1.25
Ws = [3, 5, 7, 9]   # kernel widths to sweep (TS NUFFT + HOFFT)
Ls = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # number of spatial/temporal factors to sweep (TS NUFFT + HOFFT)
num_als_iter = 100
num_time_reps = 3  # repeats per timed block, reduced via `time_reduction` to smooth GPU noise
time_reduction = 'mean'
gamma_bar = 42.58e6 # Hz/T
max_phase_wraps = 4
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# ------------ Prepare Data ------------
kwargs = {'weights_only': True, 'map_location': torch_dev}
img_gt = torch.load('./data/sim_spiral/img.pt', **kwargs)
trj = torch.load('./data/sim_spiral/trj.pt', **kwargs)
dcf = torch.load('./data/sim_spiral/dcf.pt', **kwargs)
mask = torch.load('./data/sim_spiral/mask.pt', **kwargs)
img_gt_cpu = img_gt.cpu()
im_size = img_gt.shape

# Dummy coil maps
mps = img_gt[None, :] * 0 + 1
mps *= mask

# Build quadratic phase evolution
crds = gen_grd(im_size).to(torch_dev) # in [-1/2, 1/2]
X, Y = crds[..., 0], crds[..., 1]
phis = (X ** 2 + Y ** 2)[None, :, :] # 1 *im_size
phis /= phis.max() # in [0, 1]
alphas = torch.linspace(0, 1, dcf.shape[0],
                        device=torch_dev,
                        dtype=torch.float32)
alphas = alphas[None, :] * max_phase_wraps # in [0, max_phase_wraps]

# Siimulate the long way
phis_sim   = torch.cat([crds.moveaxis(-1, 0), phis], dim=0)
alphas_sim = torch.cat([trj.moveaxis(-1, 0), alphas], dim=0)
A_sim = encoding_matrix(mps, phis_sim, alphas_sim,
                       temporal_batch_size=2**10)
ksp = A_sim(img_gt)
cg_kwargs = {'max_iter': 5, 'max_eigen': 1.0, 'verbose': False}

# ------------ Uncorrected Reconstruction ------------
# Fixed reference width -- Uncorrected/Expanded Encoding don't depend on the
# swept (W, L) below, so they're computed once with a representative width.
W_nominal = 6
nufft_uncorr = sigpy_nufft(im_size, width=W_nominal)
nufft_uncorr.beta = nufft_uncorr.optimal_beta(torch_dev=torch_dev)
A_uncorr = sense_linop(trj, mps, dcf=dcf, nufft=nufft_uncorr)
img_uncorr = CG_SENSE_recon(A_uncorr, ksp, **cg_kwargs)
img_uncorr = normalize(img_uncorr, img_gt)

# ------------ Expanded Encoding Reconstruction ------------
img_ee = CG_SENSE_recon(A_sim, ksp, **cg_kwargs)
img_ee = normalize(img_ee, img_gt)

# ------------ Shared HOFFT preprocessing (independent of W, L) ------------
# Stack quadratic phase and grid deviation phase. NOTE: `phis`/`alphas` (the
# quadratic-phase-only terms) are kept unmutated below since TS NUFFT's
# alpha_segementation call (inside the sweep) uses them directly -- only
# HOFFT needs the extra grid-deviation term folded in.
trj_grd = (trj * os).round() / os
phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, os)
phis_stack = torch.cat([phis, phis_dev], dim=0)
alphas_stack = torch.cat([alphas, alphas_dev], dim=0)

# Normalize phase coefficients
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack, whiten=True)
# phis_nrm = phis_stack
# alphas_nrm = alphas_stack
# phis_mp *= 0
# alphas_mp *= 0
spat, temp = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp)
# phis_nrm, alphas_nrm = whiten_phis_alphas(phis_nrm, alphas_nrm)

# ------------ Sweep over (L, W) for TS NUFFT and HOFFT ------------
# img_ref = img_gt
img_ref = img_ee
nL, nW = len(Ls), len(Ws)
results = {
    'nrmse_ts': torch.zeros(nL, nW),
    'nrmse_hofft': torch.zeros(nL, nW),
    'decomp_time_ts': torch.zeros(nL, nW),
    'recon_time_ts': torch.zeros(nL, nW),
    'decomp_time_hofft': torch.zeros(nL, nW),
    'recon_time_hofft': torch.zeros(nL, nW),
}
imgs_ts = torch.zeros(nL, nW, *im_size, dtype=torch.complex64)
imgs_hofft = torch.zeros(nL, nW, *im_size, dtype=torch.complex64)

for i, L in enumerate(tqdm(Ls, desc='Sweeping L')):
    for j, W in enumerate(tqdm(Ws, desc='Sweeping W', leave=False)):
        
        # HOFFT parameters
        hparams = hofft_params((W,)*2, os, L,
                            #    spatial_init=f'500_alphas_{num_als_iter}',
                                spatial_init='seg',
                               matvec_kwargs={
                                   'spatial_batch_size': 2**10,
                               },
                                # matvec_type=matvec_cur,
                                # matvec_kwargs={
                                #     'rank_phi': 500,
                                #     'rank_alpha': 500,
                                # },
                            #    reduced_im_size=(80,)*2,
                                verbose=False,)

        # ------------ TS NUFFT ------------
        def decomp_ts():
            return time_seg_decomp_linop(phis, alphas, 
                                         mps=mps, trj=trj, dcf=dcf,
                                         spatial_mask=mask,
                                         hparams=hparams,)
        results['decomp_time_ts'][i, j], A_ts = time_repeated(
            decomp_ts, torch_dev, n_reps=num_time_reps, reduction=time_reduction)

        if i == 0 and j == 0:
            # Untimed dummy pass: the very first CG_SENSE_recon call pays for
            # one-time CUDA warm-up (context init, kernel/plan caching), which
            # would otherwise inflate this specific timing.
            CG_SENSE_recon(A_ts, ksp, **cg_kwargs)
            if torch_dev.type == 'cuda':
                torch.cuda.synchronize()

        def recon_ts():
            return normalize(CG_SENSE_recon(A_ts, ksp, **cg_kwargs), img_ref)
        results['recon_time_ts'][i, j], img_ts = time_repeated(
            recon_ts, torch_dev, n_reps=num_time_reps, reduction=time_reduction)
        results['nrmse_ts'][i, j] = nrmse(img_ts, img_ref, mask=mask)
        imgs_ts[i, j] = img_ts.cpu()

        # ------------ HOFFT ------------
        def decomp_hofft():
            spatial_factors, kern_weights = als_hofft(phis_nrm, alphas_nrm,
                                                      spatial_mask=mask,
                                                      hparams=hparams,
                                                      num_als_iter=num_als_iter*0)
            spatial_factors_full = spatial_factors * spat
            kern_weights_full = kern_weights * temp
            return hofft_linop(trj_grd, mps,
                                  dcf=dcf,
                                  spatial_factors=spatial_factors_full,
                                  kern_weights=kern_weights_full,
                                  os_grid=os)
        results['decomp_time_hofft'][i, j], A_hofft = time_repeated(
            decomp_hofft, torch_dev, n_reps=num_time_reps, reduction=time_reduction)

        if i == 0 and j == 0:
            # Same untimed warm-up pass as TS NUFFT above, for HOFFT's own
            # first-call kernels (its forward operator differs from A_ts's).
            CG_SENSE_recon(A_hofft, ksp, **cg_kwargs)
            if torch_dev.type == 'cuda':
                torch.cuda.synchronize()

        def recon_hofft():
            return normalize(CG_SENSE_recon(A_hofft, ksp, **cg_kwargs), img_ref)
        results['recon_time_hofft'][i, j], img_hofft = time_repeated(
            recon_hofft, torch_dev, n_reps=num_time_reps, reduction=time_reduction)
        results['nrmse_hofft'][i, j] = nrmse(img_hofft, img_ref, mask=mask)
        imgs_hofft[i, j] = img_hofft.cpu()

# ------------ Save + Summarize ------------
save_path = './paper_experiments/quad_phase/sweep_results.pt'
torch.save({
    **results,
    'imgs_ts': imgs_ts,
    'imgs_hofft': imgs_hofft,
    'img_ee': img_ee.cpu(),
    'img_gt': img_gt.cpu(),
    'Ls': Ls,
    'Ws': Ws,
    'os': os,
}, save_path)
print(f'Saved sweep results to {save_path}')

print(f'\n{"L":>4} {"W":>4} {"NRMSE(ts)":>10} {"NRMSE(hofft)":>13} '
      f'{"decomp(ts)":>11} {"recon(ts)":>10} {"decomp(hofft)":>14} {"recon(hofft)":>13}')
for i, L in enumerate(Ls):
    for j, W in enumerate(Ws):
        print(f'{L:>4} {W:>4} '
              f'{results["nrmse_ts"][i,j]:>10.4f} {results["nrmse_hofft"][i,j]:>13.4f} '
              f'{results["decomp_time_ts"][i,j]:>10.3f}s {results["recon_time_ts"][i,j]:>9.3f}s '
              f'{results["decomp_time_hofft"][i,j]:>13.3f}s {results["recon_time_hofft"][i,j]:>12.3f}s')

# ------------ Plot: metrics vs L, one line per (method, W) ------------
# TS NUFFT and HOFFT share the same axes per metric (color = W, linestyle =
# method) so the two methods are directly comparable.
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
panels = [
    ('nrmse', 'NRMSE'), ('decomp_time', 'Decomp time [s]'), ('recon_time', 'Recon time [s]'),
]
methods = [('ts', 'TS NUFFT', '-', 'o'), ('hofft', 'HOFFT', '--', 's')]
for ax, (key, title) in zip(axes[:3], panels):
    for j, W in enumerate(Ws):
        color = f'C{j}'
        for suffix, method_label, linestyle, marker in methods:
            ax.plot(Ls, results[f'{key}_{suffix}'][:, j].numpy(),
                    color=color, linestyle=linestyle, marker=marker,
                    label=f'{method_label}, W={W}')
    ax.set_xlabel('L')
    ax.set_ylabel(title)
    ax.set_title(title)
    if key == 'nrmse':
        ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both' if key == 'nrmse' else 'major')

# NRMSE vs recon time -- the accuracy/speed tradeoff, points connected in
# increasing-L order per (method, W) line.
ax = axes[3]
for j, W in enumerate(Ws):
    color = f'C{j}'
    for suffix, method_label, linestyle, marker in methods:
        ax.plot(results[f'recon_time_{suffix}'][:, j].numpy(),
                results[f'nrmse_{suffix}'][:, j].numpy(),
                color=color, linestyle=linestyle, marker=marker,
                label=f'{method_label}, W={W}')
ax.set_yscale('log')
ax.set_xlabel('Recon time [s]')
ax.set_ylabel('NRMSE')
ax.set_title('NRMSE vs recon time')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('./paper_experiments/quad_phase/sweep_lineplots.png', dpi=150)

# ------------ Plot: qualitative comparison at the largest (L, W) ------------
imgs = [img_uncorr.cpu(), img_ee.cpu(), imgs_ts[-1, -1], imgs_hofft[-1, -1]]
vmin = 0
img_ref = img_ref.cpu()
vmax = img_gt_cpu.abs().median() + 3 * img_gt_cpu.abs().std()
titles = ['Uncorrected', 'Expanded Encoding', f'TS NUFFT (L={Ls[-1]}, W={Ws[-1]})', f'HOFFT (L={Ls[-1]}, W={Ws[-1]})']
plt.figure(figsize=(14, 7))
M = 5
for i, img in enumerate(imgs):
    img = normalize(img, img_ref)
    plt.subplot(2, len(imgs), i+1)
    plt.imshow(img.abs().cpu().rot90(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title(titles[i])
    
    err = img.abs() - img_ref.abs()
    plt.subplot(2, len(imgs), i+1+len(imgs))
    plt.imshow(err.abs().cpu().rot90(), cmap='gray', vmin=vmin/M, vmax=vmax/M)
    plt.axis('off')
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.show()
