import gc
import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from time import perf_counter
from typing import Optional
from tqdm import tqdm

from mr_recon.recons import CG_SENSE_recon
from mr_recon.linops import sense_linop, batching_params
from mr_recon.fourier import sigpy_nufft
from mr_recon.algs import density_compensation

from hofft.forward_model import expanded_encoding
from hofft.matvec import matvec_cur, matvec_rnd_fast
from hofft.decomp import hofft_params
from hofft.utils import normalize, gen_grd
from hofft.pipelines import time_seg_decomp_linop, hofft_decomp_linop
from hofft.sparse_fit import sparse_params


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
Ls_nufft = [5, 20, 40, 80, 120]
Ls_hofft = [5, 10, 15, 20]
# Ls_nufft = [20]
# Ls_hofft = [2]
Ws_nufft = [3]
Ws_hofft = [3]
num_als_iter = 10
num_time_reps = 1  # repeats per timed block, reduced via `time_reduction` to smooth GPU noise
time_reduction = 'mean'
phase_norm = True
mask_thresh = 0.9  # espirit eigenvalue threshold for the spatial support mask
B_compressed = 10  # number of SVD-compressed field bases to fit HOFFT/TS NUFFT against
B_compressed = None
cg_kwargs = {'max_iter': 10, 'max_eigen': 1.0, 'verbose': False}
hparams = hofft_params((Ws_hofft[0],)*2, os, Ls_hofft[0],
                        reduced_im_size=(100*4, 100*4),
                        # reduced_im_size=(150*4, 150*4),
                        spatial_init='seg',
                        # spatial_init=f'500_alphas_{num_als_iter}',
                        # matvec_kwargs={
                        #     'temporal_batch_size': 2**10,
                        # },
                        kalpha_method='kmeans',
                        # spatial_batch_size=2**14,
                        matvec_type=matvec_cur,
                        matvec_kwargs={
                            'rank_phi': 500,
                            'rank_alpha': 500,
                        },
                        verbose=True)
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# ------------ Prepare Data ------------
fpath = './data/highres_spiral'
kwargs = {'weights_only': True, 'map_location': torch_dev}
img_gt = torch.load(f'{fpath}/img_gt.pt', **kwargs)
trj = torch.load(f'{fpath}/trj.pt', **kwargs)
dcf = torch.load(f'{fpath}/dcf.pt', **kwargs)
mps = torch.load(f'{fpath}/mps.pt', **kwargs)
ksp = torch.load(f'{fpath}/ksp.pt', **kwargs)
evals = torch.load(f'{fpath}/evals.pt', **kwargs)
phis = torch.load(f'{fpath}/phis.pt', **kwargs)
alphas = torch.load(f'{fpath}/alphas.pt', **kwargs)
img_gt_cpu = img_gt.cpu()
im_size = mps.shape[1:]
C = mps.shape[0]
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0]  # round to nearest even oversampled grid
bparams = batching_params(coil_batch_size=C//2)


# # SVD down
# from hofft.phase_coeffs import compress_phis_alphas, rescale_phis_alphas, apply_phase_midpoints
# from einops import einsum
# from hofft.utils import reduce_spatial
# import imageio.v2 as imageio
# import numpy as np

# phis = reduce_spatial(phis, (256, 256), order=3)
# phis_plt, phis_mp, alphas_plt, alphas_mp = rescale_phis_alphas(phis, alphas)
# spat, temp = apply_phase_midpoints(phis_plt, alphas_plt, phis_mp, alphas_mp)
# phis_plt, alphas_plt = compress_phis_alphas(phis_plt, alphas_plt, B_compressed=2)
# phis_plt, _, alphas_plt, _ = rescale_phis_alphas(phis_plt, alphas_plt)

# alphas_plt = alphas_plt[:, ::50, 0]  # (2, T)
# temp = temp[::50, 0]

# a0 = alphas_plt[0].cpu().numpy()
# a1 = alphas_plt[1].cpu().numpy()
# phz = torch.exp(-2j * torch.pi * einsum(phis_plt, alphas_plt, 'B ..., B T -> T ...'))
# phz *= spat
# phz *= temp[:, None, None]
# ang = phz.angle().cpu().numpy()

# fig, (ax_trj, ax_phz) = plt.subplots(1, 2, figsize=(10, 4.5))
# ax_trj.plot(a0, a1, color='#2a6f97', lw=1.5, alpha=0.5)
# point, = ax_trj.plot([], [], 'o', color='#e76f51', ms=8, animated=True)
# trail, = ax_trj.plot([], [], color='#e76f51', lw=2.0, alpha=0.9, animated=True)
# ax_trj.set_xlim(-6, 6)
# ax_trj.set_ylim(-6, 6)
# ax_trj.set_aspect('equal')
# ax_trj.set_xlabel(r'$\alpha_0$')
# ax_trj.set_ylabel(r'$\alpha_1$')
# ax_trj.set_title(r'SVD $\alpha$-space')

# im = ax_phz.imshow(ang[0], cmap='jet', vmin=-np.pi, vmax=np.pi, animated=True)
# ax_phz.set_title('phase')
# ax_phz.axis('off')
# fig.tight_layout()

# fig.canvas.draw()
# background = fig.canvas.copy_from_bbox(fig.bbox)
# frames = []
# for i in range(len(a0)):
#     fig.canvas.restore_region(background)
#     point.set_data([a0[i]], [a1[i]])
#     trail.set_data(a0[: i + 1], a1[: i + 1])
#     im.set_data(ang[i])
#     ax_trj.draw_artist(trail)
#     ax_trj.draw_artist(point)
#     ax_phz.draw_artist(im)
#     fig.canvas.blit(fig.bbox)
#     frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

# out_gif = './paper_experiments/highres_sp
# iral/phase_evolution.gif'
# imageio.mimsave(out_gif, frames, duration=40, loop=0)
# print(f'Saved {out_gif} ({len(frames)} frames)')
# quit()

# Undersample
R = 1
trj = trj[:, ::R]
ksp = ksp[:, :, ::R]
alphas = alphas[:, :, ::R]
dcf = density_compensation(trj, im_size) # recompute

# Spatial support mask from the espirit eigenvalue map
mask = (evals > mask_thresh).float()
mps = mps * mask

# Drop negligible field terms (b0 / kspha bases with ~zero spatial or temporal support)
B = phis.shape[0]
phi_energy = phis.reshape((B, -1)).abs().mean(dim=1)
alpha_energy = alphas.reshape((B, -1)).abs().mean(dim=1)
energy = phi_energy * alpha_energy
idxs = torch.argwhere(energy > 1e-6)[:, 0]
phis = phis[idxs]
alphas = alphas[idxs]

# ------------ Uncorrected Reconstruction ------------
# Fixed reference width -- Uncorrected/Expanded Encoding don't depend on the
# swept (W, L) below, so they're computed once with a representative width.
W_nominal = 6
nufft_uncorr = sigpy_nufft(im_size, width=W_nominal)
nufft_uncorr.beta = nufft_uncorr.optimal_beta(torch_dev=torch_dev)
A_uncorr = sense_linop(trj, mps, dcf=dcf, nufft=nufft_uncorr, bparams=bparams)
img_uncorr = CG_SENSE_recon(A_uncorr, ksp, **cg_kwargs)
img_uncorr = normalize(img_uncorr, img_gt)

# ------------ Expanded Encoding Reconstruction ------------
# crds = gen_grd(im_size).to(torch_dev) # in [-1/2, 1/2]
# phis_ee   = torch.cat([crds.moveaxis(-1, 0), phis], dim=0)
# alphas_ee = torch.cat([trj.moveaxis(-1, 0), alphas], dim=0)
# bparams.coil_batch_size = C
# A_ee = expanded_encoding(mps, phis_ee, alphas_ee,
#                          dcf=dcf,
#                          spatial_mask=mask,
#                          temporal_batch_size=2**10,
#                          bparams=bparams,
#                          verbose=False)
# img_ee = CG_SENSE_recon(A_ee, ksp, **cg_kwargs)
# img_ee = normalize(img_ee, img_gt)
# torch.save(img_ee, f'./paper_experiments/highres_spiral/img_ee_R{R}.pt')
# quit()
img_ee = torch.load(f'./paper_experiments/highres_spiral/img_ee_R{R}.pt', 
                    map_location=torch_dev)
# img_ee = img_gt
img_ref = img_ee

# ------------ Sweep over (L, W) for TS NUFFT ------------
nL_nufft, nW_nufft = len(Ls_nufft), len(Ws_nufft)
results = {
    'nrmse_ts': torch.zeros(nL_nufft, nW_nufft),
    'decomp_time_ts': torch.zeros(nL_nufft, nW_nufft),
    'recon_time_ts': torch.zeros(nL_nufft, nW_nufft),
}
imgs_ts = torch.zeros(nL_nufft, nW_nufft, *im_size, dtype=torch.complex64)

for j, W in enumerate(tqdm(Ws_nufft, desc='Sweeping W', leave=True)):
    for i, L in enumerate(tqdm(Ls_nufft, desc='Sweeping L', leave=True)):
        hparams.kern_size = (W, W)
        hparams.L = L

        def decomp_ts():
            return time_seg_decomp_linop(phis, alphas, 
                                         mps=mps, trj=trj, dcf=dcf,
                                         spatial_mask=mask,
                                         normalize_coeffs=phase_norm,
                                        #  use_sigpy=True,    # use sigpy for the NUFFT
                                         hparams=hparams, bparams=bparams,)
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
        
        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()

# ------------ Sweep over (L, W) for HOFFT ------------
nL_hofft, nW_hofft = len(Ls_hofft), len(Ws_hofft)
results.update({
    'nrmse_hofft': torch.zeros(nL_hofft, nW_hofft),
    'decomp_time_hofft': torch.zeros(nL_hofft, nW_hofft),
    'recon_time_hofft': torch.zeros(nL_hofft, nW_hofft),
})
imgs_hofft = torch.zeros(nL_hofft, nW_hofft, *im_size, dtype=torch.complex64)

for j, W in enumerate(tqdm(Ws_hofft, desc='Sweeping W', leave=True)):
    for i, L in enumerate(tqdm(Ls_hofft, desc='Sweeping L', leave=True)):
        hparams.kern_size = (W, W)
        hparams.L = L

        def decomp_hofft():
            # sparams = sparse_params(Q=500, S=16, interp_type='lstsq', num_validation=100)
            sparams = None
            fact = 0 if 'alphas' in hparams.spatial_init else 1
            A =  hofft_decomp_linop(phis, alphas, 
                                    mps=mps, trj=trj, dcf=dcf, 
                                    hparams=hparams, 
                                    sparams=sparams,
                                    B_compressed=B_compressed,
                                    normalize_coeffs=phase_norm,
                                    num_als_iter=num_als_iter*fact,
                                    spatial_mask=mask,
                                    bparams=bparams)
            # torch.manual_seed(0)
            # A2 = hofft_decomp_linop(phis, alphas, 
            #                         mps=mps, trj=trj, dcf=dcf, 
            #                         hparams=hparams, 
            #                         sparams=sparams,
            #                         B_compressed=B_compressed,
            #                         num_als_iter=num_als_iter*0,
            #                         spatial_mask=mask,
            #                         bparams=bparams)
            # from mr_recon.utils import cvplot
            # cvplot(A.spatial_factors, A2.spatial_factors, A.spatial_factors / A2.spatial_factors)
            # quit()
            
            return A
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
        
        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()

# ------------ Save + Summarize ------------
save_path = './paper_experiments/highres_spiral/sweep_results.pt'
torch.save({
    **results,
    'img_uncorr': img_uncorr.cpu(),
    'imgs_ts': imgs_ts,
    'imgs_hofft': imgs_hofft,
    'img_ee': img_ee.cpu(),
    'img_gt': img_gt.cpu(),
    'Ls_nufft': Ls_nufft,
    'Ws_nufft': Ws_nufft,
    'Ls_hofft': Ls_hofft,
    'Ws_hofft': Ws_hofft,
    'os': os,
}, save_path)
print(f'Saved sweep results to {save_path}')
# quit()

# ------------ Plot: metrics vs L, one line per (method, W) ------------
# Color encodes method (TS NUFFT vs HOFFT); linestyle encodes kernel width W.
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
panels = [
    ('nrmse', 'NRMSE'), ('decomp_time', 'Decomp time [s]'), ('recon_time', 'Recon time [s]'),
]
methods = [
    ('ts', 'TS NUFFT', 'red', 'o', Ls_nufft, Ws_nufft),
    ('hofft', 'HOFFT', 'green', 's', Ls_hofft, Ws_hofft),
]
linestyles = ['-', '--', ':', '-.']
all_Ws = sorted(set(Ws_nufft) | set(Ws_hofft))
W_to_ls = {W: linestyles[i % len(linestyles)] for i, W in enumerate(all_Ws)}

nrmse_max = 0.33
for ax, (key, title) in zip(axes[:3], panels):
    for suffix, method_label, color, marker, Ls, Ws in methods:
        for j, W in enumerate(Ws):
            ax.plot(Ls, results[f'{key}_{suffix}'][:, j].numpy(),
                    color=color, linestyle=W_to_ls[W], marker=marker,
                    label=f'{method_label}, W={W}')
    ax.set_xlabel('L')
    ax.set_ylabel(title)
    ax.set_title(title)
    if key == 'nrmse':
        ax.set_ylim(0, nrmse_max)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both' if key == 'nrmse' else 'major')

# NRMSE vs recon time -- the accuracy/speed tradeoff, points connected in
# increasing-L order per (method, W) line.
ax = axes[3]
for suffix, method_label, color, marker, Ls, Ws in methods:
    for j, W in enumerate(Ws):
        total_time = results[f'decomp_time_{suffix}'][:, j].numpy() + \
                     results[f'recon_time_{suffix}'][:, j].numpy()
        ax.plot(total_time,
                results[f'nrmse_{suffix}'][:, j].numpy(),
                color=color, linestyle=W_to_ls[W], marker=marker,
                label=f'{method_label}, W={W}')
# ax.set_yscale('log')
ax.set_xlabel('Total time [s]')
ax.set_ylabel('NRMSE')
ax.set_title('NRMSE vs Total Time')
ax.set_xlim(0, 30)
ax.set_ylim(0, nrmse_max)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
extra = '_kalphas' if 'alphas' in hparams.spatial_init else ''
plt.savefig(f'./paper_experiments/highres_spiral/sweep_lineplots_{num_als_iter}{extra}.png', dpi=150)

# ------------ Plot: qualitative comparison at the largest (L, W) ------------
# from mr_recon.utils import cvplot
# cvplot('./config.yaml', img_uncorr.cpu(), img_ee.cpu(), imgs_ts[-1, -1], imgs_hofft[-1, -1])
# quit()
imgs = [img_uncorr.cpu(), img_ee.cpu(), imgs_ts[-1, -1], imgs_hofft[-1, -1]]
vmin = 0
img_ref = img_ref.cpu()
vmax = img_gt_cpu.abs().median() + 4 * img_gt_cpu.abs().std()
# pl, pr = 0.6, 0.85
# pu, pd = 0.35, 0.6
pl, pr = 0.0, 1.0
pu, pd = 0.0, 1.0
img_slc = (slice(round(pu*im_size[1]), round(pd*im_size[1])),
           slice(round(pl*im_size[0]), round(pr*im_size[0]))) 
titles = ['Uncorrected', 'Expanded Encoding (precomputed)', f'TS NUFFT (L={Ls_nufft[-1]}, W={Ws_nufft[-1]})', f'HOFFT (L={Ls_hofft[-1]}, W={Ws_hofft[-1]})']
plt.figure(figsize=(14, 7))
M = 5
for i, img in enumerate(imgs):
    img = normalize(img, img_ref)
    plt.subplot(2, len(imgs), i+1)
    plt.imshow(img.abs().cpu().rot90()[img_slc], cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title(titles[i])

    err = img.abs() - img_ref.abs()
    plt.subplot(2, len(imgs), i+1+len(imgs))
    plt.imshow(err.abs().cpu().rot90()[img_slc], cmap='gray', vmin=vmin/M, vmax=vmax/M)
    plt.axis('off')
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.show()
