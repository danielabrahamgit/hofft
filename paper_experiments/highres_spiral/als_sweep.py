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

from hofft.matvec import matvec_cur
from hofft.decomp import hofft_params
from hofft.utils import normalize
from hofft.pipelines import time_seg_decomp_linop, hofft_decomp_linop


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
W = 3   # HOFFT/TS NUFFT kernel width -- held fixed while sweeping ALS iters
L = 10  # number of spatial/temporal factors -- held fixed while sweeping ALS iters
num_als_iters = [1, 3, 5, 7, 10]  # HOFFT ALS iteration counts to sweep
num_time_reps = 3  # repeats per timed block, reduced via `time_reduction` to smooth GPU noise
time_reduction = 'mean'
mask_thresh = 0.9  # espirit eigenvalue threshold for the spatial support mask
B_compressed = 10  # number of SVD-compressed field bases to fit HOFFT against
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
im_size = mps.shape[1:]
C = mps.shape[0]
bparams = batching_params(coil_batch_size=C)

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

cg_kwargs = {'max_iter': 10, 'max_eigen': 1.0, 'verbose': False}
img_ref = img_gt

hparams = hofft_params((W,)*2, os, L,
                        reduced_im_size=(150, 150),
                        spatial_init='seg',
                        matvec_type=matvec_cur,
                        matvec_kwargs={
                            'rank_phi': 500,
                            'rank_alpha': 500,
                        },
                        verbose=False)
hparams.os = 2 * round(os * im_size[0] / 2) / im_size[0]  # round to nearest even oversampled grid

# ------------ TS NUFFT reference (doesn't depend on num_als_iter) ------------
nufft = sigpy_nufft(im_size, width=W)
nufft.beta = nufft.optimal_beta(torch_dev=torch_dev)

def decomp_ts():
    return time_seg_decomp_linop(phis, alphas,
                                 mps=mps, trj=trj, dcf=dcf,
                                 spatial_mask=mask,
                                 hparams=hparams, bparams=bparams)
decomp_time_ts, A_ts = time_repeated(decomp_ts, torch_dev, n_reps=num_time_reps, reduction=time_reduction)

# Untimed dummy pass: the very first CG_SENSE_recon call pays for one-time
# CUDA warm-up (context init, kernel/plan caching), which would otherwise
# inflate this specific timing.
CG_SENSE_recon(A_ts, ksp, **cg_kwargs)
if torch_dev.type == 'cuda':
    torch.cuda.synchronize()

def recon_ts():
    return normalize(CG_SENSE_recon(A_ts, ksp, **cg_kwargs), img_ref)
recon_time_ts, img_ts = time_repeated(recon_ts, torch_dev, n_reps=num_time_reps, reduction=time_reduction)
nrmse_ts = nrmse(img_ts, img_ref, mask=mask)
total_time_ts = decomp_time_ts + recon_time_ts

# ------------ Sweep over num_als_iter for HOFFT ------------
n = len(num_als_iters)
results = {
    'nrmse_hofft': torch.zeros(n),
    'decomp_time_hofft': torch.zeros(n),
    'recon_time_hofft': torch.zeros(n),
}

for i, num_als_iter in enumerate(tqdm(num_als_iters, desc='Sweeping num_als_iter')):

    def decomp_hofft():
        hparams.spatial_init = f'500_alphas_{num_als_iter}'
        return hofft_decomp_linop(phis, alphas,
                                  mps=mps, trj=trj, dcf=dcf,
                                  hparams=hparams,
                                  B_compressed=B_compressed,
                                  num_als_iter=num_als_iter*0,
                                  spatial_mask=mask,
                                  bparams=bparams)
    results['decomp_time_hofft'][i], A_hofft = time_repeated(
        decomp_hofft, torch_dev, n_reps=num_time_reps, reduction=time_reduction)

    if i == 0:
        # Same untimed warm-up pass as TS NUFFT above, for HOFFT's own
        # first-call kernels (its forward operator differs from A_ts's).
        CG_SENSE_recon(A_hofft, ksp, **cg_kwargs)
        if torch_dev.type == 'cuda':
            torch.cuda.synchronize()

    def recon_hofft():
        return normalize(CG_SENSE_recon(A_hofft, ksp, **cg_kwargs), img_ref)
    results['recon_time_hofft'][i], img_hofft = time_repeated(
        recon_hofft, torch_dev, n_reps=num_time_reps, reduction=time_reduction)
    results['nrmse_hofft'][i] = nrmse(img_hofft, img_ref, mask=mask)

results['total_time_hofft'] = results['decomp_time_hofft'] + results['recon_time_hofft']

# ------------ Save + Summarize ------------
save_path = './paper_experiments/highres_spiral/als_sweep_results.pt'
torch.save({
    **results,
    'num_als_iters': num_als_iters,
    'L': L,
    'W': W,
    'os': os,
    'nrmse_ts': nrmse_ts,
    'decomp_time_ts': decomp_time_ts,
    'recon_time_ts': recon_time_ts,
    'total_time_ts': total_time_ts,
}, save_path)
print(f'Saved ALS sweep results to {save_path}')

print(f'\nTS NUFFT (L={L}, W={W}): NRMSE={nrmse_ts:.4f}, '
      f'decomp={decomp_time_ts:.3f}s, recon={recon_time_ts:.3f}s, total={total_time_ts:.3f}s\n')
print(f'{"num_als_iter":>12} {"NRMSE":>10} {"decomp":>10} {"recon":>10} {"total":>10}')
for i, num_als_iter in enumerate(num_als_iters):
    print(f'{num_als_iter:>12} '
          f'{results["nrmse_hofft"][i]:>10.4f} '
          f'{results["decomp_time_hofft"][i]:>9.3f}s '
          f'{results["recon_time_hofft"][i]:>9.3f}s '
          f'{results["total_time_hofft"][i]:>9.3f}s')

# ------------ Plot: NRMSE and time vs num_als_iter, and NRMSE vs total time ------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(num_als_iters, results['nrmse_hofft'].numpy(), '-o', color='C0', label='HOFFT')
ax.axhline(nrmse_ts, color='red', linestyle='--', label='TS NUFFT')
ax.set_xlabel('num_als_iter')
ax.set_ylabel('NRMSE')
# ax.set_yscale('log')
ax.set_title(f'NRMSE vs ALS iterations (L={L}, W={W})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

ax = axes[1]
ax.plot(num_als_iters, results['decomp_time_hofft'].numpy(), '-o', color='green', label='HOFFT decomp')
ax.plot(num_als_iters, results['total_time_hofft'].numpy(), '-s', color='green', label='HOFFT total')
ax.axhline(total_time_ts, color='red', linestyle='--', label='TS NUFFT total')
ax.set_xlabel('num_als_iter')
ax.set_ylabel('Time [s]')
ax.set_title(f'Time vs ALS iterations (L={L}, W={W})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# NRMSE vs total (decomp + recon) time -- the accuracy/speed tradeoff as ALS
# iterations increase, points connected in increasing-num_als_iter order.
ax = axes[2]
ax.plot(results['total_time_hofft'].numpy(), results['nrmse_hofft'].numpy(),
        '-o', color='green', label='HOFFT')
for x, y, k in zip(results['total_time_hofft'].numpy(), results['nrmse_hofft'].numpy(), num_als_iters):
    ax.annotate(str(k), (x, y), textcoords='offset points', xytext=(4, 4), fontsize=8)
ax.plot(total_time_ts, nrmse_ts, '*', color='red', markersize=14, label='TS NUFFT')
# ax.set_yscale('log')
ax.set_xlabel('Total time (decomp + recon) [s]')
ax.set_ylabel('NRMSE')
ax.set_title('NRMSE vs total time')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('./paper_experiments/highres_spiral/als_sweep_plots.png', dpi=150)
print('Saved plot to ./paper_experiments/highres_spiral/als_sweep_plots.png')
plt.show()
