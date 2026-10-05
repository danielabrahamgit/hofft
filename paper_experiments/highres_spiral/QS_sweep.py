"""
Sparse HOFFT (Q, S) sweep on highres_spiral.

Same data / hparams / recon setup as run.py; varies only sparams.Q and sparams.S.
"""
import gc
import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from time import perf_counter
from typing import Optional
from tqdm import tqdm

from mr_recon.recons import CG_SENSE_recon
from mr_recon.linops import batching_params
from mr_recon.algs import density_compensation

from hofft.matvec import matvec_cur
from hofft.decomp import hofft_params
from hofft.utils import normalize
from hofft.pipelines import hofft_decomp_linop
from hofft.sparse_fit import sparse_params


class GPUTimer:
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
            self.elapsed = self._start_evt.elapsed_time(self._end_evt) / 1000
        else:
            self.elapsed = perf_counter() - self._t0
        return False


def time_repeated(fn, torch_dev, n_reps=1, reduction='mean'):
    times = []
    result = None
    for _ in range(n_reps):
        with GPUTimer(torch_dev) as t:
            result = fn()
        times.append(t.elapsed)
    times = torch.tensor(times)
    elapsed = {'mean': times.mean, 'median': times.median, 'min': times.min}[reduction]().item()
    return elapsed, result


def nrmse(img, ref, mask=None):
    if mask is not None:
        img = img * mask
        ref = ref * mask
    return (torch.linalg.norm(img.abs() - ref.abs()) / torch.linalg.norm(ref.abs())).item()


# ---------------------------------------------------------------------------
# Same setup as run.py (fixed L, W; sweep Q, S)
# ---------------------------------------------------------------------------
os = 1.25
L = 20
W = 3
Qs = [200, 500, 1000, 2500]
Ss = [1, 4, 8, 16, 32]
num_als_iter = 10
num_time_reps = 1
time_reduction = 'mean'
mask_thresh = 0.9
B_compressed = 10
interp_type = 'lstsq'
num_validation = 100
cg_kwargs = {'max_iter': 10, 'max_eigen': 1.0, 'verbose': False}

hparams = hofft_params(
    (W, W), os, L,
    reduced_im_size=(100 * 2, 100 * 2),
    spatial_init='seg',
    matvec_type=matvec_cur,
    matvec_kwargs={'rank_phi': 500, 'rank_alpha': 500},
    verbose=False,
)
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

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
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0]
bparams = batching_params(coil_batch_size=C // 2)

R = 1
trj = trj[:, ::R]
ksp = ksp[:, :, ::R]
alphas = alphas[:, :, ::R]
dcf = density_compensation(trj, im_size)

mask = (evals > mask_thresh).float()
mps = mps * mask

B = phis.shape[0]
energy = (phis.reshape((B, -1)).abs().mean(dim=1)
          * alphas.reshape((B, -1)).abs().mean(dim=1))
idxs = torch.argwhere(energy > 1e-6)[:, 0]
phis, alphas = phis[idxs], alphas[idxs]

img_ee = torch.load(
    f'./paper_experiments/highres_spiral/img_ee_R{R}.pt',
    map_location=torch_dev,
)
img_ref = img_ee

# ---------------------------------------------------------------------------
# Sweep (Q, S)
# ---------------------------------------------------------------------------
nQ, nS = len(Qs), len(Ss)
results = {
    'nrmse': torch.zeros(nQ, nS),
    'decomp_time': torch.zeros(nQ, nS),
    'recon_time': torch.zeros(nQ, nS),
}
imgs = torch.zeros(nQ, nS, *im_size, dtype=torch.complex64)
fact = 0 if 'alphas' in hparams.spatial_init else 1
warmed_up = False

for i, Q in enumerate(tqdm(Qs, desc='Sweeping Q')):
    for j, S in enumerate(tqdm(Ss, desc=f'Q={Q} S', leave=False)):
        assert S <= Q, f'S={S} must be <= Q={Q}'

        def decomp():
            sparams = sparse_params(
                Q=Q, S=S, interp_type=interp_type, num_validation=num_validation,
            )
            return hofft_decomp_linop(
                phis, alphas,
                mps=mps, trj=trj, dcf=dcf,
                hparams=hparams,
                sparams=sparams,
                B_compressed=B_compressed,
                normalize_coeffs=True,
                num_als_iter=num_als_iter * fact,
                spatial_mask=mask,
                bparams=bparams,
            )

        results['decomp_time'][i, j], A = time_repeated(
            decomp, torch_dev, n_reps=num_time_reps, reduction=time_reduction)

        if not warmed_up:
            CG_SENSE_recon(A, ksp, **cg_kwargs)
            if torch_dev.type == 'cuda':
                torch.cuda.synchronize()
            warmed_up = True

        def recon():
            return normalize(CG_SENSE_recon(A, ksp, **cg_kwargs), img_ref)

        results['recon_time'][i, j], img = time_repeated(
            recon, torch_dev, n_reps=num_time_reps, reduction=time_reduction)
        results['nrmse'][i, j] = nrmse(img, img_ref, mask=mask)
        imgs[i, j] = img.cpu()

        print(
            f'Q={Q:<5d} S={S:<3d}: '
            f'NRMSE={results["nrmse"][i, j]:.4f}, '
            f'decomp={results["decomp_time"][i, j]:.2f}s, '
            f'recon={results["recon_time"][i, j]:.2f}s'
        )
        gc.collect()
        torch.cuda.empty_cache()

save_path = './paper_experiments/highres_spiral/QS_sweep_results.pt'
torch.save({
    **results,
    'imgs': imgs,
    'Qs': Qs,
    'Ss': Ss,
    'L': L,
    'W': W,
    'interp_type': interp_type,
    'os': hparams.os,
}, save_path)
print(f'Saved {save_path}')

# ---------------------------------------------------------------------------
# Plots (slide-friendly styling, same as plot_ablations.py)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'axes.linewidth': 2.0,
    'lines.linewidth': 4.0,
    'lines.markersize': 14,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

fig, axes = plt.subplots(1, 4, figsize=(28, 6.5))
Ss_arr = torch.tensor(Ss, dtype=torch.float32).numpy()
plot_kw = dict(lw=4.5, ms=16, mew=2.0, markeredgecolor='white', zorder=3)
q_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

for i, Q in enumerate(Qs):
    color = q_colors[i % len(q_colors)]
    label = f'Q={Q}'
    nrmse_q = results['nrmse'][i].numpy()
    decomp_q = results['decomp_time'][i].numpy()
    recon_q = results['recon_time'][i].numpy()
    total_q = decomp_q + recon_q

    axes[0].plot(Ss_arr, nrmse_q, color=color, marker='o', label=label,
                 markerfacecolor=color, **plot_kw)
    axes[1].plot(Ss_arr, decomp_q, color=color, marker='o', label=label,
                 markerfacecolor=color, **plot_kw)
    axes[2].plot(Ss_arr, recon_q, color=color, marker='o', label=label,
                 markerfacecolor=color, **plot_kw)
    axes[3].plot(total_q, nrmse_q, color=color, marker='o', label=label,
                 markerfacecolor=color, **plot_kw)

panels = [
    (axes[0], 'S', 'NRMSE', 'NRMSE vs S'),
    (axes[1], 'S', 'Decomp time [s]', 'Decomp time vs S'),
    (axes[2], 'S', 'Recon time [s]', 'Recon time vs S'),
    (axes[3], 'Total time [s]', 'NRMSE', 'NRMSE vs total time'),
]
for ax, xlabel, ylabel, title in panels:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, which='major', alpha=0.35, lw=1.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    leg = ax.legend(
        loc='best', frameon=True, fancybox=False,
        edgecolor='0.4', framealpha=0.95, borderpad=0.5,
        handlelength=2.0, markerscale=1.05,
    )
    leg.get_frame().set_linewidth(1.5)

fig.suptitle(
    f'Sparse HOFFT  (L={L}, W={W}, {interp_type})',
    fontsize=28, fontweight='bold', y=1.02,
)
fig.tight_layout()
out_png = './paper_experiments/highres_spiral/QS_sweep_plots.png'
fig.savefig(out_png, dpi=200)
print(f'Saved {out_png}')
