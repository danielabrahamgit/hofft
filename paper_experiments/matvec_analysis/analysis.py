"""
Compare three approximations of the high-order phase matrix
P[r,t] = exp(-j 2π φ(r) · α(t)) on coco_spiral:

1. CUR without SVD whitening — cluster on the rescaled φ/α, P ≈ C R
2. CUR with phase whitening — cluster in (Σ V^T, Σ U^T), P ≈ C_w R_w
3. Spatial / temporal downsample by (Rn, Rm) — reconstruct via
   φ'' = upsamp(downsamp(φ)), α'' = upsamp(downsamp(α))

Accuracy is ||P - P_hat||_F^2 / ||P||_F^2 on a fixed random subset of
spatial and temporal indices.
Timing is the cheap apply: C R (or C_w R_w) for CUR, and
exp(-j 2π φ' · α') on the downsampled grid for (3).
"""
import gc
import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from time import perf_counter
from tqdm import tqdm
from einops import einsum

from hofft.matvec import matvec_naive, matvec_cur
from hofft.phase_coeffs import rescale_phis_alphas, whiten_phis_alphas
from hofft.utils import reduce_spatial, expand_spatial, reduce_temporal, expand_temporal


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

def time_repeated(fn, torch_dev, n_reps=5, reduction='median'):
    times = []
    result = None
    for _ in range(n_reps):
        with GPUTimer(torch_dev) as t:
            result = fn()
        times.append(t.elapsed)
    times = torch.tensor(times)
    elapsed = {'mean': times.mean, 'median': times.median, 'min': times.min}[reduction]().item()
    return elapsed, result

def rel_frob_sq(P_hat: torch.Tensor, P_ref: torch.Tensor) -> float:
    """||P - P_hat||_F^2 / ||P||_F^2 on the sampled block."""
    return ((P_hat - P_ref).abs().square().sum() / P_ref.abs().square().sum()).item()


def phase_block(phis, alphas, r_idx, t_idx) -> torch.Tensor:
    """P[r_idx, t_idx] = exp(-j 2π φ(r) · α(t)), shape (n_s, n_t)."""
    B = phis.shape[0]
    phi_s = phis.reshape(B, -1)[:, r_idx]
    alpha_t = alphas.reshape(B, -1)[:, t_idx]
    return torch.exp(-2j * torch.pi * (phi_s.T @ alpha_t))


def cur_block(R, C, r_idx, t_idx) -> torch.Tensor:
    """P_hat[r, t] = sum_k R[k, r] C[k, t] on the sampled indices."""
    return einsum(R[:, r_idx], C[:, t_idx], 'K Ns, K Nt -> Ns Nt')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
# fpath = './data/coco_spiral'
fpath = './data/highres_spiral'
kwargs = {'weights_only': True, 'map_location': torch_dev}

phis = torch.load(f'{fpath}/phis.pt', **kwargs).float()
alphas = torch.load(f'{fpath}/alphas.pt', **kwargs).float()[:, :, 0]
evals = torch.load(f'{fpath}/evals.pt', **kwargs).float()
mask = (evals > 0.9).float()

# Remove low-energy bases
B = phis.shape[0]
energy = phis.reshape(B, -1).abs().mean(1) * alphas.reshape(B, -1).abs().mean(1)
keep = torch.argwhere(energy > 1e-6)[:, 0]
phis, alphas = phis[keep], alphas[keep]

# # Compress
# phis, alphas = whiten_phis_alphas(phis, alphas, B_compressed=10)
# B = phis.shape[0]

# Consts
im_size = phis.shape[1:]
trj_size = alphas.shape[1:]
alphas = alphas.reshape(alphas.shape[0], -1)
B, T = alphas.shape[0], alphas.shape[1]
N = int(torch.tensor(im_size).prod().item())

print(f'phis {tuple(phis.shape)}, alphas {tuple(alphas.shape)} on {torch_dev}')

x = torch.randn((1, *im_size), device=torch_dev, dtype=torch.complex64)
n_reps = 10
n_spatial = min(2048, N)
n_temporal = min(2048, T)
cur_ranks = [10, 25, 50, 100, 200, 400, 800]
ds_factors = [(i,j) for i in range(1, 5) for j in range(10, 101, 30)]

r_idx = torch.randperm(N, device=torch_dev)[:n_spatial]
t_idx = torch.randperm(T, device=torch_dev)[:n_temporal]
P_ref = phase_block(phis, alphas, r_idx, t_idx)
print(f'error subset: {n_spatial} spatial x {n_temporal} temporal')

results = {
    'cur': {'times': [], 'err_f2': [], 'labels': [], 'ranks': []},
    'cur_white': {'times': [], 'err_f2': [], 'labels': [], 'ranks': []},
    'downsample': {'times': [], 'err_f2': [], 'labels': [], 'Rn': [], 'Rm': []},
    'N': N,
    'T': T,
}

# ---------------------------------------------------------------------------
# 1) CUR without phase whitening
# ---------------------------------------------------------------------------
cluster_method = 'maxmin'
for rank in tqdm(cur_ranks, desc='CUR (no whitening)'):
    mv = matvec_cur(phis, alphas.reshape(B, T), cur_rank=rank, cluster_method=cluster_method, normalize_coeffs=False)
    _ = mv.normal(x)
    if torch_dev.type == 'cuda':
        torch.cuda.synchronize()
    t, _ = time_repeated(lambda: mv.normal(x), torch_dev, n_reps=n_reps)
    R = mv.curR.reshape(mv.curR.shape[0], -1)
    C = mv.curC.reshape(mv.curC.shape[0], -1)
    err = rel_frob_sq(cur_block(R, C, r_idx, t_idx), P_ref)
    results['cur']['times'].append(t)
    results['cur']['err_f2'].append(err)
    results['cur']['labels'].append(f'k={rank}')
    results['cur']['ranks'].append(rank)
    print(f'cur        k={rank:<4d}: time={t*1e3:.2f} ms, rel F^2={err:.4e}')
    del R, C, mv
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 2) CUR with phase whitening
# ---------------------------------------------------------------------------
for rank in tqdm(cur_ranks, desc='CUR (whitened)'):
    mv = matvec_cur(phis, alphas.reshape(B, T), cur_rank=rank, cluster_method=cluster_method, normalize_coeffs=True,)
    _ = mv.normal(x)
    if torch_dev.type == 'cuda':
        torch.cuda.synchronize()
    t, _ = time_repeated(lambda: mv.normal(x), torch_dev, n_reps=n_reps)
    R = mv.curR.reshape(mv.curR.shape[0], -1)
    C = mv.curC.reshape(mv.curC.shape[0], -1)
    err = rel_frob_sq(cur_block(R, C, r_idx, t_idx), P_ref)
    results['cur_white']['times'].append(t)
    results['cur_white']['err_f2'].append(err)
    results['cur_white']['labels'].append(f'k={rank}')
    results['cur_white']['ranks'].append(rank)
    print(f'cur_white  k={rank:<4d}: time={t*1e3:.2f} ms, rel F^2={err:.4e}')
    del R, C, mv
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 3) Spatial / temporal downsample
# ---------------------------------------------------------------------------
for Rn, Rm in tqdm(ds_factors, desc='Downsample'):
    im_low = tuple(max(2, s // Rn) for s in im_size)
    T_low = max(2, trj_size[0] // Rm)
    phis_ds = reduce_spatial(phis, im_low, order=3)
    phis_up = expand_spatial(phis_ds, im_size, order=3)
    alphas_reshaped = alphas.reshape(B, *trj_size)
    alphas_ds = reduce_temporal(alphas_reshaped, T_low, dim=1, order=3)
    alphas_up = expand_temporal(alphas_ds, trj_size[0], dim=1, order=3)
    alphas_up = alphas_up.reshape(B, -1)
    
    x_ds = reduce_spatial(x, im_low, order=3)
    mv = matvec_naive(phis_ds, alphas_ds)
    _ = mv.normal(x_ds)
    if torch_dev.type == 'cuda':
        torch.cuda.synchronize()
    t, _ = time_repeated(lambda: mv.normal(x_ds), torch_dev, n_reps=n_reps)
    err = rel_frob_sq(phase_block(phis_up, alphas_up, r_idx, t_idx), P_ref)
    results['downsample']['times'].append(t)
    results['downsample']['err_f2'].append(err)
    results['downsample']['labels'].append(f'Rn={Rn}, Rm={Rm}')
    results['downsample']['Rn'].append(Rn)
    results['downsample']['Rm'].append(Rm)
    print(f'downsample Rn={Rn:<2d} Rm={Rm:<2d}: time={t*1e3:.2f} ms, '
          f'rel F^2={err:.4e}  grid={im_low}+{T_low}')
    del mv
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14,
    'axes.linewidth': 2.0,
    'lines.linewidth': 4.0,
    'lines.markersize': 14,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

fig, ax = plt.subplots(figsize=(14, 7))
cur_color = '#2ca02c'
ds_color = '#ff7f0e'
ds_linestyles = ['-', '--', '-.', ':']

ax.plot(
    [t * 1e3 for t in results['cur']['times']],
    results['cur']['err_f2'],
    color=cur_color, marker='s', linestyle='--', label='CUR',
)
ax.plot(
    [t * 1e3 for t in results['cur_white']['times']],
    results['cur_white']['err_f2'],
    color=cur_color, marker='D', linestyle='-', label='CUR + whiten',
)

Rn_vals = sorted(set(results['downsample']['Rn']))
for i, Rn in enumerate(Rn_vals):
    idxs = [j for j, r in enumerate(results['downsample']['Rn']) if r == Rn]
    idxs = sorted(idxs, key=lambda j: results['downsample']['Rm'][j])
    ax.plot(
        [results['downsample']['times'][j] * 1e3 for j in idxs],
        [results['downsample']['err_f2'][j] for j in idxs],
        color=ds_color,
        linestyle=ds_linestyles[i % len(ds_linestyles)],
        marker='o',
        label=fr'downsample $R_n$={Rn}',
    )

ax.set_xlabel('Apply time [ms]')
ax.set_ylabel(r'$\|P - \widehat{P}\|_F^2 / \|P\|_F^2$')
ax.set_title('coco_spiral phase matrix: error vs apply time')
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
fig.tight_layout()

out_png = './paper_experiments/matvec_analysis/nrmse_vs_time.png'
out_pt = './paper_experiments/matvec_analysis/results.pt'
fig.savefig(out_png, dpi=150)
torch.save(results, out_pt)
print(f'Saved {out_png}')
print(f'Saved {out_pt}')
plt.show()
