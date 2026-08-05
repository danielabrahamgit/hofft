"""
Compare matvec_naive (GT) vs matvec_rnd / matvec_cur:
NRMSE (vs naive) vs forward evaluation time on coco_spiral phis/alphas.
"""
import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from time import perf_counter
from typing import Optional

from einops import einsum

from hofft.matvec import matvec_naive, matvec_rnd, matvec_cur, matvec_rnd_fast
from hofft.phase_coeffs import rescale_phis_alphas, compress_phis_alphas
from hofft.utils import reduce_spatial


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


def nrmse(y: torch.Tensor, y_ref: torch.Tensor) -> float:
    return (torch.linalg.norm(y - y_ref) / torch.linalg.norm(y_ref)).item()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
fpath = './data/coco_spiral'
kwargs = {'weights_only': True, 'map_location': torch_dev}

phis = torch.load(f'{fpath}/phis.pt', **kwargs).float()
alphas = torch.load(f'{fpath}/alphas.pt', **kwargs).float()

# Downsample phis
phis = reduce_spatial(phis, (100, 100), order=3)

# Drop negligible bases
B = phis.shape[0]
energy = phis.reshape(B, -1).abs().mean(1) * alphas.reshape(B, -1).abs().mean(1)
keep = torch.argwhere(energy > 1e-6)[:, 0]
phis, alphas = phis[keep], alphas[keep]

# Same rescale HOFFT uses before matvecs (relative approx quality is invariant
# to global midpoint phases, so we ignore midpoints here).
phis, _, alphas, _ = rescale_phis_alphas(phis, alphas, whiten=False)
phis, alphas = compress_phis_alphas(phis, alphas, B_compressed=phis.shape[0])

print(f'phis {tuple(phis.shape)}, alphas {tuple(alphas.shape)} on {torch_dev}')

# Random input image (N=1 batch)
x = torch.randn((1, *phis.shape[1:]), device=torch_dev, dtype=torch.complex64)

# Batch sizes keep naive's temporary R x T blocks manageable
n_reps = 100

# ---------------------------------------------------------------------------
# Ground truth: matvec_naive
# ---------------------------------------------------------------------------
mv_naive = matvec_naive(
    phis, alphas,
)
# Warmup
_ = mv_naive(x)
if torch_dev.type == 'cuda':
    torch.cuda.synchronize()

t_naive, y_ref = time_repeated(lambda: mv_naive(x), torch_dev, n_reps=n_reps)
print(f'naive:  time={t_naive*1e3:.2f} ms')

results = {
    'naive': {'times': [t_naive], 'nrmses': [0.0], 'labels': ['naive']},
    'rnd': {'times': [], 'nrmses': [], 'labels': []},
    'cur': {'times': [], 'nrmses': [], 'labels': []},
}
# ---------------------------------------------------------------------------
# matvec_rnd sweep (forward uses rnd_frac_phis)
# ---------------------------------------------------------------------------
rnd_fracs = [0.01, 0.1, 0.5, 0.75]
for frac in rnd_fracs:
    mv = matvec_rnd_fast(phis, alphas, 
                         rnd_frac_phis=frac, rnd_frac_alphas=frac)
    _ = mv(x)
    if torch_dev.type == 'cuda':
        torch.cuda.synchronize()
    # Average NRMSE over a few redraws; time a single forward (median of reps)
    nrmses = []
    for _rep in range(n_reps):
        y = mv(x)
        nrmses.append(nrmse(y, y_ref))
    t, _ = time_repeated(lambda: mv(x), torch_dev, n_reps=n_reps)
    err = sum(nrmses) / len(nrmses)
    results['rnd']['times'].append(t)
    results['rnd']['nrmses'].append(err)
    results['rnd']['labels'].append(f'frac={frac:g}')
    print(f'rnd frac={frac:<4g}: time={t*1e3:.2f} ms, NRMSE={err:.3e}')

# ---------------------------------------------------------------------------
# matvec_cur sweep
# ---------------------------------------------------------------------------
cur_ranks = [10, 25, 50, 100, 200, 400, 800]
for rank in cur_ranks:
    mv = matvec_cur(phis, alphas, cur_rank=rank)
    _ = mv(x)
    if torch_dev.type == 'cuda':
        torch.cuda.synchronize()
    t, y = time_repeated(lambda: mv(x), torch_dev, n_reps=n_reps)
    err = nrmse(y, y_ref)
    results['cur']['times'].append(t)
    results['cur']['nrmses'].append(err)
    results['cur']['labels'].append(f'rank={rank}')
    print(f'cur  rank={rank:<4d}: time={t*1e3:.2f} ms, NRMSE={err:.3e}')

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
styles = {
    'naive': dict(color='C0', marker='*', markersize=14, linestyle='none', label='naive'),
    'rnd': dict(color='C1', marker='o', linestyle='-', label='rnd'),
    'cur': dict(color='C2', marker='s', linestyle='-', label='cur'),
}
for key, style in styles.items():
    ax.plot(
        [t * 1e3 for t in results[key]['times']],
        results[key]['nrmses'],
        **style,
    )
    # Annotate sweep points lightly
    if key != 'naive':
        for t, e, lab in zip(results[key]['times'], results[key]['nrmses'], results[key]['labels']):
            ax.annotate(lab, (t * 1e3, e), textcoords='offset points',
                        xytext=(4, 4), fontsize=17, alpha=0.8)
            
# Vertical line for naive 
ax.axvline(x=t_naive * 1e3, color='C0', linestyle='--', alpha=0.5)
ax.annotate('naive', (t_naive * 1e3, 1e-6), textcoords='offset points',
            xytext=(4, 4), fontsize=7, alpha=0.8)
ax.set_xlabel('Forward time [ms]')
ax.set_ylabel('NRMSE vs naive')
ax.set_title('coco_spiral matvec: NRMSE vs forward time')
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid(True, which='both', alpha=0.3)
# make lines thicker and markers bigger and text bigger
for line in ax.lines:
    line.set_linewidth(5)
    line.set_markersize(10)
ax.tick_params(axis='both', which='major', labelsize=24)
# ax.legend()
fig.tight_layout()

out_png = './paper_experiments/matvec_analysis/nrmse_vs_time.png'
out_pt = './paper_experiments/matvec_analysis/results.pt'
fig.savefig(out_png, dpi=150)
torch.save(results, out_pt)
print(f'Saved {out_png}')
print(f'Saved {out_pt}')
plt.show()
