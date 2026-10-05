"""
Sweep spatial / temporal downsample (Rn, Rm). For each coarse grid, compare:

  - no CUR:        exp(-j 2π φ' · α') on the coarse grid
  - CUR:           no coefficient normalization
  - CUR + cov:     ZCA-whiten each side (pivot metric)

Speed is the coarse-grid apply (naive or CUR).
CUR rank is swept; all ranks for a given (Rn, Rm) share one marker.
Accuracy is ||P - P_hat||_F^2 / ||P||_F^2 on a fixed random (r, t) subset,
where P_hat comes from the downsampled-then-upsampled coefficients φ'', α'':
  - no CUR:        P_hat = exp(-j 2π φ'' · α'')
  - CUR methods:   P_hat = C R built from (φ'', α'')

Replot a saved run without the sweep:

  python analysis2.py --plot-only
  python analysis2.py --plot-only --results path/to/analysis2_results.pt

Writes analysis2_nrmse_vs_time.png (x = apply time [ms]) and
analysis2_nrmse_vs_memory.png (x = coefficient memory [GB]).
Both have y = relative F^2.
"""
import argparse
import gc
import colorsys
import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable

from time import perf_counter
from tqdm import tqdm
from einops import einsum

from hofft.matvec import matvec_naive, matvec_cur
from hofft.utils import reduce_spatial, expand_spatial, reduce_temporal, expand_temporal
from hofft.phase_coeffs import whiten_phis_alphas, remove_linear_terms, visualize_alpha_space, rescale_phis_alphas


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


def time_repeated(fn, torch_dev, n_reps=5, reduction='min'):
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
    err = (P_hat * mask - P_ref * mask).norm().square()
    return (err / (P_ref * mask).norm().square()).item()
    # return (((P_hat - P_ref) * mask).abs().sum() / (P_ref * mask).abs().sum()).item()


def phase_block(phis, alphas, r_idx, t_idx) -> torch.Tensor:
    B = phis.shape[0]
    phi_s = phis.reshape(B, -1)[:, r_idx]
    alpha_t = alphas.reshape(B, -1)[:, t_idx]
    return torch.exp(-2j * torch.pi * (phi_s.T @ alpha_t))


def cur_block(R, C, r_idx, t_idx) -> torch.Tensor:
    return einsum(R[:, r_idx], C[:, t_idx], 'K Ns, K Nt -> Ns Nt')


def empty_result():
    return {'times': [], 'err_f2': [], 'labels': [], 'Rn': [], 'Rm': [], 'ranks': [],
            'N_low': [], 'T_ds': []}


def pairs_from_results(res):
    if res.get('ds'):
        return [tuple(p) for p in res['ds']]
    pairs = []
    skip = {'N', 'T', 'cur_ranks', 'ds', 'im_size', 'trj_size'}
    for key, rec in res.items():
        if key in skip or not isinstance(rec, dict) or 'Rn' not in rec:
            continue
        for r, m in zip(rec['Rn'], rec['Rm']):
            p = (int(r), int(m))
            if p not in pairs:
                pairs.append(p)
    return pairs


# (results key, normalize_method, linestyle, legend label)
CUR_SPECS = [
    ('cur', '', ':', 'CUR'),
    ('cur_svd', 'svd', '-', 'CUR + SVD'),
]
PIVOT_SEED = 0


parser = argparse.ArgumentParser()
parser.add_argument('--plot-only', action='store_true',
                    help='Skip the sweep and plot from --results')
parser.add_argument('--results',
                    default='./paper_experiments/matvec_analysis/analysis2_results.pt')
args = parser.parse_args()
out_png = './paper_experiments/matvec_analysis/analysis2_nrmse_vs_time.png'
out_png_mem = './paper_experiments/matvec_analysis/analysis2_nrmse_vs_memory.png'
out_pt = args.results

if args.plot_only:
    results = torch.load(out_pt, map_location='cpu', weights_only=False)
    ds = pairs_from_results(results)
    print(f'Loaded {out_pt}, downsample {ds}')
else:
    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    # fpath = './data/tilt_spi_invivo'
    # fpath = './data/tilt_spi'
    fpath = './data/axial_spi_invivo'
    # fpath = './data/highres_spiral'
    kwargs = {'weights_only': True, 'map_location': torch_dev}
    
    phis = torch.load(f'{fpath}/phis.pt', **kwargs).float()
    alphas = torch.load(f'{fpath}/alphas.pt', **kwargs).float()#[:, :, 0]
    evals = torch.load(f'{fpath}/evals.pt', **kwargs).float()
    mask = (evals > 0.9).float()
    # phis *= mask
    
    # Remove first order terms
    phis, trj_term, zeroth_order = remove_linear_terms(phis, alphas, mask=mask)
    
    B = phis.shape[0]
    energy = phis.reshape(B, -1).abs().mean(1) * alphas.reshape(B, -1).abs().mean(1)
    keep = torch.argwhere(energy > 1e-5)[:, 0]
    phis, alphas = phis[keep], alphas[keep]
    
    # # Compress
    # phis, alphas = whiten_phis_alphas(phis, alphas, B_compressed=7)
    # B = phis.shape[0]
    
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    alphas_flat = alphas.reshape(alphas.shape[0], -1)
    B, T = alphas_flat.shape[0], alphas_flat.shape[1]
    N = int(torch.tensor(im_size).prod().item())
    
    print(f'phis {tuple(phis.shape)}, alphas {tuple(alphas.shape)} on {torch_dev}')
    
    x = torch.randn((1, *im_size), device=torch_dev, dtype=torch.complex64)
    n_reps = 10
    n_spatial = min(2048, N)
    n_temporal = min(2048, T)
    cur_ranks = list(range(120, 701, 20))
    spatial_batch_size = 2 ** 14
    cluster_method = 'maxmin'
    ds = [(2, 10), (3, 50), (4, 100)]

    r_idx = torch.randperm(N, device=torch_dev)[:n_spatial]
    t_idx = torch.randperm(T, device=torch_dev)[:n_temporal]
    mask = mask.flatten()[r_idx, None]
    P_ref = phase_block(phis, alphas_flat, r_idx, t_idx)
    print(f'error subset: {n_spatial} spatial x {n_temporal} temporal')
    print(f'CUR ranks {cur_ranks}, downsample {ds}')

    results = {
        'naive': empty_result(),
        **{key: empty_result() for key, *_ in CUR_SPECS},
        'N': N,
        'T': T,
        'cur_ranks': cur_ranks,
        'ds': ds,
        'im_size': tuple(im_size),
        'trj_size': tuple(trj_size),
    }

    def record(key, t, err, Rn, Rm, rank=None, N_low=None, T_ds=None):
        results[key]['times'].append(t)
        results[key]['err_f2'].append(err)
        results[key]['labels'].append(f'Rn={Rn}, Rm={Rm}')
        results[key]['Rn'].append(Rn)
        results[key]['Rm'].append(Rm)
        results[key]['ranks'].append(rank)
        results[key]['N_low'].append(N_low)
        results[key]['T_ds'].append(T_ds)

    def time_normal(mv, x_in):
        _ = mv.normal(x_in)
        if torch_dev.type == 'cuda':
            torch.cuda.synchronize()
        t, _ = time_repeated(lambda: mv.normal(x_in), torch_dev, n_reps=n_reps)
        return t

    def reseed():
        torch.manual_seed(PIVOT_SEED)
        if torch_dev.type == 'cuda':
            torch.cuda.manual_seed_all(PIVOT_SEED)

    def cur_err_on_upsampled(phis_up, alphas_up, normalize_method, k):
        reseed()
        mv_acc = matvec_cur(
            phis_up, alphas_up, cur_rank=k,
            cluster_method=cluster_method,
            normalize_method=normalize_method,
            seed=PIVOT_SEED,
        )
        R = mv_acc.curR.reshape(mv_acc.curR.shape[0], -1)
        C = mv_acc.curC.reshape(mv_acc.curC.shape[0], -1)
        err = rel_frob_sq(cur_block(R, C, r_idx, t_idx), P_ref)
        del mv_acc, R, C
        return err

    # ---------------------------------------------------------------------------
    # Sweep
    # ---------------------------------------------------------------------------
    for Rn, Rm in tqdm(ds, desc='Downsample sweep'):
        im_low = tuple(max(2, s // Rn) for s in im_size)
        T_low = max(2, trj_size[0] // Rm)
        phis_ds = reduce_spatial(phis, im_low, order=3)
        phis_up = expand_spatial(phis_ds, im_size, order=3)
        alphas_ds = reduce_temporal(alphas.reshape(B, *trj_size), T_low, dim=1, order=3)
        alphas_up = expand_temporal(alphas_ds, trj_size[0], dim=1, order=3).reshape(B, -1)
        x_ds = reduce_spatial(x, im_low, order=3)
        N_low = phis_ds[0].numel()
        T_ds = alphas_ds.reshape(B, -1).shape[1]

        # no CUR (one point per (Rn, Rm))
        mv = matvec_naive(phis_ds, alphas_ds, spatial_batch_size=spatial_batch_size)
        t = time_normal(mv, x_ds)
        err = rel_frob_sq(phase_block(phis_up, alphas_up, r_idx, t_idx), P_ref)
        record('naive', t, err, Rn, Rm, N_low=N_low, T_ds=T_ds)
        print(f'naive      Rn={Rn} Rm={Rm}: time={t*1e3:.2f} ms, rel F^2={err:.4e}')
        del mv
        
        # # Uhhhhh whats going on here
        # ts = torch.randperm(alphas.shape[1])[:2].sort().values
        # alphas_prime = alphas_up
        # phis_prime = phis_up
        # mv = matvec_cur(phis_prime, alphas_prime, cur_rank=cur_ranks[-1],
        #                 cluster_method=cluster_method, normalize_coeffs=True)
        # phz_cur = einsum(mv.curR, mv.curC[:, ts], 'K ..., K T -> T ...')
        # phz_actual = torch.exp(-2j * torch.pi * einsum(phis, alphas[:, ts], 'B ..., B T -> T ...'))
        # for i in range(len(ts)):
        #     plt.figure(figsize=(14,7))
        #     plt.subplot(131)
        #     plt.imshow(phz_actual[i].angle().cpu(), cmap='jet', vmin=-torch.pi, vmax=torch.pi)
        #     plt.axis('off')
        #     plt.subplot(132)
        #     plt.imshow(phz_cur[i].angle().cpu(), cmap='jet', vmin=-torch.pi, vmax=torch.pi)
        #     plt.axis('off')
        #     plt.subplot(133)
        #     plt.imshow((phz_actual[i] - phz_cur[i]).abs().cpu(), cmap='jet', vmin=0, vmax=0.1)
        #     plt.axis('off')
        #     plt.tight_layout()
        # plt.show()
        # quit()
            
        seen_k = set()
        for rank in cur_ranks:
            k = max(2, min(rank, N_low, T_ds))
            if k in seen_k:
                continue
            seen_k.add(k)

            for key, method, _, name in CUR_SPECS:
                reseed()
                mv = matvec_cur(phis_ds, alphas_ds, cur_rank=k,
                                cluster_method=cluster_method,
                                normalize_method=method, seed=PIVOT_SEED)
                t = time_normal(mv, x_ds)
                del mv
                err = cur_err_on_upsampled(phis_up, alphas_up, method, k)
                record(key, t, err, Rn, Rm, rank=k, N_low=N_low, T_ds=T_ds)
                print(f'{name:<14} Rn={Rn} Rm={Rm} k={k}: time={t*1e3:.2f} ms, rel F^2={err:.4e}')

        gc.collect()
        if torch_dev.type == 'cuda':
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Hue = (Rn, Rm). Brightness + thickness = CUR rank. Linestyle = normalize method.
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 12,
    'axes.linewidth': 2.0,
    'lines.linewidth': 4.0,
    'lines.markersize': 14,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})
marker = 'x'
pair_colors = {pair: plt.cm.tab10(i) for i, pair in enumerate(ds)}

all_ks = [
    k for key, *_ in CUR_SPECS if key in results
    for k in results[key]['ranks'] if k is not None
]
rank_norm = Normalize(vmin=min(all_ks, default=0), vmax=max(all_ks, default=1))


def hue_rank_cmap(base):
    """Pale/bright (low k) → saturated (mid) → dark (high k) in the same hue."""
    h, s, v = colorsys.rgb_to_hsv(*to_rgb(base))
    light = colorsys.hsv_to_rgb(h, max(s * 0.45, 0.35), 0.95)
    mid = colorsys.hsv_to_rgb(h, min(s + 0.15, 1.0), 0.72)
    dark = colorsys.hsv_to_rgb(h, 1.0, 0.28)
    return LinearSegmentedColormap.from_list('rank_hue', [light, mid, dark])


pair_cmaps = {pair: hue_rank_cmap(color) for pair, color in pair_colors.items()}


def idxs_for(key, Rn, Rm):
    return [j for j, (r, m) in enumerate(zip(results[key]['Rn'], results[key]['Rm']))
            if r == Rn and m == Rm]


def coarse_NM(key, j):
    """N' (coarse spatial) and M' (coarse temporal) for a recorded point."""
    rec = results[key]
    if rec.get('N_low') and rec['N_low'][j] is not None:
        return int(rec['N_low'][j]), int(rec['T_ds'][j])
    Rn, Rm = rec['Rn'][j], rec['Rm'][j]
    im_size = results.get('im_size')
    trj_size = results.get('trj_size')
    if im_size is not None:
        N_low = 1
        for s in im_size:
            N_low *= max(2, int(s) // int(Rn))
    else:
        H = int(round(results['N'] ** 0.5))
        N_low = max(2, H // int(Rn)) ** 2
    if trj_size is not None:
        T_ds = max(2, int(trj_size[0]) // int(Rm))
        for s in trj_size[1:]:
            T_ds *= int(s)
    else:
        T_ds = max(2, int(results['T']) // int(Rm))
    return N_low, T_ds


def memory_coeff(key, j):
    N_low, T_ds = coarse_NM(key, j)
    k = results[key]['ranks'][j]
    if k is None:
        return 8 * N_low * T_ds / 2 ** 30
    return 8 * int(k) * (N_low + T_ds) / 2 ** 30


def make_tradeoff_fig(x_of, xlabel, out_path, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(14, 7))

    def plot_rank_segments(key, Rn, Rm, linestyle):
        idxs = sorted(idxs_for(key, Rn, Rm), key=lambda j: results[key]['ranks'][j])
        if len(idxs) < 2:
            return
        cmap = pair_cmaps[(Rn, Rm)]
        xs = [x_of(key, j) for j in idxs]
        ys = [results[key]['err_f2'][j] for j in idxs]
        ks = [results[key]['ranks'][j] for j in idxs]
        for i in range(len(idxs) - 1):
            k_mid = 0.5 * (ks[i] + ks[i + 1])
            t = float(rank_norm(k_mid))
            ax.plot(xs[i:i + 2], ys[i:i + 2],
                    color=cmap(t), linestyle=linestyle,
                    linewidth=2.0 + 4.0 * t)

    for Rn, Rm in ds:
        color = pair_colors[(Rn, Rm)]
        for key, _, ls, _ in CUR_SPECS:
            if key in results:
                plot_rank_segments(key, Rn, Rm, ls)
        naive_idxs = idxs_for('naive', Rn, Rm)
        if naive_idxs:
            ax.scatter(
                [x_of('naive', j) for j in naive_idxs],
                [results['naive']['err_f2'][j] for j in naive_idxs],
                c=[color], marker=marker, zorder=4,
            )

    method_handles = [
        *[Line2D([0], [0], color='k', linestyle=ls, label=label)
          for _, _, ls, label in CUR_SPECS],
        Line2D([0], [0], color='k', linestyle='None', marker=marker,
               markersize=12, label='no CUR'),
    ]
    fig.subplots_adjust(right=0.78)
    n_cb = max(len(ds), 1)
    for i, (Rn, Rm) in enumerate(ds):
        height = 0.72 / n_cb - 0.05
        bottom = 0.12 + (n_cb - 1 - i) * (0.72 / n_cb)
        cax = fig.add_axes([0.80, bottom, 0.02, height])
        sm = ScalarMappable(norm=rank_norm, cmap=pair_cmaps[(Rn, Rm)])
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(fr'CUR rank  ($R_n$={Rn}, $R_m$={Rm})', fontsize=12)

    # ax.legend(handles=method_handles, loc='upper right', title='Method')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\|P - \widehat{P}\|_F^2 / \|P\|_F^2$')
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, which='both', alpha=0.3)
    fig.savefig(out_path, dpi=150)
    print(f'Saved {out_path}')
    return fig


make_tradeoff_fig(
    x_of=lambda key, j: results[key]['times'][j] * 1e3,
    xlabel='Apply time [ms]',
    ylim=(1e-4, 1e-1),
    out_path=out_png,
)
make_tradeoff_fig(
    x_of=memory_coeff,
    xlabel='Memory [GB]',
    out_path=out_png_mem,
)

if not args.plot_only:
    torch.save(results, out_pt)
    print(f'Saved {out_pt}')
plt.show()
