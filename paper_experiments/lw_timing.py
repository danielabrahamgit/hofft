"""
L vs W forward-model cost: time and memory of the normal operator (A^H A).

Uses coco_spiral trajectory / coil / image sizes with random spatial factors
and kernel weights (no decomposition). HOFFT stores a dense (L, W, W) kernel
per trajectory point; SVD uses KB-NUFFT kernels of width W times L temporal
weights — the same ``hofft_linop`` used in recon.
"""
import argparse
import gc

import torch
import numpy as np

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

from time import perf_counter
from tqdm import tqdm

from mr_recon.linops import batching_params
from mr_recon.fourier import sigpy_nufft

from hofft.pipelines import kb_nufft
from hofft.forward_model import hofft_linop


# ---------------------------------------------------------------------------
# Timing / memory helpers
# ---------------------------------------------------------------------------
class GPUTimer:
    """Time a (possibly GPU-async) block; use CUDA events when available."""

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


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_paper_style():
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


# ---------------------------------------------------------------------------
# Fake linops (real coco_spiral dimensions)
# ---------------------------------------------------------------------------
def _snap_os(os: float, n: int) -> float:
    return 2 * round(os * n / 2) / n


def make_hofft_linop(trj_grd, mps, dcf, os, L, W, bparams):
    """HOFFT: independent (L, W, W) interpolation weights per k-space sample."""
    im_size = mps.shape[1:]
    trj_size = trj_grd.shape[:-1]
    kern_weights = torch.randn(
        (L, W, W, *trj_size), dtype=torch.complex64, device=trj_grd.device,
    )
    spatial_factors = torch.randn(
        (L, *im_size), dtype=torch.complex64, device=trj_grd.device,
    )
    return hofft_linop(
        trj=trj_grd, mps=mps, dcf=dcf,
        kern_weights=kern_weights, spatial_factors=spatial_factors,
        os_grid=os, bparams=bparams,
    )


def make_svd_linop(trj_grd, mps, dcf, os, L, spatial_kb, kern_kb, bparams):
    """SVD: L temporal weights broadcasting over a shared KB kernel of width W."""
    im_size = mps.shape[1:]
    trj_size = trj_grd.shape[:-1]
    d = trj_grd.shape[-1]
    temporal = torch.randn(
        (L, *([1] * d), *trj_size), dtype=torch.complex64, device=trj_grd.device,
    )
    spatial_funcs = torch.randn(
        (L, *im_size), dtype=torch.complex64, device=trj_grd.device,
    )
    return hofft_linop(
        trj=trj_grd, mps=mps, dcf=dcf,
        kern_weights=kern_kb * temporal,
        spatial_factors=spatial_kb * spatial_funcs,
        os_grid=os, bparams=bparams,
    )


def measure_normal(A, torch_dev, n_warmup, n_reps):
    """Return (mean A^H A time [s], peak extra GPU memory [GiB])."""
    img = torch.randn(A.ishape, dtype=torch.complex64, device=torch_dev)
    for _ in range(n_warmup):
        A.normal(img)
    if torch_dev.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_reps):
        with GPUTimer(torch_dev) as t:
            A.normal(img)
        times.append(t.elapsed)

    if torch_dev.type == 'cuda':
        peak = torch.cuda.max_memory_allocated(torch_dev)
        mem_gib = peak / (1024 ** 3)
    else:
        mem_gib = float('nan')
    return float(np.mean(times)), mem_gib


def run_sweep(args):
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch_dev.type != 'cuda':
        print('Warning: no CUDA device; timings will be CPU and memory will be NaN.')
    torch.manual_seed(0)

    Ls = list(range(args.Lmin, args.Lmax + 1))
    Ws = list(range(args.Wmin, args.Wmax + 1))
    nL, nW = len(Ls), len(Ws)

    fpath = args.data_path
    load_kw = {'weights_only': True, 'map_location': torch_dev}
    trj = torch.load(f'{fpath}/trj.pt', **load_kw).float()
    mps = torch.load(f'{fpath}/mps.pt', **load_kw).to(torch.complex64)
    dcf = torch.load(f'{fpath}/dcf.pt', **load_kw).float()

    im_size = mps.shape[1:]
    C = mps.shape[0]
    os = _snap_os(args.os, im_size[0])
    R = args.R
    trj = trj[:, ::R]
    dcf = dcf[:, ::R]
    trj_grd = (os * trj).round() / os
    bparams = batching_params(coil_batch_size=max(1, C // 2))

    print(f'device={torch_dev}, im_size={tuple(im_size)}, C={C}, '
          f'trj={tuple(trj.shape)}, os={os:.4f}, R={R}')
    print(f'L={Ls[0]}..{Ls[-1]} ({nL}), W={Ws[0]}..{Ws[-1]} ({nW}), '
          f'warmup={args.warmup}, reps={args.reps}')

    time = {k: torch.full((nL, nW), float('nan')) for k in ('hofft', 'svd')}
    mem = {k: torch.full((nL, nW), float('nan')) for k in ('hofft', 'svd')}

    def build_linop(method, L, W, spatial_kb, kern_kb):
        if method == 'hofft':
            return make_hofft_linop(trj_grd, mps, dcf, os, L, W, bparams)
        return make_svd_linop(trj_grd, mps, dcf, os, L, spatial_kb, kern_kb, bparams)

    warmed = False
    for method in ('hofft', 'svd'):
        for j, W in enumerate(tqdm(Ws, desc=f'{method} W', leave=True)):
            spatial_kb = kern_kb = None
            if method == 'svd':
                nft = sigpy_nufft(im_size, oversamp=os, width=W)
                nft.beta = nft.optimal_beta(torch_dev=torch_dev)
                spatial_kb, kern_kb = kb_nufft(
                    trj, im_size, (W, W), os=os, beta=nft.beta,
                )
            for i, L in enumerate(tqdm(Ls, desc=f'{method} L', leave=False)):
                clear_gpu()
                if torch_dev.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(torch_dev)
                try:
                    A = build_linop(method, L, W, spatial_kb, kern_kb)
                    if not warmed:
                        measure_normal(A, torch_dev, n_warmup=1, n_reps=1)
                        if torch_dev.type == 'cuda':
                            torch.cuda.reset_peak_memory_stats(torch_dev)
                        warmed = True
                    t_s, m_gib = measure_normal(
                        A, torch_dev, n_warmup=args.warmup, n_reps=args.reps,
                    )
                    time[method][i, j] = t_s
                    mem[method][i, j] = m_gib
                    print(
                        f'{method:<6s} L={L:<3d} W={W}: '
                        f'time={t_s:.4f}s, mem={m_gib:.3f} GiB'
                    )
                    del A
                except RuntimeError as exc:
                    if 'out of memory' not in str(exc).lower():
                        raise
                    print(f'{method:<6s} L={L:<3d} W={W}: OOM')
                    clear_gpu()
            del spatial_kb, kern_kb

    results = {
        'time_hofft': time['hofft'],
        'time_svd': time['svd'],
        'mem_hofft': mem['hofft'],
        'mem_svd': mem['svd'],
        'Ls': Ls,
        'Ws': Ws,
        'os': os,
        'R': R,
        'im_size': im_size,
        'trj_shape': tuple(trj.shape),
        'C': C,
    }
    torch.save(results, args.save_path)
    print(f'Saved timing results to {args.save_path}')
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _style_axes(ax):
    ax.grid(True, which='major', alpha=0.35, lw=1.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)


def plot_heatmaps(results, out_path):
    Ls = results['Ls']
    Ws = results['Ws']
    panels = [
        ('time_hofft', 'HOFFT time [s]'),
        ('time_svd', 'SVD time [s]'),
        ('mem_hofft', 'HOFFT memory [GiB]'),
        ('mem_svd', 'SVD memory [GiB]'),
    ]
    tmin = np.nanmin(torch.stack([results['time_hofft'], results['time_svd']]).numpy())
    tmax = np.nanmax(torch.stack([results['time_hofft'], results['time_svd']]).numpy())
    mmin = np.nanmin(torch.stack([results['mem_hofft'], results['mem_svd']]).numpy())
    mmax = np.nanmax(torch.stack([results['mem_hofft'], results['mem_svd']]).numpy())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    extent = [Ls[0] - 0.5, Ls[-1] + 0.5, Ws[0] - 0.5, Ws[-1] + 0.5]
    ims = []
    for ax, (key, title) in zip(axes.flat, panels):
        data = np.ma.masked_invalid(results[key].numpy().T)
        vmin, vmax = (tmin, tmax) if key.startswith('time') else (mmin, mmax)
        im = ax.imshow(
            data, origin='lower', extent=extent, aspect='auto',
            cmap='magma', vmin=vmin, vmax=vmax,
        )
        ims.append(im)
        ax.set_title(title)
        ax.set_xlabel('L')
        ax.set_ylabel('W')
        ax.set_xticks([L for L in Ls if L == Ls[0] or L == Ls[-1] or L % 5 == 0])
        ax.set_yticks(Ws)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)

    fig.colorbar(ims[0], ax=axes[0, :].tolist(), fraction=0.046, pad=0.02)
    fig.colorbar(ims[2], ax=axes[1, :].tolist(), fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=200)
    print(f'Saved {out_path}')
    return fig


def plot_lines(results, out_path):
    Ls = results['Ls']
    Ws = results['Ws']
    cmap = plt.cm.viridis
    norm = Normalize(vmin=Ws[0], vmax=Ws[-1])
    plot_kw = dict(lw=4.5, zorder=3)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    metrics = [
        (axes[0], 'time_hofft', 'time_svd', 'Time [s]', 'A$^H$A time vs L'),
        (axes[1], 'mem_hofft', 'mem_svd', 'Memory [GiB]', 'A$^H$A memory vs L'),
    ]
    for ax, key_h, key_s, ylabel, title in metrics:
        for W, j in zip(Ws, range(len(Ws))):
            color = cmap(norm(W))
            ax.plot(Ls, results[key_h][:, j].numpy(), '-', color=color, **plot_kw)
            ax.plot(Ls, results[key_s][:, j].numpy(), '--', color=color, **plot_kw)
        ax.set_xlabel('L')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        _style_axes(ax)
        method_handles = [
            Line2D([0], [0], color='0.2', ls='-', lw=4.5, label='HOFFT'),
            Line2D([0], [0], color='0.2', ls='--', lw=4.5, label='SVD'),
        ]
        leg = ax.legend(
            handles=method_handles, loc='best', frameon=True, fancybox=False,
            edgecolor='0.4', framealpha=0.95, borderpad=0.5, handlelength=2.4,
        )
        leg.get_frame().set_linewidth(1.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('W')
    cbar.set_ticks(Ws)
    fig.savefig(out_path, dpi=200)
    print(f'Saved {out_path}')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=str, default='./data/coco_spiral')
    parser.add_argument('--save_path', type=str,
                        default='./paper_experiments/coco_spiral_lw_timing.pt')
    parser.add_argument('--heatmap_path', type=str,
                        default='./paper_experiments/lw_timing_heatmaps.png')
    parser.add_argument('--line_path', type=str,
                        default='./paper_experiments/lw_timing_lines.png')
    parser.add_argument('--os', type=float, default=1.25)
    parser.add_argument('--R', type=int, default=3)
    parser.add_argument('--Lmin', type=int, default=1)
    parser.add_argument('--Lmax', type=int, default=30)
    parser.add_argument('--Wmin', type=int, default=1)
    parser.add_argument('--Wmax', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--plot_only', action='store_true',
                        help='Reload save_path and remake figures.')
    args = parser.parse_args()

    apply_paper_style()
    if args.plot_only:
        results = torch.load(args.save_path, weights_only=False, map_location='cpu')
    else:
        results = run_sweep(args)

    plot_heatmaps(results, args.heatmap_path)
    plot_lines(results, args.line_path)
    plt.show()


if __name__ == '__main__':
    main()
