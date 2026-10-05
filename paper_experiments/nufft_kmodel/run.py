"""
Compare three non-Cartesian forwards against analytical Shepp-Logan k-space:

  1. KB-NUFFT via ``sigpy_nufft`` (optimal beta)
  2. KB-NUFFT via ``hofft_linop`` (same kernels; faster, denser)
  3. Chan-Haldar B-spline k-space model

Reference is ``shepp_logan.ksp(trj)``, not a DFT/NUFFT of the discrete image,
so KB's convergence to ``matrix_nufft`` does not artificially win.

  python paper_experiments/nufft_kmodel/run.py
  python paper_experiments/nufft_kmodel/run.py --quick
  python paper_experiments/nufft_kmodel/run.py --plot_only --no-show
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import matplotlib as mpl
mpl.use('agg' if '--no-show' in sys.argv else 'webagg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tqdm import tqdm

from mr_recon.fourier import sigpy_nufft
from mr_sim.phantoms import shepp_logan

from hofft.forward_model import hofft_linop
from hofft.pipelines import kb_nufft

from kspace_nufft import kspace_nufft


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Timing / memory / metrics
# ---------------------------------------------------------------------------
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


def time_repeated(fn, torch_dev, n_warmup=1, n_reps=3):
    for _ in range(n_warmup):
        fn()
    if torch_dev.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    result = None
    for _ in range(n_reps):
        with GPUTimer(torch_dev) as t:
            result = fn()
        times.append(t.elapsed)
    return float(np.median(times)), result


def scaled_nrmse(approx: torch.Tensor, ref: torch.Tensor):
    """NRMSE after a global complex scale (ignores FT convention mismatch)."""
    num = (approx.conj() * ref).sum()
    den = (approx.conj() * approx).sum().real.clamp_min(1e-12)
    scale = num / den
    err = (scale * approx - ref).norm() / ref.norm().clamp_min(1e-12)
    return err.item(), scale.item()


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def reset_peak_mem(torch_dev):
    clear_gpu()
    if torch_dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(torch_dev)


def _rss_gib():
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return float('nan')


def _cupy_used_bytes():
    try:
        import cupy as cp
        return int(cp.get_default_memory_pool().used_bytes())
    except Exception:
        return 0


def tensor_nbytes(t):
    if t is None:
        return 0
    if t.layout == torch.strided:
        return t.numel() * t.element_size()
    if t.layout == torch.sparse_csc:
        return (
            t.values().numel() * t.values().element_size()
            + t.row_indices().numel() * t.row_indices().element_size()
            + t.ccol_indices().numel() * t.ccol_indices().element_size()
        )
    if t.layout == torch.sparse_coo:
        return (
            t.values().numel() * t.values().element_size()
            + t.indices().numel() * t.indices().element_size()
        )
    return t.numel() * t.element_size()


def operator_mem_gib(op):
    """Bytes held by the operator (kernels / sparse matrices), not the process."""
    n = 0
    if isinstance(op, hofft_linop):
        for name in ('kern_weights', 'spatial_factors', 'idx_kerns', 'mps', 'dcf'):
            n += tensor_nbytes(getattr(op, name, None))
    elif isinstance(op, kspace_nufft):
        for name in ('Hmat', 'Hmath', 'apod'):
            n += tensor_nbytes(getattr(op, name, None))
    return n / (1024 ** 3)


def read_peak_mem_gib(torch_dev, op=None):
    """GPU: torch peak + CuPy pool. CPU: operator storage (RSS is too noisy)."""
    op_gib = operator_mem_gib(op) if op is not None else 0.0
    if torch_dev.type == 'cuda':
        torch_b = torch.cuda.max_memory_allocated(torch_dev)
        peak = (torch_b + _cupy_used_bytes()) / (1024 ** 3)
        return max(peak, op_gib)
    return op_gib


def parse_floats(s: str):
    return [float(x) for x in s.split(',') if x]


def parse_ints(s: str):
    return [int(x) for x in s.split(',') if x]


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


def _style_axes(ax):
    ax.grid(True, which='major', alpha=0.35, lw=1.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def snap_os(os: float, n: int) -> float:
    return 2 * round(os * n / 2) / n


_beta_cache = {}


def optimal_beta(im_size, os, W, torch_dev):
    key = (tuple(im_size), float(os), int(W))
    if key not in _beta_cache:
        nft = sigpy_nufft(im_size, oversamp=os, width=W)
        _beta_cache[key] = nft.optimal_beta(torch_dev=torch_dev)
    return _beta_cache[key]


def build_sigpy(im_size, trj, os, W, torch_dev):
    os = snap_os(os, im_size[0])
    beta = optimal_beta(im_size, os, W, torch_dev)
    nft = sigpy_nufft(im_size, oversamp=os, width=W, beta=beta)
    return nft, beta, os


def build_hofft(im_size, trj, os, W, torch_dev):
    os = snap_os(os, im_size[0])
    beta = optimal_beta(im_size, os, W, torch_dev)
    spatial_factor, kern_weights = kb_nufft(
        trj, im_size, (W,) * len(im_size), os=os, beta=beta,
    )
    trj_grd = (os * trj).round() / os
    mps = torch.ones((1, *im_size), dtype=torch.complex64, device=trj.device)
    A = hofft_linop(trj_grd, mps, kern_weights, spatial_factor, os_grid=os)
    return A, beta, os


def build_kmodel(im_size, trj, rho, W, torch_dev):
    nft = kspace_nufft(im_size, trj, oversamp=rho, width=W, device=torch_dev)
    return nft, float('nan'), float(nft.oversamp)


def apply_fwd(op, img, trj):
    if isinstance(op, hofft_linop):
        return op.forward(img)[0]
    return op.forward(img[None], trj[None])[0]


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(args):
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    print(f'device={torch_dev}')
    if torch_dev.type != 'cuda':
        print('Warning: no CUDA; peak memory is process RSS and timings are CPU.')

    load_kw = {'weights_only': True, 'map_location': torch_dev}
    trj = torch.load(args.data_path / 'trj.pt', **load_kw).float()
    im_size = (args.N,) * 2
    phantom = shepp_logan(torch_dev)
    img = phantom.img(im_size)
    print('Computing analytical Shepp-Logan k-space ...')
    ksp_gt = phantom.ksp(trj)
    print(f'img={im_size}, trj={tuple(trj.shape)}')

    methods = [
        {
            'key': 'sigpy',
            'label': 'KB-NUFFT (SigPy)',
            'os_list': args.os_list,
            'W_list': args.W_list,
            'build': lambda os, W: build_sigpy(im_size, trj, os, W, torch_dev),
        },
        {
            'key': 'hofft',
            'label': 'KB-NUFFT (HOFFT)',
            'os_list': args.os_list,
            'W_list': args.W_list,
            'build': lambda os, W: build_hofft(im_size, trj, os, W, torch_dev),
        },
        {
            'key': 'kmodel',
            'label': 'B-spline k-model',
            'os_list': args.rho_list,
            'W_list': args.W_kmodel,
            'build': lambda os, W: build_kmodel(im_size, trj, os, W, torch_dev),
        },
    ]

    rows = []
    warmed = False
    for spec in methods:
        combos = list(product(spec['os_list'], spec['W_list']))
        for os, W in tqdm(combos, desc=spec['label']):
            reset_peak_mem(torch_dev)
            t_setup0 = perf_counter()
            nft, extra, os_used = spec['build'](os, W)
            t_setup = perf_counter() - t_setup0

            def fwd():
                return apply_fwd(nft, img, trj)

            if not warmed:
                fwd()
                if torch_dev.type == 'cuda':
                    torch.cuda.synchronize()
                warmed = True
                reset_peak_mem(torch_dev)
                del nft
                nft, extra, os_used = spec['build'](os, W)

            t_fwd, ksp = time_repeated(
                fwd, torch_dev, n_warmup=args.warmup, n_reps=args.reps,
            )
            mem_gib = read_peak_mem_gib(torch_dev, op=nft)
            nrmse, scale = scaled_nrmse(ksp, ksp_gt)
            rows.append({
                'method': spec['key'],
                'label': spec['label'],
                'os': os_used,
                'os_requested': float(os),
                'W': int(W),
                'beta': extra if spec['key'] != 'kmodel' else float('nan'),
                'degree': (int(W) - 1) if spec['key'] == 'kmodel' else float('nan'),
                'nrmse': nrmse,
                'scale_real': float(np.real(scale)),
                'scale_imag': float(np.imag(scale)),
                't_fwd': t_fwd,
                't_setup': t_setup,
                'mem_gib': mem_gib,
            })
            extra_s = f' beta={extra:.3f}' if spec['key'] != 'kmodel' else f' p={int(W)-1}'
            print(
                f'{spec["label"]:<20s} os={os_used:<7.4g} W={W}{extra_s}: '
                f'NRMSE={nrmse:.4e}, fwd={t_fwd*1e3:.2f} ms, '
                f'mem={mem_gib:.3f} GiB, setup={t_setup:.2f}s'
            )
            del nft, ksp
            clear_gpu()

    results = {
        'rows': rows,
        'im_size': im_size,
        'trj_shape': tuple(trj.shape),
        'gt': 'shepp_logan_analytical',
        'device': str(torch_dev),
    }
    torch.save(results, args.save_path)
    print(f'Saved {args.save_path}')
    return results


def print_highlights(results):
    rows = results['rows']

    def find(method, os, W, os_tol=0.02):
        hits = [r for r in rows
                if r['method'] == method and r['W'] == W and abs(r['os'] - os) <= os_tol]
        if not hits:
            return None
        return min(hits, key=lambda r: abs(r['os'] - os))

    targets = [
        ('sigpy', 1.25, 3, 'KB-NUFFT (SigPy)   os=1.25  W=3'),
        ('hofft', 1.25, 3, 'KB-NUFFT (HOFFT)   os=1.25  W=3'),
        ('kmodel', 1.3, 4, 'k-model            rho=1.30  W=4 (cubic)'),
        ('kmodel', 1.25, 3, 'k-model            rho=1.25  W=3'),
    ]
    print('\n--- Highlighted comparisons (analytical Shepp-Logan k-space) ---')
    for method, os, W, desc in targets:
        r = find(method, os, W)
        if r is None:
            print(f'  (missing) {desc}')
            continue
        print(
            f'  {desc}\n'
            f'      NRMSE={r["nrmse"]:.4e}   fwd={r["t_fwd"]*1e3:.2f} ms   '
            f'mem={r["mem_gib"]:.3f} GiB'
        )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
METHOD_STYLE = {
    'sigpy': dict(color='#1f77b4', marker='o', label='KB-NUFFT (SigPy)'),
    'hofft': dict(color='#2ca02c', marker='D', label='KB-NUFFT (HOFFT)'),
    'kmodel': dict(color='#d62728', marker='s', label='B-spline k-model'),
}


def _rows_by_method(results):
    out = {k: [] for k in METHOD_STYLE}
    for r in results['rows']:
        if np.isfinite(r.get('nrmse', np.nan)):
            out.setdefault(r['method'], []).append(r)
    return out


def _plot_vs_nrmse(results, y_key, ylabel, title, out_path, yscale='log'):
    grouped = _rows_by_method(results)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    lss = ['-', '--', ':', '-.', (0, (5, 1))]
    for key, style in METHOD_STYLE.items():
        rows = grouped.get(key, [])
        if not rows:
            continue
        Ws = sorted({r['W'] for r in rows})
        for i, W in enumerate(Ws):
            sub = sorted(
                [r for r in rows if r['W'] == W],
                key=lambda r: r['nrmse'],
            )
            if not sub:
                continue
            ys = [r[y_key] for r in sub]
            if y_key == 't_fwd':
                ys = [y * 1e3 for y in ys]
            if y_key == 'mem_gib':
                ys = [max(y, 1e-5) for y in ys]
            ax.plot(
                [r['nrmse'] for r in sub], ys,
                color=style['color'], linestyle=lss[i % len(lss)],
                marker=style['marker'], label=f'{style["label"]}, W={W}',
                lw=4.5, ms=14, mew=2.0, markeredgecolor='white', zorder=3,
            )
    # ax.set_xscale('log')
    # ax.set_yscale(yscale)
    ax.set_xlabel('k-space NRMSE')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style_axes(ax)
    leg = ax.legend(loc='best', frameon=True, fancybox=False,
                    edgecolor='0.4', framealpha=0.95, fontsize=11)
    leg.get_frame().set_linewidth(1.5)
    fig.savefig(out_path, dpi=200)
    print(f'Saved {out_path}')
    return fig


def plot_tradeoff_panels(results, out_path):
    """Forward time and peak memory vs NRMSE, one panel each."""
    grouped = _rows_by_method(results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    lss = ['-', '--', ':', '-.', (0, (5, 1))]
    panels = [
        (axes[0], 't_fwd', 1e3, 'Forward time [ms]', 'Time vs NRMSE'),
        (axes[1], 'mem_gib', 1.0, 'Peak memory [GiB]', 'Memory vs NRMSE'),
    ]
    for ax, y_key, yscale_mul, ylabel, title in panels:
        for key, style in METHOD_STYLE.items():
            rows = grouped.get(key, [])
            if not rows:
                continue
            Ws = sorted({r['W'] for r in rows})
            for i, W in enumerate(Ws):
                sub = sorted(
                    [r for r in rows if r['W'] == W and np.isfinite(r['nrmse'])],
                    key=lambda r: r['nrmse'],
                )
                if not sub:
                    continue
                ax.plot(
                    [r['nrmse'] for r in sub],
                    [max(r[y_key] * yscale_mul, 1e-5) for r in sub],
                    color=style['color'], linestyle=lss[i % len(lss)],
                    marker=style['marker'], label=f'{style["label"]}, W={W}',
                    lw=4.5, ms=14, mew=2.0, markeredgecolor='white', zorder=3,
                )
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xlabel('k-space NRMSE')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style_axes(ax)
        leg = ax.legend(loc='best', frameon=True, fancybox=False,
                        edgecolor='0.4', framealpha=0.95, fontsize=10)
        leg.get_frame().set_linewidth(1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved {out_path}')
    return fig


def plot_nrmse_vs_os(results, out_path):
    grouped = _rows_by_method(results)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    lss = ['-', '--', ':', '-.', (0, (5, 1))]
    for key, style in METHOD_STYLE.items():
        rows = grouped.get(key, [])
        if not rows:
            continue
        Ws = sorted({r['W'] for r in rows})
        for i, W in enumerate(Ws):
            sub = sorted(
                [r for r in rows if r['W'] == W and np.isfinite(r['nrmse'])],
                key=lambda r: r['os'],
            )
            if not sub:
                continue
            ax.semilogy(
                [r['os'] for r in sub], [r['nrmse'] for r in sub],
                color=style['color'], linestyle=lss[i % len(lss)],
                marker=style['marker'], label=f'{style["label"]}, W={W}',
                lw=4.5, ms=14, mew=2.0, markeredgecolor='white', zorder=3,
            )
    ax.set_xlabel('Oversampling')
    ax.set_ylabel('k-space NRMSE')
    ax.set_title('Accuracy vs oversampling')
    _style_axes(ax)
    leg = ax.legend(loc='best', frameon=True, fancybox=False,
                    edgecolor='0.4', framealpha=0.95, fontsize=11)
    leg.get_frame().set_linewidth(1.5)
    fig.savefig(out_path, dpi=200)
    print(f'Saved {out_path}')
    return fig


def plot_scatter(results, out_path):
    grouped = _rows_by_method(results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, y_key, ymul, ylabel, title in [
        (axes[0], 't_fwd', 1e3, 'Forward time [ms]', 'Time vs NRMSE'),
        (axes[1], 'mem_gib', 1.0, 'Peak memory [GiB]', 'Memory vs NRMSE'),
    ]:
        for key, style in METHOD_STYLE.items():
            rows = grouped.get(key, [])
            if not rows:
                continue
            ax.scatter(
                [r['nrmse'] for r in rows],
                [r[y_key] * ymul for r in rows],
                c=style['color'], marker=style['marker'], s=160,
                edgecolors='white', linewidths=1.6, zorder=3, label=style['label'],
            )
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xlabel('k-space NRMSE')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style_axes(ax)
        extra = [
            Line2D([0], [0], marker='*', color='0.15', lw=0, markersize=18,
                   markerfacecolor='none', markeredgewidth=2.4,
                   label='KB W=3, os=1.25'),
            Line2D([0], [0], marker='P', color='0.15', lw=0, markersize=16,
                   markerfacecolor='none', markeredgewidth=2.4,
                   label='k-model W=4, ρ=1.3'),
        ]
        for method, os, W, marker in [
            ('sigpy', 1.25, 3, '*'), ('hofft', 1.25, 3, '*'),
            ('kmodel', 1.3, 4, 'P'),
        ]:
            hits = [r for r in results['rows']
                    if r['method'] == method and r['W'] == W and abs(r['os'] - os) < 0.02]
            if not hits:
                continue
            r = min(hits, key=lambda x: abs(x['os'] - os))
            ax.plot(
                r['nrmse'], r[y_key] * ymul, marker=marker,
                markersize=20, markerfacecolor='none', markeredgewidth=2.8,
                markeredgecolor='0.15', zorder=5,
            )
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(
            handles + extra, labels + [e.get_label() for e in extra],
            loc='best', frameon=True, fancybox=False,
            edgecolor='0.4', framealpha=0.95, fontsize=11,
        )
        leg.get_frame().set_linewidth(1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved {out_path}')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=Path, default=ROOT / 'data' / 'sim_spiral')
    parser.add_argument('--save_path', type=Path, default=OUT_DIR / 'nufft_kmodel_results.pt')
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--os_list', type=parse_floats, default=[1.0, 1.125, 1.25, 1.5, 2.0])
    parser.add_argument('--rho_list', type=parse_floats, default=[1.0, 1.125, 1.25, 1.3, 1.5, 2.0])
    parser.add_argument('--W_list', type=parse_ints, default=[2, 3, 4, 5, 6])
    parser.add_argument('--W_kmodel', type=parse_ints, default=[2, 3, 4, 5, 6],
                        help='B-spline support W=p+1 (degree p). Default cubic is W=4.')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--plot_only', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.os_list = [1.25, 1.5]
        args.rho_list = [1.25, 1.3, 1.5]
        args.W_list = [3, 4]
        args.W_kmodel = [3, 4]
        args.reps = 2

    apply_paper_style()
    if args.plot_only:
        results = torch.load(args.save_path, weights_only=False, map_location='cpu')
    else:
        results = run_sweep(args)

    print_highlights(results)
    plot_nrmse_vs_os(results, OUT_DIR / 'nrmse_vs_oversamp.png')
    plot_tradeoff_panels(results, OUT_DIR / 'time_mem_vs_nrmse.png')
    plot_scatter(results, OUT_DIR / 'time_mem_vs_nrmse_scatter.png')
    _plot_vs_nrmse(
        results, 't_fwd', 'Forward time [ms]', 'Forward time vs NRMSE',
        OUT_DIR / 'time_vs_nrmse.png',
    )
    _plot_vs_nrmse(
        results, 'mem_gib', 'Peak memory [GiB]', 'Peak memory vs NRMSE',
        OUT_DIR / 'mem_vs_nrmse.png', yscale='log',
    )
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
