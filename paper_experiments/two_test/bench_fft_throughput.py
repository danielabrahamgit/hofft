"""
Part A of math_docs/two_test.md: cuFFT throughput at full vs block size.

Standalone. No HOFFT / NUFFT. torch.fft.fftn only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)

N_WARMUP = 5
N_REPS = 20
LS = (8, 32, 80, 160)
Q = 8

# Spec table / prose. tilt full grid already includes sigma=2.
GEOMS = {
    'tilt_spi_invivo': dict(full=(1468, 1468), block=(260, 260), d=2),
    'coco_spiral':     dict(full=(276, 276), block=(98, 98), d=2),
    'coco_7t_spi':     dict(full=(320, 320, 320), block=(160, 160, 160), d=3),
}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median_s(fn, warmup=N_WARMUP, reps=N_REPS) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync()
        t0 = perf_counter()
        fn()
        _sync()
        ts.append(perf_counter() - t0)
    return float(np.median(ts))


def _metrics(n_trans: int, grid: tuple, d: int, t_s: float) -> dict:
    npts = int(np.prod(grid))
    bytes_moved = 2 * d * n_trans * npts * 8
    flops = 5.0 * n_trans * npts * math.log2(max(npts, 2))
    gbs = (bytes_moved / t_s) / 1e9 if t_s > 0 else 0.0
    gflops = (flops / t_s) / 1e9 if t_s > 0 else 0.0
    return dict(
        n_trans=int(n_trans), grid=list(grid), npts=npts,
        time_s=t_s, bytes=bytes_moved, GBps=gbs, GFLOPps=gflops,
    )


def _max_batch(grid: tuple, frac: float = 0.30) -> int:
    npts = int(np.prod(grid))
    free, _ = torch.cuda.mem_get_info()
    # input + output + FFT workspace
    per = 8 * npts * 3
    return max(1, int(free * frac / max(per, 1)))


def _fftn(x: torch.Tensor, d: int) -> torch.Tensor:
    dims = tuple(range(-d, 0))
    return torch.fft.fftn(x, dim=dims)


def _try_graph(fn, x_ref) -> tuple[float | None, str]:
    """Replay a CUDA graph of fn if the capture succeeds."""
    if not torch.cuda.is_available():
        return None, 'no_cuda'
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        t = _median_s(lambda: g.replay(), warmup=2, reps=N_REPS)
        return t, 'ok'
    except Exception as exc:
        return None, f'fail:{type(exc).__name__}'


def bench_full(grid: tuple, L: int, d: int) -> dict:
    n = min(int(L), _max_batch(grid))
    x = torch.randn((n, *grid), dtype=torch.complex64, device='cuda')
    t = _median_s(lambda: _fftn(x, d))
    rec = _metrics(n, grid, d, t)
    rec['requested_n_trans'] = int(L)
    rec['capped'] = n < L
    tg, note = _try_graph(lambda: _fftn(x, d), x)
    rec['graph'] = None if tg is None else _metrics(n, grid, d, tg)
    rec['graph_note'] = note
    del x
    torch.cuda.empty_cache()
    return rec


def bench_block_batched(grid: tuple, Lq: int, Q_kept: int, d: int) -> dict:
    n = Lq * Q_kept
    n_fit = min(n, _max_batch(grid))
    x = torch.randn((n_fit, *grid), dtype=torch.complex64, device='cuda')
    t = _median_s(lambda: _fftn(x, d))
    rec = _metrics(n_fit, grid, d, t)
    rec['requested_n_trans'] = int(n)
    rec['L_q'] = int(Lq)
    rec['Q_kept'] = int(Q_kept)
    rec['capped'] = n_fit < n
    tg, note = _try_graph(lambda: _fftn(x, d), x)
    rec['graph'] = None if tg is None else _metrics(n_fit, grid, d, tg)
    rec['graph_note'] = note
    del x
    torch.cuda.empty_cache()
    return rec


def bench_per_block(grid: tuple, Lq: int, Q_kept: int, d: int,
                    streams: bool) -> dict:
    n_fit = min(int(Lq), _max_batch(grid))
    xs = [torch.randn((n_fit, *grid), dtype=torch.complex64, device='cuda')
          for _ in range(Q_kept)]
    if streams:
        sts = [torch.cuda.Stream() for _ in range(Q_kept)]

        def run():
            for st, x in zip(sts, xs):
                with torch.cuda.stream(st):
                    _fftn(x, d)
            torch.cuda.synchronize()
    else:
        def run():
            for x in xs:
                _fftn(x, d)

    t = _median_s(run)
    rec = _metrics(n_fit * Q_kept, grid, d, t)
    rec['requested_n_trans'] = int(Lq * Q_kept)
    rec['L_q'] = int(Lq)
    rec['Q_kept'] = int(Q_kept)
    rec['capped'] = n_fit < Lq
    rec['streams'] = bool(streams)
    rec['graph'] = None
    rec['graph_note'] = 'n/a'
    del xs
    torch.cuda.empty_cache()
    return rec


def sweep_eta(d: int, grids: list[tuple], n_trans: int = 32) -> list[dict]:
    rows = []
    for grid in grids:
        n = min(n_trans, _max_batch(grid))
        if n < 1:
            continue
        x = torch.randn((n, *grid), dtype=torch.complex64, device='cuda')
        t = _median_s(lambda: _fftn(x, d))
        rec = _metrics(n, grid, d, t)
        rec['side'] = int(grid[0])
        rows.append(rec)
        print(f'    sweep {grid}  n={n}  {rec["GBps"]:.1f} GB/s  '
              f'{rec["GFLOPps"]:.1f} GFLOP/s  {t*1e3:.2f} ms', flush=True)
        del x
        torch.cuda.empty_cache()
    return rows


def _print_arm(tag: str, rec: dict):
    g = rec.get('graph')
    extra = ''
    if g:
        extra = f'  graph {g["GBps"]:.1f} GB/s'
    print(f'    {tag:<28s} n={rec["n_trans"]:<4d}  '
          f'{rec["time_s"]*1e3:8.2f} ms  {rec["GBps"]:7.1f} GB/s  '
          f'{rec["GFLOPps"]:8.1f} GFLOP/s{extra}', flush=True)


def main():
    if not torch.cuda.is_available():
        raise SystemExit('Part A needs a GPU')
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    print(f'GPU {props.name}  {props.total_memory/2**30:.1f} GiB', flush=True)

    report = dict(gpu=props.name, gpu_gib=props.total_memory / 2**30,
                  warmup=N_WARMUP, reps=N_REPS, Q=Q, geoms={})

    for name, geom in GEOMS.items():
        full, block, d = geom['full'], geom['block'], geom['d']
        print(f'\n======== {name}  full={full}  block={block} ========', flush=True)
        rec_name = dict(full=full, block=block, d=d, by_L={})
        for L in LS:
            Lq = max(int(L) // Q, 1)
            print(f'  L={L}  L_q={Lq}  Q_kept={Q}', flush=True)
            full_r = bench_full(full, L, d)
            bat = bench_block_batched(block, Lq, Q, d)
            ser = bench_per_block(block, Lq, Q, d, streams=False)
            stm = bench_per_block(block, Lq, Q, d, streams=True)
            eta = (bat['GBps'] / full_r['GBps']) if full_r['GBps'] > 0 else None
            _print_arm('full', full_r)
            _print_arm('block batched', bat)
            _print_arm('block per-block', ser)
            _print_arm('block streams', stm)
            print(f'    eta (batched/full GB/s) = {eta}', flush=True)
            rec_name['by_L'][str(L)] = dict(
                full=full_r, batched=bat, per_block=ser, streams=stm, eta=eta,
            )
        report['geoms'][name] = rec_name

    # eta vs grid size
    print('\n======== eta vs grid (2D) ========', flush=True)
    sides_2d = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1468]
    sweep2 = sweep_eta(2, [(s, s) for s in sides_2d], n_trans=32)
    g_full = next(r['GBps'] for r in sweep2 if r['side'] == 1468)
    for r in sweep2:
        r['eta_vs_1468'] = r['GBps'] / g_full if g_full else None
    report['sweep_2d'] = sweep2

    print('\n======== eta vs grid (3D) ========', flush=True)
    sides_3d = [32, 48, 64, 80, 96, 128, 160, 192, 256, 320]
    sweep3 = sweep_eta(3, [(s, s, s) for s in sides_3d], n_trans=8)
    g320 = next(r['GBps'] for r in sweep3 if r['side'] == 320)
    for r in sweep3:
        r['eta_vs_320'] = r['GBps'] / g320 if g320 else None
    report['sweep_3d'] = sweep3

    # Decision from A.3: batched-arm eta at the production sizes.
    etas = []
    for name, rec in report['geoms'].items():
        # prefer L that is not memory-capped
        pick = None
        for L in LS:
            row = rec['by_L'][str(L)]
            if not row['full']['capped'] and not row['batched']['capped']:
                pick = row['eta']
        if pick is None:
            pick = rec['by_L'][str(LS[0])]['eta']
        rec['eta_decision'] = pick
        etas.append((name, pick))
        print(f'  {name} eta={pick}', flush=True)

    vals = [e for _, e in etas if e is not None]
    med = float(np.median(vals)) if vals else None
    if med is not None and med > 0.7:
        branch = 'eta>0.7: transforms are efficient; shortfall is plan fragmentation'
    elif med is not None and med < 0.4:
        branch = 'eta<0.4: small transforms are bandwidth-inefficient; flop model fails'
    else:
        branch = 'in between: discount predicted speedups by measured eta'
    report['eta_median'] = med
    report['a3_branch'] = branch
    print(f'\nA.3 {branch}  (median eta={med})', flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['side'] for r in sweep2], [r['GBps'] for r in sweep2], 'o-')
    axes[0].set_xlabel('grid side')
    axes[0].set_ylabel('GB/s')
    axes[0].set_title('2D cuFFT bandwidth')
    axes[0].set_xscale('log', base=2)
    axes[1].plot([r['side'] for r in sweep2],
                 [r['eta_vs_1468'] for r in sweep2], 'o-')
    axes[1].axhline(0.7, color='k', ls='--', lw=0.8)
    axes[1].axhline(0.4, color='k', ls=':', lw=0.8)
    axes[1].set_xlabel('grid side')
    axes[1].set_ylabel(r'$\eta$ vs $1468^2$')
    axes[1].set_title('2D eta vs full tilt grid')
    axes[1].set_xscale('log', base=2)
    fig.tight_layout()
    fig.savefig(OUT / 'eta_vs_grid_2d.png', dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['side'] for r in sweep3], [r['GBps'] for r in sweep3], 'o-')
    axes[0].set_xlabel('grid side')
    axes[0].set_ylabel('GB/s')
    axes[0].set_title('3D cuFFT bandwidth')
    axes[1].plot([r['side'] for r in sweep3],
                 [r['eta_vs_320'] for r in sweep3], 'o-')
    axes[1].axhline(0.7, color='k', ls='--', lw=0.8)
    axes[1].axhline(0.4, color='k', ls=':', lw=0.8)
    axes[1].set_xlabel('grid side')
    axes[1].set_ylabel(r'$\eta$ vs $320^3$')
    axes[1].set_title('3D eta vs 320^3')
    fig.tight_layout()
    fig.savefig(OUT / 'eta_vs_grid_3d.png', dpi=140)
    plt.close(fig)

    (OUT / 'part_a.json').write_text(json.dumps(report, indent=2))
    print(f'\nwrote {OUT / "part_a.json"}', flush=True)


if __name__ == '__main__':
    main()
