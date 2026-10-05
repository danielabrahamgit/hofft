"""
Item 2 of math_docs/two_test_follow.md: cuFINUFFT grid size vs 5-smooth cuFFT.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import cufinufft

from bench_fft_throughput import _median_s, _metrics
from mr_recon.fourier.cufi_nufft import cufi_nufft, eps_from_width

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)


def next235even(n: int) -> int:
    n = max(2, int(n))
    if n % 2:
        n += 1
    x = n
    while True:
        t = x
        while t % 2 == 0:
            t //= 2
        while t % 3 == 0:
            t //= 3
        while t % 5 == 0:
            t //= 5
        if t == 1:
            return x
        x += 2


def factorize(n: int) -> str:
    n = int(n)
    parts = []
    p = 2
    while p * p <= n:
        k = 0
        while n % p == 0:
            n //= p
            k += 1
        if k:
            parts.append(f'{p}' if k == 1 else f'{p}^{k}')
        p += 1 if p == 2 else 2
    if n > 1:
        parts.append(str(n))
    return '*'.join(parts) if parts else '1'


def snap_os(os_nominal: float, n: int) -> float:
    return 2 * round(os_nominal * n / 2) / n


def predicted_nf(n: int, sigma: float) -> dict:
    raw = sigma * n
    ceiled = int(math.ceil(raw - 1e-12))
    nf = next235even(ceiled)
    return dict(sigma=float(sigma), raw=float(raw), ceil=ceiled,
                next235even=nf, ceil_factors=factorize(ceiled),
                nf_factors=factorize(nf))


def _try_read_nf(plan) -> list | None:
    """Best-effort read of cuFINUFFT allocated (nf1, nf2, nf3)."""
    for attr in ('nf', 'nfs', '_nf', 'n_freqs'):
        if hasattr(plan, attr):
            v = getattr(plan, attr)
            try:
                return [int(x) for x in v]
            except Exception:
                pass
    # peek common C layout: after a few ints, int64 nf1,nf2,nf3
    ptr = getattr(plan, '_plan', None)
    if ptr is None:
        return None
    try:
        import ctypes
        addr = ctypes.c_void_p.from_buffer(ptr).value
        if not addr:
            addr = int(ptr.value) if hasattr(ptr, 'value') else None
        if not addr:
            return None
        # Conservative: don't dereference an unknown struct.
        return None
    except Exception:
        return None


def probe_cufinufft_grid(im_size: tuple, sigma: float, width: int = 3):
    """Build a real cufi plan and report requested vs next235even vs memory."""
    eps = eps_from_width(width, sigma)
    torch.cuda.empty_cache()
    free0, total = torch.cuda.mem_get_info()
    nft = cufi_nufft(im_size, oversamp=sigma, width=width, n_trans=1)
    # dummy trajectory so plan() actually builds
    d = len(im_size)
    trj = torch.zeros((8, d), device='cuda', dtype=torch.float32)
    nft.plan(trj[None], n_trans=1)
    free1, _ = torch.cuda.mem_get_info()
    used = int(free0 - free1)
    pred = predicted_nf(im_size[0], sigma)
    nf_guess = pred['next235even']
    # type-2 workspace is about n_trans * prod(nf) * 8 plus extras
    npts_from_mem = used / 8.0
    inferred = None
    if d == 2 and npts_from_mem > 100:
        side = int(round(math.sqrt(npts_from_mem)))
        # try nearby 5-smooth
        for cand in range(max(2, side - 32), side + 33, 2):
            if next235even(cand) == cand and abs(cand * cand * 8 - used) < 0.25 * used:
                inferred = [cand, cand]
                break
    elif d == 3 and npts_from_mem > 100:
        side = int(round(npts_from_mem ** (1 / 3)))
        for cand in range(max(2, side - 32), side + 33, 2):
            if next235even(cand) == cand and abs(cand ** 3 * 8 - used) < 0.35 * used:
                inferred = [cand, cand, cand]
                break

    plan = next(iter(nft._fwd.values()), None)
    read_nf = _try_read_nf(plan) if plan is not None else None
    rec = dict(
        im_size=list(im_size), sigma=float(sigma), width=width, eps=float(eps),
        ceil_sigma_N=pred['ceil'], next235even=nf_guess,
        ceil_factors=pred['ceil_factors'], nf_factors=pred['nf_factors'],
        plan_bytes=used, inferred_grid=inferred, plan_nf=read_nf,
        upsampfac=nft.plan_kwargs.get('upsampfac'),
    )
    nft.clear_plans()
    del nft
    torch.cuda.empty_cache()
    return rec


def bench_grid(grid: tuple, n_trans: int, d: int) -> dict:
    x = torch.randn((n_trans, *grid), dtype=torch.complex64, device='cuda')
    t = _median_s(lambda: torch.fft.fftn(x, dim=tuple(range(-d, 0))))
    rec = _metrics(n_trans, grid, d, t)
    rec['factors'] = '*'.join(factorize(n) for n in grid)
    rec['five_smooth'] = all(next235even(n) == n for n in grid)
    del x
    torch.cuda.empty_cache()
    return rec


def main():
    torch.device('cuda')
    report = dict(grids=[], ffts=[])

    geoms = [
        dict(name='tilt_spi_invivo', N=734, d=2, sigma=snap_os(1.25, 734)),
        dict(name='coco_spiral', N=276, d=2, sigma=snap_os(1.25, 276)),
        dict(name='coco_7t_spi', N=320, d=3, sigma=snap_os(1.25, 320)),
    ]
    print('==== cuFINUFFT allocated grids ====', flush=True)
    for g in geoms:
        im = (g['N'],) * g['d']
        rec = probe_cufinufft_grid(im, g['sigma'])
        rec['name'] = g['name']
        rec['snap_os'] = g['sigma']
        report['grids'].append(rec)
        print(f'  {g["name"]} N={g["N"]} sigma={g["sigma"]:.5f}  '
              f'ceil={rec["ceil_sigma_N"]}={rec["ceil_factors"]}  '
              f'next235even={rec["next235even"]}={rec["nf_factors"]}  '
              f'inferred={rec["inferred_grid"]}  bytes={rec["plan_bytes"]}',
              flush=True)

    print('\n==== batched cuFFT ====', flush=True)
    cases = [
        ('tilt_918', (918, 918), 80, 2),
        ('tilt_960', (960, 960), 80, 2),
        ('tilt_972', (972, 972), 80, 2),
        ('tilt_1024', (1024, 1024), 80, 2),
        ('coco_345', (345, 345), 80, 2),
        ('coco_360', (360, 360), 80, 2),
        ('coco_384', (384, 384), 80, 2),
        ('3d_400', (400, 400, 400), 8, 3),
        ('3d_384', (384, 384, 384), 8, 3),
        ('3d_432', (432, 432, 432), 8, 3),
    ]
    for name, grid, nt, d in cases:
        rec = bench_grid(grid, nt, d)
        rec['name'] = name
        report['ffts'].append(rec)
        print(f'  {name:12s} {grid}  {rec["factors"]:20s}  '
              f'{rec["GBps"]:7.1f} GB/s  {rec["time_s"]*1e3:.2f} ms  '
              f'5smooth={rec["five_smooth"]}', flush=True)

    # Recommended sigma if ceil grid is awkward and next235even already
    # lands on a 5-smooth size — no change. If 918 is used and slow, bump.
    tilt_918 = next(r for r in report['ffts'] if r['name'] == 'tilt_918')
    tilt_960 = next(r for r in report['ffts'] if r['name'] == 'tilt_960')
    tilt_nf = report['grids'][0]['next235even']
    action = 'none'
    if tilt_nf == 918 and tilt_918['GBps'] < 0.7 * tilt_960['GBps']:
        action = 'raise_sigma_to_960/734'
    elif tilt_nf == 960:
        action = 'already_960_via_next235even'
    report['action'] = action
    report['tilt_sigma_960'] = 960 / 734
    report['tilt_sigma_972'] = 972 / 734
    print(f'\naction: {action}', flush=True)

    (OUT / 'item2.json').write_text(json.dumps(report, indent=2))
    print(f'wrote {OUT / "item2.json"}', flush=True)


if __name__ == '__main__':
    main()
