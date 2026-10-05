"""
rho for cufinufft on the same three problems as rho_sigpy.py.

Splits wall-clock the same way: oversampled FFT vs cufinufft's binned interpolator
(type-2 Plan with gpu_spreadinterponly=1). rho = gather / FFT.

cufinufft width is set by ``eps``, not a Kaiser-Bessel W. At the Horner kernel
(upsampfac=2) the usual table is nspread ~ 3 at 1e-2 and ~7 at 1e-6, so those two
tolerances stand in for W=3 and W=6. Default MRI wrap (cufi_nufft) uses eps=1e-4.

The FFT grid is still os=1.25 so the denominator matches the sigpy / HOFFT profiles.
The interpolator is cufinufft's, not sigpy's.

Run with:
    paper_experiments/qblock_feas/run.sh rho_cufinufft.py
"""
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter

import cufinufft
import numpy as np
import torch

OUT = Path(__file__).resolve().parent / 'results'
OS = 1.25

# Horner ES kernel (upsampfac=2): nspread ≈ 3 at 1e-2, ≈5 at 1e-4, ≈7 at 1e-6.
EPS_W3 = 1e-2
EPS_W6 = 1e-6
EPS_DEFAULT = 1e-4


class _Timer:
    def __init__(self):
        self.t = {}

    def add(self, name, seconds):
        self.t[name] = self.t.get(name, 0.0) + seconds

    def mean(self, n):
        return {k: v / n for k, v in self.t.items()}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    try:
        import cupy as cp
        cp.cuda.Device().synchronize()
    except Exception:
        pass


def _cuda_s(fn):
    _sync()
    t0 = perf_counter()
    out = fn()
    _sync()
    return perf_counter() - t0, out


def _load_2d(name, torch_dev, sample_r=1):
    cfg = {'coco_spiral': 3, 'tilt_spi_invivo': 2}[name]
    fpath = f'./data/{name}'
    kw = {'weights_only': True, 'map_location': torch_dev}
    trj = torch.load(f'{fpath}/trj.pt', **kw).float()[:, ::cfg]
    d = trj.shape[-1]
    # Flattened stride so coco (already one interleave) still loses half its samples.
    if sample_r > 1:
        trj = trj.reshape(-1, d)[::sample_r]
    mps = torch.load(f'{fpath}/mps.pt', **kw).type(torch.complex64)
    return trj.contiguous(), mps


def _load_3d(torch_dev, sample_r=1):
    fpath = './data/coco_7t_spi'
    kw = {'weights_only': True, 'map_location': torch_dev}
    trj = torch.load(f'{fpath}/trj.pt', **kw).float()
    if sample_r > 1:
        trj = trj[:, :, ::sample_r]
    mps = torch.load(f'{fpath}/mps.pt', **kw).type(torch.complex64)
    return trj.contiguous(), mps


def _flop_ratio(im_size, M, os, nspread):
    n_os = [ceil(os * n) for n in im_size]
    nfft = float(np.prod(n_os))
    k = nspread ** len(im_size)
    return dict(N_os=n_os, Nfft=nfft, M=int(M), taps=int(M * k),
                M_over_Nfft=M / nfft, taps_over_Nfft=M * k / nfft,
                nspread=nspread)


def _nspread_horner(eps):
    """FINUFFT Horner table at upsampfac=2 (docs / common.cpp)."""
    if eps >= 1e-1:
        return 2
    if eps >= 1e-2:
        return 3
    if eps >= 1e-3:
        return 4
    if eps >= 1e-4:
        return 5
    if eps >= 1e-6:
        return 7
    return 9


def _pts_cufi(trj, im_size):
    """Map stored cycles/FOV coords into the open interval (-pi, pi)."""
    d = trj.shape[-1]
    trj_f = trj.reshape(-1, d).contiguous()
    im = torch.tensor(im_size, device=trj.device, dtype=trj_f.dtype)
    pts = torch.pi * trj_f / (im * 0.5)
    lim = torch.pi - 1e-4
    return pts.clamp(-lim, lim).contiguous()


def _make_interp_plan(n_os, n_trans, eps, pts):
    """
    Type-2 interpolate-only plan. n_os is the grid we interpolate from
    (already oversampled); upsampfac=2 keeps the Horner kernel.
    """
    kwargs = dict(
        n_trans=n_trans,
        eps=eps,
        isign=-1,
        dtype='complex64',
        gpu_spreadinterponly=1,
        gpu_method=1,
        gpu_sort=1,
        gpu_kerevalmeth=1,
        upsampfac=2.0,
        modeord=1,
    )
    try:
        plan = cufinufft.Plan(2, tuple(n_os), **kwargs)
    except RuntimeError:
        kwargs['upsampfac'] = 1.0
        kwargs['gpu_kerevalmeth'] = 0
        plan = cufinufft.Plan(2, tuple(n_os), **kwargs)
        kwargs['_fallback'] = 'upsampfac=1,kerevalmeth=0'
    d = pts.shape[-1]
    args = [pts[:, i].contiguous() for i in range(d)]
    plan.setpts(*args)
    return plan, kwargs


def profile_cufi(trj, mps, os, eps, coil_batch, n_reps=5):
    torch_dev = mps.device
    im_size = tuple(mps.shape[1:])
    C = mps.shape[0]
    d = trj.shape[-1]
    n_os = [ceil(os * n) for n in im_size]
    M = int(trj.reshape(-1, d).shape[0])
    # Use coil maps as the image so we do not keep a second  C * N^d  copy.
    img = mps
    pts = _pts_cufi(trj, im_size)

    n_trans = min(coil_batch, C)
    plan, plan_kw = _make_interp_plan(n_os, n_trans, eps, pts)

    def _fft_batch(c0, c1):
        return torch.fft.fftn(img[c0:c1], s=n_os, dim=tuple(range(-d, 0))).contiguous()

    # Warm-up: compile kernels, populate bins
    k_w = _fft_batch(0, n_trans)
    _ = plan.execute(k_w)
    del k_w
    _sync()

    acc = _Timer()
    for _ in range(n_reps):
        for c0 in range(0, C, coil_batch):
            c1 = min(c0 + coil_batch, C)
            if (c1 - c0) != plan.n_trans:
                plan, plan_kw = _make_interp_plan(n_os, c1 - c0, eps, pts)
            t_ft, k_os = _cuda_s(lambda c0=c0, c1=c1: _fft_batch(c0, c1))
            t_g, _ = _cuda_s(lambda p=plan, x=k_os: p.execute(x))
            acc.add('fft_all', t_ft)
            acc.add('gather', t_g)
            del k_os
    t = acc.mean(n_reps)
    t['rho'] = t['gather'] / t['fft_all']
    t['C'] = C
    t['coil_batch'] = coil_batch
    t['eps'] = eps
    t['W'] = _nspread_horner(eps)
    t['os'] = os
    t['im_size'] = list(im_size)
    t['plan'] = {k: v for k, v in plan_kw.items() if k != 'dtype'}
    t.update(_flop_ratio(im_size, M, os, t['W']))
    del plan
    return t


def _print(key, t):
    print(f'\n{key}')
    print(f'  im_size={tuple(t["im_size"])}  C={t["C"]}  M={t["M"]}  '
          f'N_os={t["N_os"]}  eps={t["eps"]}  nspread~{t["W"]}')
    print(f'  M/Nfft={t["M_over_Nfft"]:.3f}  taps/Nfft={t["taps_over_Nfft"]:.2f}')
    print(f'  fft             {t["fft_all"]*1e3:8.2f} ms')
    print(f'  gather/interp   {t["gather"]*1e3:8.2f} ms')
    print(f'  rho = gather/FFT = {t["rho"]:.3f}   '
          f'cap (1+rho)/rho = {(1 + t["rho"]) / t["rho"]:.2f}x')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-3d', action='store_true')
    ap.add_argument('--skip-3d-w6', action='store_true')
    ap.add_argument('--sample-r', type=int, default=1,
                    help='Extra readout stride after the dataset R (2 halves M).')
    ap.add_argument('--tag', default='',
                    help='Suffix for the output json, e.g. R2 -> rho_cufinufft_R2.json')
    args, _ = ap.parse_known_args()

    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}  cufinufft {cufinufft.__version__}  '
          f'sample_r={args.sample_r}')
    if torch_dev.type != 'cuda':
        raise SystemExit('cufinufft rho profiler needs a GPU')
    OUT.mkdir(exist_ok=True)
    out = {}

    for name in ('coco_spiral', 'tilt_spi_invivo'):
        trj, mps = _load_2d(name, torch_dev, sample_r=args.sample_r)
        print(f'\nloaded {name}: trj {tuple(trj.shape)}  mps {tuple(mps.shape)}  '
              f'trj range {trj.amin().item():.3f} .. {trj.amax().item():.3f}')
        for eps, tag in ((EPS_W3, 'W3'), (EPS_DEFAULT, 'eps1e-4'), (EPS_W6, 'W6')):
            key = f'cufi_{name}_{tag}'
            t = profile_cufi(trj, mps, OS, eps, coil_batch=mps.shape[0], n_reps=5)
            out[key] = t
            _print(key, t)
            del t
            torch.cuda.empty_cache()
        del trj, mps
        torch.cuda.empty_cache()

    if not args.skip_3d:
        print('\n===== 3D coco_7t_spi =====')
        trj, mps = _load_3d(torch_dev, sample_r=args.sample_r)
        print(f'  loaded trj {tuple(trj.shape)}  mps {tuple(mps.shape)}  '
              f'trj range {trj.amin().item():.3f} .. {trj.amax().item():.3f}')
        jobs = [(EPS_W3, 'W3', 1), (EPS_DEFAULT, 'eps1e-4', 1)]
        if not args.skip_3d_w6:
            jobs.append((EPS_W6, 'W6', 1))
        for eps, tag, cb in jobs:
            key = f'cufi_coco_7t_spi_{tag}'
            try:
                t = profile_cufi(trj, mps, OS, eps, coil_batch=cb, n_reps=3)
            except (RuntimeError, MemoryError) as e:
                print(f'  {key} failed at coil_batch={cb}: {e}')
                torch.cuda.empty_cache()
                t = profile_cufi(trj, mps, OS, eps, coil_batch=1, n_reps=2)
            out[key] = t
            _print(key, t)
            del t
            torch.cuda.empty_cache()

    tag = f'_{args.tag}' if args.tag else ''
    dest = OUT / f'rho_cufinufft{tag}.json'
    for v in out.values():
        v['sample_r'] = args.sample_r
    with open(dest, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'\nwrote {dest}')


if __name__ == '__main__':
    main()
