"""
rho for sigpy_nufft, and for a 3D problem matching coco_7t_spi.

Splits the NUFFT the same way the HOFFT profiler did: wall-clock of the oversampled FFT
versus wall-clock of the KB gather (sigpy's interpolate). rho = gather / FFT.

2D cases reuse the run_sweep grids and undersampling. The 3D case uses the stored
coco_7t_spi tensors as-is (320^3, 16 coils, 22k x 6 x 500 trajectory).

Run with:
    paper_experiments/qblock_feas/run.sh rho_sigpy.py
"""
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from mr_recon.fourier import sigpy_nufft

OUT = Path(__file__).resolve().parent / 'results'
OS = 1.25


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
    # perf_counter after a full torch+cupy sync. CUDA events go negative here because
    # sigpy's interpolate runs on a cupy stream the events do not wait on.
    _sync()
    t0 = perf_counter()
    out = fn()
    _sync()
    return perf_counter() - t0, out


def _load_2d(name, torch_dev):
    cfg = {'coco_spiral': 3, 'tilt_spi_invivo': 2}[name]
    fpath = f'./data/{name}'
    kw = {'weights_only': True, 'map_location': torch_dev}
    trj = torch.load(f'{fpath}/trj.pt', **kw).float()[:, ::cfg].contiguous()
    mps = torch.load(f'{fpath}/mps.pt', **kw).type(torch.complex64)
    return trj, mps


def _load_3d(torch_dev):
    fpath = './data/coco_7t_spi'
    kw = {'weights_only': True, 'map_location': torch_dev}
    trj = torch.load(f'{fpath}/trj.pt', **kw).float()
    mps = torch.load(f'{fpath}/mps.pt', **kw).type(torch.complex64)
    return trj, mps


def _flop_ratio(im_size, M, os, W):
    n_os = [ceil(os * n) for n in im_size]
    nfft = float(np.prod(n_os))
    k = W ** len(im_size)
    return dict(N_os=n_os, Nfft=nfft, M=int(M), taps=int(M * k),
                M_over_Nfft=M / nfft, taps_over_Nfft=M * k / nfft)


def profile_sigpy(trj, mps, os, W, coil_batch, n_reps=5):
    """
    Time sigpy_nufft.forward_FT_only vs forward_interp_only.

    Coils are the image-batch dim (one interpolate call). coil_batch limits GPU memory;
    times are summed over coil batches so they are totals for the full coil set.
    """
    torch_dev = mps.device
    im_size = tuple(mps.shape[1:])
    C = mps.shape[0]
    d = trj.shape[-1]
    trj_b = trj.reshape((1, -1, d)).contiguous()
    M = trj_b.shape[1]
    img = (mps * torch.randn(im_size, dtype=torch.complex64, device=torch_dev))

    nft = sigpy_nufft(im_size, oversamp=os, width=W)
    # Same beta the rest of the repo uses; skip the expensive optimal_beta sweep
    nft.beta = float(np.pi * max(((W / os) * (os - 0.5)) ** 2 - 0.8, 0.0) ** 0.5) or 1.0

    # Warm-up one coil so cuFFT / cupy kernels are compiled
    k0 = nft.forward_FT_only(img[:1][None])
    _ = nft.forward_interp_only(k0, trj_b)
    del k0
    _sync()

    acc = _Timer()
    for _ in range(n_reps):
        for c0 in range(0, C, coil_batch):
            c1 = min(c0 + coil_batch, C)
            x = img[c0:c1][None]
            t_ft, k_os = _cuda_s(lambda: nft.forward_FT_only(x))
            t_g, _ = _cuda_s(lambda: nft.forward_interp_only(k_os, trj_b))
            acc.add('fft_all', t_ft)
            acc.add('gather', t_g)
            del k_os
    t = acc.mean(n_reps)
    t['rho'] = t['gather'] / t['fft_all']
    t['C'] = C
    t['coil_batch'] = coil_batch
    t['W'] = W
    t['os'] = os
    t['im_size'] = list(im_size)
    t.update(_flop_ratio(im_size, M, os, W))
    return t


def _print(key, t):
    print(f'\n{key}')
    print(f'  im_size={tuple(t["im_size"])}  C={t["C"]}  M={t["M"]}  '
          f'N_os={t["N_os"]}  W={t["W"]}')
    print(f'  M/Nfft={t["M_over_Nfft"]:.3f}  taps/Nfft={t["taps_over_Nfft"]:.2f}')
    print(f'  fft+apod+pad {t["fft_all"]*1e3:8.2f} ms')
    print(f'  gather/interp {t["gather"]*1e3:8.2f} ms')
    print(f'  rho = gather/FFT = {t["rho"]:.3f}   '
          f'cap (1+rho)/rho = {(1 + t["rho"]) / t["rho"]:.2f}x')


def main():
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')
    OUT.mkdir(exist_ok=True)
    out = {}

    for name in ('coco_spiral', 'tilt_spi_invivo'):
        trj, mps = _load_2d(name, torch_dev)
        for W in (3, 6):
            key = f'sigpy_{name}_W{W}'
            t = profile_sigpy(trj, mps, OS, W, coil_batch=mps.shape[0], n_reps=5)
            out[key] = t
            _print(key, t)
            del t
            torch.cuda.empty_cache()
        del trj, mps
        torch.cuda.empty_cache()

    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-3d', action='store_true')
    ap.add_argument('--skip-3d-w6', action='store_true')
    args, _ = ap.parse_known_args()

    if not args.skip_3d:
        print('\n===== 3D coco_7t_spi =====')
        trj, mps = _load_3d(torch_dev)
        print(f'  loaded trj {tuple(trj.shape)}  mps {tuple(mps.shape)}')
        widths = ((3, 2),) if args.skip_3d_w6 else ((3, 2), (6, 1))
        for W, cb in widths:
            key = f'sigpy_coco_7t_spi_W{W}'
            try:
                t = profile_sigpy(trj, mps, OS, W, coil_batch=cb, n_reps=3)
            except RuntimeError as e:
                print(f'  {key} failed at coil_batch={cb}: {e}')
                torch.cuda.empty_cache()
                t = profile_sigpy(trj, mps, OS, W, coil_batch=1, n_reps=2)
            out[key] = t
            _print(key, t)
            del t
            torch.cuda.empty_cache()

    with open(OUT / 'rho_sigpy.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'\nwrote {OUT / "rho_sigpy.json"}')


if __name__ == '__main__':
    main()
