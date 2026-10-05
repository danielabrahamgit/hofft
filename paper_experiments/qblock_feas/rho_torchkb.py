"""
rho for torchkbnufft (LUT gather) on the same 2D problems as rho_sigpy.py.

torchkbnufft does not expose FT-only / interp-only, so FFT time is a matched
apod+pad+fftn of the same oversampled grid, and gather = full_forward - FFT.
That over-assigns any torchkb overhead to gather, so rho is an upper bound.

Run with:
    paper_experiments/qblock_feas/run.sh rho_torchkb.py
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from mr_recon.fourier.torchkb_nufft import torchkb_nufft

OUT = Path(__file__).resolve().parent / 'results'
OS = 1.25


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, n_reps=5):
    _sync(); fn(); _sync()
    ts = []
    for _ in range(n_reps):
        _sync()
        t0 = perf_counter()
        fn()
        _sync()
        ts.append(perf_counter() - t0)
    return float(np.median(ts))


def _load_2d(name, torch_dev):
    R = {'coco_spiral': 3, 'tilt_spi_invivo': 2}[name]
    kw = {'weights_only': True, 'map_location': torch_dev}
    trj = torch.load(f'./data/{name}/trj.pt', **kw).float()[:, ::R].contiguous()
    mps = torch.load(f'./data/{name}/mps.pt', **kw).type(torch.complex64)
    return trj, mps


def profile(name, W, torch_dev, n_reps=5):
    trj, mps = _load_2d(name, torch_dev)
    im_size = tuple(mps.shape[1:])
    C = mps.shape[0]
    d = trj.shape[-1]
    img = (mps * torch.randn(im_size, dtype=torch.complex64, device=torch_dev))
    trj_b = trj.reshape((1, -1, d)).contiguous()
    img_b = img[None]
    M = trj_b.shape[1]

    nft = torchkb_nufft(im_size, torch_dev=torch_dev, oversamp=OS, numpoints=W)
    trj_rs = nft.rescale_trajectory(trj_b)

    # Matched FFT: same oversampled size torchkb uses
    n_os = tuple(round(s * OS) for s in im_size)

    def _fft():
        x = torch.fft.fftshift(img_b, dim=tuple(range(-d, 0)))
        pad = [0, 0] * d
        # torch.nn.functional.pad is last-dim-first
        pads = []
        for i in reversed(range(d)):
            extra = n_os[i] - im_size[i]
            lo, hi = extra // 2, extra - extra // 2
            pads.extend([lo, hi])
        x = torch.nn.functional.pad(x, pads)
        return torch.fft.fftn(x, dim=tuple(range(-d, 0)))

    def _full():
        return nft.forward(img_b, trj_rs)

    t_fft = _timed(_fft, n_reps)
    t_full = _timed(_full, n_reps)
    t_g = max(t_full - t_fft, 0.0)
    tables = [getattr(nft.kb_ob, f'table_{i}') for i in range(d)]
    return dict(
        fft_all=t_fft, gather=t_g, full=t_full,
        rho=t_g / t_fft if t_fft > 0 else None,
        C=C, W=W, os=OS, im_size=list(im_size), M=int(M),
        N_os=list(n_os), table_bytes=int(sum(t.numel() * t.element_size() for t in tables)),
        table_lens=[int(t.numel()) for t in tables],
    )


def main():
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')
    OUT.mkdir(exist_ok=True)
    out = {}
    for name in ('coco_spiral', 'tilt_spi_invivo'):
        for W in (3, 6):
            key = f'torchkb_{name}_W{W}'
            t = profile(name, W, torch_dev)
            out[key] = t
            print(f'\n{key}')
            print(f'  im={tuple(t["im_size"])} C={t["C"]} M={t["M"]} W={t["W"]}  '
                  f'LUT {t["table_lens"]} ({t["table_bytes"]} B)')
            print(f'  fft (matched) {t["fft_all"]*1e3:8.2f} ms')
            print(f'  full NUFFT    {t["full"]*1e3:8.2f} ms')
            print(f'  gather est    {t["gather"]*1e3:8.2f} ms')
            print(f'  rho <= {t["rho"]:.3f}')
            torch.cuda.empty_cache()
    with open(OUT / 'rho_torchkb.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'\nwrote {OUT / "rho_torchkb.json"}')


if __name__ == '__main__':
    main()
