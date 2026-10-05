"""
Sec. 3.3 -- split the existing ``hofft_linop`` forward wall-clock into FFT vs gather vs
everything else, and report ``rho = gather / FFT``.

This number matters independently of the block study: ``rho`` caps the payoff of *any*
method that buys speed by shrinking FFTs. The block model's gather cost never decreases
(Sec. 1.3), so its best possible speedup is ``(1 + rho) / rho``.

The forward pass is replicated section by section rather than timed as a whole, because
the sections are the quantity of interest. It mirrors ``hofft_linop.forward`` exactly:
apodize -> pad -> FFT -> gather stencil blocks -> contract with kernel weights.

Run with:
    paper_experiments/qblock_feas/run.sh rho_profile.py
"""
import json
import sys

import numpy as np
import torch

from pathlib import Path

from einops import einsum
from mr_recon._func.indexing import multi_index
from mr_recon.fourier import fft
from mr_recon.utils import batch_iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REAL, load_real  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'


class _Sect:
    """CUDA-event timer for one named section, accumulated over calls."""

    def __init__(self):
        self.t = {}

    def time(self, name, fn):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        self.t[name] = self.t.get(name, 0.0) + s.elapsed_time(e) / 1e3
        return out


def profile_forward(linop, img, n_reps=5):
    """
    Time ``hofft_linop.forward`` section by section.

    Args
    ----
    linop : hofft_linop
        Operator to profile
    img : torch.Tensor
        Input image with shape ``im_size``
    n_reps : int
        Repetitions after one warm-up

    Returns
    -------
    dict
        Seconds per section plus ``rho`` and ``total``
    """
    D = linop.idx_kerns.shape[-1]
    C = linop.mps.shape[0]
    L = linop.kern_weights.shape[0]
    cbs = linop.bparams.coil_batch_size or C
    fbs = linop.bparams.field_batch_size or L

    for rep in range(n_reps + 1):
        sect = _Sect() if rep else _Sect()          # discard the warm-up
        ksp = torch.zeros(linop.oshape, device=img.device, dtype=torch.complex64)
        for c1, c2 in batch_iterator(C, cbs):
            Sx = sect.time('coils', lambda: linop.mps[c1:c2] * img)
            for l1, l2 in batch_iterator(L, fbs):
                MSx = sect.time('apod', lambda: einsum(
                    Sx, linop.spatial_factors[l1:l2], 'C ..., L ... -> C L ...'))

                def _fft():
                    z = linop.padder(MSx)
                    Z = fft(z, dim=tuple(range(-D, 0)))
                    Z *= np.prod(Z.shape[-D:]) ** 0.5 / np.prod(linop.im_size) ** 0.5
                    return Z
                FMSx = sect.time('fft', _fft)

                def _gather():
                    b = multi_index(FMSx, D, linop.idx_kerns)
                    return b.moveaxis(-1, 2)
                blocks = sect.time('gather', _gather)

                sect.time('kern', lambda: einsum(
                    blocks, linop.kern_weights[l1:l2],
                    'C L K ..., L K ... -> C ...'))
        del ksp

    t = {k: v / n_reps for k, v in sect.t.items()}
    t['total'] = sum(t.values())
    # The stencil contraction is the second half of the gather: it touches L*K*M taps and
    # disappears along with them, so it is charged to gather, not to "other".
    t['gather_all'] = t['gather'] + t['kern']
    t['rho'] = t['gather_all'] / t['fft']
    return t


def build_linop(name, torch_dev, L=8, W=3):
    """A representative ``hofft_linop`` for one dataset, via the published pipeline."""
    from mr_recon.linops import batching_params
    from mr_recon.algs import density_compensation
    from hofft.decomp import hofft_params
    from hofft.pipelines import svd_decomp_linop

    cfg = REAL[name]
    fpath = f'./data/{name}'
    kw = {'weights_only': True, 'map_location': torch_dev}
    trj = torch.load(f'{fpath}/trj.pt', **kw).float()[:, ::cfg['R']].contiguous()
    mps = torch.load(f'{fpath}/mps.pt', **kw).type(torch.complex64)
    evals = torch.load(f'{fpath}/evals.pt', **kw).float()
    phis = torch.load(f'{fpath}/phis.pt', **kw).float()
    alphas = torch.load(f'{fpath}/alphas.pt', **kw).float()[:, :, ::cfg['R']].contiguous()

    im_size = tuple(mps.shape[1:])
    d = trj.shape[-1]
    mask = (evals > 0.9).float()
    mps = mps * mask

    from hofft.phase_coeffs import remove_linear_terms
    phis, trj_term, _ = remove_linear_terms(phis, alphas, mask=mask)
    trj = trj + trj_term
    phis = phis * mask
    B = phis.shape[0]
    energy = (phis.reshape((B, -1)).abs().mean(dim=1)
              * alphas.reshape((B, -1)).abs().mean(dim=1))
    keep = torch.argwhere(energy > 1e-6)[:, 0]
    phis, alphas = phis[keep].contiguous(), alphas[keep].contiguous()

    dcf = density_compensation(trj, im_size)
    bparams = batching_params(coil_batch_size=mps.shape[0] // 2)
    hparams = hofft_params(
        (W,) * d, 1.25, L, reduced_im_size=cfg['im_size_s'], spatial_init='seg',
        normalize_coeffs=True, kalpha_method='maxmin', time_reduction_factor=10,
        num_compressed_bases=None, cur_rank=500, solver='solve', lamda=1e-3,
        max_als_iter=10, verbose=False)
    hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0]
    hparams.matvec_kwargs = {'spatial_batch_size': 2 ** 10}

    torch.manual_seed(0)
    A = svd_decomp_linop(phis=phis, alphas=alphas, mps=mps, trj=trj, dcf=dcf,
                         spatial_mask=mask, hparams=hparams, bparams=bparams,
                         svd_method='cur')
    return A, im_size, trj.reshape((-1, d)).shape[0]


def main():
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')
    OUT.mkdir(exist_ok=True)
    out = {}
    for name in REAL:
        for L, W in ((8, 3), (30, 3)):
            A, im_size, M = build_linop(name, torch_dev, L=L, W=W)
            inner = A.linops[-1] if hasattr(A, 'linops') else A
            while not hasattr(inner, 'idx_kerns'):
                inner = getattr(inner, 'linop', None) or getattr(inner, 'A', None)
                if inner is None:
                    raise RuntimeError('could not find the hofft_linop inside the sense op')
            img = torch.randn(im_size, dtype=torch.complex64, device=torch_dev)
            t = profile_forward(inner, img)
            key = f'{name}_L{L}_W{W}'
            out[key] = dict(t, im_size=list(im_size), M=int(M), L=L, W=W)
            frac = {k: t[k] / t['total'] for k in ('fft', 'gather_all', 'apod', 'coils')}
            print(f'\n{key}: im_size={im_size}, M={M}')
            print(f'  fft        {t["fft"] * 1e3:8.2f} ms  ({frac["fft"]:5.1%})')
            print(f'  gather     {t["gather"] * 1e3:8.2f} ms')
            print(f'  stencil    {t["kern"] * 1e3:8.2f} ms')
            print(f'  gather_all {t["gather_all"] * 1e3:8.2f} ms  ({frac["gather_all"]:5.1%})')
            print(f'  apod       {t["apod"] * 1e3:8.2f} ms  ({frac["apod"]:5.1%})')
            print(f'  coils      {t["coils"] * 1e3:8.2f} ms  ({frac["coils"]:5.1%})')
            print(f'  rho = gather/FFT = {t["rho"]:.3f}   '
                  f'cap on any FFT-only speedup = {(1 + t["rho"]) / t["rho"]:.1f}x')
            del A, inner, img
            torch.cuda.empty_cache()

    with open(OUT / 'rho_profile.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'\nwrote {OUT / "rho_profile.json"}')


if __name__ == '__main__':
    main()
