"""Find a CG stopping rule whose optimization error is below 0.1τ."""
import sys
from pathlib import Path
from time import perf_counter

import torch

from mr_recon.algs import power_method_operator
from mr_recon.recons import CG_SENSE_recon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_dataset, nrmse
from cufi_op import factor_phase_cur, make_dense_linop

BUDGET = 0.001


def _recon(A, ksp, iters, max_eigen, verbose=False):
    return CG_SENSE_recon(A, ksp, max_iter=iters, max_eigen=max_eigen,
                          tolerance=1e-8, clear_gpu_mem=False, verbose=verbose)


def run_one(name, L, iters_list, n_power=6):
    torch_dev = torch.device('cuda')
    ds = load_dataset(name, torch_dev)
    print(f'\n==== {name} L={L} ====', flush=True)
    spatial, temporal = factor_phase_cur(ds, L)
    A, nft = make_dense_linop(ds, spatial, temporal, eps=1e-6)

    print(f'  power method {n_power} iters on A.normal', flush=True)
    x0 = torch.randn(A.ishape, dtype=torch.complex64, device=torch_dev)
    t0 = perf_counter()
    _, lam = power_method_operator(A.normal, x0, num_iter=n_power, verbose=True)
    print(f'  λ_max ≈ {lam:.4g}  ({perf_counter()-t0:.1f}s)', flush=True)

    imgs = {}
    for eigen_tag, eigen in (('fixed1', 1.0), ('power', lam)):
        prev = None
        for n in iters_list:
            t0 = perf_counter()
            img = _recon(A, ds.ksp, n, eigen)
            dt = perf_counter() - t0
            imgs[(eigen_tag, n)] = img
            line = f'  CG={n:<3d} eigen={eigen_tag:<6s}  {dt:.1f}s'
            if prev is not None:
                d = nrmse(img, prev, ds.mask)
                line += f'  Δ(vs {n//2})={d:.4f}{"  OK" if d <= BUDGET else ""}'
            print(line, flush=True)
            prev = img

    nft.clear_plans()
    return dict(name=name, L=L, lam=lam, iters=iters_list)


def main():
    # coco is cheap; tilt uses L=32 so a normal is ~15s not 75s.
    run_one('coco_spiral', 12, [20, 40, 80])
    run_one('tilt_spi_invivo', 32, [20, 40, 80], n_power=5)


if __name__ == '__main__':
    main()
