"""Diagnose why the tilt reference is not 0.1τ-stable."""
import sys
from pathlib import Path
from time import perf_counter

import torch

from mr_recon.recons import CG_SENSE_recon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, load_dataset, nrmse
from cufi_op import factor_phase_cur, make_dense_linop

TAU_BUDGET = 0.001


def recon(ds, spatial, temporal, eps, iters):
    A, nft = make_dense_linop(ds, spatial, temporal, eps=eps)
    img = CG_SENSE_recon(A, ds.ksp, max_iter=iters, max_eigen=1.0,
                         tolerance=1e-8, clear_gpu_mem=False, verbose=False)
    nft.clear_plans()
    return img


def main():
    torch_dev = torch.device('cuda')
    name = 'tilt_spi_invivo'
    print('probe tilt reference', flush=True)
    ds = load_dataset(name, torch_dev)
    mask = ds.mask
    print(f'  grid {ds.im_size} reduced={ds.reduced_im_size}  '
          f'time_red={ds.time_reduction_factor}', flush=True)

    print('  factor L=160', flush=True)
    spat160, temp160 = factor_phase_cur(ds, 160)
    print('  recon L=160 eps=1e-6 CG=20', flush=True)
    t0 = perf_counter()
    img20 = recon(ds, spat160, temp160, 1e-6, 20)
    print(f'    {perf_counter()-t0:.1f}s', flush=True)
    print('  recon L=160 eps=1e-6 CG=40', flush=True)
    t0 = perf_counter()
    img40 = recon(ds, spat160, temp160, 1e-6, 40)
    print(f'    {perf_counter()-t0:.1f}s', flush=True)
    d_cg = nrmse(img20, img40, mask)
    print(f'  Δ(CG 20 vs 40) = {d_cg:.4f}  budget={TAU_BUDGET}', flush=True)

    print('  factor L=224', flush=True)
    spat224, temp224 = factor_phase_cur(ds, 224)
    print('  recon L=224 eps=1e-6 CG=20', flush=True)
    t0 = perf_counter()
    img224 = recon(ds, spat224, temp224, 1e-6, 20)
    print(f'    {perf_counter()-t0:.1f}s', flush=True)
    d_L = nrmse(img20, img224, mask)
    print(f'  Δ(L=160 vs 224, CG=20) = {d_L:.4f}', flush=True)

    # Finer factorization grid
    print('\n  retry factorization at 400^2', flush=True)
    DATASETS[name]['reduced_im_size'] = (400, 400)
    ds4 = load_dataset(name, torch_dev)
    s160, t160 = factor_phase_cur(ds4, 160)
    s224, t224 = factor_phase_cur(ds4, 224)
    print('  recon 400^2 L=160', flush=True)
    i160 = recon(ds4, s160, t160, 1e-6, 20)
    print('  recon 400^2 L=224', flush=True)
    i224 = recon(ds4, s224, t224, 1e-6, 20)
    d4 = nrmse(i160, i224, mask)
    print(f'  Δ(L=160 vs 224 | 400^2 factors) = {d4:.4f}', flush=True)

    print('\n=== probe ===')
    print(f'  CG solver  {d_cg:.4f}  {"OK" if d_cg <= TAU_BUDGET else "NOT CONVERGED"}')
    print(f'  rank 200^2 {d_L:.4f}  {"OK" if d_L <= TAU_BUDGET else "UNSTABLE"}')
    print(f'  rank 400^2 {d4:.4f}  {"OK" if d4 <= TAU_BUDGET else "UNSTABLE"}')


if __name__ == '__main__':
    main()
