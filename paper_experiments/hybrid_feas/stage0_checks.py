"""
Priority 0 of math_docs/hybrid_feas.md.

Inventory both datasets, estimate memory, establish a stock-cuFINUFFT dense
forward/adjoint, and run tiny direct-sum plus adjoint-identity checks.

Stop performance work if these fail.

Run with:
    paper_experiments/hybrid_feas/run.sh stage0_checks.py
"""
import json
import sys

from pathlib import Path

import cufinufft
import numpy as np
import torch

from hofft.utils import gen_grd
from mr_recon.fourier.misc_nuffts import cufi_nufft as StockCufi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, load_dataset, memory_estimate  # noqa: E402
from cufi_op import CufiNUFFT, factor_phase_cur, make_dense_linop  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

ADJ_TOL_F32 = 1e-5
DIRECT_TOL = 2e-3          # tight-eps NUFFT vs explicit sum, relative


def _versions(torch_dev):
    info = dict(
        cufinufft=getattr(cufinufft, '__version__', 'unknown'),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=str(torch_dev),
    )
    if torch_dev.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(0)
        info['gpu_mem_GiB'] = torch.cuda.get_device_properties(0).total_memory / 2**30
    return info


def _explicit_nufft(img, trj, im_size):
    """Direct sum oracle: y_m = sum_n x_n exp(-2j pi k_m·r_n) / sqrt(N)."""
    rs = gen_grd(im_size).to(img.device).reshape(-1, len(im_size))
    ks = trj.reshape(-1, len(im_size))
    x = img.reshape(-1)
    y = torch.exp(-2j * np.pi * (ks @ rs.T)) @ x
    return y / float(np.prod(im_size)) ** 0.5


def test_tiny_direct(torch_dev, eps=1e-6):
    """Random 16x16 image, 48 samples. Plan NUFFT vs explicit sum."""
    print('\n=== tiny direct-sum (no phase) ===')
    im_size = (16, 16)
    M = 48
    gen = torch.Generator(device=torch_dev)
    gen.manual_seed(0)
    img = (torch.randn(im_size, generator=gen, device=torch_dev)
           + 1j * torch.randn(im_size, generator=gen, device=torch_dev)).type(torch.complex64)
    trj = (torch.rand(M, 2, generator=gen, device=torch_dev) - 0.5) * torch.tensor(
        im_size, device=torch_dev)
    nft = CufiNUFFT(im_size, eps=eps)
    trj_pi = nft.rescale_trajectory(trj)
    y_op = nft.forward(img[None, None], trj_pi[None])[0, 0]
    y_ex = _explicit_nufft(img, trj, im_size)
    rel = float((y_op - y_ex).norm() / y_ex.norm())
    print(f'  ||NUFFT - DFT|| / ||DFT|| = {rel:.3e}  (eps={eps})')
    nft.clear_plans()
    return rel < DIRECT_TOL, rel


def test_tiny_adjoint(torch_dev, eps=1e-6):
    print('\n=== tiny adjoint identity ===')
    im_size = (16, 16)
    M = 48
    gen = torch.Generator(device=torch_dev)
    gen.manual_seed(1)
    img = (torch.randn(im_size, generator=gen, device=torch_dev)
           + 1j * torch.randn(im_size, generator=gen, device=torch_dev)).type(torch.complex64)
    z = (torch.randn(M, generator=gen, device=torch_dev)
         + 1j * torch.randn(M, generator=gen, device=torch_dev)).type(torch.complex64)
    trj = (torch.rand(M, 2, generator=gen, device=torch_dev) - 0.5) * torch.tensor(
        im_size, device=torch_dev)
    nft = CufiNUFFT(im_size, eps=eps)
    trj_pi = nft.rescale_trajectory(trj)
    Ax = nft.forward(img[None, None], trj_pi[None])[0, 0]
    Ahz = nft.adjoint(z[None, None], trj_pi[None])[0, 0]
    lhs = torch.vdot(Ax.reshape(-1), z.reshape(-1))
    rhs = torch.vdot(img.reshape(-1), Ahz.reshape(-1))
    rel = float((lhs - rhs).abs() / (lhs.abs() + rhs.abs()).clamp(min=1e-30) * 2)
    print(f'  <Ax,z>={lhs.item():.6e}  <x,A*z>={rhs.item():.6e}  rel={rel:.3e}')
    nft.clear_plans()
    return rel < ADJ_TOL_F32, rel


def test_stock_matches_plan(torch_dev, eps=1e-4):
    """Plan wrapper vs the installed cufi_nufft one-shot API."""
    print('\n=== Plan wrapper vs stock cufi_nufft ===')
    im_size = (32, 32)
    M = 200
    gen = torch.Generator(device=torch_dev)
    gen.manual_seed(2)
    # Stock cufi_nufft.forward mishandles img_batch > 1 (reshape includes N).
    # Compare on the (N=1, no extra batch) layout it actually supports.
    img = (torch.randn((1, *im_size), generator=gen, device=torch_dev)
           + 1j * torch.randn((1, *im_size), generator=gen, device=torch_dev)
           ).type(torch.complex64)
    trj = (torch.rand(1, M, 2, generator=gen, device=torch_dev) - 0.5) * torch.tensor(
        im_size, device=torch_dev)
    plan = CufiNUFFT(im_size, eps=eps)
    stock = StockCufi(im_size, eps=eps)
    trj_pi = plan.rescale_trajectory(trj)
    y_p = plan.forward(img, trj_pi)
    y_s = stock.forward(img, trj_pi)
    rel = float((y_p - y_s).norm() / y_s.norm())
    print(f'  ||plan - stock|| / ||stock|| = {rel:.3e}')
    plan.clear_plans()
    return rel < 5e-3, rel


def test_dataset_adjoint(ds, L=2, eps=1e-4):
    print(f'\n=== dataset adjoint ({ds.name}, L={L}) ===')
    spatial, temporal = factor_phase_cur(ds, L)
    A, nft = make_dense_linop(ds, spatial, temporal, eps=eps,
                              coil_batch=1, field_batch=1)
    x = (torch.randn(ds.im_size, device=ds.mps.device)
         + 1j * torch.randn(ds.im_size, device=ds.mps.device)).type(torch.complex64)
    x = x * ds.mask
    z = (torch.randn_like(ds.ksp)
         + 1j * torch.randn_like(ds.ksp))
    Ax = A.forward(x)
    Ahz = A.adjoint(z)
    # sense_linop.adjoint applies DCF: A.H(z) := F^H W z. Inner product is <Ax, Wz>.
    Wz = z * ds.dcf[None]
    lhs = torch.vdot(Ax.reshape(-1), Wz.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), Ahz.reshape(-1))
    rel = float((lhs - rhs).abs() / (lhs.abs() + rhs.abs()).clamp(min=1e-30) * 2)
    print(f'  <Ax,Wz>={lhs.item():.6e}  <x,A*z>={rhs.item():.6e}  rel={rel:.3e}')
    nft.clear_plans()
    return rel < ADJ_TOL_F32 * 20, rel   # full operator, slightly looser


def inventory(ds):
    mem = memory_estimate(ds)
    print(f'\n===== {ds.name} =====')
    print(f'  im_size {ds.im_size}  C={ds.C}  B={ds.B}  trj {ds.trj_size}  '
          f'M={ds.M}  N={ds.N}  n_mask={ds.n_mask}')
    print(f'  trj range {float(ds.trj.amin()):.3f} .. {float(ds.trj.amax()):.3f}  '
          f'os={ds.os:.4f}  R={ds.R}')
    print(f'  phis {tuple(ds.phis.shape)}  alphas {tuple(ds.alphas.shape)}')
    print(f'  ksp {tuple(ds.ksp.shape)}  dcf {tuple(ds.dcf.shape)}')
    print(f'  img_gt={"yes" if ds.img_gt is not None else "no"}  '
          f'img_ee={"yes" if ds.img_ee is not None else "no"}')
    print(f'  factorization grid {ds.reduced_im_size}  '
          f'time_stride={ds.time_reduction_factor}')
    print(f'  dense P would be {mem["P_dense_GiB"]:.2f} GiB  '
          f'({"INFEASIBLE — use CUR/sampled SVD" if mem["P_infeasible"] else "fits"})')
    print(f'  factors@L=32 {mem["factors_MiB"]:.1f} MiB  '
          f'NUFFT batch {mem["nufft_batch_MiB"]:.1f} MiB  '
          f'ksp {mem["ksp_MiB"]:.1f} MiB')
    return mem


def main():
    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ver = _versions(torch_dev)
    print('Priority 0 — inventory + cuFINUFFT dense F/A checks')
    print(f'  {ver}')
    if torch_dev.type != 'cuda':
        print('BLOCKER: no GPU. Do not invent timings.')
        return 2

    checks = []
    ok, rel = test_tiny_direct(torch_dev)
    checks.append(('tiny_direct', ok, rel))
    ok, rel = test_tiny_adjoint(torch_dev)
    checks.append(('tiny_adjoint', ok, rel))
    ok, rel = test_stock_matches_plan(torch_dev)
    checks.append(('plan_vs_stock', ok, rel))

    out = dict(versions=ver, checks={}, datasets={})
    for name in DATASETS:
        ds = load_dataset(name, torch_dev)
        mem = inventory(ds)
        ok, rel = test_dataset_adjoint(ds, L=2, eps=1e-4)
        checks.append((f'adjoint_{name}', ok, rel))
        out['datasets'][name] = dict(
            im_size=list(ds.im_size), C=ds.C, B=ds.B, M=ds.M, N=ds.N,
            n_mask=ds.n_mask, trj_size=list(ds.trj_size), os=ds.os, R=ds.R,
            memory=mem, adjoint_rel=rel,
        )
        del ds
        torch.cuda.empty_cache()

    print('\n================ PRIORITY 0 ================')
    all_ok = True
    for name, ok, rel in checks:
        tag = 'PASS' if ok else 'FAIL'
        all_ok &= ok
        print(f'  [{tag}] {name:<22s}  rel={rel:.3e}')
        out['checks'][name] = dict(ok=ok, rel=rel)
    out['passed'] = all_ok
    (OUT / 'stage0.json').write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote {OUT / "stage0.json"}')
    if not all_ok:
        print('Priority 0 FAIL. Stop performance work.')
        return 1
    print('Priority 0 PASS. Dense cuFINUFFT F/A is usable.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
