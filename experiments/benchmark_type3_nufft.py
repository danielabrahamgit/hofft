"""
Benchmark type3_nufft accuracy vs naive reference and profile timing breakdown.
"""
import time
import torch
import numpy as np

from mr_recon.utils import gen_grd
from mr_recon.linops import type3_nufft, type3_nufft_naive, encoding_matrix
from hofft.phase_coeffs import coco_to_phis_alphas


def make_coco_problem(im_size, trj_shape, fov=0.22, dev='cuda:0'):
    torch_dev = torch.device(dev)
    trj = torch.randn(*trj_shape, 3, device=torch_dev) * im_size[0] * 0.4
    spatial_crds = gen_grd(im_size, (fov,) * len(im_size)).to(torch_dev)
    phis, alphas = coco_to_phis_alphas(
        trj / fov, spatial_crds, field_strength=7.0, ro_dim=0, dt=1e-6)
    img = torch.randn(im_size, dtype=torch.complex64, device=torch_dev)
    return phis, alphas, img


def rel_err(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def adjoint_inner_test(A, img, ksp=None):
    if ksp is None:
        ksp = torch.randn(A.oshape, dtype=torch.complex64, device=img.device)
    Ax = A.forward(img[None])[0]
    Aty = A.adjoint(ksp[None])[0]
    lhs = (Ax.conj() * ksp).sum()
    rhs = (img.conj() * Aty).sum()
    return abs(lhs - rhs).item() / (abs(lhs).item() + 1e-12)


def benchmark_case(im_size, trj_shape, dev='cuda:0', width=4.0, oversamp=1.25):
    phis, alphas, img = make_coco_problem(im_size, trj_shape, dev=dev)
    R = int(np.prod(im_size))
    T = int(np.prod(alphas.shape[1:]))

    naive = type3_nufft_naive(phis, alphas)
    t3 = type3_nufft(phis, alphas, oversamp=oversamp, width=width)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    y_naive = naive.forward(img[None])[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_naive = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    y_t3 = t3.forward(img[None])[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_t3 = time.perf_counter() - t0

    fwd_err = rel_err(y_t3, y_naive)
    adj_err = adjoint_inner_test(t3, img)

    # Adjoint accuracy vs naive (small only)
    if T <= 50000:
        At_naive = naive.adjoint(y_naive[None])[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        At_t3 = t3.adjoint(y_naive[None])[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_adj = time.perf_counter() - t0
        adj_vs_naive = rel_err(At_t3, At_naive)
    else:
        t_adj = float('nan')
        adj_vs_naive = float('nan')

    print(f'im_size={im_size}, trj={trj_shape}, R={R:,}, T={T:,}, B={phis.shape[0]}')
    print(f'  grd_N_os={t3.grd_N_os}, width={width}, oversamp={oversamp}')
    print(f'  forward: naive={t_naive:.3f}s  type3={t_t3:.3f}s  speedup={t_naive/t_t3:.1f}x')
    print(f'  forward rel err vs naive: {fwd_err:.2e}')
    print(f'  adjoint inner-product test: {adj_err:.2e}')
    if not np.isnan(adj_vs_naive):
        print(f'  adjoint rel err vs naive: {adj_vs_naive:.2e}  adjoint time={t_adj:.3f}s')
    print()


def sweep_width_oversamp(im_size=(48, 48, 48), trj_shape=(128, 4, 2), dev='cuda:0'):
    phis, alphas, img = make_coco_problem(im_size, trj_shape, dev=dev)
    naive = type3_nufft_naive(phis, alphas)
    y_ref = naive.forward(img[None])[0]
    print('Width / oversamp sweep (forward rel err vs naive):')
    for width in [2.0, 3.0, 4.0, 6.0, 8.0]:
        for oversamp in [1.25, 1.5, 2.0]:
            t3 = type3_nufft(phis, alphas, oversamp=oversamp, width=width)
            y = t3.forward(img[None])[0]
            print(f'  W={width:.1f} os={oversamp:.2f}  err={rel_err(y, y_ref):.2e}  grd={t3.grd_N_os}')
    print()


if __name__ == '__main__':
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {dev}\n')

    sweep_width_oversamp(dev=dev)

    benchmark_case((32, 32, 32), (64, 2, 2), dev=dev)
    benchmark_case((48, 48, 48), (128, 4, 2), dev=dev)
    benchmark_case((64, 64, 64), (256, 4, 4), dev=dev)
    benchmark_case((64, 64, 64), (512, 8, 4), dev=dev)
