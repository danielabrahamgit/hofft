"""
Benchmark type3_nufft on the coco_3d pipeline (rescale + phase midpoints).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from mr_recon.utils import gen_grd
from mr_recon.linops import type3_nufft, encoding_matrix
from hofft.phase_coeffs import coco_to_phis_alphas, rescale_phis_alphas, apply_phase_midpoints
from mr_sim.phantoms import shepp_logan

# coco_3d trajectory helpers (avoid importing gen_data.py — it runs module-level sim)
DEFAULT_MAT = (
    '/local_mount/space/mayday/data/users/xc/share/for_Daniel/spiral_7T_flash/'
    'knew_v2_fov240_res075_int6_2213624000_t1t10_g10s400.mat'
)
N_TP, N_SPIRAL, N_SPIRAL_CIRCLE = 22136, 18, 6
BETA_STEP = 0.131267444 * np.pi
K_NORM, FOV = 0.5, 0.22


def load_2d_ktrajectory(mat_path, n_tp=N_TP):
    mat = loadmat(mat_path)
    k_adc = mat['k_adc']
    k_complex = k_adc[:n_tp, 0] + 1j * k_adc[:n_tp, 1]
    k_complex = k_complex / np.abs(k_complex).max() * K_NORM
    return k_complex.real.copy(), k_complex.imag.copy()


def build_k3d(kx, ky, n_spiral=N_SPIRAL, n_spiral_circle=N_SPIRAL_CIRCLE, n_rot=500,
              init_xita=0.0, init_beta=0.0, beta_step=BETA_STEP):
    n_tp = kx.shape[0]
    k_3d = np.zeros((n_tp, 3, n_spiral, n_rot), dtype=np.float32)
    kx0, ky0 = kx.astype(np.float64), ky.astype(np.float64)
    for nn in range(n_rot):
        for ii in range(n_spiral):
            beta = ((ii % n_spiral_circle) + 1 + nn - 1) * beta_step + init_beta
            xita = (ii % n_spiral_circle) * 2 * np.pi / n_spiral_circle + init_xita
            cos_x, sin_x = np.cos(xita), np.sin(xita)
            cos_b, sin_b = np.cos(beta), np.sin(beta)
            k_temp_x = cos_x * kx0 + sin_x * ky0
            k_temp_y = -sin_x * kx0 + cos_x * ky0
            if ii < n_spiral_circle:
                k_3d[:, 0, ii, nn] = k_temp_x
                k_3d[:, 1, ii, nn] = cos_b * k_temp_y
                k_3d[:, 2, ii, nn] = -sin_b * k_temp_y
            elif ii < 2 * n_spiral_circle:
                k_3d[:, 0, ii, nn] = -sin_b * k_temp_y
                k_3d[:, 1, ii, nn] = k_temp_x
                k_3d[:, 2, ii, nn] = cos_b * k_temp_y
            else:
                k_3d[:, 0, ii, nn] = cos_b * k_temp_y
                k_3d[:, 1, ii, nn] = -sin_b * k_temp_y
                k_3d[:, 2, ii, nn] = k_temp_x
    return k_3d


def load_coco_trj(n_rot, im_size=(310,) * 3, dev='cuda:0'):
    kx, ky = load_2d_ktrajectory(DEFAULT_MAT)
    trj = build_k3d(kx, ky, n_rot=n_rot)
    trj = torch.from_numpy(trj).float().to(dev).moveaxis(1, -1) * im_size[0]
    return trj


def rel_err(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def prep_coco(trj, im_size, fov=0.22, field_strength=7.0, ro_dim=0, dt=1e-6):
    spatial_crds = gen_grd(im_size, (fov,) * len(im_size)).to(trj.device)
    phis, alphas = coco_to_phis_alphas(trj / fov, spatial_crds, field_strength, ro_dim, dt)
    phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)
    spat, temp = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp)
    return phis_nrm, alphas_nrm, spat, temp


def simulate_type3(mps, phis, alphas, spat, temp, img):
    t3n = type3_nufft(phis, alphas)
    C = mps.shape[0]
    ksp = torch.zeros((C, *alphas.shape[1:]), dtype=torch.complex64, device=img.device)
    for c in range(C):
        ksp[c] = t3n.forward((mps[c] * spat * img)[None])[0]
    return ksp * temp


def simulate_naive(mps, phis, alphas, spat, temp, img, temporal_batch_size=512):
    A = encoding_matrix(mps * spat, phis, alphas, temporal_batch_size=temporal_batch_size)
    return A(img) * temp


def adjoint_test(t3n, img):
    ksp = torch.randn(t3n.oshape, dtype=torch.complex64, device=img.device)
    Ax = t3n.forward(img[None])[0]
    Atx = t3n.adjoint(ksp[None])[0]
    lhs = (Ax.conj() * ksp).sum()
    rhs = (img.conj() * Atx).sum()
    return abs(lhs - rhs).item() / (abs(lhs).item() + 1e-12)


def benchmark(trj, im_size, label, dev='cuda:0', run_naive=True):
    torch_dev = torch.device(dev)
    trj = trj.to(torch_dev)
    phis, alphas, spat, temp = prep_coco(trj, im_size)
    mps = torch.ones((1, *im_size), dtype=torch.complex64, device=torch_dev)
    img = torch.randn(im_size, dtype=torch.complex64, device=torch_dev)

    R = int(np.prod(im_size))
    T = int(np.prod(alphas.shape[1:]))
    t3n = type3_nufft(phis, alphas)

    print(f'\n=== {label} ===')
    print(f'  im_size={im_size}, trj={tuple(trj.shape[:-1])}, R={R:,}, T={T:,}')
    print(f'  phis max per basis: {phis.abs().amax(dim=tuple(range(1, phis.ndim))).tolist()}')
    print(f'  alphas max per basis: {alphas.abs().amax(dim=tuple(range(1, alphas.ndim))).tolist()}')
    print(f'  type3 grd_N_os={t3n.grd_N_os}, grid_voxels={int(np.prod(t3n.grd_N_os)):,}')

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    y_t3 = simulate_type3(mps, phis, alphas, spat, temp, img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_t3 = time.perf_counter() - t0
    print(f'  type3 forward: {t_t3:.3f}s')

    adj_err = adjoint_test(t3n, mps[0] * spat * img)
    print(f'  type3 adjoint inner-product test: {adj_err:.2e}')

    if run_naive and R * T <= 5e7:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_nv = simulate_naive(mps, phis, alphas, spat, temp, img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_nv = time.perf_counter() - t0
        err = rel_err(y_t3, y_nv)
        print(f'  naive forward: {t_nv:.3f}s  rel err={err:.2e}  speedup={t_nv/t_t3:.1f}x')
    elif run_naive:
        print(f'  naive skipped (R*T={R*T:.2e} too large)')


if __name__ == '__main__':
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    im_size = (310,) * 3

    # subsampled real trajectory — naive reference fits in memory
    for n_rot in (5, 20):
        trj = load_coco_trj(n_rot, im_size=im_size, dev=dev)
        benchmark(trj, im_size, f'real coco_3d, N_ROT={n_rot}', dev=dev, run_naive=(n_rot <= 5))

    # full coco_3d scale (500 rots) — type3 only
    trj_full = load_coco_trj(500, im_size=im_size, dev=dev)
    benchmark(trj_full, im_size, 'real coco_3d FULL (500 rots, 310^3)', dev=dev, run_naive=False)
