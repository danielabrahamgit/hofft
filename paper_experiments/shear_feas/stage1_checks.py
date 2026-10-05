"""
Follow-up checks on the Stage 1 result.

1. Is the gather index periodic modulo the oversampled grid? feas_test.md Sec 1.6 assumes
   it is not ("wrapping is not an option -- it aliases") and charges the shear a grid
   padding factor rho_l. For a discrete voxel image on an even grid with integer os*N,
   the model term is exactly periodic, which would make the shear free.
2. Has the Stage 1 ALS converged, or is the sheared variant just under-iterated?
3. Does the shear increase the marginal benefit of a wider stencil W, as the
   local-linearization story predicts?

Run with:
    PYTHONPATH=src python paper_experiments/shear_feas/stage1_checks.py
"""
import torch
import numpy as np

from pathlib import Path
from einops import einsum

from hofft.utils import gen_grd, reduce_spatial
from hofft.decomp import build_kern_bases
from hofft.shear import (
    alpha_second_moment, spatial_jacobian, kaffine_clustering,
    init_label_candidates, shear_trajectory, soft_membership,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_dataset  # noqa: E402
from stage1_dense import build_dense_problem, als_dense, svd_init, CONFIG  # noqa: E402


def check_wrap_exactness(torch_dev):
    """The HOFFT gather term must be invariant to shifting the index by os*N."""
    print('\n--- 1. gather-index periodicity ---')
    for N, os_nom in ((276, 1.25), (734, 1.25), (200, 1.5)):
        os = 2 * round(os_nom * N / 2) / N
        grid = os * N
        r = gen_grd((N,)).to(torch_dev)[:, 0]
        z = torch.tensor([-3.0, 0.0, 7.0, 41.0], device=torch_dev)
        base = torch.exp(-2j * np.pi * r[None] * z[:, None] / os)
        wrapped = torch.exp(-2j * np.pi * r[None] * (z[:, None] + grid) / os)
        err = float((base - wrapped).abs().max())
        print(f'  N={N:<4d} os={os:.4f}  os*N={grid:g} (int={grid == round(grid)})  '
              f'max|term(z) - term(z + os*N)| = {err:.3e}')


def check_convergence(name, torch_dev, L, iters=(4, 8, 16, 32)):
    """Rerun the Stage 1 fit at several ALS budgets to see if the ranking is stable."""
    print(f'\n--- 2. ALS convergence, {name}, L={L} ---')
    cfg = CONFIG[name]
    ds = load_dataset(name, torch_dev)
    prob = build_dense_problem(ds, cfg['im_size_s'], cfg['M'])
    P, Kw, rs = prob['P'], prob['Kw'], prob['rs']
    d = ds.trj.shape[-1]

    g0 = svd_init(P, L)
    _, sigma_sqrt = alpha_second_moment(ds.alphas)
    jac = spatial_jacobian(ds.phis)
    inits = init_label_candidates(ds.phis, jac, ds.weights, sigma_sqrt, L)
    labels, _, grads, _ = kaffine_clustering(
        ds.phis, ds.weights, sigma_sqrt, L, inits, include_linear=True)
    memb = soft_membership(labels, L, ds.im_size, smooth_iters=2)
    memb = reduce_spatial(memb, im_size_low=cfg['im_size_s'], order=1)
    memb = memb.reshape((L, -1))[:, prob['sel']].type(torch.complex64)
    memb = memb * g0.abs().mean()

    kappa, _, tau = shear_trajectory(ds.trj, ds.alphas, grads, ds.os,
                                     cap=None, im_size=ds.im_size_full)
    a_anch = ds.alphas.reshape((ds.alphas.shape[0], -1))[:, prob['t_idx']]
    shear_a = einsum(grads * tau[:, None, None], a_anch, 'L B d, B M -> L M d')
    k_a = ds.trj.reshape((-1, d))[prob['t_idx']]
    delta = ((ds.os * (k_a[None] + shear_a)).round() - (ds.os * k_a[None]).round())

    print(f'  {"iters":>6} {"HOFFT":>12} {"shear/svd":>12} {"shear/memb":>12}')
    for n in iters:
        e_h = als_dense(P, Kw, g0, None, ds.os, rs, n)
        e_s = als_dense(P, Kw, g0, delta, ds.os, rs, n)
        e_m = als_dense(P, Kw, memb, delta, ds.os, rs, n)
        print(f'  {n:>6d} {e_h:12.4e} {e_s:12.4e} {e_m:12.4e}')
    return ds, prob, delta, g0, memb


def check_stencil_benefit(name, ds, prob, torch_dev, L, Ws=(1, 3, 5)):
    """Marginal value of a wider stencil, with and without shear."""
    print(f'\n--- 3. stencil width benefit, {name}, L={L} ---')
    cfg = CONFIG[name]
    P, rs = prob['P'], prob['rs']
    d = ds.trj.shape[-1]
    g0 = svd_init(P, L)

    _, sigma_sqrt = alpha_second_moment(ds.alphas)
    jac = spatial_jacobian(ds.phis)
    inits = init_label_candidates(ds.phis, jac, ds.weights, sigma_sqrt, L)
    _, _, grads, _ = kaffine_clustering(
        ds.phis, ds.weights, sigma_sqrt, L, inits, include_linear=True)
    _, _, tau = shear_trajectory(ds.trj, ds.alphas, grads, ds.os,
                                 cap=None, im_size=ds.im_size_full)
    a_anch = ds.alphas.reshape((ds.alphas.shape[0], -1))[:, prob['t_idx']]
    shear_a = einsum(grads * tau[:, None, None], a_anch, 'L B d, B M -> L M d')
    k_a = ds.trj.reshape((-1, d))[prob['t_idx']]
    delta = ((ds.os * (k_a[None] + shear_a)).round() - (ds.os * k_a[None]).round())

    print(f'  {"W":>3} {"HOFFT":>12} {"sheared":>12} {"gain":>7}')
    prev = {}
    for W in Ws:
        Kw = build_kern_bases((W,) * d, cfg['im_size_s'], ds.os).to(torch_dev)
        Kw = Kw.reshape((Kw.shape[0], -1))[:, prob['sel']].type(torch.complex64)
        e_h = als_dense(P, Kw, g0, None, ds.os, rs, 10)
        e_s = als_dense(P, Kw, g0, delta, ds.os, rs, 10)
        msg = ''
        if prev:
            msg = (f'   (W benefit: HOFFT {prev["h"] / e_h:.2f}x, '
                   f'sheared {prev["s"] / e_s:.2f}x)')
        print(f'  {W:>3d} {e_h:12.4e} {e_s:12.4e} {e_h / e_s:7.2f}x{msg}')
        prev = dict(h=e_h, s=e_s)
        del Kw
        torch.cuda.empty_cache()


def main():
    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_wrap_exactness(torch_dev)
    for name, L, Lw in (('coco_spiral', 9, 5), ('tilt_spi_invivo', 30, 10)):
        ds, prob, *_ = check_convergence(name, torch_dev, L)
        check_stencil_benefit(name, ds, prob, torch_dev, Lw)
        del ds, prob
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
