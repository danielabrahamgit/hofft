"""
Stage 1 of the per-factor trajectory shear feasibility study (math_docs/feas_test.md).

Dense reference implementation: no NUFFT, no coils, no CG. Builds the exact phase matrix

    P(t, r) = exp(-2i pi [ r . k_dev(t) + phi(r) . alpha(t) ])

on a reduced spatial grid at Q anchor times, then measures how well each model
approximates it in relative Frobenius norm:

  * SVD          -- optimal rank-L approximation (the error floor for L fields)
  * HOFFT        -- sum_l g_l(r) sum_w h_lw(t) e^{-2i pi r.z_w/os}
  * Sheared HOFFT-- same, with a per-factor integer grid offset Delta_l(t)

Both HOFFT variants are fit by the same dense ALS from the same initializations, so the
only difference is the shear. Error is plotted against ``L_eff``, which charges the
sheared variant for the grid padding its shear requires.

Run with:
    PYTHONPATH=src python paper_experiments/shear_feas/stage1_dense.py
"""
import json
import torch
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from time import perf_counter
from einops import einsum

from hofft.utils import gen_grd, reduce_spatial, maxmin_indices
from hofft.decomp import build_kern_bases
from hofft.phase_coeffs import trj_dev_to_phis_alphas
from hofft.shear import (
    alpha_second_moment,
    spatial_jacobian,
    kaffine_clustering,
    init_label_candidates,
    shear_trajectory,
    grid_growth,
    soft_membership,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_dataset  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

CONFIG = {
    'coco_spiral':     dict(im_size_s=(100, 100), M=256, Ls=[1, 2, 3, 5, 7, 9], n_als=32),
    'tilt_spi_invivo': dict(im_size_s=(96, 96),   M=256, Ls=[5, 10, 20, 30],    n_als=32),
}
CAPS = [0.1, None]


def build_dense_problem(ds, im_size_s, M, seed=0):
    """Assemble the anchor phase matrix P and everything the models need to fit it."""
    torch_dev = ds.phis.device
    d = ds.trj.shape[-1]

    phis = reduce_spatial(ds.phis, im_size_low=im_size_s, order=3)
    mask = reduce_spatial(ds.mask, im_size_low=im_size_s, order=3) > 0.5
    phis = phis * mask

    # Trajectory grid deviation enters as extra (r, k_dev) basis pairs, exactly as
    # hofft_decomp_linop does before the decomposition
    phis_dev, alphas_dev = trj_dev_to_phis_alphas(ds.trj, ds.im_size_full, ds.os)
    phis_dev = gen_grd(im_size_s).to(torch_dev).moveaxis(-1, 0)

    B = phis.shape[0]
    phis_all = torch.cat([phis_dev, phis], dim=0)
    alphas_all = torch.cat([alphas_dev.reshape((d, -1)),
                            ds.alphas.reshape((B, -1))], dim=0)

    # Anchor times: farthest point in the whitened stacked coefficient space, so the
    # anchors span both the field coefficients and the sub-grid trajectory deviation
    _, sig_sqrt_all = alpha_second_moment(alphas_all)
    aw = (sig_sqrt_all @ alphas_all).T.contiguous()
    gen = torch.Generator(device=torch_dev)
    gen.manual_seed(seed)
    pool = torch.randperm(aw.shape[0], generator=gen, device=torch_dev)[:20_000]
    t_idx = pool[maxmin_indices(aw[pool], M, seed=seed)]

    sel = torch.argwhere(mask.reshape(-1))[:, 0]
    rs = gen_grd(im_size_s).to(torch_dev).reshape((-1, d))[sel]
    phis_flt = phis_all.reshape((phis_all.shape[0], -1))[:, sel]
    P = torch.exp(-2j * np.pi * (alphas_all[:, t_idx].T @ phis_flt))    # M N

    Kw = build_kern_bases(ds.kern_size, im_size_s, ds.os).to(torch_dev)
    Kw = Kw.reshape((Kw.shape[0], -1))[:, sel].type(torch.complex64)

    return dict(P=P.type(torch.complex64), Kw=Kw, rs=rs, sel=sel, t_idx=t_idx,
                phis=phis, mask=mask, im_size_s=im_size_s)


def als_dense(P, Kw, g_init, delta, os, rs, n_iter, lamda=1e-4):
    """
    Alternating least squares for the dense HOFFT model.

    Args
    ----
    P : torch.Tensor
        Target phase matrix with shape (M, N)
    Kw : torch.Tensor
        Stencil bases with shape (K, N)
    g_init : torch.Tensor
        Initial spatial factors with shape (L, N)
    delta : Optional[torch.Tensor]
        Integer grid offsets with shape (L, M, d); None for standard HOFFT
    os : float
        Grid oversampling factor
    rs : torch.Tensor
        Voxel coordinates with shape (N, d)
    n_iter : int
        ALS sweeps
    lamda : float
        Relative ridge on both least squares solves

    Returns
    -------
    err : float
        Relative Frobenius error of the fitted model
    """
    M, N = P.shape
    K = Kw.shape[0]
    L = g_init.shape[0]
    torch_dev = P.device

    if delta is None:
        D = None
    else:
        D = torch.exp(-2j * np.pi * einsum(delta / os, rs, 'L M d, N d -> L M N')
                      ).type(torch.complex64)

    g = g_init.clone()
    nrm = P.norm()
    best = None
    mbs = max(1, min(M, int(2 ** 26 / max(N * L * K, 1))))

    for _ in range(n_iter):
        # ---- kernel weights, one (LK x LK) solve per anchor -----------------
        H = torch.empty((L, M, K), dtype=torch.complex64, device=torch_dev)
        for m1 in range(0, M, mbs):
            m2 = min(m1 + mbs, M)
            E = g[None, :, None, :] * Kw[None, None, :, :]              # 1 L K N
            E = E.expand(m2 - m1, L, K, N)
            if D is not None:
                E = E * D[:, m1:m2].permute(1, 0, 2)[:, :, None, :]
            E = E.reshape((m2 - m1, L * K, N))
            A = einsum(E.conj(), E, 'M P N, M Q N -> M P Q')
            b = einsum(E.conj(), P[m1:m2], 'M P N, M N -> M P')
            ridge = lamda * A.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)
            A = A + ridge[:, None, None] * torch.eye(L * K, device=torch_dev)
            H[:, m1:m2] = torch.linalg.solve(A, b).reshape(
                (m2 - m1, L, K)).permute(1, 0, 2)
            del E, A, b

        # ---- spatial factors, one (L x L) solve per voxel -------------------
        S = einsum(H, Kw, 'L M K, K N -> L M N')
        if D is not None:
            S = S * D
        A = einsum(S.conj(), S, 'L M N, Q M N -> N L Q')
        b = einsum(S.conj(), P, 'L M N, M N -> N L')
        ridge = lamda * A.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)
        A = A + ridge[:, None, None] * torch.eye(L, device=torch_dev)
        g = torch.linalg.solve(A, b).T.contiguous()
        del A, b

        # ALS on this bilinear model is not monotone and occasionally diverges, so
        # score every sweep and keep the best rather than whatever the last one gave
        err = float((P - einsum(g, S, 'L N, L M N -> M N')).norm() / nrm)
        best = err if best is None else min(best, err)
        del S

    del D
    return best


def svd_init(P, L):
    """Rank-L SVD spatial factors, used as a common ALS initialization."""
    U, S, Vh = torch.linalg.svd(P, full_matrices=False)
    return (S[:L, None] * Vh[:L]).conj().contiguous()


def svd_error_curve(P, Ls):
    """Relative Frobenius error of the optimal rank-L approximation."""
    s = torch.linalg.svdvals(P)
    tot = s.square().sum()
    return {L: float((s[L:].square().sum() / tot).sqrt()) for L in Ls}


def run_dataset(name, torch_dev):
    cfg = CONFIG[name]
    ds = load_dataset(name, torch_dev)
    prob = build_dense_problem(ds, cfg['im_size_s'], cfg['M'])
    P, Kw, rs = prob['P'], prob['Kw'], prob['rs']
    d = ds.trj.shape[-1]
    print(f'\n===== {name} =====')
    print(f'  P {tuple(P.shape)}  K {Kw.shape[0]}  masked voxels {rs.shape[0]}')

    svd_err = svd_error_curve(P, cfg['Ls'])
    _, sigma_sqrt = alpha_second_moment(ds.alphas)
    jac = spatial_jacobian(ds.phis)

    rows = []
    for L in cfg['Ls']:
        g0 = svd_init(P, L)

        t0 = perf_counter()
        e_hofft = als_dense(P, Kw, g0, None, ds.os, rs, cfg['n_als'])
        t_hofft = perf_counter() - t0
        rows.append(dict(L=L, method='hofft', cap=None, err=e_hofft,
                         L_eff=float(L), t=t_hofft))
        print(f'  L={L:<3d} SVD {svd_err[L]:.4e}   HOFFT {e_hofft:.4e} '
              f'({t_hofft:.1f}s)')

        # Sheared: partition and affine fit chosen to minimize the post-fit residual
        inits = init_label_candidates(ds.phis, jac, ds.weights, sigma_sqrt, L)
        labels, consts, grads, _ = kaffine_clustering(
            ds.phis, ds.weights, sigma_sqrt, L, inits, include_linear=True)
        memb = soft_membership(labels, L, ds.im_size, smooth_iters=2)
        memb = reduce_spatial(memb, im_size_low=cfg['im_size_s'], order=1)
        memb = memb.reshape((L, -1))[:, prob['sel']].type(torch.complex64)

        for cap in CAPS:
            kappa, _, tau = shear_trajectory(ds.trj, ds.alphas, grads, ds.os,
                                             cap=cap, im_size=ds.im_size_full)
            _, leff, _ = grid_growth(ds.trj, kappa, ds.os, ds.kern_size, ds.im_size_full)
            shear_a = einsum(grads * tau[:, None, None],
                             ds.alphas.reshape((ds.alphas.shape[0], -1))[:, prob['t_idx']],
                             'L B d, B M -> L M d')
            k_a = ds.trj.reshape((-1, d))[prob['t_idx']]
            delta = ((ds.os * (k_a[None] + shear_a)).round()
                     - (ds.os * k_a[None]).round())

            best_e, best_init = None, None
            for init_name, gi in (('svd', g0), ('memb', memb * g0.abs().mean())):
                e = als_dense(P, Kw, gi, delta, ds.os, rs, cfg['n_als'])
                if best_e is None or e < best_e:
                    best_e, best_init = e, init_name
            rows.append(dict(L=L, method='shear', cap=(-1 if cap is None else cap),
                             err=best_e, L_eff=leff, t=0.0, init=best_init,
                             shear_pk=float((kappa - ds.trj[None]).abs().amax())))
            print(f'        cap={str(cap):<5} shear {best_e:.4e}  L_eff={leff:6.2f} '
                  f'({leff / L:.2f}L)  init={best_init}  '
                  f'gain={e_hofft / best_e:5.2f}x')
            del kappa, delta
            torch.cuda.empty_cache()

    return dict(dataset=name, svd=svd_err, rows=rows,
                M=cfg['M'], N=int(rs.shape[0]), im_size_s=list(cfg['im_size_s']))


def make_figure(out):
    fig, ax = plt.subplots(1, len(out), figsize=(6.0 * len(out), 4.4), squeeze=False)
    for j, (name, res) in enumerate(out.items()):
        a = ax[0, j]
        Ls = sorted(res['svd'])
        a.plot(Ls, [res['svd'][L] for L in Ls], 'o-', color='k', label='SVD (rank L)')
        h = [r for r in res['rows'] if r['method'] == 'hofft']
        a.plot([r['L_eff'] for r in h], [r['err'] for r in h], 's-',
               color='tab:blue', label='HOFFT')
        for cap, c in zip(CAPS, ['tab:orange', 'tab:green', 'tab:red']):
            key = -1 if cap is None else cap
            s = sorted([r for r in res['rows']
                        if r['method'] == 'shear' and r['cap'] == key],
                       key=lambda r: r['L_eff'])
            a.plot([r['L_eff'] for r in s], [r['err'] for r in s], '^--', color=c,
                   label=f'sheared HOFFT (cap={cap})')
        a.set_yscale('log')
        a.set_xlabel(r'$L_{eff}$  (FFT-equivalents; $=L$, the gather index wraps)')
        a.set_ylabel('relative Frobenius error of $P$')
        a.set_title(name)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle('Stage 1: phase-matrix approximation error vs effective cost')
    fig.tight_layout()
    fig.savefig(OUT / 'stage1_error_vs_Leff.png', dpi=140)
    plt.close(fig)


def main():
    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')
    out = {}
    for name in CONFIG:
        out[name] = run_dataset(name, torch_dev)
        torch.cuda.empty_cache()
    torch.save(out, OUT / 'stage1_results.pt')
    with open(OUT / 'stage1_results.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)
    make_figure(out)

    print('\n================ GATE 1 SUMMARY ================')
    for name, res in out.items():
        print(f'\n{name}')
        for L in sorted(res['svd']):
            hof = [r for r in res['rows'] if r['method'] == 'hofft' and r['L'] == L][0]
            best = min([r for r in res['rows']
                        if r['method'] == 'shear' and r['L'] == L],
                       key=lambda r: r['err'])
            print(f'  L={L:<3d} SVD {res["svd"][L]:.3e}  HOFFT {hof["err"]:.3e}  '
                  f'shear {best["err"]:.3e} @ L_eff={best["L_eff"]:6.2f} '
                  f'({best["L_eff"] / L:.2f}L)  gain={hof["err"] / best["err"]:5.2f}x')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
