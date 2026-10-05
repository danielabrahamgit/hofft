"""
Part B of math_docs/two_test.md: joint coil-phase floor.

Measurements only. No new production linop.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from hofft.decomp import hofft_params
from hofft.pipelines import (
    _build_reduced_terms,
    _process_phase_coefficients,
    svd_decomp_linop,
)
from hofft.utils import expand_spatial, expand_temporal, reduce_spatial
from mr_recon.fourier import cufi_nufft
from mr_recon.linops import linop, batching_params
from mr_recon.recons import CG_SENSE_recon

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)


def _load_hybrid():
    path = HERE.parent / 'hybrid_feas' / 'common.py'
    spec = importlib.util.spec_from_file_location('hybrid_feas_common', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hybrid = _load_hybrid()

N_Q_SWITCH = 10000
CG_ITERS = 10
CG_KW = dict(max_iter=CG_ITERS, max_eigen=1.0, verbose=False)


def nrmse_mag(img, ref, mask=None):
    if mask is not None:
        img, ref = img * mask, ref * mask
    return float((img.abs() - ref.abs()).norm() / ref.abs().norm().clamp_min(1e-30))


def coil_compress(mps, ksp, C_comp: int):
    C = mps.shape[0]
    if C_comp is None or C_comp <= 0 or C_comp >= C:
        return mps, ksp
    S = mps.reshape(C, -1)
    U, _, _ = torch.linalg.svd(S, full_matrices=False)
    U = U[:, :int(C_comp)]
    mps_c = (U.mH @ S).reshape(int(C_comp), *mps.shape[1:])
    ksp_c = torch.einsum('kc,c...->k...', U.mH, ksp)
    return mps_c.contiguous(), ksp_c.contiguous()


def _hparams(ds, L: int) -> hofft_params:
    d = ds.d
    return hofft_params(
        (3,) * d, ds.os, int(L),
        reduced_im_size=ds.reduced_im_size,
        time_reduction_factor=ds.time_reduction_factor,
        num_compressed_bases=ds.num_compressed_bases,
        cur_rank=ds.cur_rank,
        normalize_coeffs=True,
        verbose=False,
    )


def reduced_joint_factors(ds, mps):
    """Reduced-grid P (M, N) and coil maps S (C, N), plus bookkeeping."""
    hp = _hparams(ds, 8)
    phis, alphas, spat, temp = _process_phase_coefficients(
        ds.phis, ds.alphas,
        normalize_coeffs=hp.normalize_coeffs,
        num_compressed_bases=hp.num_compressed_bases,
    )
    phis_r, alphas_r, _, mask_r = _build_reduced_terms(
        phis, alphas, hp, ds.mask)
    B = phis_r.shape[0]
    mask_f = (mask_r.abs() if mask_r.is_complex() else mask_r) > 0.5
    sel = torch.argwhere(mask_f.reshape(-1))[:, 0]
    phi_f = phis_r.reshape(B, -1)[:, sel]
    alp_f = alphas_r.reshape(B, -1)
    P = torch.exp(-2j * math.pi * (alp_f.T @ phi_f)).contiguous()
    mps_r = reduce_spatial(mps, tuple(phis_r.shape[1:]), order=3)
    S = mps_r.reshape(mps.shape[0], -1)[:, sel].contiguous()
    return dict(
        P=P, S=S, sel=sel, mask_r=mask_f, red_size=tuple(phis_r.shape[1:]),
        phis_r=phis_r, alphas_r=alphas_r, phis=phis, alphas=alphas,
        spat=spat, temp=temp, n_vox=int(sel.numel()),
        M=int(P.shape[0]), N=int(P.shape[1]), C=int(S.shape[0]),
    )


@torch.no_grad()
def joint_svd(P: torch.Tensor, S: torch.Tensor, requested_S: int) -> dict:
    """Thin SVD of A[(c,t), n] = S[c,n] * P[t,n] without forming A."""
    M, N = P.shape
    C = S.shape[0]
    L_max = int(requested_S)
    q_os = L_max + 10
    path = 'gram' if N <= N_Q_SWITCH else 'randomized'
    n_a = C * M

    if path == 'gram':
        G = (P.mH @ P) * (S.mH @ S)
        G = 0.5 * (G + G.mH)
        ridge = 1e-8 * G.diagonal().abs().mean().clamp_min(1e-30)
        G = G + ridge * torch.eye(N, dtype=G.dtype, device=G.device)
        try:
            evals, V = torch.linalg.eigh(G)
            idx = torch.argsort(evals.real, descending=True)
            n_sv = int(min(L_max, N, n_a, evals.numel()))
            V = V[:, idx[:n_sv]]
            s = evals.real[idx[:n_sv]].clamp(min=0).sqrt()
        except Exception:
            path = 'randomized'
    if path == 'gram':
        U = P.new_zeros((n_a, n_sv))
        for c in range(C):
            U[c * M:(c + 1) * M] = (P * S[c]) @ V
        U = U / s.clamp_min(1e-12)
        q_used = int(N)
    else:
        q_used = int(max(min(q_os, N, n_a), 1))
        Omega = torch.randn(N, q_used, dtype=P.dtype, device=P.device)
        Y = P.new_empty((n_a, q_used))
        for c in range(C):
            Y[c * M:(c + 1) * M] = (P * S[c]) @ Omega
        Qb, _ = torch.linalg.qr(Y, mode='reduced')
        Bmat = P.new_zeros((q_used, N))
        for c in range(C):
            Bmat = Bmat + (Qb[c * M:(c + 1) * M].mH @ P) * S[c]
        Ub, s, Vh = torch.linalg.svd(Bmat, full_matrices=False)
        n_sv = int(min(L_max, s.numel()))
        U = Qb @ Ub[:, :n_sv]
        s = s[:n_sv]
        V = Vh[:n_sv].mH
        del Y, Omega, Bmat, Qb, Ub, Vh

    a2 = (P.abs().square().sum(0) * S.abs().square().sum(0)).sum()
    captured = s.square().sum()
    ratio = float((captured / a2.clamp_min(1e-30)).real)
    frob = float(torch.sqrt((a2 - captured).clamp(min=0) / a2.clamp_min(1e-30)))
    return dict(
        U=U, s=s, V=V, err=frob,
        requested_S=int(requested_S),
        realized=int(s.numel()),
        L_max=L_max,
        q_oversample=int(q_os),
        q_used=int(q_used),
        n_sv=int(s.numel()),
        path=path,
        N=int(N), M=int(M), C=int(C),
        a_norm=float(a2.sqrt()),
        energy_ratio=ratio,
    )


def fullres_refit(ds, mps, red: dict, fit: dict, chunk: int = 512,
                  max_cols: int | None = None) -> dict:
    """B.4: keep reduced-grid H, LS spatial maps on full-res s_c and phi."""
    L = int(fit['s'].numel())
    C = red['C']
    H = _temporal_from_fit(fit, red, ds.trj_size)
    H = H.reshape(L, C, -1).permute(1, 2, 0).reshape(C * int(np.prod(ds.trj_size)), L)
    H = H.contiguous()
    gram = H.mH @ H
    eye = torch.eye(L, dtype=gram.dtype, device=gram.device)
    gram = gram + 1e-6 * eye * gram.diagonal().abs().mean().clamp_min(1e-30)

    phis = red['phis'].reshape(red['phis'].shape[0], -1)
    alphas = red['alphas'].reshape(red['alphas'].shape[0], -1)
    Sfull = mps.reshape(C, -1)
    mask = ds.mask.reshape(-1) > 0
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if max_cols is not None and idx.numel() > max_cols:
        pick = torch.randperm(idx.numel(), device=idx.device)[:max_cols]
        idx = idx[pick]
    N = int(idx.numel())
    M = alphas.shape[1]
    rhs = H.new_zeros((L, N))
    for n0 in range(0, N, chunk):
        n1 = min(n0 + chunk, N)
        cols = idx[n0:n1]
        Pch = torch.exp(-2j * math.pi * (alphas.T @ phis[:, cols]))
        acc = H.new_zeros((L, n1 - n0))
        for c in range(C):
            Hc = H[c * M:(c + 1) * M]
            acc = acc + Sfull[c, cols] * (Hc.mH @ Pch)
        rhs[:, n0:n1] = acc
    B = torch.linalg.solve(gram, rhs)
    a2 = torch.zeros((), dtype=torch.float64, device=H.device)
    r2 = torch.zeros((), dtype=torch.float64, device=H.device)
    for n0 in range(0, N, chunk):
        n1 = min(n0 + chunk, N)
        cols = idx[n0:n1]
        Pch = torch.exp(-2j * math.pi * (alphas.T @ phis[:, cols]))
        A = Sfull[:, cols][:, None, :] * Pch[None]
        recon = A.new_zeros(A.shape)
        for c in range(C):
            Hc = H[c * M:(c + 1) * M]
            recon[c] = Hc @ B[:, n0:n1]
        a2 = a2 + A.norm().square().double()
        r2 = r2 + (A - recon).norm().square().double()
    err = float(torch.sqrt(r2 / a2.clamp_min(1e-30)))
    return dict(B=B, H=H, err_full=err, L=L, N=N)


class JointLinop(linop):
    """y_c = sum_l h_{c,l} NUFFT(g_l x). Measurement harness, not a product linop."""

    def __init__(self, g, h, trj, dcf, im_size, os, width=3, n_trans=8):
        C = h.shape[1]
        super().__init__(im_size, (C, *h.shape[2:]))
        self.g = g.contiguous()
        self.h = h.contiguous()
        self.dcf = dcf
        self.L = int(g.shape[0])
        self.C = C
        self.n_trans = min(int(n_trans), self.L)
        self.nft = cufi_nufft(im_size, oversamp=os, width=width,
                              n_trans=self.n_trans)
        self.trj = trj.contiguous()
        self.nft.plan(self.trj, n_trans=self.n_trans)
        self.scale = 1.0

    def forward(self, img):
        ksp = img.new_zeros(self.oshape)
        gimg = self.g * img
        for l0 in range(0, self.L, self.n_trans):
            l1 = min(l0 + self.n_trans, self.L)
            y = self.nft.forward(gimg[l0:l1][None], self.trj[None])[0]
            ksp = ksp + (y[:, None] * self.h[l0:l1]).sum(0)
        return ksp

    def adjoint(self, ksp):
        img = ksp.new_zeros(self.ishape)
        wy = ksp * self.dcf
        for l0 in range(0, self.L, self.n_trans):
            l1 = min(l0 + self.n_trans, self.L)
            Hy = (self.h[l0:l1].conj() * wy).sum(1)
            x = self.nft.adjoint(Hy[None], self.trj[None])[0]
            img = img + (x * self.g[l0:l1].conj()).sum(0)
        return img

    def normal(self, img):
        return self.adjoint(self.forward(img))


def _spatial_from_fit(fit, red, im_size):
    """Upsample V (N_red, L) onto the full grid (zeros off-mask).

    A ≈ U diag(s) V^H, so g_l = conj(V_l) * s_l.
    """
    L = int(fit['s'].numel())
    canvas = fit['V'].new_zeros((red['mask_r'].numel(), L))
    canvas[red['sel']] = fit['V'].conj() * fit['s']
    canvas = canvas.T.reshape(L, *red['red_size'])
    if red['red_size'] != tuple(im_size):
        canvas = expand_spatial(canvas, im_size, order=3)
    spat = red['spat']
    if spat is not None:
        canvas = canvas * spat
    return canvas


def _temporal_from_fit(fit, red, trj_size):
    C = red['C']
    L = int(fit['s'].numel())
    trj_r = tuple(red['alphas_r'].shape[1:])
    H = fit['U'].reshape(C, *trj_r, L)
    H = H.permute(H.ndim - 1, 0, *range(1, H.ndim - 1)).contiguous()
    if H.shape[2:] != tuple(trj_size):
        H = expand_temporal(H, num_time_high=trj_size[0], dim=2, order=3)
    temp = red['temp']
    if temp is not None:
        H = H * temp
    return H


def _adjoint_rel(A, im_size, ksp_shape):
    x = torch.randn(im_size, dtype=torch.complex64, device=A.g.device)
    y = torch.randn(ksp_shape, dtype=torch.complex64, device=A.g.device)
    # Adjoint includes DCF, so the identity is ⟨Ax, W y⟩ = ⟨x, A* y⟩.
    lhs = torch.vdot(A.forward(x).reshape(-1), (A.dcf * y).reshape(-1))
    rhs = torch.vdot(x.reshape(-1), A.adjoint(y).reshape(-1))
    return float(abs(lhs - rhs) / abs(lhs).clamp_min(1e-30))


def recon_joint(ds, mps, ksp, red, fit, n_trans=8):
    g = _spatial_from_fit(fit, red, ds.im_size)
    h = _temporal_from_fit(fit, red, ds.trj_size)
    A = JointLinop(g, h, ds.trj, ds.dcf, ds.im_size, ds.os, n_trans=n_trans)
    rel = _adjoint_rel(A, ds.im_size, ksp.shape)
    print(f'    adjoint rel={rel:.2e}', flush=True)
    img = CG_SENSE_recon(A, ksp, **CG_KW)
    return img, A


def recon_svd(ds, mps, ksp, L: int):
    hp = _hparams(ds, L)
    bparams = batching_params(coil_batch_size=mps.shape[0], field_batch_size=L)
    A = svd_decomp_linop(
        ds.phis, ds.alphas, mps, ds.trj, hp,
        svd_method='cur', spatial_mask=ds.mask, dcf=ds.dcf, bparams=bparams,
    )
    A.bparams.field_batch_size = L
    img = CG_SENSE_recon(A, ksp, **CG_KW)
    return img, A


def _gpu_timer(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return perf_counter() - t0, out


def run_coco_floor(torch_dev):
    name = 'coco_spiral'
    print(f'\n======== B coco Q=1 floor ({name}) ========', flush=True)
    ds = hybrid.load_dataset(name, torch_dev)
    mps, ksp = ds.mps, ds.ksp
    C = mps.shape[0]
    print(f'  C={C}  grid={ds.im_size}  reduced={ds.reduced_im_size}', flush=True)
    red = reduced_joint_factors(ds, mps)
    print(f'  reduced P {tuple(red["P"].shape)}  N={red["N"]}  '
          f'path-switch at N>{N_Q_SWITCH}', flush=True)

    Ss = [10 * C, 15 * C, 20 * C, 25 * C, 30 * C]
    rows = []
    for Sreq in Ss:
        fit = joint_svd(red['P'], red['S'], Sreq)
        print(f'  S={Sreq}  realized={fit["realized"]}  L_max={fit["L_max"]}  '
              f'q={fit["q_oversample"]} used={fit["q_used"]}  '
              f'n_sv={fit["n_sv"]}  path={fit["path"]}  '
              f'frob={fit["err"]:.4e}  energy={fit["energy_ratio"]:.6f}', flush=True)
        t, (img, _) = _gpu_timer(lambda: recon_joint(ds, mps, ksp, red, fit))
        e_img = nrmse_mag(img, ds.img_ee, ds.mask)
        print(f'    image |NRMSE|={e_img:.4e}  recon {t:.2f}s', flush=True)
        rows.append(dict(
            requested_S=Sreq, realized=fit['realized'], L_max=fit['L_max'],
            q=fit['q_oversample'], q_used=fit['q_used'], n_sv=fit['n_sv'],
            path=fit['path'], frob=fit['err'], nrmse=e_img, recon_s=t,
        ))
        del img, fit
        torch.cuda.empty_cache()

    # SVD at the matching L = S/C
    svd_rows = []
    for L in (10, 15, 20, 25, 30):
        t, (img, _) = _gpu_timer(lambda L=L: recon_svd(ds, mps, ksp, L))
        e_img = nrmse_mag(img, ds.img_ee, ds.mask)
        print(f'  SVD L={L}  |NRMSE|={e_img:.4e}  recon {t:.2f}s', flush=True)
        svd_rows.append(dict(L=L, nrmse=e_img, recon_s=t))
        del img
        torch.cuda.empty_cache()

    refit_rows = []
    for Sreq in (Ss[0], Ss[2], Ss[-1]):
        fit = joint_svd(red['P'], red['S'], Sreq)
        ref = fullres_refit(ds, mps, red, fit, max_cols=8192)
        print(f'  B.4 refit S={Sreq}  frob_full={ref["err_full"]:.4e}  '
              f'(reduced frob={fit["err"]:.4e})', flush=True)
        refit_rows.append(dict(
            requested_S=Sreq, frob_red=fit['err'], frob_full=ref['err_full'],
        ))
        del fit, ref
        torch.cuda.empty_cache()

    _plot_frob_img(name, rows, OUT / f'b3_{name}.png')
    return dict(name=name, C=C, Q=1, joint=rows, svd=svd_rows, refit=refit_rows,
                N=red['N'], M=red['M'])


def run_tilt_q1(torch_dev, C_comp=8):
    name = 'tilt_spi_invivo'
    print(f'\n======== B.5 tilt Q=1 C={C_comp} ========', flush=True)
    ds = hybrid.load_dataset(name, torch_dev)
    mps, ksp = coil_compress(ds.mps, ds.ksp, C_comp)
    C = mps.shape[0]
    print(f'  C_full={ds.C} -> C={C}  grid={ds.im_size}', flush=True)
    red = reduced_joint_factors(ds, mps)
    print(f'  reduced P {tuple(red["P"].shape)}  N={red["N"]}', flush=True)

    Ls = (30, 55, 80)
    rows = []
    for L in Ls:
        Sreq = C * L
        fit = joint_svd(red['P'], red['S'], Sreq)
        print(f'  joint S={Sreq} (=C*L, L={L})  realized={fit["realized"]}  '
              f'path={fit["path"]}  q={fit["q_oversample"]}  '
              f'n_sv={fit["n_sv"]}  frob={fit["err"]:.4e}', flush=True)
        t, (img, _) = _gpu_timer(lambda: recon_joint(ds, mps, ksp, red, fit))
        e_img = nrmse_mag(img, ds.img_ee, ds.mask)
        print(f'    image |NRMSE|={e_img:.4e}  recon {t:.2f}s', flush=True)
        rows.append(dict(
            tag='joint', L=L, requested_S=Sreq, realized=fit['realized'],
            L_max=fit['L_max'], q=fit['q_oversample'], q_used=fit['q_used'],
            n_sv=fit['n_sv'], path=fit['path'], frob=fit['err'],
            nrmse=e_img, recon_s=t,
        ))
        del img, fit
        torch.cuda.empty_cache()

        t, (img, _) = _gpu_timer(lambda L=L: recon_svd(ds, mps, ksp, L))
        e_svd = nrmse_mag(img, ds.img_ee, ds.mask)
        print(f'  SVD L={L}  |NRMSE|={e_svd:.4e}  recon {t:.2f}s', flush=True)
        ok = rows[-1]['nrmse'] <= e_svd + 0.002
        rows[-1]['svd_nrmse'] = e_svd
        rows[-1]['svd_recon_s'] = t
        rows[-1]['pass'] = bool(ok)
        print(f'    joint <= SVD+0.002 ? {ok}', flush=True)
        del img
        torch.cuda.empty_cache()

    refit_rows = []
    for L in (Ls[0], Ls[-1]):
        Sreq = C * L
        fit = joint_svd(red['P'], red['S'], Sreq)
        ref = fullres_refit(ds, mps, red, fit, max_cols=4096)
        print(f'  B.4 refit S={Sreq}  frob_full={ref["err_full"]:.4e}  '
              f'(4096-voxel probe)', flush=True)
        refit_rows.append(dict(
            requested_S=Sreq, frob_red=fit['err'], frob_full=ref['err_full'],
        ))
        del fit, ref
        torch.cuda.empty_cache()

    _plot_frob_img(f'{name}_C{C}', rows, OUT / f'b3_{name}.png')
    return dict(name=name, C=C, C_comp=C_comp, Q=1, rows=rows, refit=refit_rows,
                N=red['N'], M=red['M'])


def _plot_frob_img(title, rows, path):
    xs = [r.get('requested_S', r.get('L')) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    if any('frob' in r for r in rows):
        ax.semilogy(xs, [r['frob'] for r in rows], 'o-', label='||A-HB||_F / ||A||_F')
    ax.semilogy(xs, [r['nrmse'] for r in rows], 's-', label='image |NRMSE|')
    ax.set_xlabel('S' if 'requested_S' in rows[0] else 'L')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['coco_spiral', 'tilt_spi_invivo'])
    args = p.parse_args()
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prev_path = OUT / 'part_b.json'
    report = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    if 'coco_spiral' in args.datasets:
        report['coco_spiral'] = run_coco_floor(torch_dev)
    if 'tilt_spi_invivo' in args.datasets:
        report['tilt_spi_invivo'] = run_tilt_q1(torch_dev, C_comp=8)

    # B.3 branch
    for name, rec in report.items():
        rows = rec.get('joint') or rec.get('rows')
        frobs = [r['frob'] for r in rows if 'frob' in r]
        imgs = [r['nrmse'] for r in rows]
        f_span = (max(frobs) / min(frobs)) if frobs and min(frobs) > 0 else None
        i_span = (max(imgs) / min(imgs)) if imgs and min(imgs) > 0 else None
        # pinned image, falling frob -> linop; both pinned -> decomp
        img_flat = (max(imgs) - min(imgs)) < 0.003 if imgs else None
        frob_zero = all(f < 1e-4 for f in frobs) if frobs else None
        frob_flat = (
            frob_zero
            or ((max(frobs) / min(frobs) < 1.3) if frobs and min(frobs) > 0 else None)
        )
        if img_flat and frob_zero:
            branch = 'Frobenius already ~0; image pinned -> downstream of the SVD'
        elif img_flat and (frob_flat is False):
            branch = 'Frobenius keeps falling; image pinned -> linop'
        elif img_flat and frob_flat:
            branch = 'Frobenius also floors -> decomposition is capped'
        else:
            branch = 'image NRMSE still moves with S'
        rec['b3_branch'] = branch
        rec['frob_span'] = f_span
        rec['img_span'] = i_span
        print(f'\nB.3 {name}: {branch}', flush=True)

    (OUT / 'part_b.json').write_text(json.dumps(report, indent=2, default=str))
    print(f'\nwrote {OUT / "part_b.json"}', flush=True)


if __name__ == '__main__':
    main()
