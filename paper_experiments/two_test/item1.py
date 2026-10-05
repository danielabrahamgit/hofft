"""
Item 1 of math_docs/two_test_follow.md: exact-operator comparison.

No CG. 1.4 delta first, then 1.2 y_joint vs y_svd, then 1.5 numerical rank.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from hofft.pipelines import qblock_svd_decomp_linop, svd_decomp_linop
from hofft.utils import expand_spatial, gen_grd
from mr_recon.fourier import cufi_nufft
from mr_recon.linops import batching_params

from part_b import (
    OUT, coil_compress, joint_svd, reduced_joint_factors,
    _hparams, _spatial_from_fit, _temporal_from_fit, JointLinop, hybrid,
)

HERE = Path(__file__).resolve().parent


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _ratio_stats(num, den, floor=0.05):
    mag = den.abs()
    keep = mag > floor * mag.median().clamp_min(1e-30)
    nkeep = int(keep.sum())
    if nkeep == 0:
        return dict(n=0, mean=None, std=None, cv=None)
    r = (num[keep] / den[keep])
    mu = r.mean()
    sd = r.std()
    return dict(
        n=nkeep,
        mean_re=float(mu.real),
        mean_im=float(mu.imag),
        mean_abs=float(mu.abs()),
        std_abs=float(sd.abs()),
        cv=float(sd.abs() / mu.abs().clamp_min(1e-30)),
        angle_std=float(r.angle().std()),
    )


def _per_coil(a, b):
    out = []
    for c in range(a.shape[0]):
        out.append(_rel(a[c], b[c]))
    return out


def _pick_delta(mask, mps0):
    """In-mask voxel near the center with large |s_0|, off the FOV wrap."""
    im = tuple(mask.shape)
    grd = gen_grd(im).to(mask.device)
    rad = grd.norm(dim=-1)
    sl = (slice(4, -4),) * mask.ndim
    frame = torch.zeros_like(mask, dtype=torch.bool)
    frame[sl] = True
    inside = (mask > 0) & frame
    if mps0 is not None:
        inside = inside & (mps0.abs() > 0.1 * mps0.abs().max())
    if int(inside.sum()) == 0:
        inside = (mask > 0) & frame
    if int(inside.sum()) == 0:
        inside = mask > 0
    score = torch.where(inside, rad, rad.new_full((), 1e9))
    idx = int(score.reshape(-1).argmin())
    ijk = tuple(int(x) for x in torch.unravel_index(torch.tensor(idx), im))
    r0 = grd[ijk]
    return ijk, r0


def _y_exact(r0, s0, trj, phis_r0, alphas):
    """Closed form from the spec (no 1/sqrt(N))."""
    rk = (trj * r0).sum(dim=-1)
    pa = (alphas * phis_r0.reshape((-1,) + (1,) * (alphas.ndim - 1))).sum(0)
    return s0 * torch.exp(-2j * math.pi * (rk + pa))


def _classify(rel, per_coil, ratio, edge_frac):
    cv = ratio.get('cv')
    mag = ratio.get('mean_abs')
    pc = [p for p in per_coil if p is not None]
    coil_span = (max(pc) / min(pc)) if pc and min(pc) > 1e-12 else 1.0
    scale = (cv is not None and cv < 0.08 and mag is not None
             and (abs(mag - 1.0) > 0.05))
    coil = len(pc) > 1 and coil_span > 2.0 and rel > 0.02
    edge = edge_frac is not None and edge_frac > 0.6
    if rel < 0.02 and mag is not None and abs(mag - 1.0) < 0.05:
        return 'match', (
            f'ratio ~ 1 (cv={cv}), rel={rel:.3e}; per_coil span {coil_span:.2f}x '
            f'but all coils are small'
        )
    if scale and not coil:
        return 'scale', ('ratio is near-constant and |mean| != 1')
    if coil:
        return 'coil', f'per_coil span {coil_span:.2f}x'
    if edge and rel < 0.05:
        return 'mask', f'rel={rel:.3e} but residual is edge-localized ({edge_frac:.2f})'
    return 'other', (
        f'rel={rel:.3e} cv={cv} coil_span={coil_span:.2f} edge={edge_frac}'
    )


def _edge_frac(residual_img, mask, ring=4):
    mag = residual_img.abs()
    m = mask > 0
    if int(m.sum()) == 0:
        return None
    ext = (~m).float()[None, None]
    for _ in range(ring):
        ext = torch.nn.functional.max_pool2d(ext, 3, stride=1, padding=1)
    ring_m = m & (ext.squeeze() > 0)
    interior = m & (~ring_m)
    e_r = float(mag[ring_m].square().sum())
    e_i = float(mag[interior].square().sum())
    tot = e_r + e_i
    return e_r / tot if tot > 0 else None


def _build_svd(ds, mps, L):
    hp = _hparams(ds, L)
    bparams = batching_params(coil_batch_size=mps.shape[0], field_batch_size=L)
    A = svd_decomp_linop(
        ds.phis, ds.alphas, mps, ds.trj, hp,
        svd_method='cur', spatial_mask=ds.mask, dcf=ds.dcf, bparams=bparams,
    )
    A.bparams.field_batch_size = L
    return A


def _build_joint(ds, mps, red, fit, n_trans=8):
    g = _spatial_from_fit(fit, red, ds.im_size)
    h = _temporal_from_fit(fit, red, ds.trj_size)
    return JointLinop(g, h, ds.trj, ds.dcf, ds.im_size, ds.os, n_trans=n_trans)


def _nufft_delta(ds, ijk):
    nft = cufi_nufft(ds.im_size, oversamp=ds.os, width=3)
    trj = ds.trj
    nft.plan(trj[None] if trj.ndim == trj.shape[-1] + 1 else trj[None])
    x = ds.trj.new_zeros(ds.im_size, dtype=torch.complex64)
    x[ijk] = 1
    y = nft.forward(x[None, None], ds.trj[None])[0, 0]
    return y


def run_delta(ds, mps, tag, L_svd=1, S_joint=1):
    print(f'\n---- 1.4 delta  {tag}  C=1 L={L_svd} S={S_joint} ----', flush=True)
    mps1 = mps[:1].contiguous()
    ksp1 = ds.ksp[:1] if hasattr(ds, 'ksp') else None
    ijk, r0 = _pick_delta(ds.mask, mps1[0])
    s0 = mps1[(0,) + ijk]
    phis_r0 = ds.phis[(slice(None),) + ijk]
    print(f'  voxel {ijk}  r0={r0.tolist()}  |s0|={float(s0.abs()):.4e}',
          flush=True)

    y_exact = _y_exact(r0, s0, ds.trj, phis_r0, ds.alphas)
    N = float(np.prod(ds.im_size))
    y_exact_n = y_exact / math.sqrt(N)

    y_nft = _nufft_delta(ds, ijk) * s0
    # exact temporal phase on top of the NUFFT of a delta
    pa = (ds.alphas * phis_r0.reshape((-1,) + (1,) * (ds.alphas.ndim - 1))).sum(0)
    y_nft_ph = y_nft * torch.exp(-2j * math.pi * pa)

    rec = dict(ijk=list(ijk), r0=[float(x) for x in r0],
               s0_abs=float(s0.abs()), N=N, inv_sqrtN=1.0 / math.sqrt(N))

    rec['nufft_vs_exact'] = dict(
        rel=_rel(y_nft, y_exact),
        rel_scaled=_rel(y_nft, y_exact_n),
        ratio=_ratio_stats(y_nft, y_exact),
        ratio_scaled=_ratio_stats(y_nft, y_exact_n),
    )
    rec['nufft_phase_vs_exact'] = dict(
        rel=_rel(y_nft_ph, y_exact),
        rel_scaled=_rel(y_nft_ph, y_exact_n),
        ratio=_ratio_stats(y_nft_ph, y_exact),
        ratio_scaled=_ratio_stats(y_nft_ph, y_exact_n),
    )
    print(f'  NUFFT(delta)*s0 vs exact:     rel={rec["nufft_vs_exact"]["rel"]:.4e}  '
          f'rel/sqrtN={rec["nufft_vs_exact"]["rel_scaled"]:.4e}  '
          f'ratio_abs={rec["nufft_vs_exact"]["ratio"]["mean_abs"]}', flush=True)
    print(f'  NUFFT(delta)*s0*phase vs exact: rel={rec["nufft_phase_vs_exact"]["rel"]:.4e}  '
          f'rel/sqrtN={rec["nufft_phase_vs_exact"]["rel_scaled"]:.4e}', flush=True)

    x = mps1.new_zeros(ds.im_size)
    x[ijk] = 1

    A_svd = _build_svd(ds, mps1, L_svd)
    _sync()
    y_svd = A_svd.forward(x)
    rec['svd'] = dict(
        L=L_svd,
        rel=_rel(y_svd[0], y_exact),
        rel_scaled=_rel(y_svd[0], y_exact_n),
        ratio=_ratio_stats(y_svd[0], y_exact),
        ratio_scaled=_ratio_stats(y_svd[0], y_exact_n),
        vs_nufft_phase=_rel(y_svd[0], y_nft_ph),
    )
    print(f'  A_svd L={L_svd}: rel={rec["svd"]["rel"]:.4e}  '
          f'rel/sqrtN={rec["svd"]["rel_scaled"]:.4e}  '
          f'ratio_abs={rec["svd"]["ratio"]["mean_abs"]}  '
          f'vs_nufft+ph={rec["svd"]["vs_nufft_phase"]:.4e}', flush=True)

    red = reduced_joint_factors(ds, mps1)
    fit = joint_svd(red['P'], red['S'], S_joint)
    A_j = _build_joint(ds, mps1, red, fit)
    _sync()
    y_j = A_j.forward(x)
    rec['joint'] = dict(
        S=S_joint, realized=fit['realized'], frob=fit['err'],
        rel=_rel(y_j[0], y_exact),
        rel_scaled=_rel(y_j[0], y_exact_n),
        ratio=_ratio_stats(y_j[0], y_exact),
        ratio_scaled=_ratio_stats(y_j[0], y_exact_n),
        vs_nufft_phase=_rel(y_j[0], y_nft_ph),
        vs_svd=_rel(y_j[0], y_svd[0]),
    )
    print(f'  A_joint S={S_joint}: rel={rec["joint"]["rel"]:.4e}  '
          f'rel/sqrtN={rec["joint"]["rel_scaled"]:.4e}  '
          f'ratio_abs={rec["joint"]["ratio"]["mean_abs"]}  '
          f'vs_nufft+ph={rec["joint"]["vs_nufft_phase"]:.4e}  '
          f'vs_svd={rec["joint"]["vs_svd"]:.4e}', flush=True)

    del A_svd, A_j, y_svd, y_j, fit, red
    torch.cuda.empty_cache()
    return rec


def run_y_compare(ds, mps, tag, L_svd, S_joint, x):
    print(f'\n---- 1.2 y-compare  {tag}  L={L_svd} S={S_joint} ----', flush=True)
    A_svd = _build_svd(ds, mps, L_svd)
    _sync()
    y_svd = A_svd.forward(x)
    red = reduced_joint_factors(ds, mps)
    fit = joint_svd(red['P'], red['S'], S_joint)
    print(f'  joint realized={fit["realized"]} frob={fit["err"]:.4e}  '
          f'energy={fit["energy_ratio"]:.6f}  N={red["N"]}', flush=True)
    A_j = _build_joint(ds, mps, red, fit)
    _sync()
    y_j = A_j.forward(x)

    rel = _rel(y_j, y_svd)
    per = _per_coil(y_j, y_svd)
    ratio = _ratio_stats(y_j, y_svd)
    print(f'  rel={rel:.4e}', flush=True)
    print(f'  per_coil={[f"{p:.4e}" for p in per]}', flush=True)
    print(f'  ratio |mean|={ratio["mean_abs"]}  cv={ratio["cv"]}  '
          f'angle_std={ratio["angle_std"]}', flush=True)

    # residual in image space via SVD adjoint (includes DCF)
    rimg = A_svd.adjoint(y_j - y_svd)
    edge = _edge_frac(rimg, ds.mask)
    print(f'  residual edge-energy frac={edge}', flush=True)

    # masks: upsampled reduced support vs full ESPIRiT
    canvas = torch.zeros(red['mask_r'].numel(), device=x.device)
    canvas[red['sel']] = 1
    canvas = canvas.reshape(red['red_size'])
    if red['red_size'] != tuple(ds.im_size):
        up = expand_spatial(canvas[None].to(torch.complex64), ds.im_size, order=3)[0]
        up_m = up.abs() > 0.5
    else:
        up_m = canvas > 0.5
    full_m = ds.mask > 0
    n_sym = int((up_m ^ full_m).sum())
    print(f'  mask xor voxels={n_sym}  up={int(up_m.sum())}  full={int(full_m.sum())}',
          flush=True)

    sig, why = _classify(rel, per, ratio, edge)
    print(f'  SIGNATURE: {sig}  ({why})', flush=True)

    # histogram of |ratio|
    mag = y_svd.abs()
    keep = mag > 0.05 * mag.median().clamp_min(1e-30)
    rr = (y_j[keep] / y_svd[keep]).abs().detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(rr, bins=60, range=(0, max(3.0, float(np.percentile(rr, 99)))))
    ax.axvline(1.0, color='k', lw=0.8)
    ax.set_xlabel('|y_joint / y_svd|')
    ax.set_title(f'{tag}  rel={rel:.3e}  sig={sig}')
    fig.tight_layout()
    fig.savefig(OUT / f'item1_ratio_{tag}.png', dpi=130)
    plt.close(fig)

    rec = dict(
        L=L_svd, S=S_joint, realized=fit['realized'], frob=fit['err'],
        energy=fit['energy_ratio'], N=red['N'],
        rel=rel, per_coil=per, ratio=ratio, edge_frac=edge,
        mask_xor=n_sym, mask_up=int(up_m.sum()), mask_full=int(full_m.sum()),
        signature=sig, signature_why=why,
    )
    del A_svd, A_j, y_svd, y_j, rimg, fit, red
    torch.cuda.empty_cache()
    return rec


def _sv_from_joint(P, S, max_s=None):
    """Singular values of A[(c,t),n] = S[c,n] P[t,n] via Gram or randomized."""
    M, N = P.shape
    C = S.shape[0]
    n_a = C * M
    s = None
    path = None
    if N <= 12000:
        G = (P.mH @ P) * (S.mH @ S)
        G = 0.5 * (G + G.mH)
        ridge = 1e-8 * G.diagonal().abs().mean().clamp_min(1e-30)
        G = G + ridge * torch.eye(N, dtype=G.dtype, device=G.device)
        try:
            evals = torch.linalg.eigvalsh(G).real.clamp(min=0)
            s = evals.flip(0).sqrt()
            path = 'gram'
        except Exception:
            s = None
    if s is None:
        q = int(min(N, n_a, max_s or 800))
        Omega = torch.randn(N, q, dtype=P.dtype, device=P.device)
        Y = P.new_empty((n_a, q))
        for c in range(C):
            Y[c * M:(c + 1) * M] = (P * S[c]) @ Omega
        Qb, _ = torch.linalg.qr(Y, mode='reduced')
        Bmat = P.new_zeros((q, N))
        for c in range(C):
            Bmat = Bmat + (Qb[c * M:(c + 1) * M].mH @ P) * S[c]
        s = torch.linalg.svdvals(Bmat)
        path = f'randomized_q={q}'
    return s, path


def _rank_table(s):
    s = s.real.clamp(min=0)
    s0 = float(s[0].clamp(min=1e-30))
    energy = (s.square().cumsum(0) / s.square().sum().clamp_min(1e-30)).cpu()
    out = dict(s0=s0, n=int(s.numel()),
               energy_170=float(energy[min(169, len(energy) - 1)]),
               energy_340=float(energy[min(339, len(energy) - 1)]),
               energy_510=float(energy[min(509, len(energy) - 1)]))
    for t in (1e-3, 1e-6, 1e-8, 1e-12):
        out[f'rank_{t:g}'] = int((s / s0 > t).sum())
    # smallest k with energy >= 1 - 1e-6
    need = 1.0 - 1e-6
    k = int((energy < need).sum()) + 1
    out['rank_energy_1em6'] = min(k, int(s.numel()))
    return out


def run_rank(ds, mps, tag, sizes):
    print(f'\n---- 1.5 numerical rank  {tag} ----', flush=True)
    rows = []
    orig = ds.reduced_im_size
    for sz in sizes:
        ds.reduced_im_size = sz
        red = reduced_joint_factors(ds, mps)
        print(f'  reduced={sz}  P={tuple(red["P"].shape)}  N={red["N"]}', flush=True)
        s, path = _sv_from_joint(red['P'], red['S'])
        tab = _rank_table(s)
        tab.update(reduced=list(sz), N=red['N'], M=red['M'], C=red['C'], path=path)
        print(f'    path={path}  rank@1e-6={tab["rank_1e-06"]}  '
              f'energy-rank={tab["rank_energy_1em6"]}  '
              f'E(170)={tab["energy_170"]:.6f}  E(340)={tab["energy_340"]:.6f}',
              flush=True)
        rows.append(tab)
        del red, s
        torch.cuda.empty_cache()
    ds.reduced_im_size = orig
    return rows


def run_qblock_check(ds, mps, x, L=40, Q=4):
    """1.6: if SVD/joint disagree, is Q-block on the same scale as SVD?"""
    print(f'\n---- 1.6 Q-block vs SVD  L={L} Q={Q} ----', flush=True)
    A_s = _build_svd(ds, mps, L)
    y_s = A_s.forward(x)
    hp = _hparams(ds, L)
    bparams = batching_params(coil_batch_size=max(1, mps.shape[0] // 2),
                              field_batch_size=1)
    A_q = qblock_svd_decomp_linop(
        ds.phis, ds.alphas, mps, ds.trj, hp,
        Q=Q, shear=True, overlap=0.05,
        svd_method='cur', spatial_mask=ds.mask, dcf=ds.dcf, bparams=bparams,
    )
    y_q = A_q.forward(x)
    rel = _rel(y_q, y_s)
    ratio = _ratio_stats(y_q, y_s)
    print(f'  rel={rel:.4e}  ratio|mean|={ratio["mean_abs"]}  cv={ratio["cv"]}',
          flush=True)
    rec = dict(L=L, Q=Q, rel=rel, ratio=ratio, per_coil=_per_coil(y_q, y_s))
    if hasattr(A_q, 'clear_plans'):
        A_q.clear_plans()
    del A_s, A_q, y_s, y_q
    torch.cuda.empty_cache()
    return rec


def _dump(report):
    (OUT / 'item1.json').write_text(json.dumps(report, indent=2, default=str))


def main():
    torch_dev = torch.device('cuda')
    report = {}

    # ---------- coco ----------
    name = 'coco_spiral'
    ds = hybrid.load_dataset(name, torch_dev)
    mps = ds.mps
    print(f'coco {ds.im_size} C={ds.C} os={ds.os} reduced={ds.reduced_im_size}',
          flush=True)
    report[name] = {}
    report[name]['delta'] = run_delta(ds, mps, name, L_svd=1, S_joint=1)
    # also a large-rank delta so factorization error is not the story
    report[name]['delta_hi'] = run_delta(ds, mps, name + '_hi', L_svd=30, S_joint=170)
    x = ds.img_ee.to(torch.complex64)
    report[name]['y'] = run_y_compare(ds, mps, name, L_svd=30, S_joint=170, x=x)
    _dump(report)
    report[name]['rank'] = run_rank(
        ds, mps, name,
        sizes=[(100, 100), (150, 150), (200, 200), (276, 276)],
    )
    report[name]['qblock'] = run_qblock_check(ds, mps, x, L=15, Q=4)
    _dump(report)

    del ds, mps, x
    torch.cuda.empty_cache()

    # ---------- tilt ----------
    name = 'tilt_spi_invivo'
    ds = hybrid.load_dataset(name, torch_dev)
    mps8, _ = coil_compress(ds.mps, ds.ksp, 8)
    print(f'tilt {ds.im_size} C_full={ds.C} C=8 os={ds.os} '
          f'reduced={ds.reduced_im_size}', flush=True)
    report[name] = {}
    report[name]['delta'] = run_delta(ds, ds.mps, name, L_svd=1, S_joint=1)
    report[name]['delta_hi'] = run_delta(ds, mps8, name + '_hi', L_svd=80, S_joint=240)
    x = ds.img_ee.to(torch.complex64)
    report[name]['y'] = run_y_compare(ds, mps8, name, L_svd=80, S_joint=240, x=x)
    _dump(report)
    report[name]['rank'] = run_rank(
        ds, mps8, name,
        sizes=[(200, 200), (280, 280), (360, 360)],
    )
    report[name]['qblock'] = run_qblock_check(ds, mps8, x, L=40, Q=4)
    _dump(report)
    print(f'\nwrote {OUT / "item1.json"}', flush=True)


if __name__ == '__main__':
    main()
