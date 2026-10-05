#!/usr/bin/env python
"""
2D type-3 NUFFT + residual HOFFT feasibility (math_docs/t3n_synergy.md).

Stages A–C only. A type-3 execution path is implemented only if a dataset
clears the conservative 1.5× predicted-speedup gate at matched accuracy.

Run from the repo root:
    experiments/type3_2d/run.sh
    JOBID=8020 experiments/type3_2d/run.sh --datasets coco_spiral tilt_spi_invivo
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    BETA_CONS, BETA_OPT, BLOCK_NS, BLOCK_NT, COND_T_MAX, DATASETS, EPSILONS,
    ETA_SWEEP, GAMMA_CONS, GAMMA_OPT, HELD_NS, HELD_NT, PRIMARY_EPS, REL_QR_TOL,
    SEEDS, W_CONS, W_OPT, Candidate, apply_T, balanced_factors, bandwidth,
    bandwidth_S, best_rotation_cleanup, candidate_metrics, cond2, dense_phase,
    fold_midpoint_offsets, gpu_mem_GiB, ho_mix_directions, load_phase_data,
    memory_ok, min_rank_heldout, mix_FQ, pareto_mask, phase_tail, predict_cost,
    qr_core_svd, rank_epsilon, rank_report, reconstruct_sampled_phase,
    residual_exp_block, residual_exp_from_psi, residual_phase_block,
    rotation_T, rms_from_tail, scale_mix_to_eta, separable_from_FQ, shear_T,
    stratified_indices, sv_tail_rel,
)
from common import _nystrom_from_svd  # noqa: E402

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 140,
    'savefig.bbox': 'tight',
})


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True
        ).strip()
    except Exception:
        return 'unknown'


def _env(device) -> dict:
    info = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=str(device),
        git=_git_hash(),
        python=sys.version.split()[0],
    )
    try:
        import cufinufft
        info['cufinufft'] = getattr(cufinufft, '__version__', 'unknown')
    except Exception as e:
        info['cufinufft'] = f'unavailable: {e}'
    if device.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(0)
        info['gpu_mem_GiB'] = gpu_mem_GiB(device)
    return info


def _check(name, ok, detail, records):
    rec = dict(name=name, ok=bool(ok), detail=detail)
    records.append(rec)
    flag = 'PASS' if ok else 'FAIL'
    print(f'  [{flag}] {name}: {detail}')
    return ok


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, torch.Tensor):
        return _jsonable(x.detach().cpu())
    if isinstance(x, Path):
        return str(x)
    return x


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------
def stage_a(data, device) -> dict:
    print('\n=== Stage A — discovery and numerical controls ===')
    checks = []
    B, D = data.B, data.D
    n, m, d = B.shape[0], D.shape[0], B.shape[1]
    print(f'  {data.name}: im={data.im_size}  Nsrc={n}  M={m}  K={data.n_high_order}  '
          f'd={d}  C={data.n_coils}  R={data.R_undersample}  '
          f'Nfft_base={data.n_fft_base}  os={data.os:.4f}')

    # Synthetic well-scaled QR vs dense SVD.
    g = torch.Generator(device=device).manual_seed(0)
    Bs = torch.randn(80, 5, generator=g, device=device, dtype=torch.float64)
    Ds = torch.randn(100, 5, generator=g, device=device, dtype=torch.float64)
    dense = torch.linalg.svdvals(Bs @ Ds.mT)
    core_s = qr_core_svd(Bs, Ds, center=False)
    rel = float((core_s.sigma[:5] - dense[:5]).norm() / dense[:5].norm())
    _check('qr_vs_dense_svd_synthetic', rel < REL_QR_TOL, f'rel={rel:.3e}', checks)

    # Dataset block: dense Ψ vs QR reconstruction.
    g = torch.Generator(device=device).manual_seed(1)
    Ii = torch.randperm(n, generator=g, device=device)[:40]
    Jj = torch.randperm(m, generator=g, device=device)[:50]
    Psi = dense_phase(B, D, Ii, Jj)
    core_raw = qr_core_svd(B, D, center=False)
    Psi_qr = reconstruct_sampled_phase(core_raw, B, D, Ii, Jj)
    rel = float((Psi - Psi_qr).norm() / Psi.norm().clamp(min=1e-30))
    cond = float((core_raw.sigma[0] / core_raw.sigma[core_raw.rank - 1].clamp(min=1e-30)).cpu())
    _check('qr_vs_dense_phase_block', rel < REL_QR_TOL,
           f'rel={rel:.3e}  cond(CΨ)={cond:.3e}  rank={core_raw.rank}/{d}', checks)

    # Offset identity.
    core_c = qr_core_svd(B, D, center=True)
    Psi_c = reconstruct_sampled_phase(core_c, B, D, Ii, Jj)
    rel = float((Psi - Psi_c).norm() / Psi.norm().clamp(min=1e-30))
    _check('separable_offset_identity', rel < REL_QR_TOL, f'rel={rel:.3e}', checks)

    # Factor recentering.
    F, Q = balanced_factors(core_c, 2)
    a, c = core_c.a, core_c.c
    Fc, Qc, a2, c2 = fold_midpoint_offsets(F, Q, a, c)
    lhs = F[Ii] @ Q[Jj].mT + a[Ii][:, None] + c[Jj][None, :]
    rhs = Fc[Ii] @ Qc[Jj].mT + a2[Ii][:, None] + c2[Jj][None, :]
    rel = float((lhs - rhs).norm() / lhs.norm().clamp(min=1e-30))
    _check('factor_recentering', rel < REL_QR_TOL, f'rel={rel:.3e}', checks)

    # Convention: HO columns of B,D vs Φ,C; total vs r·k + H.
    H = data.Phi[Ii] @ data.C[Jj].mT
    Hbd = B[Ii, 2:] @ D[Jj, 2:].mT
    G = data.R[Ii] @ data.Ktraj[Jj].mT
    rel_h = float((H - Hbd).norm() / H.norm().clamp(min=1e-30)) if H.norm() > 0 else 0.0
    rel_tot = float((Psi - (G + H)).norm() / Psi.norm().clamp(min=1e-30))
    _check('phase_convention_HO', rel_h < REL_QR_TOL, f'rel={rel_h:.3e}', checks)
    _check('phase_convention_total', rel_tot < REL_QR_TOL, f'rel={rel_tot:.3e}', checks)

    # Geometry residual == H (a=c=0). Exponential SVD vs baseline E0.
    geo = Candidate('geometry', 'geometry', 2, data.R, data.Ktraj,
                    torch.zeros(n, device=device, dtype=B.dtype),
                    torch.zeros(m, device=device, dtype=D.dtype))
    I = stratified_indices(n, min(BLOCK_NS, n), data.energy_src, data.R, seed=0)
    J = stratified_indices(m, min(BLOCK_NT, m), data.energy_tgt, data.Ktraj, seed=0)
    E0 = torch.exp(-2j * math.pi * (data.Phi[I] @ data.C[J].mT))
    ER = torch.exp(-2j * math.pi * residual_phase_block(B, D, geo, I, J))
    rel_e = float((E0 - ER).norm() / E0.norm().clamp(min=1e-30))
    _check('geometry_residual_equals_HO', rel_e < REL_QR_TOL, f'rel={rel_e:.3e}', checks)
    s0 = torch.linalg.svdvals(E0)
    sr = torch.linalg.svdvals(ER)
    L0 = {f'{e:.0e}': rank_epsilon(s0, e) for e in EPSILONS}
    Lgeo = {f'{e:.0e}': rank_epsilon(sr, e) for e in EPSILONS}
    _check('geometry_rank_matches_baseline', L0 == Lgeo,
           f'L0={L0}  Lgeo={Lgeo}', checks)

    # Zero high-order: residual rank 1.
    Bz, Dz = data.R, data.Ktraj
    core_z = qr_core_svd(Bz, Dz, center=False)
    geo_z = Candidate('zero_ho', 'geometry', 2, data.R, data.Ktraj,
                      torch.zeros(n, device=device, dtype=B.dtype),
                      torch.zeros(m, device=device, dtype=D.dtype))
    Ez = residual_exp_block(Bz, Dz, geo_z, I[: min(256, I.numel())],
                            J[: min(512, J.numel())])
    sz = torch.linalg.svdvals(Ez)
    Lz = rank_epsilon(sz, PRIMARY_EPS)
    # Constant kernel of ones: first SV is sqrt(NM), rest ~0.
    rel_ones = float((Ez - 1).abs().max().cpu())
    _check('zero_HO_constant_kernel', rel_ones < 1e-12 and Lz <= 1,
           f'max|E-1|={rel_ones:.3e}  L={Lz}  sigma0={float(sz[0]):.4g}', checks)

    spectra = dict(
        raw=core_raw.sigma.detach().cpu().numpy().tolist(),
        offset=core_c.sigma.detach().cpu().numpy().tolist(),
        tail_raw_p2=phase_tail(core_raw.sigma, 2),
        tail_offset_p2=phase_tail(core_c.sigma, 2),
        rms_raw_p2=rms_from_tail(core_raw.sigma, 2, n, m),
        rms_offset_p2=rms_from_tail(core_c.sigma, 2, n, m),
        L0_block=L0,
        rank_raw=core_raw.rank,
        rank_offset=core_c.rank,
        cond_core=cond,
    )
    all_ok = all(c['ok'] for c in checks)
    print(f'  Stage A: {"PASS" if all_ok else "FAIL"}')
    return dict(checks=checks, pass_=all_ok, spectra=spectra,
                manifest=dict(
                    dataset=data.name, im_size=list(data.im_size),
                    Nsrc=n, M=m, K=data.n_high_order, coils=data.n_coils,
                    Nfft_base=data.n_fft_base, os=data.os,
                    R=data.R_undersample, dtype='float64',
                    phase_units='cycles', mask='ESPIRiT evals>0.9',
                    n_fft_note='upsampfac=2 on native grid (cufinufft type-1/2)',
                ),
                cores=dict(raw=core_raw, offset=core_c),
                geometry=geo)


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------
def _materialize(cid, family, F, Q, a, c, core_c, n, m, S_geo, T=None, notes='', p=2):
    Fc, Qc, a2, c2 = fold_midpoint_offsets(F, Q, a, c)
    cand = Candidate(cid, family, p, Fc.detach(), Qc.detach(),
                     a2.detach(), c2.detach(),
                     None if T is None else T.detach().cpu().numpy(), notes)
    cand.metrics = candidate_metrics(cand, core_c, n, m, S_geo)
    return cand


def _budgeted(core, F0, Q0, Smax, steps=180, lr=0.08):
    X = (core.QB.mT @ F0).detach().clone().requires_grad_(True)
    Y = (core.QD.mT @ Q0).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([X, Y], lr=lr)
    last = (F0.detach(), Q0.detach(), float('inf'), float('inf'))
    Smax = max(float(Smax), 1.0)
    for _ in range(steps):
        opt.zero_grad()
        F = core.QB @ X
        Q = core.QD @ Y
        loss = torch.linalg.norm(core.core - X @ Y.mT) ** 2
        s = []
        for j in range(2):
            s.append((F[:, j].max() - F[:, j].min()) * (Q[:, j].max() - Q[:, j].min()))
        S = (1 + s[0]) * (1 + s[1])
        pen = torch.relu(S / Smax - 1.0).square()
        obj = loss + (loss.detach() + 1.0) * 25.0 * pen
        if not torch.isfinite(obj):
            break
        obj.backward()
        opt.step()
        last = (F.detach(), Q.detach(), float(loss.detach().cpu()), float(S.detach().cpu()))
    return last


def stage_b(data, cores, device) -> dict:
    print('\n=== Stage B — phase-rank / bandwidth screen ===')
    n, m = data.n_src, data.n_tgt
    core_r, core_c = cores['raw'], cores['offset']
    geo_bw = bandwidth(data.R, data.Ktraj)
    S_geo = geo_bw['S']
    print(f'  S_geometry={S_geo:.4g}  s={geo_bw["s"]}')

    z_n = torch.zeros(n, device=device, dtype=data.B.dtype)
    z_m = torch.zeros(m, device=device, dtype=data.D.dtype)
    a_c, c_c = core_c.a, core_c.c

    cands: list[Candidate] = []
    cands.append(_materialize('geometry', 'geometry', data.R, data.Ktraj,
                              z_n, z_m, core_c, n, m, S_geo,
                              notes='F=R, Q=K; residual is HO phase'))

    F_raw, Q_raw = balanced_factors(core_r, 2)
    cands.append(_materialize('svd_raw_p2', 'svd_raw', F_raw, Q_raw,
                              z_n, z_m, core_c, n, m, S_geo,
                              notes='unconstrained rank-2 SVD of raw Ψ'))
    F1, Q1 = balanced_factors(core_r, 1)
    # Rank-1 diagnostic: pad a zero column so bandwidth helper stays 2D.
    F1p = torch.cat([F1, torch.zeros_like(F1)], dim=1)
    Q1p = torch.cat([Q1, torch.zeros_like(Q1)], dim=1)
    cands.append(_materialize('svd_raw_p1', 'svd_raw', F1p, Q1p,
                              z_n, z_m, core_c, n, m, S_geo, notes='rank-1 diagnostic', p=1))

    F_off, Q_off = balanced_factors(core_c, 2)
    cands.append(_materialize('svd_offset_p2', 'svd_offset', F_off, Q_off,
                              a_c, c_c, core_c, n, m, S_geo,
                              notes='offset-aware rank-2 SVD (preferred)'))
    F1o, Q1o = balanced_factors(core_c, 1)
    F1op = torch.cat([F1o, torch.zeros_like(F1o)], dim=1)
    Q1op = torch.cat([Q1o, torch.zeros_like(Q1o)], dim=1)
    cands.append(_materialize('svd_offset_p1', 'svd_offset', F1op, Q1op,
                              a_c, c_c, core_c, n, m, S_geo,
                              notes='rank-1 diagnostic', p=1))

    # Basis-preserving T: rotations then shears of the offset-aware rank-2 factors.
    # Metrics only; materialize the best few.
    F0, Q0 = F_off, Q_off
    dtype, dev = F0.dtype, F0.device
    t_records = []
    thetas = [math.radians(t) for t in range(0, 180, 15)]
    shears = [0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0]
    best_S, best_T, best_note = math.inf, None, ''
    for th in thetas:
        Rth = rotation_T(th, dev, dtype)
        for ax in (0, 1):
            for sg in shears:
                T = Rth @ shear_T(ax, sg, dev, dtype)
                cd = cond2(T)
                if cd > COND_T_MAX:
                    continue
                Ft, Qt = apply_T(F0, Q0, T)
                bw = bandwidth(Ft, Qt)
                t_records.append(dict(
                    theta_deg=th * 180 / math.pi, shear_axis=ax, shear=sg,
                    cond_T=cd, S=bw['S'], s=bw['s'],
                ))
                if bw['S'] < best_S:
                    best_S, best_T, best_note = bw['S'], T, (
                        f'rot={th * 180 / math.pi:.0f}deg shear{ax}={sg} cond={cd:.2f}')
    print(f'  T-search: {len(t_records)} feasible (cond<= {COND_T_MAX:.0f}); '
          f'best S={best_S:.4g} vs SVD S={bandwidth(F0, Q0)["S"]:.4g}')
    if best_T is not None:
        Ft, Qt = apply_T(F0, Q0, best_T)
        cands.append(_materialize('svd_offset_Tbest', 'T_search', Ft, Qt,
                                  a_c, c_c, core_c, n, m, S_geo,
                                  T=best_T, notes=best_note))
        # Also keep a few mid-bandwidth rotations (phase-preserving).
        for th in (0.0, math.pi / 8, math.pi / 4, 3 * math.pi / 8):
            T = rotation_T(th, dev, dtype)
            Ft, Qt = apply_T(F0, Q0, T)
            cands.append(_materialize(
                f'svd_offset_rot{int(th * 180 / math.pi)}', 'T_search',
                Ft, Qt, a_c, c_c, core_c, n, m, S_geo, T=T,
                notes=f'rotation {th * 180 / math.pi:.0f} deg'))

    # Budgeted core fits (small X,Y).
    for ratio in SMAX_RATIOS:
        Smax = ratio * S_geo
        for tag, Finit, Qinit in (('svd', F0, Q0), ('geo', data.R, data.Ktraj)):
            Fb, Qb, loss, S_end = _budgeted(core_c, Finit, Qinit, Smax)
            cands.append(_materialize(
                f'budget_r{ratio:g}_{tag}', 'budgeted', Fb, Qb, a_c, c_c,
                core_c, n, m, S_geo,
                notes=f'Smax/Sgeo={ratio:g} init={tag} S_end={S_end:.4g}'))

    rows = [c.metrics for c in cands]
    rms = np.array([r['phase_rms_cycles'] for r in rows])
    S = np.array([r['S'] for r in rows])
    nd = pareto_mask(rms, S)
    for c, flag in zip(cands, nd):
        c.metrics['pareto'] = bool(flag)
    print(f'  {len(cands)} candidates, {int(nd.sum())} on Pareto(phase RMS, S)')
    for c in cands:
        mtr = c.metrics
        print(f'    {c.cid:28s}  rms={mtr["phase_rms_cycles"]:.3e}  '
              f'S/Sgeo={mtr["S_over_Sgeo"]:.3f}  s={np.array(mtr["s"])}')

    must = {'geometry', 'svd_offset_p2', 'svd_raw_p2'}
    short = []
    seen_keys = set()
    for c in cands:
        if c.p != 2:
            continue
        key = (round(c.metrics['phase_rms_cycles'], 4),
               round(c.metrics['S_over_Sgeo'], 3))
        keep = c.cid in must or c.metrics.get('pareto')
        if not keep:
            continue
        if c.cid not in must and key in seen_keys:
            continue
        seen_keys.add(key)
        short.append(c)
    req = [c for c in short if c.cid in must]
    rest = [c for c in short if c.cid not in must]
    # Prefer T-search / low-S budgeted; drop near-duplicate geometry clones.
    rest = sorted(rest, key=lambda c: (c.metrics['phase_rms_cycles'], c.metrics['S']))
    out = req + rest[: max(0, 8 - len(req))]
    print(f'  shortlist ({len(out)}): {[c.cid for c in out]}')
    return dict(candidates=cands, shortlist=out, S_geo=S_geo,
                geo_bw=geo_bw, t_search=t_records)


from stage_b_search import stage_b  # noqa: E402  (replaces the phase-only Stage B above)


# ---------------------------------------------------------------------------
# Stage C
# ---------------------------------------------------------------------------
def _shared_blocks(data, seed):
    """One (I,J) pair plus held-out indices; Ψ tiles are formed once per seed."""
    I = stratified_indices(data.n_src, min(BLOCK_NS, data.n_src),
                           data.energy_src, data.R, seed=seed)
    J = stratified_indices(data.n_tgt, min(BLOCK_NT, data.n_tgt),
                           data.energy_tgt, data.Ktraj, seed=seed + 17)
    g = torch.Generator(device=data.B.device).manual_seed(seed + 99)
    Iho = torch.randperm(data.n_src, generator=g, device=data.B.device)
    Jho = torch.randperm(data.n_tgt, generator=g, device=data.B.device)
    Iho = Iho[~torch.isin(Iho, I)][: min(HELD_NS, max(32, data.n_src // 8))]
    Jho = Jho[~torch.isin(Jho, J)][: min(HELD_NT, max(32, data.n_tgt // 8))]
    Psi = data.B[I] @ data.D[J].mT
    Psi_Iho_J = data.B[Iho] @ data.D[J].mT
    Psi_I_Jho = data.B[I] @ data.D[Jho].mT
    Psi_ho = data.B[Iho] @ data.D[Jho].mT
    return dict(I=I, J=J, Iho=Iho, Jho=Jho, Psi=Psi,
                Psi_Iho_J=Psi_Iho_J, Psi_I_Jho=Psi_I_Jho, Psi_ho=Psi_ho)


def _eval_residual(cand, blk):
    E = residual_exp_from_psi(blk['Psi'], cand, blk['I'], blk['J'])
    U, s, Vh = torch.linalg.svd(E, full_matrices=False)
    E_Iho_J = residual_exp_from_psi(blk['Psi_Iho_J'], cand, blk['Iho'], blk['J'])
    E_I_Jho = residual_exp_from_psi(blk['Psi_I_Jho'], cand, blk['I'], blk['Jho'])
    E_ho = residual_exp_from_psi(blk['Psi_ho'], cand, blk['Iho'], blk['Jho'])
    return E, s, U, Vh, E_Iho_J, E_I_Jho, E_ho


def _stable(vals):
    med = float(np.median(vals))
    spread = float(np.max(vals) - np.min(vals)) if vals else 0.0
    return spread <= max(2.0, 0.10 * med), med, spread


def _score_candidate(cand, blk_svals, L_med, env, data):
    """Turn per-seed ranks / held-out operating ranks into a gate row."""
    gpu = env.get('gpu_mem_GiB', 0.0)
    s = cand.metrics['s']
    row = dict(cid=cand.cid, family=cand.family, metrics=cand.metrics,
               ranks={f'{e:.0e}': blk_svals['ranks'][e] for e in EPSILONS},
               L_held={f'{e:.0e}': blk_svals['L_held'][e] for e in EPSILONS},
               heldout_at_Lblock=blk_svals['held_block'],
               heldout_at_Lop=blk_svals['held_op'],
               svals_seed0=blk_svals['svals_seed0'])
    for e in EPSILONS:
        ok_b, med_b, _ = _stable(blk_svals['ranks'][e])
        Lp = int(round(med_b))
        L = int(round(L_med[e]))
        opt = predict_cost(L, Lp, s, data.n_src, data.n_tgt, data.n_fft_base,
                           GAMMA_OPT, BETA_OPT, W_OPT)
        cons = predict_cost(L, Lp, s, data.n_src, data.n_tgt, data.n_fft_base,
                            GAMMA_CONS, BETA_CONS, W_CONS)
        mem = memory_ok(max(Lp, 1), data.n_src, data.n_tgt, cons['Nprime'],
                        gpu, data.n_coils)
        key = f'{e:.0e}'
        row[f'L_{key}'] = L
        row[f'Lp_{key}'] = Lp
        row[f'Lblock_{key}'] = int(round(med_b))
        row[f'Lp_stable_{key}'] = ok_b
        row[f'opt_{key}'] = opt
        row[f'cons_{key}'] = cons
        row[f'mem_{key}'] = mem
        row[f'speedup_cons_{key}'] = 1.0 / max(cons['T_new_over_T_base'], 1e-12)
        row[f'speedup_opt_{key}'] = 1.0 / max(opt['T_new_over_T_base'], 1e-12)
    ekey = f'{PRIMARY_EPS:.0e}'
    held_med = float(np.median(blk_svals['held_op'])) if blk_svals['held_op'] else float('nan')
    row['heldout_med'] = held_med
    Lp, L = row[f'Lp_{ekey}'], row[f'L_{ekey}']
    spd = row[f'speedup_cons_{ekey}']
    reasons = []
    if not (held_med <= PRIMARY_EPS):
        reasons.append(f'held-out kernel err {held_med:.3e} > {PRIMARY_EPS:.0e}')
    if not (Lp < L):
        reasons.append(f"L'={Lp} not < L={L}")
    if not row[f'Lp_stable_{ekey}']:
        reasons.append('rank estimates unstable')
    if not row[f'mem_{ekey}']['ok']:
        reasons.append('memory headroom')
    if spd < 1.5:
        tag = 'marginal' if 1.0 <= spd < 1.5 else 'fail'
        reasons.append(f'conservative speedup {spd:.2f}x ({tag}; need ≥1.5)')
    row['gate_pass'] = len(reasons) == 0
    row['gate_reasons'] = reasons
    row['gate_label'] = 'pass' if row['gate_pass'] else (
        'marginal' if (Lp < L and 1.0 <= spd < 1.5) else 'fail')
    print(f'    L={L} (block {row[f"Lblock_{ekey}"]})  L\'={Lp}  '
          f'held={held_med:.3e}  speedup cons={spd:.2f}x '
          f'opt={row[f"speedup_opt_{ekey}"]:.2f}x  {row["gate_label"]}')
    if reasons:
        print('     ' + '; '.join(reasons))
    return row


def stage_c(data, baseline_geo: Candidate, shortlist, env) -> dict:
    print('\n=== Stage C — residual exponential rank and predicted speedup ===')
    blocks = [_shared_blocks(data, seed) for seed in SEEDS]

    def collect(cand):
        ranks = {e: [] for e in EPSILONS}
        L_held = {e: [] for e in EPSILONS}
        held_block, held_op = [], []
        svals_ex = None
        for si, blk in enumerate(blocks):
            E, svals, U, Vh, E_Iho_J, E_I_Jho, E_ho = _eval_residual(cand, blk)
            if si == 0:
                svals_ex = svals.detach().cpu().numpy()[:min(64, svals.numel())].tolist()
            fac = (U, svals, Vh)
            for e in EPSILONS:
                Lblk = rank_epsilon(svals, e)
                ranks[e].append(Lblk)
                L_held[e].append(Lblk)
                if e == PRIMARY_EPS:
                    held_op.append(_nystrom_from_svd(
                        U, svals, Vh, E_Iho_J, E_I_Jho, E_ho, Lblk))
            held_block.append(held_op[-1] if held_op else float('nan'))
        return dict(ranks=ranks, L_held=L_held, held_block=held_block,
                    held_op=held_op, svals_seed0=svals_ex)

    base = collect(baseline_geo)
    L_med, L_stable = {}, {}
    for e in EPSILONS:
        ok_b, med_b, sp_b = _stable(base['ranks'][e])
        L_med[e] = med_b
        L_stable[e] = ok_b
        print(f'  baseline {e:.0e}: L_block={base["ranks"][e]}  '
              f'med={med_b:.1f}  spread={sp_b:.1f}  stable={ok_b}')

    # Phase-preserving T-search shares the offset-aware SVD residual.
    cache = {}
    results = []
    for cand in shortlist:
        print(f'  -- {cand.cid} --')
        key = 'svd_offset' if cand.family == 'T_search' else cand.cid
        if key not in cache:
            cache[key] = collect(cand)
            cache[key]['L_held_base'] = base['L_held']
        blob = cache[key]
        blob = dict(blob)
        blob['L_held_base'] = base['L_held']
        results.append(_score_candidate(cand, blob, L_med, env, data))

    return dict(
        baseline_ranks={f'{e:.0e}': base['ranks'][e] for e in EPSILONS},
        baseline_L_held={f'{e:.0e}': base['L_held'][e] for e in EPSILONS},
        baseline_L_med={f'{e:.0e}': L_med[e] for e in EPSILONS},
        baseline_stable={f'{e:.0e}': L_stable[e] for e in EPSILONS},
        baseline_held=base['held_op'],
        candidates=results,
        gate_pass=any(r['gate_pass'] for r in results),
    )


# ---------------------------------------------------------------------------
# Plots + report
# ---------------------------------------------------------------------------
def plot_all(data, stageA, stageB, stageC, out: Path):
    spec_r = np.array(stageA['spectra']['raw'])
    spec_c = np.array(stageA['spectra']['offset'])
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for a, s, lab in ((ax[0], spec_r, 'raw total'), (ax[0], spec_c, 'offset-aware')):
        a.semilogy(np.arange(1, len(s) + 1), s / s[0], marker='o', ms=3, label=lab)
    ax[0].set(xlabel='k', ylabel=r'$\sigma_k / \sigma_1$', title=f'{data.name} phase spectrum')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    tail_r = np.sqrt(np.cumsum(spec_r[::-1] ** 2)[::-1] / np.sum(spec_r ** 2))
    tail_c = np.sqrt(np.cumsum(spec_c[::-1] ** 2)[::-1] / np.sum(spec_c ** 2))
    ax[1].semilogy(np.arange(len(tail_r)), tail_r, marker='o', ms=3, label='raw')
    ax[1].semilogy(np.arange(len(tail_c)), tail_c, marker='o', ms=3, label='offset-aware')
    ax[1].set(xlabel='p (absorbed rank)', ylabel='rel Frobenius tail', title='phase-rank tail')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.savefig(out / 'phase_spectrum.png')
    plt.close(fig)

    rows = [c.metrics for c in stageB['candidates']]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    families = {}
    for r in rows:
        families.setdefault(r['family'], []).append(r)
    markers = dict(geometry='*', svd_raw='o', svd_offset='s', T_search='^',
                   mix='D', refine='P')
    for fam, rs in families.items():
        ax.scatter([r['S_over_Sgeo'] for r in rs],
                   [r['phase_rms_cycles'] for r in rs],
                   marker=markers.get(fam, 'x'), label=fam, s=36)
        for r in rs:
            if r.get('pareto_residual') or r['cid'] in ('geometry', 'svd_offset_p2'):
                ax.annotate(r['cid'], (r['S_over_Sgeo'], r['phase_rms_cycles']),
                            fontsize=6, xytext=(3, 3), textcoords='offset points')
    ax.set(xlabel=r'$S / S_{\mathrm{geometry}}$', ylabel='phase RMS (cycles)',
           title=f'{data.name} phase error vs bandwidth', yscale='log', xscale='log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')
    fig.savefig(out / 'pareto_phase_bandwidth.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.4))
    ekey = f'{PRIMARY_EPS:.0e}'
    for r in stageC['candidates']:
        sv = np.array(r.get('svals_seed0') or [1.0])
        ax[0].semilogy(np.arange(1, len(sv) + 1), sv / sv[0], label=r['cid'], lw=1)
    ax[0].set(xlabel='ℓ', ylabel=r'$\sigma_\ell / \sigma_1$', title='residual exp. spectrum')
    ax[0].legend(fontsize=6)
    ax[0].grid(True, alpha=0.3)
    infl, Lp, L0 = [], [], None
    for r in stageC['candidates']:
        infl.append(r['metrics']['S_over_Sgeo'])
        Lp.append(r[f'Lp_{ekey}'])
        L0 = r[f'L_{ekey}']
        ax[1].scatter(r['metrics']['S_over_Sgeo'], r[f'Lp_{ekey}'], s=40)
        ax[1].annotate(r['cid'], (r['metrics']['S_over_Sgeo'], r[f'Lp_{ekey}']),
                       fontsize=6)
    if L0 is not None:
        ax[1].axhline(L0, color='k', ls='--', lw=1, label=f'baseline L={L0}')
    ax[1].set(xlabel=r'$S/S_{\mathrm{geo}}$', ylabel="L' (ε=1e-3)",
              title="residual rank vs achieved bandwidth", xscale='log')
    ax[1].legend(fontsize=7)
    ax[1].grid(True, alpha=0.3, which='both')
    fig.savefig(out / 'residual_rank.png')
    plt.close(fig)

    # Speedup heatmap over (rho, L'/L).
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    Lref = int(round(stageC['baseline_L_med'][ekey]))
    N = float(data.n_fft_base)
    M = float(data.n_tgt)
    Ns = float(data.n_src)
    rhos = np.logspace(-0.3, 1.2, 80)
    ratios = np.linspace(0.05, 1.4, 80)
    W = W_CONS
    Z = np.zeros((len(ratios), len(rhos)))
    t_base = Lref * (N * np.log(N) + W * M)
    for i, rr in enumerate(ratios):
        for j, rho in enumerate(rhos):
            Np = rho * N
            t_new = (rr * Lref) * (Np * np.log(max(Np, 2)) + W * Ns + W * M)
            Z[i, j] = t_base / max(t_new, 1e-30)
    im = ax.pcolormesh(rhos, ratios, Z, shading='auto', cmap='viridis',
                       vmin=0.25, vmax=3.0)
    cs = ax.contour(rhos, ratios, Z, levels=[1.0, 1.5], colors=['w', 'C1'], linewidths=1.2)
    ax.clabel(cs, fmt=lambda v: f'{v:.1f}×', fontsize=7)
    fig.colorbar(im, ax=ax, label='predicted T_base / T_new (conservative)')
    for r in stageC['candidates']:
        rho = r[f'cons_{ekey}']['rho']
        rr = r[f'Lp_{ekey}'] / max(r[f'L_{ekey}'], 1)
        ax.scatter(rho, rr, c='red', s=28, zorder=5)
        ax.annotate(r['cid'], (rho, rr), fontsize=6, color='w',
                    path_effects=[])
    ax.set(xscale='log', xlabel=r'$\rho = N\' / N_{\mathrm{fft}}$',
           ylabel="L'/L", title=f'{data.name} predicted speedup (conservative)')
    fig.savefig(out / 'speedup_heatmap.png')
    plt.close(fig)


def write_tables(stageA, stageB, stageC, out: Path):
    man = stageA['manifest']
    with (out / 'manifest.json').open('w') as f:
        json.dump(_jsonable(man), f, indent=2)
    fields = ['cid', 'family', 'p', 'phase_rms_cycles', 'S', 'S_over_Sgeo',
              's', 'eta', 'feasible', 'train_L', 'train_tail', 'val_L', 'notes']
    with (out / 'candidates.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in stageB['candidates']:
            m = c.metrics
            w.writerow({k: m.get(k, getattr(c, k, '')) for k in fields})
    with (out / 'coverage.json').open('w') as f:
        json.dump(_jsonable(stageB.get('coverage', [])), f, indent=2)
    ekey = f'{PRIMARY_EPS:.0e}'
    cfields = ['cid', 'family', 'L', 'Lp', 'Lp_over_L', 'heldout_med',
               'speedup_cons', 'speedup_opt', 'rho_cons', 'Nprime_cons',
               'gate_label', 'gate_reasons']
    with (out / 'stageC_screen.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cfields)
        w.writeheader()
        for r in stageC['candidates']:
            w.writerow(dict(
                cid=r['cid'], family=r['family'],
                L=r[f'L_{ekey}'], Lp=r[f'Lp_{ekey}'],
                Lp_over_L=r[f'Lp_{ekey}'] / max(r[f'L_{ekey}'], 1),
                heldout_med=r['heldout_med'],
                speedup_cons=r[f'speedup_cons_{ekey}'],
                speedup_opt=r[f'speedup_opt_{ekey}'],
                rho_cons=r[f'cons_{ekey}']['rho'],
                Nprime_cons=r[f'cons_{ekey}']['Nprime'],
                gate_label=r['gate_label'],
                gate_reasons='; '.join(r['gate_reasons']),
            ))


def dataset_summary_row(name, stageA, stageB, stageC) -> dict:
    ekey = f'{PRIMARY_EPS:.0e}'
    ranked = sorted(stageC['candidates'],
                    key=lambda r: -r[f'speedup_cons_{ekey}'])
    best = ranked[0] if ranked else None
    passed = [r for r in stageC['candidates'] if r['gate_pass']]
    winner = (max(passed, key=lambda r: r[f'speedup_cons_{ekey}']) if passed else best)
    return dict(
        dataset=name,
        stageA='PASS' if stageA['pass_'] else 'FAIL',
        L=None if winner is None else winner[f'L_{ekey}'],
        best_cid=None if winner is None else winner['cid'],
        Lp=None if winner is None else winner[f'Lp_{ekey}'],
        heldout=None if winner is None else winner['heldout_med'],
        speedup_cons=None if winner is None else winner[f'speedup_cons_{ekey}'],
        speedup_opt=None if winner is None else winner[f'speedup_opt_{ekey}'],
        gate=None if winner is None else winner['gate_label'],
        reasons='' if winner is None else '; '.join(winner['gate_reasons']),
        any_pass=stageC['gate_pass'],
        type3='skip — gate not passed' if not stageC['gate_pass']
        else 'would run Stages D–E',
    )


def write_report(run_id, env, rows, out_root: Path, per_ds: dict):
    lines = [
        '# 2D type-3 + residual HOFFT feasibility',
        '',
        f'Implements `math_docs/t3n_synergy.md` Stages A–C on **coco_spiral** and '
        f'**tilt_spi_invivo**. Imaging and the latent transform stay 2D. Rank-1 fits '
        f'are diagnostic; rank 2 is the proposed type-3 path.',
        '',
        f'**Run:** `{run_id}`  **git:** `{env.get("git")}`  '
        f'**GPU:** {env.get("gpu", "cpu")} ({env.get("gpu_mem_GiB", 0):.1f} GiB)  '
        f'**torch:** {env.get("torch")}  **cufinufft:** {env.get("cufinufft")}',
        '',
        'Preprocessing matches `paper_experiments/run_sweep.py` / `hybrid_feas` '
        '(same R, ESPIRiT 0.9 mask, `remove_linear_terms`). Phase is in cycles. '
        'The primary search spends η = S/S_geo ∈ {1.5, 2, 4, 8} by mixing '
        'high-order bases into the two latent coordinates (F=R+ΦU, Q=K+CV) and '
        'scoring residual exponential rank. Phase-SVD / rotations are controls; '
        'they cannot test the bandwidth–compression tradeoff.',
        '',
        '**Implementation gate:** held-out kernel error ≤ 1e-3, L\' < L, stable ranks, '
        'conservative memory with 20% headroom, predicted conservative speedup ≥ 1.5×. '
        'A prediction in [1, 1.5) is **marginal**, not a pass. If neither dataset '
        'passes, stop without a type-3 adapter.',
        '',
        '## Verdict',
        '',
        '| dataset | Stage A | baseline L (ε=1e-3) | best candidate | L\' | held-out | '
        'speedup cons / opt | gate | type-3 |',
        '|---|---|---:|---|---:|---:|---:|---|---|',
    ]
    for r in rows:
        held = '—' if r['heldout'] is None else f"{r['heldout']:.2e}"
        spd = '—' if r['speedup_cons'] is None else (
            f"{r['speedup_cons']:.2f}× / {r['speedup_opt']:.2f}×")
        lines.append(
            f"| {r['dataset']} | {r['stageA']} | {r['L']} | `{r['best_cid']}` | "
            f"{r['Lp']} | {held} | {spd} | **{r['gate']}** | {r['type3']} |"
        )
    n_pass = sum(1 for r in rows if r['any_pass'])
    if n_pass == 0:
        rec = ('**Recommendation:** do not implement a type-3 path. Residual-rank '
               'reduction does not outweigh type-3 source spreading and grid inflation '
               'under the conservative cost model on either dataset.')
    elif n_pass == 1:
        rec = ('**Recommendation:** dataset-specific feasibility only. Benchmark the '
               'passing dataset (Stages D–E); keep the other as a negative result.')
    else:
        rec = ('**Recommendation:** both datasets pass the A–C gate; a 2D type-3 '
               'adapter is justified.')
    lines += ['', rec, '',
              'Operation-count ratios are **predicted**, not measured wall-clock. '
              'Stages D–E (cuFINUFFT `nufft2d3`, isign=-1, hybrid fwd/adj) were not '
              'started unless a gate pass is recorded above.',
              '']
    for name, blob in per_ds.items():
        A, B, C = blob['A'], blob['B'], blob['C']
        lines += [f'## {name}', '']
        fails = [c for c in A['checks'] if not c['ok']]
        lines.append(f"Stage A: **{'PASS' if A['pass_'] else 'FAIL'}** "
                     f"({sum(c['ok'] for c in A['checks'])}/{len(A['checks'])} checks).")
        if fails:
            for c in fails:
                lines.append(f"- FAIL `{c['name']}`: {c['detail']}")
        man = A['manifest']
        lines += [
            '',
            f"- grid {man['im_size']}, Nsrc={man['Nsrc']}, M={man['M']}, "
            f"K={man['K']}, coils={man['coils']}, Nfft_base={man['Nfft_base']}, "
            f"R={man['R']}",
            f"- phase rank raw={A['spectra']['rank_raw']}, "
            f"offset-aware={A['spectra']['rank_offset']}; "
            f"p=2 rel Frobenius tail raw={A['spectra']['tail_raw_p2']:.3e}, "
            f"offset={A['spectra']['tail_offset_p2']:.3e}",
            f"- geometry S={B['S_geo']:.4g}, s={B['geo_bw']['s']}",
            f"- search coverage: L_base={B.get('L_base')}, "
            f"shortlist S/Sgeo="
            f"{[round(c.metrics['S_over_Sgeo'], 2) for c in B.get('shortlist', [])]}",
            '',
            'Shortlist at ε=1e-3:',
            '',
            '| candidate | phase RMS | S/Sgeo | L\' | held-out | speedup cons | gate |',
            '|---|---:|---:|---:|---:|---:|---|',
        ]
        ekey = f'{PRIMARY_EPS:.0e}'
        for r in C['candidates']:
            lines.append(
                f"| `{r['cid']}` | {r['metrics']['phase_rms_cycles']:.3e} | "
                f"{r['metrics']['S_over_Sgeo']:.3f} | {r[f'Lp_{ekey}']} | "
                f"{r['heldout_med']:.2e} | {r[f'speedup_cons_{ekey}']:.2f}× | "
                f"{r['gate_label']} |"
            )
        lines += [
            '',
            f"Figures: `{name}/{run_id}/phase_spectrum.png`, "
            f"`pareto_phase_bandwidth.png`, `residual_rank.png`, `speedup_heatmap.png`.",
            '',
        ]
    text = '\n'.join(lines) + '\n'
    (HERE / 'REPORT.md').write_text(text)
    (out_root / 'REPORT.md').write_text(text)
    print(f'\nWrote {HERE / "REPORT.md"}')


def run_one(name, device, run_id, out_root, env):
    print('\n' + '=' * 72)
    print(f'DATASET {name}')
    print('=' * 72)
    data = load_phase_data(name, device, repo=REPO)
    out = out_root / name / run_id
    out.mkdir(parents=True, exist_ok=True)
    A = stage_a(data, device)
    cores = A.pop('cores')
    geo = A.pop('geometry')
    B = stage_b(data, cores, device)
    C = stage_c(data, geo, B['shortlist'], env)
    plot_all(data, A, B, C, out)
    write_tables(A, B, C, out)
    blob = dict(A=A, B=dict(
        S_geo=B['S_geo'], geo_bw=B['geo_bw'],
        t_search_n=len(B.get('t_search') or []),
        coverage=B.get('coverage', []),
        L_base=B.get('L_base'),
        candidates=[c.metrics for c in B['candidates']],
        shortlist=[c.cid for c in B['shortlist']],
    ), C=C, env=env)
    # Keep Stage B candidate metrics only in JSON (no tensors).
    with (out / 'results.json').open('w') as f:
        json.dump(_jsonable(blob), f, indent=2)
    # Restore objects needed for the report (full C, A, B with Candidate list).
    return dict(A=A, B=B, C=C, out=out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument('--run-id', default=None)
    args = p.parse_args()
    for name in args.datasets:
        if name not in DATASETS:
            raise SystemExit(f'unknown dataset {name}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = _env(device)
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_root = REPO / 'results' / 'type3_2d'
    out_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps(_jsonable(env), indent=2))
    print(f'run_id={run_id}  out={out_root}')

    rows, per = [], {}
    for name in args.datasets:
        try:
            blob = run_one(name, device, run_id, out_root, env)
            per[name] = blob
            rows.append(dataset_summary_row(name, blob['A'], blob['B'], blob['C']))
            with (out_root / name / run_id / 'summary.json').open('w') as f:
                json.dump(_jsonable(rows[-1]), f, indent=2)
        except Exception:
            traceback.print_exc()
            rows.append(dict(
                dataset=name, stageA='FAIL', L=None, best_cid=None, Lp=None,
                heldout=None, speedup_cons=None, speedup_opt=None,
                gate='blocked/inconclusive', reasons='exception',
                any_pass=False, type3='skip — run failed',
            ))
            per[name] = dict(
                A=dict(pass_=False, checks=[], spectra=dict(
                    rank_raw=None, rank_offset=None, tail_raw_p2=float('nan'),
                    tail_offset_p2=float('nan')), manifest=dict(
                    im_size=None, Nsrc=None, M=None, K=None, coils=None,
                    Nfft_base=None, R=None)),
                B=dict(S_geo=float('nan'), geo_bw=dict(s=None), candidates=[]),
                C=dict(candidates=[], gate_pass=False),
            )
    write_report(run_id, env, rows, out_root, per)
    print('\nDone.')


if __name__ == '__main__':
    os.chdir(REPO)
    main()
