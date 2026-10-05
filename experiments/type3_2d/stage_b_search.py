"""
Stage B: QR controls plus the required subspace-changing exponential search.

The primary experiment spends η = S / S_geo ∈ {1.5, 2, 4, 8} bandwidth by
mixing high-order bases into F,Q (F = R + ΦU, Q = K + CV) and minimizing
sampled residual-exponential rank. Phase-SVD / rotations cannot test that
tradeoff: unconstrained rank-2 SVD already minimizes phase RMS, and
enlarging its budget cannot improve it.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from common import (
    BLOCK_NS, BLOCK_NT, ETA_SWEEP, PRIMARY_EPS, Candidate, balanced_factors,
    bandwidth, bandwidth_S, best_rotation_cleanup, candidate_metrics,
    fold_midpoint_offsets, ho_mix_directions, mix_FQ, pareto_mask,
    rank_report, residual_exp_from_psi, scale_mix_to_eta, separable_from_FQ,
    stratified_indices,
)


def _train_block(data, seed=0):
    I = stratified_indices(data.n_src, min(BLOCK_NS, data.n_src),
                           data.energy_src, data.R, seed=seed)
    J = stratified_indices(data.n_tgt, min(BLOCK_NT, data.n_tgt),
                           data.energy_tgt, data.Ktraj, seed=seed + 3)
    Psi = data.B[I] @ data.D[J].mT
    Ival = stratified_indices(data.n_src, min(BLOCK_NS, data.n_src),
                              data.energy_src, data.R, seed=seed + 11)
    Jval = stratified_indices(data.n_tgt, min(BLOCK_NT, data.n_tgt),
                              data.energy_tgt, data.Ktraj, seed=seed + 13)
    Psiv = data.B[Ival] @ data.D[Jval].mT
    return dict(I=I, J=J, Psi=Psi, Ival=Ival, Jval=Jval, Psiv=Psiv)


def _block_L(cand, Psi, I, J, eps=PRIMARY_EPS):
    E = residual_exp_from_psi(Psi, cand, I, J)
    svals = torch.linalg.svdvals(E)
    return rank_report(svals, eps), svals


def _materialize(cid, family, F, Q, a, c, core_c, n, m, S_geo, T=None, notes='', p=2):
    Fc, Qc, a2, c2 = fold_midpoint_offsets(F, Q, a, c)
    cand = Candidate(cid, family, p, Fc.detach(), Qc.detach(),
                     a2.detach(), c2.detach(),
                     None if T is None else T.detach().cpu().numpy(), notes)
    cand.metrics = candidate_metrics(cand, core_c, n, m, S_geo)
    return cand


def _make_cand(cid, family, F, Q, data, core_c, S_geo, notes='', eta=None):
    a, c = separable_from_FQ(data.B, data.D, F, Q)
    Fc, Qc, a2, c2 = fold_midpoint_offsets(F, Q, a, c)
    cand = Candidate(cid, family, 2, Fc.detach(), Qc.detach(),
                     a2.detach(), c2.detach(), None, notes)
    cand.metrics = candidate_metrics(cand, core_c, data.n_src, data.n_tgt, S_geo)
    if eta is not None:
        cand.metrics['eta'] = float(eta)
        cand.metrics['feasible'] = cand.metrics['S'] <= eta * S_geo * 1.05
    return cand


def _refine_mix(data, U, V, l, Smax, blk, steps=40, outer=3):
    U = U.detach().clone().requires_grad_(True)
    V = V.detach().clone().requires_grad_(True)
    I, J, Psi = blk['I'], blk['J'], blk['Psi']
    R, Phi, K, C = data.R, data.Phi, data.Ktraj, data.C
    last = (U.detach(), V.detach())
    for _ in range(outer):
        with torch.no_grad():
            F, Q = mix_FQ(R, Phi, K, C, U.detach(), V.detach())
            a, c = separable_from_FQ(data.B, data.D, F, Q)
            tmp = Candidate('tmp', 'mix', 2, F, Q, a, c)
            E = residual_exp_from_psi(Psi, tmp, I, J)
            UU, ss, Vh = torch.linalg.svd(E, full_matrices=False)
            Luse = max(1, min(int(l), int(ss.numel())))
            sc = ss[:Luse].clamp(min=0).sqrt().to(E.dtype)
            target = (UU[:, :Luse] * sc) @ (Vh[:Luse, :].mH * sc).mH
        opt = torch.optim.Adam([U, V], lr=0.04)
        for _s in range(steps):
            opt.zero_grad()
            F, Q = mix_FQ(R, Phi, K, C, U, V)
            ph = Psi - F[I] @ Q[J].mT
            E = torch.exp(-2j * math.pi * ph.float())
            match = (E - target).norm() ** 2 / max(E.numel(), 1)
            S = bandwidth_S(F, Q)
            pen = torch.relu(S / max(float(Smax), 1.0) - 1.0).square()
            obj = match + 8.0 * pen
            if not torch.isfinite(obj):
                break
            obj.backward()
            opt.step()
            last = (U.detach(), V.detach())
    return last


def stage_b(data, cores, device) -> dict:
    print('\n=== Stage B — QR controls + exponential residual search ===')
    n, m = data.n_src, data.n_tgt
    core_r, core_c = cores['raw'], cores['offset']
    geo_bw = bandwidth(data.R, data.Ktraj)
    S_geo = geo_bw['S']
    print(f'  S_geometry={S_geo:.4g}  s={geo_bw["s"]}')
    blk = _train_block(data, seed=0)

    z_n = torch.zeros(n, device=device, dtype=data.B.dtype)
    z_m = torch.zeros(m, device=device, dtype=data.D.dtype)
    a_c, c_c = core_c.a, core_c.c
    coverage = []
    cands: list[Candidate] = []

    geo = _materialize('geometry', 'geometry', data.R, data.Ktraj,
                       z_n, z_m, core_c, n, m, S_geo,
                       notes='F=R, Q=K; residual is HO phase')
    cands.append(geo)
    F_raw, Q_raw = balanced_factors(core_r, 2)
    cands.append(_materialize('svd_raw_p2', 'svd_raw', F_raw, Q_raw,
                              z_n, z_m, core_c, n, m, S_geo,
                              notes='unconstrained rank-2 SVD of raw Ψ'))
    F_off, Q_off = balanced_factors(core_c, 2)
    cands.append(_materialize('svd_offset_p2', 'svd_offset', F_off, Q_off,
                              a_c, c_c, core_c, n, m, S_geo,
                              notes='offset-aware rank-2 SVD'))
    F1, Q1 = balanced_factors(core_c, 1)
    F1p = torch.cat([F1, torch.zeros_like(F1)], dim=1)
    Q1p = torch.cat([Q1, torch.zeros_like(Q1)], dim=1)
    cands.append(_materialize('svd_offset_p1', 'svd_offset', F1p, Q1p,
                              a_c, c_c, core_c, n, m, S_geo,
                              notes='rank-1 diagnostic', p=1))
    Ft, Qt, Tbest, Scl = best_rotation_cleanup(F_off, Q_off)
    cands.append(_materialize('svd_offset_Tbest', 'T_search', Ft, Qt,
                              a_c, c_c, core_c, n, m, S_geo, T=Tbest,
                              notes=f'rotation cleanup S={Scl:.4g}'))

    geo_rep, _ = _block_L(geo, blk['Psi'], blk['I'], blk['J'])
    L_base = max(int(geo_rep['L']), 1)
    geo.metrics['train_L'] = L_base
    print(f'  geometry train-block L(ε=1e-3)={L_base}  tail={geo_rep["tail"]:.3e}')
    trial_ranks = sorted({max(1, int(round(f * L_base))) for f in (0.25, 0.5, 0.75, 1.0)})
    print(f'  trial ranks relative to L={L_base}: {trial_ranks}')

    for c in cands:
        if c.p != 2:
            continue
        rep, _ = _block_L(c, blk['Psi'], blk['I'], blk['J'])
        c.metrics['train_L'] = rep['L']
        c.metrics['train_tail'] = rep['tail']
        coverage.append(dict(cid=c.cid, family=c.family, stage='control',
                             S_over_Sgeo=c.metrics['S_over_Sgeo'],
                             train_L=rep['L'], train_tail=rep['tail']))

    dirs = ho_mix_directions(data.Phi, data.C, seed=0)
    modes = ('spat', 'temp', 'joint')
    mix_pool = []
    for eta in ETA_SWEEP:
        if eta < 1.05:
            continue
        for mode in modes:
            for dname, U0, V0 in dirs:
                alpha, F, Q, U, V, _Sach = scale_mix_to_eta(
                    data.R, data.Phi, data.Ktraj, data.C, U0, V0, eta, S_geo, mode)
                F, Q, _Tcl, _Scl = best_rotation_cleanup(F, Q)
                cid = f'mix_{mode}_{dname}_e{eta:g}'
                cand = _make_cand(cid, 'mix', F, Q, data, core_c, S_geo,
                                  notes=f'mode={mode} dir={dname} α={alpha:.3g} η={eta:g}',
                                  eta=eta)
                cand.metrics['alpha'] = alpha
                cand.metrics['mode'] = mode
                cand.metrics['dir'] = dname
                cand.U, cand.V = U.detach(), V.detach()
                mix_pool.append(cand)

    print(f'  mix starts: {len(mix_pool)}')
    for cand in mix_pool:
        rep, _ = _block_L(cand, blk['Psi'], blk['I'], blk['J'])
        cand.metrics['train_L'] = rep['L']
        cand.metrics['train_tail'] = rep['tail']
        cands.append(cand)
        coverage.append(dict(
            cid=cand.cid, family='mix_start', mode=cand.metrics['mode'],
            dir=cand.metrics['dir'], eta=cand.metrics['eta'],
            alpha=cand.metrics['alpha'], S_over_Sgeo=cand.metrics['S_over_Sgeo'],
            feasible=cand.metrics.get('feasible'), train_L=rep['L'],
            train_tail=rep['tail']))

    for eta in ETA_SWEEP:
        if eta < 1.05:
            continue
        band = [c for c in mix_pool if c.metrics.get('eta') == eta]
        if not band:
            continue
        band = sorted(band, key=lambda c: (c.metrics.get('train_L', 10**9),
                                           c.metrics.get('train_tail', 1.0)))
        seed_c = band[0]
        print(f'  η={eta:g} best start {seed_c.cid}  '
              f'S/Sgeo={seed_c.metrics["S_over_Sgeo"]:.3f}  '
              f'train L={seed_c.metrics["train_L"]}')
        U0, V0 = seed_c.U, seed_c.V
        Smax = eta * S_geo
        for l in trial_ranks:
            print(f'    refine l={l} ...', flush=True)
            Ur, Vr = _refine_mix(data, U0, V0, l, Smax, blk)
            F, Q = mix_FQ(data.R, data.Phi, data.Ktraj, data.C, Ur, Vr)
            F, Q, _Tcl, _Scl = best_rotation_cleanup(F, Q)
            cid = f'refine_e{eta:g}_l{l}'
            cand = _make_cand(cid, 'refine', F, Q, data, core_c, S_geo,
                              notes=f'refine {seed_c.cid} trial_rank={l} η={eta:g}',
                              eta=eta)
            cand.metrics['trial_rank'] = l
            cand.metrics['init'] = seed_c.cid
            rep, _ = _block_L(cand, blk['Psi'], blk['I'], blk['J'])
            cand.metrics['train_L'] = rep['L']
            cand.metrics['train_tail'] = rep['tail']
            vrep, _ = _block_L(cand, blk['Psiv'], blk['Ival'], blk['Jval'])
            cand.metrics['val_L'] = vrep['L']
            cand.metrics['val_tail'] = vrep['tail']
            cands.append(cand)
            coverage.append(dict(
                cid=cid, family='refine', eta=eta, trial_rank=l,
                S_over_Sgeo=cand.metrics['S_over_Sgeo'],
                feasible=cand.metrics.get('feasible'),
                train_L=rep['L'], val_L=vrep['L'],
                train_tail=rep['tail'], val_tail=vrep['tail']))
            print(f'      S/Sgeo={cand.metrics["S_over_Sgeo"]:.3f}  '
                  f'feas={cand.metrics.get("feasible")}  '
                  f'train L={rep["L"]}  val L={vrep["L"]}')

    p2 = [c for c in cands if c.p == 2]
    Ls = np.array([c.metrics.get('train_L', 10**6) for c in p2], dtype=float)
    Ss = np.array([c.metrics['S_over_Sgeo'] for c in p2])
    nd = pareto_mask(Ls, Ss)
    for c, flag in zip(p2, nd):
        c.metrics['pareto_residual'] = bool(flag)
    print(f'  {len(cands)} candidates, {int(nd.sum())} on residual-L vs S/Sgeo Pareto')
    for c in cands:
        if c.p != 2 or c.family not in ('geometry', 'svd_raw', 'svd_offset',
                                         'T_search', 'refine'):
            continue
        mtr = c.metrics
        print(f'    {c.cid:32s}  rms={mtr["phase_rms_cycles"]:.3e}  '
              f'S/Sgeo={mtr["S_over_Sgeo"]:.3f}  trainL={mtr.get("train_L", "-")}')

    must_ids = {'geometry', 'svd_offset_p2', 'svd_raw_p2'}
    short = [c for c in cands if c.cid in must_ids]
    for eta in ETA_SWEEP:
        if eta < 1.05:
            band = []
        else:
            band = [c for c in p2 if c.family in ('mix', 'refine')
                    and c.metrics.get('eta') == eta
                    and c.metrics.get('feasible', True)]
        if not band:
            continue
        best = min(band, key=lambda c: (c.metrics.get('train_L', 10**9),
                                        c.metrics.get('train_tail', 1),
                                        c.metrics['S']))
        if best.cid not in {c.cid for c in short}:
            short.append(best)
    for c in p2:
        if (c.metrics.get('pareto_residual') and c.metrics['S_over_Sgeo'] > 1.2
                and c.cid not in {x.cid for x in short} and c.family == 'refine'):
            short.append(c)
    print(f'  shortlist ({len(short)}): {[c.cid for c in short]}')
    print(f'  shortlist S/Sgeo: '
          f'{[round(c.metrics["S_over_Sgeo"], 2) for c in short]}')
    return dict(candidates=cands, shortlist=short, S_geo=S_geo,
                geo_bw=geo_bw, t_search=[], coverage=coverage,
                L_base=L_base, trial_ranks=trial_ranks)
