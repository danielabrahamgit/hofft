#!/usr/bin/env python
"""
Follow-up to math_docs/t3n_followup.md.

Experiment A: trustworthy dense-block SVD ranks and factor-extension errors.
Experiment B: optimize joint / one-sided / general mixing against the
              residual-exponential SVD tail (not scale-to-bandwidth scoring).
Experiment C: freeze shortlisted coordinates; validate on held-out blocks.
Cost screen only after matched held-out accuracy.

Does not implement a type-3 adapter. Writes a new report; does not overwrite
experiments/type3_2d/REPORT.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'experiments' / 'type3_2d'))
from common import (  # noqa: E402
    BETA_CONS, BETA_OPT, BLOCK_NS, BLOCK_NT, C0, DATASETS, EPSILONS,
    GAMMA_CONS, GAMMA_OPT, PRIMARY_EPS, REL_QR_TOL, W_CONS, W_OPT,
    Candidate, bandwidth, bandwidth_S, best_rotation_cleanup, dense_phase,
    fold_midpoint_offsets, gpu_mem_GiB, ho_mix_directions, load_phase_data,
    memory_ok, mix_FQ, phase_fro_from_core, predict_cost, qr_core_svd,
    rank_epsilon, rank_report, residual_phase_block, scale_mix_to_eta,
    separable_from_FQ, stratified_indices, sv_tail_rel,
)

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 140,
    'savefig.bbox': 'tight',
})

ETA_MAX = (2.0, 4.0)
OPT_NS, OPT_NT = 256, 512
EVAL_NS, EVAL_NT = 1024, 2048
GROW_SIZES = ((512, 1024), (1024, 2048))
MAX_UPDATES = 40
AUDIT_SEEDS = (0, 1)
ALS_ITERS = 8
EXT_CAP = 48
PRIOR_RUN = '20260910T071914Z'
LR = 0.025
INNER = 10


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True).strip()
    except Exception:
        return 'unknown'


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def _env(device) -> dict:
    info = dict(
        torch=torch.__version__, cuda=torch.version.cuda, device=str(device),
        git=_git_hash(), python=sys.version.split()[0],
        prior_run=PRIOR_RUN,
    )
    try:
        import cufinufft
        info['cufinufft'] = getattr(cufinufft, '__version__', 'unknown')
    except Exception as e:
        info['cufinufft'] = f'unavailable: {e}'
    if device.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(0)
        info['gpu_mem_GiB'] = gpu_mem_GiB(device)
    tracked = [
        REPO / 'experiments' / 'type3_2d' / 'common.py',
        REPO / 'experiments' / 'type3_2d' / 'run.py',
        REPO / 'experiments' / 'type3_2d' / 'stage_b_search.py',
        HERE / 'run.py',
        REPO / 'math_docs' / 't3n_followup.md',
    ]
    info['file_hashes'] = {str(p.relative_to(REPO)): _file_hash(p)
                           for p in tracked if p.exists()}
    return info


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


def _dump(path: Path, obj):
    path.write_text(json.dumps(_jsonable(obj), indent=2))


# ---------------------------------------------------------------------------
# Ordinary-transpose SVD factors  E ≈ Z T^T
# ---------------------------------------------------------------------------
def zt_from_svd(U, s, Vh, l: int):
    """Z = U √Σ, T = conj(V) √Σ = (V^H)^T √Σ so that Z T^T = U Σ V^H."""
    l = max(1, min(int(l), int(s.numel()), U.shape[1], Vh.shape[0]))
    sc = s[:l].clamp(min=0).sqrt().to(U.dtype)
    Z = U[:, :l] * sc
    T = Vh[:l, :].mT * sc
    return Z, T


def rel_fro(A, B) -> float:
    return float((A - B).norm() / A.norm().clamp(min=1e-30))


def svd_tail_energy(s, l: int) -> torch.Tensor:
    if l >= s.numel():
        return s.sum() * 0
    return s[l:].square().sum()


def kernel_from_phase(ph, dtype=torch.complex128):
    return torch.exp((-2j * math.pi) * ph).to(dtype)


def geometry_candidate(data) -> Candidate:
    n, m = data.n_src, data.n_tgt
    z_n = torch.zeros(n, device=data.B.device, dtype=data.B.dtype)
    z_m = torch.zeros(m, device=data.B.device, dtype=data.D.dtype)
    return Candidate('geometry', 'geometry', 2, data.R, data.Ktraj, z_n, z_m,
                     notes='F=R, Q=K')


def residual_phase(data, cand, I, J, Psi=None):
    if Psi is None:
        Psi = data.B[I] @ data.D[J].mT
    return Psi - cand.a[I][:, None] - cand.c[J][None, :] - cand.F[I] @ cand.Q[J].mT


def make_cand(cid, family, F, Q, data, notes='', p=2):
    a, c = separable_from_FQ(data.B, data.D, F, Q)
    Fc, Qc, a2, c2 = fold_midpoint_offsets(F, Q, a, c)
    return Candidate(cid, family, p, Fc.detach(), Qc.detach(),
                     a2.detach(), c2.detach(), notes=notes)


def pick_indices(data, n_s, n_t, seed, kind='train'):
    off = {'train': 0, 'val': 101, 'test': 202}[kind]
    s = 10007 * int(seed) + off
    I = _strat_cpu(data.n_src, min(n_s, data.n_src), data.energy_src, data.R, s)
    J = _strat_cpu(data.n_tgt, min(n_t, data.n_tgt), data.energy_tgt, data.Ktraj, s + 7)
    return I, J


def _strat_cpu(n, n_pick, energy, coords, seed):
    """Stratified sample that covers extrema and energy without collapsing
    to the lowest index set (torch.unique sorts)."""
    n_pick = min(n_pick, n)
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    n_rand = max(n_pick // 2, 1)
    n_hi = max(n_pick // 4, 0)
    energy_cpu = energy.detach().cpu()
    coords_cpu = coords[:, :2].detach().cpu()
    rand = torch.randperm(n, generator=g)[:n_rand]
    k_hi = min(max(n_hi * 8, n_hi), n)
    hi_pool = torch.topk(energy_cpu, k_hi).indices
    hi = hi_pool[torch.randperm(hi_pool.numel(), generator=g)[:n_hi]]
    lo, hi_c = coords_cpu.min(0).values, coords_cpu.max(0).values
    corners = torch.stack([
        torch.stack([lo[0], lo[1]]), torch.stack([lo[0], hi_c[1]]),
        torch.stack([hi_c[0], lo[1]]), torch.stack([hi_c[0], hi_c[1]]),
    ])
    d2 = ((coords_cpu[:, None, :] - corners[None, :, :]) ** 2).sum(-1)
    corner_idx = d2.argmin(dim=0)
    axis_idx = torch.stack([coords_cpu[:, 0].argmin(), coords_cpu[:, 0].argmax(),
                            coords_cpu[:, 1].argmin(), coords_cpu[:, 1].argmax()])
    extra = torch.randperm(n, generator=g)
    seq = torch.cat([corner_idx, axis_idx, hi, rand, extra])
    seen = torch.zeros(n, dtype=torch.bool)
    out = []
    for i in seq.tolist():
        i = int(i)
        if not seen[i]:
            seen[i] = True
            out.append(i)
            if len(out) >= n_pick:
                break
    return torch.tensor(out, device=energy.device, dtype=torch.long)


def disjoint_indices(data, n_s, n_t, seed, kind, avoid_I, avoid_J):
    I, J = pick_indices(data, n_s + avoid_I.numel(), n_t + avoid_J.numel(),
                        seed, kind)
    I = I[~torch.isin(I, avoid_I)][:n_s]
    J = J[~torch.isin(J, avoid_J)][:n_t]
    if I.numel() < min(n_s, data.n_src // 4):
        extra = torch.randperm(data.n_src, device=data.B.device)
        extra = extra[~torch.isin(extra, avoid_I)]
        I = extra[:n_s]
    if J.numel() < min(n_t, data.n_tgt // 4):
        extra = torch.randperm(data.n_tgt, device=data.B.device)
        extra = extra[~torch.isin(extra, avoid_J)]
        J = extra[:n_t]
    return I, J


# ---------------------------------------------------------------------------
# Audit: preprocessing / prior-run meaning (runtime measurements)
# ---------------------------------------------------------------------------
def audit_preprocessing(name, device, data) -> dict:
    fpath = REPO / 'data' / name
    kw = {'weights_only': True, 'map_location': 'cpu'}
    phis = torch.load(fpath / 'phis.pt', **kw).float()
    alphas = torch.load(fpath / 'alphas.pt', **kw).float()
    evals = torch.load(fpath / 'evals.pt', **kw).float()
    from hofft.phase_coeffs import remove_linear_terms
    mask = (evals > 0.9).float()
    phis2, trj_term, zeroth = remove_linear_terms(phis, alphas, mask=mask)
    Bho = phis2.shape[0]
    energy = (phis2.reshape((Bho, -1)).abs().mean(dim=1)
              * alphas.reshape((Bho, -1)).abs().mean(dim=1))
    keep = energy > 1e-6
    return dict(
        alias_requested='tilted_spi_invivo' if name == 'tilt_spi_invivo' else name,
        alias_resolved=name,
        n_high_order_raw=int(phis.shape[0]),
        n_high_order_kept=int(keep.sum()),
        n_high_order_loader=int(data.n_high_order),
        dropped_energies=[float(e) for e in energy[~keep].cpu()],
        kept_energies=[float(e) for e in energy[keep].cpu()],
        energy_thresh=1e-6,
        mask_thresh=0.9,
        R=int(data.R_undersample),
        trj_term_rms=float(trj_term.float().pow(2).mean().sqrt()),
        zeroth_rms=float(zeroth.float().pow(2).mean().sqrt()),
        remove_linear_terms='masked LS of each spatial basis onto {1,x,y}; '
                            'subtract from phis; fold (coeff α) into trajectory '
                            '(trj_term) and a discarded zeroth-order offset. '
                            'Weights are the ESPIRiT 0/1 mask.',
    )


# ---------------------------------------------------------------------------
# Experiment A
# ---------------------------------------------------------------------------
def als_fit(E, l, iters=ALS_ITERS, init='svd'):
    n, m = E.shape
    l = max(1, min(l, min(n, m)))
    if init == 'svd':
        U, s, Vh = torch.linalg.svd(E, full_matrices=False)
        Z, T = zt_from_svd(U, s, Vh, l)
    else:
        g = torch.Generator(device=E.device).manual_seed(0)
        Z = torch.randn(n, l, generator=g, device=E.device, dtype=E.dtype)
        T = torch.randn(m, l, generator=g, device=E.device, dtype=E.dtype)
    for _ in range(iters):
        T = torch.linalg.lstsq(Z, E).solution.mT
        Z = torch.linalg.lstsq(T, E.mT).solution.mT
    return Z, T


def extend_new_rows(E_Iho_J, T):
    return torch.linalg.lstsq(T, E_Iho_J.mT).solution.mT


def extend_new_cols(E_I_Jho, Z):
    return torch.linalg.lstsq(Z, E_I_Jho).solution.mT


def four_errors(E_tr, Z, T, E_val, Ival_on_train_cols, Jval_on_train_rows,
                E_Iho_Jtr, E_Itr_Jho, E_test):
    """Train recon, val (mixed), and two-sided test using LS-extended factors."""
    rec_tr = Z @ T.mT
    train = rel_fro(E_tr, rec_tr)
    Z_ho = extend_new_rows(E_Iho_Jtr, T)
    T_ho = extend_new_cols(E_Itr_Jho, Z)
    test = rel_fro(E_test, Z_ho @ T_ho.mT)
    # validation: independent block, extend from train factors via the
    # overlapping train columns/rows if provided; else two-sided from train.
    val = test
    return dict(train=train, val=val, test=test,
                row_only=rel_fro(E_Iho_Jtr, Z_ho @ T.mT),
                col_only=rel_fro(E_Itr_Jho, Z @ T_ho.mT))


def _rank_grid(L, cap):
    L = max(1, int(L))
    cap = max(1, int(cap))
    return sorted({max(1, L // 2), L, min(cap, L + 4)})


def extension_on_grid(U, s, Vh, E_Iho_J, E_I_Jho, E_test, ranks, eps=PRIMARY_EPS):
    met_L, met_err, best_err = None, None, None
    for l in ranks:
        l = max(1, min(int(l), int(s.numel())))
        Z, T = zt_from_svd(U, s, Vh, l)
        err = rel_fro(E_test, extend_new_rows(E_Iho_J, T) @ extend_new_cols(E_I_Jho, Z).mT)
        if best_err is None or err < best_err:
            best_err = err
        if err <= eps and (met_L is None or l < met_L):
            met_L, met_err = l, err
    return met_L, met_err, best_err


def experiment_a(data, device) -> dict:
    print('\n=== Experiment A — trustworthy baseline rank ===')
    geo = geometry_candidate(data)
    rows = []
    dense_ok = True
    zt_ok = True
    geo_agree_ok = True

    for seed in AUDIT_SEEDS:
        I, J = pick_indices(data, EVAL_NS, EVAL_NT, seed, 'train')
        Ival, Jval = disjoint_indices(data, EVAL_NS // 2, EVAL_NT // 2, seed,
                                      'val', I, J)
        Itest, Jtest = disjoint_indices(data, EVAL_NS // 2, EVAL_NT // 2, seed,
                                        'test', torch.cat([I, Ival]),
                                        torch.cat([J, Jval]))
        Phi_I, C_J = data.Phi[I], data.C[J]
        H = Phi_I @ C_J.mT
        Rph = residual_phase(data, geo, I, J)
        dH = float((H - Rph).abs().max().cpu())
        E0 = kernel_from_phase(H, torch.complex128)
        ER = kernel_from_phase(Rph, torch.complex128)
        dE = float((E0 - ER).abs().max().cpu())
        geo_agree_ok = geo_agree_ok and dH < 1e-10 and dE < 1e-10
        print(f'  seed={seed} |I|={I.numel()} |J|={J.numel()}  '
              f'max|H-Rgeo|={dH:.3e}  max|E0-ER|={dE:.3e}')

        U, s, Vh = torch.linalg.svd(E0, full_matrices=False)
        # Reconstruction at reported L vs tail formula.
        for eps in EPSILONS:
            L = rank_epsilon(s, eps)
            Z, T = zt_from_svd(U, s, Vh, L)
            rec = Z @ T.mT
            meas = rel_fro(E0, rec)
            tail = sv_tail_rel(s, L)
            # unit-modulus: ||E||_F = sqrt(|I||J|); zero approx error is 1
            nrm = float(E0.norm().cpu())
            unit = math.sqrt(I.numel() * J.numel())
            ok_tail = abs(meas - tail) / max(tail, 1e-15) < 0.05 or meas < 1e-12
            dense_ok = dense_ok and ok_tail
            rows.append(dict(
                seed=seed, method='dense_svd_block', eps=eps, L=L,
                cap=int(s.numel()), target_met=bool(tail <= eps + 1e-15),
                exact_tail=tail, fitted_train=meas, block_ns=int(I.numel()),
                block_nt=int(J.numel()), unit_norm_ratio=nrm / unit,
                tail_vs_recon_ok=ok_tail, dH=dH, dE=dE,
            ))
            print(f'    ε={eps:.0e}  L_block={L}  tail={tail:.3e}  '
                  f'recon={meas:.3e}  ||E||/√NM={nrm / unit:.6f}')

        # Factor convention check at L corresponding to 1e-3.
        Lpri = rank_epsilon(s, PRIMARY_EPS)
        Z, T = zt_from_svd(U, s, Vh, Lpri)
        rec = Z @ T.mT
        # Wrong convention: T ← V √Σ  (Vh^H √Σ) yields Z T^H rather than Z T^T
        Twrong = Vh[:Lpri, :].mH * s[:Lpri].clamp(min=0).sqrt().to(U.dtype)
        rec_wrong_TH = Z @ Twrong.mH
        rec_wrong_TT = Z @ Twrong.mT
        e_right = rel_fro(E0, rec)
        e_wh = rel_fro(E0, rec_wrong_TH)
        e_wt = rel_fro(E0, rec_wrong_TT)
        print(f'    ZT^T rel={e_right:.3e}  wrong V√Σ as T^H={e_wh:.3e}  '
              f'wrong V√Σ as T^T={e_wt:.3e}')
        zt_ok = zt_ok and e_right <= e_wh + 1e-12

        # SVD factors at a few ranks vs two-sided LS extension (no ALS, no 1..cap scan).
        E64 = E0.to(torch.complex64)
        U64, s64, Vh64 = torch.linalg.svd(E64, full_matrices=False)
        E_Iho_J = kernel_from_phase(data.Phi[Ival] @ data.C[J].mT, torch.complex64)
        E_I_Jho = kernel_from_phase(data.Phi[I] @ data.C[Jval].mT, torch.complex64)
        E_val = kernel_from_phase(data.Phi[Ival] @ data.C[Jval].mT, torch.complex64)
        s_val = torch.linalg.svdvals(E_val)
        s_testb = torch.linalg.svdvals(
            kernel_from_phase(data.Phi[Itest] @ data.C[Jtest].mT, torch.complex64))
        for ltry in _rank_grid(Lpri, min(EXT_CAP, s64.numel())):
            Zs, Ts = zt_from_svd(U64, s64, Vh64, ltry)
            ext_s = four_errors(E64, Zs, Ts, E_val, None, None,
                                E_Iho_J, E_I_Jho, E_val)
            rows.append(dict(
                seed=seed, method='svd_factors', l=ltry,
                exact_tail=sv_tail_rel(s64, ltry),
                fitted_train=ext_s['train'], fitted_val_row=ext_s['row_only'],
                fitted_val_col=ext_s['col_only'], fitted_test=ext_s['test'],
                val_intrinsic_L=rank_epsilon(s_val, PRIMARY_EPS),
                test_intrinsic_L=rank_epsilon(s_testb, PRIMARY_EPS),
                extension='LS ordinary-transpose ZT^T',
            ))
            print(f'    l={ltry:3d}  SVD train={ext_s["train"]:.3e}  '
                  f'row={ext_s["row_only"]:.3e}  col={ext_s["col_only"]:.3e}  '
                  f'test={ext_s["test"]:.3e}  '
                  f'valL={rank_epsilon(s_val, PRIMARY_EPS)}  '
                  f'testL={rank_epsilon(s_testb, PRIMARY_EPS)}')

        E_Iho_Jt = kernel_from_phase(data.Phi[Itest] @ data.C[J].mT, torch.complex64)
        E_I_Jht = kernel_from_phase(data.Phi[I] @ data.C[Jtest].mT, torch.complex64)
        E_test = kernel_from_phase(data.Phi[Itest] @ data.C[Jtest].mT, torch.complex64)
        grid = _rank_grid(Lpri, min(EXT_CAP, s64.numel()))
        met_L, met_err, best_err = extension_on_grid(
            U64, s64, Vh64, E_Iho_Jt, E_I_Jht, E_test, grid)
        label = str(met_L) if met_L is not None else f'unresolved at grid={grid}'
        print(f'    validated extension rank @ ε=1e-3: {label}  '
              f'(best test err {best_err:.3e})')
        rows.append(dict(
            seed=seed, method='extension_scan', L_validated=met_L,
            cap=EXT_CAP, best_test_err=best_err, L_label=label,
        ))

    # Sample growth on seed 0.
    growth = []
    for ns, nt in GROW_SIZES:
        if ns > data.n_src or nt > data.n_tgt:
            continue
        I, J = pick_indices(data, ns, nt, 0, 'train')
        E0 = kernel_from_phase(data.Phi[I] @ data.C[J].mT, torch.complex64)
        s = torch.linalg.svdvals(E0)
        rep = {f'{e:.0e}': rank_epsilon(s, e) for e in EPSILONS}
        tails = {f'{e:.0e}': sv_tail_rel(s, rank_epsilon(s, e)) for e in EPSILONS}
        growth.append(dict(ns=int(I.numel()), nt=int(J.numel()), L=rep,
                           tail=tails))
        print(f'  grow {I.numel()}x{J.numel()}  L={rep}')

    Ls = [g['L']['1e-03'] for g in growth]
    stable = (max(Ls) - min(Ls) <= max(2.0, 0.10 * float(np.median(Ls)))) if Ls else False
    print(f'  sample-growth L(ε=1e-3)={Ls}  stable={stable}')
    print(f'  dense SVD controls: '
          f'{"PASS" if dense_ok and geo_agree_ok and zt_ok else "FAIL"}')
    return dict(
        rows=rows, growth=growth, L_growth=Ls, growth_stable=stable,
        dense_ok=dense_ok, geo_agree_ok=geo_agree_ok, zt_ok=zt_ok,
        dense_controls_pass=bool(dense_ok and geo_agree_ok and zt_ok),
        L_block_eval=next(
            (r['L'] for r in rows
             if r.get('method') == 'dense_svd_block' and r.get('seed') == 0
             and r.get('eps') == PRIMARY_EPS),
            growth[-1]['L']['1e-03'] if growth else 2),
    )


# ---------------------------------------------------------------------------
# Experiment B — SVD-tail optimization
# ---------------------------------------------------------------------------
def _synth_kernel(R, Phi, Ktr, C, U, V, Psi):
    F = R + Phi @ U
    Q = Ktr + C @ V
    ph = Psi - F @ Q.mT
    return torch.exp((-2j * math.pi) * ph).to(torch.complex64)


def grad_check(device) -> dict:
    """FD check of the *inner* objective ||E - target||^2 with target frozen.

    Direct autodiff of complex SVD tails failed this check (rel ~ 1); the
    search therefore alternates an exact truncated SVD with coordinate
    updates against that frozen reconstruction, as allowed by the protocol.
    """
    torch.manual_seed(0)
    N, M, K = 18, 24, 3
    R = torch.randn(N, 2, device=device, dtype=torch.float64)
    Ktr = torch.randn(M, 2, device=device, dtype=torch.float64)
    Phi = torch.randn(N, K, device=device, dtype=torch.float64)
    C = torch.randn(M, K, device=device, dtype=torch.float64)
    U0 = 0.05 * torch.randn(K, 2, device=device, dtype=torch.float64)
    V0 = 0.05 * torch.randn(K, 2, device=device, dtype=torch.float64)
    Psi = (torch.cat([R, Phi], 1) @ torch.cat([Ktr, C], 1).mT)
    with torch.no_grad():
        E0 = _synth_kernel(R, Phi, Ktr, C, U0, V0, Psi)
        Uu, ss, Vh = torch.linalg.svd(E0, full_matrices=False)
        Z, T = zt_from_svd(Uu, ss, Vh, 2)
        target = (Z @ T.mT).detach()

    def loss_of(U, V):
        E = _synth_kernel(R, Phi, Ktr, C, U, V, Psi)
        return (E - target).abs().square().mean()

    U = U0.clone().requires_grad_(True)
    V = V0.clone().requires_grad_(True)
    L = loss_of(U, V)
    L.backward()
    gU = U.grad.detach().clone()
    eps = 1e-6
    fdU = torch.zeros_like(U0)
    for i in range(K):
        for j in range(2):
            d = torch.zeros_like(U0)
            d[i, j] = eps
            fdU[i, j] = (loss_of(U0 + d, V0) - loss_of(U0 - d, V0)) / (2 * eps)
    rel = float((gU - fdU).norm() / fdU.norm().clamp(min=1e-30))
    ok = rel < 0.12 or float(fdU.norm()) < 1e-10
    print(f'  grad-check frozen-target match vs FD: rel={rel:.3e}  '
          f'{"PASS" if ok else "FAIL"}')
    return dict(ok=ok, rel=rel, method='frozen_truncated_svd_target')


def pack_from_UV(U, V, data, family, cid, eta, S_geo, notes='', cleanup=False):
    F, Q = mix_FQ(data.R, data.Phi, data.Ktraj, data.C, U, V)
    T = None
    if cleanup:
        F, Q, T, _Scl = best_rotation_cleanup(F, Q)
    cand = make_cand(cid, family, F, Q, data, notes=notes)
    bw = bandwidth(cand.F, cand.Q)
    cand.metrics = dict(
        **bw, family=family, cid=cid,
        S_over_Sgeo=bw['S'] / max(S_geo, 1e-18),
        eta_max=float(eta),
        feasible=bw['S'] <= eta * S_geo * 1.05,
        cond_T=None if T is None else float(torch.linalg.cond(T).cpu()),
        notes=notes,
    )
    cand.U, cand.V = U.detach(), V.detach()
    return cand


def pack_from_XY(X, Y, QB, QD, data, cid, eta, S_geo, notes='', cleanup=False):
    F, Q = QB @ X, QD @ Y
    if cleanup:
        F, Q, _T, _Scl = best_rotation_cleanup(F, Q)
    cand = make_cand(cid, 'general', F, Q, data, notes=notes)
    bw = bandwidth(cand.F, cand.Q)
    cand.metrics = dict(
        **bw, family='general', cid=cid,
        S_over_Sgeo=bw['S'] / max(S_geo, 1e-18),
        eta_max=float(eta),
        feasible=bw['S'] <= eta * S_geo * 1.05,
        notes=notes,
    )
    cand.X, cand.Y = X.detach(), Y.detach()
    return cand


def optimize_UV(data, U, V, I, J, Psi, l, Smax, family, steps=MAX_UPDATES):
    """Alternating truncated SVD + Adam on ||E - ZT^T||^2, S ≤ Smax."""
    if family == 'spatial':
        U = U.detach().clone().requires_grad_(True)
        V = torch.zeros_like(V.detach())
    elif family == 'temporal':
        U = torch.zeros_like(U.detach())
        V = V.detach().clone().requires_grad_(True)
    else:
        U = U.detach().clone().requires_grad_(True)
        V = V.detach().clone().requires_grad_(True)
    params = [p for p in (U, V) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    Phi_I, C_J = data.Phi[I], data.C[J]
    R_I, K_J = data.R[I], data.Ktraj[J]
    inner = INNER
    outer = max(1, (steps + inner - 1) // inner)
    trace = []
    best = (float('inf'), U.detach().clone(), V.detach().clone())
    stall = 0
    t0 = time.time()
    step = 0

    def current_FQ():
        FI = R_I if family == 'temporal' else R_I + Phi_I @ U
        QJ = K_J if family == 'spatial' else K_J + C_J @ V
        Ffull, Qfull = mix_FQ(data.R, data.Phi, data.Ktraj, data.C, U, V)
        return FI, QJ, Ffull, Qfull

    for _o in range(outer):
        with torch.no_grad():
            FI, QJ, _, _ = current_FQ()
            E = torch.exp((-2j * math.pi) * (Psi - FI @ QJ.mT)).to(torch.complex64)
            Uu, ss, Vh = torch.linalg.svd(E, full_matrices=False)
            Z, T = zt_from_svd(Uu, ss, Vh, l)
            target = (Z @ T.mT).detach()
            tail = float((svd_tail_energy(ss, l) / E.numel()).cpu())
            if tail + 1e-16 < best[0]:
                best = (tail, U.detach().clone(), V.detach().clone())
                stall = 0
            else:
                stall += 1
                if stall >= 2 and step > 20:
                    break
        for _i in range(inner):
            opt.zero_grad()
            FI, QJ, Ffull, Qfull = current_FQ()
            E = torch.exp((-2j * math.pi) * (Psi - FI @ QJ.mT)).to(torch.complex64)
            match = (E - target).abs().square().mean()
            S = bandwidth_S(Ffull, Qfull)
            pen = torch.relu(S / max(float(Smax), 1.0) - 1.0).square()
            obj = match + 4.0 * pen
            if not torch.isfinite(obj):
                break
            obj.backward()
            opt.step()
            mval = float(match.detach().cpu())
            trace.append(dict(step=step, loss=tail, match=mval,
                              S=float(S.detach().cpu())))
            step += 1
    # final measurement of the last iterate
    with torch.no_grad():
        FI, QJ, _, _ = current_FQ()
        E = torch.exp((-2j * math.pi) * (Psi - FI @ QJ.mT)).to(torch.complex64)
        ss = torch.linalg.svdvals(E)
        tail = float((svd_tail_energy(ss, l) / E.numel()).cpu())
        if tail + 1e-16 < best[0]:
            best = (tail, U.detach().clone(), V.detach().clone())
    return best[1], best[2], trace, best[0], time.time() - t0


def optimize_XY(data, X, Y, QB, QD, I, J, Psi, l, Smax, steps=MAX_UPDATES):
    X = X.detach().clone().requires_grad_(True)
    Y = Y.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([X, Y], lr=LR)
    QBI, QDJ = QB[I], QD[J]
    inner = INNER
    outer = max(1, (steps + inner - 1) // inner)
    trace, stall = [], 0
    best = (float('inf'), X.detach().clone(), Y.detach().clone())
    t0 = time.time()
    step = 0
    for _o in range(outer):
        with torch.no_grad():
            E = torch.exp((-2j * math.pi) * (Psi - (QBI @ X) @ (QDJ @ Y).mT)
                          ).to(torch.complex64)
            Uu, ss, Vh = torch.linalg.svd(E, full_matrices=False)
            Z, T = zt_from_svd(Uu, ss, Vh, l)
            target = (Z @ T.mT).detach()
            tail = float((svd_tail_energy(ss, l) / E.numel()).cpu())
            if tail + 1e-16 < best[0]:
                best = (tail, X.detach().clone(), Y.detach().clone())
                stall = 0
            else:
                stall += 1
                if stall >= 2 and step > 20:
                    break
        for _i in range(inner):
            opt.zero_grad()
            FI, QJ = QBI @ X, QDJ @ Y
            Ffull, Qfull = QB @ X, QD @ Y
            E = torch.exp((-2j * math.pi) * (Psi - FI @ QJ.mT)).to(torch.complex64)
            match = (E - target).abs().square().mean()
            S = bandwidth_S(Ffull, Qfull)
            pen = torch.relu(S / max(float(Smax), 1.0) - 1.0).square()
            obj = match + 4.0 * pen
            if not torch.isfinite(obj):
                break
            obj.backward()
            opt.step()
            trace.append(dict(step=step, loss=tail, match=float(match.detach().cpu()),
                              S=float(S.detach().cpu())))
            step += 1
    with torch.no_grad():
        E = torch.exp((-2j * math.pi) * (Psi - (QBI @ X) @ (QDJ @ Y).mT)
                      ).to(torch.complex64)
        ss = torch.linalg.svdvals(E)
        tail = float((svd_tail_energy(ss, l) / E.numel()).cpu())
        if tail + 1e-16 < best[0]:
            best = (tail, X.detach().clone(), Y.detach().clone())
    return best[1], best[2], trace, best[0], time.time() - t0


def block_metrics(data, cand, I, J, Psi, dtype=torch.complex64):
    ph = residual_phase(data, cand, I, J, Psi)
    E = kernel_from_phase(ph, dtype)
    s = torch.linalg.svdvals(E)
    rep = rank_report(s, PRIMARY_EPS)
    return dict(L=rep['L'], tail=rep['tail'], target_met=rep['target_met'],
                svals=s[:min(48, s.numel())].detach().cpu().numpy().tolist())


def experiment_b(data, cores, A, device) -> dict:
    print('\n=== Experiment B — joint mixing SVD-tail search ===')
    gc = grad_check(device)
    S_geo = bandwidth(data.R, data.Ktraj)['S']
    print(f'  S_geo={S_geo:.4g}')
    I, J = pick_indices(data, OPT_NS, OPT_NT, 0, 'train')
    Ival, Jval = disjoint_indices(data, OPT_NS, OPT_NT, 0, 'val', I, J)
    Psi = data.B[I] @ data.D[J].mT
    Psiv = data.B[Ival] @ data.D[Jval].mT
    L_base = int(A['L_block_eval'] or 2)
    trial = sorted({max(1, L_base // 2), L_base})
    print(f'  opt block {I.numel()}x{J.numel()}  L_block={L_base}  '
          f'trial ranks={trial}  max_updates={MAX_UPDATES}')

    core = cores['raw']
    QB, QD = core.QB, core.QD
    geo = geometry_candidate(data)
    F_svd, Q_svd = (core.QB @ (core.U[:, :2] * core.sigma[:2].clamp(min=0).sqrt()),
                    core.QD @ (core.V[:, :2] * core.sigma[:2].clamp(min=0).sqrt()))
    # LS projections of SVD factors onto mix families.
    U_svd = torch.linalg.lstsq(data.Phi, F_svd - data.R).solution
    V_svd = torch.linalg.lstsq(data.C, Q_svd - data.Ktraj).solution
    dirs = ho_mix_directions(data.Phi, data.C, seed=0)

    cands = []
    search_rows = []
    traces = {}

    def score_and_keep(cand, l, eta, family, n_upd, loss0, loss1, elapsed,
                       start_name, seed=0):
        tr = block_metrics(data, cand, I, J, Psi)
        va = block_metrics(data, cand, Ival, Jval, Psiv)
        cand.metrics.update(train_L=tr['L'], train_tail=tr['tail'],
                            val_L=va['L'], val_tail=va['tail'],
                            trial_rank=l, n_updates=n_upd,
                            loss0=loss0, loss1=loss1, elapsed_s=elapsed)
        bw = cand.metrics
        X = cores['offset'].QB.mT @ cand.F
        Y = cores['offset'].QD.mT @ cand.Q
        loss_ph = float(phase_fro_from_core(cores['offset'], X, Y).cpu())
        cand.metrics['phase_rms_cycles'] = math.sqrt(
            loss_ph / max(data.n_src * data.n_tgt, 1))
        cands.append(cand)
        row = dict(
            cid=cand.cid, family=family, trial_rank=l, seed=seed,
            start=start_name, n_updates=n_upd, loss0=loss0, loss1=loss1,
            eta_max=eta, S_over_Sgeo=bw['S_over_Sgeo'], feasible=bw['feasible'],
            phase_rms=cand.metrics['phase_rms_cycles'],
            train_L=tr['L'], train_tail=tr['tail'],
            val_L=va['L'], val_tail=va['tail'],
            constraint_violation=max(0.0, bw['S'] / (eta * S_geo) - 1.0),
            elapsed_s=elapsed,
        )
        search_rows.append(row)
        print(f'    {cand.cid:36s}  S/Sgeo={bw["S_over_Sgeo"]:.3f}  '
              f'feas={bw["feasible"]}  trainL={tr["L"]}  valL={va["L"]}  '
              f'loss {loss0:.3e}->{loss1:.3e}  {elapsed:.1f}s')
        return cand

    # Controls (no opt).
    for cid, fam, F, Q in (
        ('geometry', 'geometry', data.R, data.Ktraj),
        ('svd_raw_p2', 'phase_svd', F_svd, Q_svd),
    ):
        cand = make_cand(cid, fam, F, Q, data)
        bw = bandwidth(cand.F, cand.Q)
        cand.metrics = dict(**bw, family=fam, cid=cid,
                            S_over_Sgeo=bw['S'] / S_geo, eta_max=1.0,
                            feasible=True)
        tr = block_metrics(data, cand, I, J, Psi)
        loss = (tr['tail'] ** 2)  # relative; store tail energy proxy
        score_and_keep(cand, L_base, 1.0, fam, 0, loss, loss, 0.0, 'control')

    # One joint scale-to-eta control per budget (not optimized).
    _, U0, V0 = dirs[0]
    for eta in ETA_MAX:
        _a, F, Q, U, V, _ = scale_mix_to_eta(
            data.R, data.Phi, data.Ktraj, data.C, U0, V0, eta, S_geo, 'joint')
        cand = pack_from_UV(U, V, data, 'joint', f'control_joint_pca_e{eta:g}',
                            eta, S_geo, notes='scale-to-eta control')
        tr = block_metrics(data, cand, I, J, Psi)
        score_and_keep(cand, L_base, eta, 'joint', 0, tr['tail'], tr['tail'],
                       0.0, 'scale_to_eta')

    families_opt = ('spatial', 'temporal', 'joint', 'general')
    n_starts = {f: 0 for f in families_opt}

    for family in families_opt:
        for eta in ETA_MAX:
            Smax = eta * S_geo
            _, Udir, Vdir = dirs[0]
            _a, F, Q, Us, Vs, _ = scale_mix_to_eta(
                data.R, data.Phi, data.Ktraj, data.C, Udir, Vdir, eta, S_geo,
                'spat' if family == 'spatial' else
                'temp' if family == 'temporal' else 'joint')
            if family == 'general':
                Xs = torch.linalg.lstsq(QB, F).solution
                Ys = torch.linalg.lstsq(QD, Q).solution
                start = ('pca_scaled', Xs, Ys)
            else:
                start = ('pca_scaled', Us, Vs)
            for l in trial:
                sname, p1, p2 = start
                cid = f'{family}_e{eta:g}_l{l}_{sname}'
                n_starts[family] += 1
                if family == 'general':
                    X, Y, trc, loss1, elapsed = optimize_XY(
                        data, p1, p2, QB, QD, I, J, Psi, l, Smax)
                    cand = pack_from_XY(X, Y, QB, QD, data, cid, eta, S_geo,
                                        notes=f'opt {sname}')
                else:
                    U, V, trc, loss1, elapsed = optimize_UV(
                        data, p1, p2, I, J, Psi, l, Smax, family)
                    cand = pack_from_UV(U, V, data, family, cid, eta, S_geo,
                                        notes=f'opt {sname}')
                loss0 = trc[0]['loss'] if trc else float('nan')
                traces[cid] = trc
                score_and_keep(cand, l, eta, family, len(trc), loss0, loss1,
                               elapsed, sname)

    print(f'  starts per family: {n_starts}')
    short = []
    seen = set()
    for cid in ('geometry', 'svd_raw_p2'):
        for c in cands:
            if c.cid == cid and c.cid not in seen:
                short.append(c)
                seen.add(c.cid)
    for family in families_opt:
        band = [c for c in cands
                if c.family == family and c.metrics.get('n_updates', 0) > 0]
        if not band:
            continue
        best = min(band, key=lambda c: (c.metrics.get('val_L', 99),
                                        c.metrics.get('val_tail', 1)))
        if best.cid not in seen:
            short.append(best)
            seen.add(best.cid)
    print(f'  shortlist ({len(short)}): {[c.cid for c in short]}')
    print(f'  shortlist S/Sgeo: '
          f'{[round(c.metrics["S_over_Sgeo"], 2) for c in short]}')
    return dict(
        candidates=cands, shortlist=short, search_rows=search_rows,
        traces=traces, S_geo=S_geo, trial_ranks=trial, L_base=L_base,
        n_starts=n_starts, grad_check=gc, I=I, J=J, Ival=Ival, Jval=Jval,
        opt_ns=int(I.numel()), opt_nt=int(J.numel()),
    )


# ---------------------------------------------------------------------------
# Experiment C + cost
# ---------------------------------------------------------------------------
def experiment_c(data, B, env) -> dict:
    print('\n=== Experiment C — frozen-coordinate validation ===')
    results = []
    S_geo = B['S_geo']
    for cand in B['shortlist']:
        per_seed = []
        for seed in AUDIT_SEEDS:
            I, J = pick_indices(data, EVAL_NS, EVAL_NT, seed, 'train')
            Ite, Jte = disjoint_indices(
                data, min(EVAL_NS // 2, 512), min(EVAL_NT // 2, 1024),
                seed, 'test', I, J)
            Psi = data.B[I] @ data.D[J].mT
            Psit = data.B[Ite] @ data.D[Jte].mT
            tr = block_metrics(data, cand, I, J, Psi)
            te = block_metrics(data, cand, Ite, Jte, Psit)
            Etr = kernel_from_phase(residual_phase(data, cand, I, J, Psi),
                                    torch.complex64)
            E_Iho_J = kernel_from_phase(
                residual_phase(data, cand, Ite, J, data.B[Ite] @ data.D[J].mT),
                torch.complex64)
            E_I_Jho = kernel_from_phase(
                residual_phase(data, cand, I, Jte, data.B[I] @ data.D[Jte].mT),
                torch.complex64)
            Ete = kernel_from_phase(residual_phase(data, cand, Ite, Jte, Psit),
                                    torch.complex64)
            U, s, Vh = torch.linalg.svd(Etr, full_matrices=False)
            lblk = tr['L']
            Z, T = zt_from_svd(U, s, Vh, lblk)
            Zh = extend_new_rows(E_Iho_J, T)
            Th = extend_new_cols(E_I_Jho, Z)
            ext = rel_fro(Ete, Zh @ Th.mT)
            grid = _rank_grid(lblk, min(EXT_CAP, s.numel()))
            met_L, met_err, _best = extension_on_grid(
                U, s, Vh, E_Iho_J, E_I_Jho, Ete, grid)
            met = None if met_L is None else (met_L, met_err)
            per_seed.append(dict(
                seed=seed, L_block=tr['L'], tail_block=tr['tail'],
                L_test_intrinsic=te['L'], tail_test=te['tail'],
                fitted_test_at_Lblock=ext,
                L_validated=None if met is None else met[0],
                err_validated=None if met is None else met[1],
            ))
        Lb = [p['L_block'] for p in per_seed]
        Lt = [p['L_test_intrinsic'] for p in per_seed]
        ext_e = [p['fitted_test_at_Lblock'] for p in per_seed]
        Lval = [p['L_validated'] for p in per_seed]
        row = dict(
            cid=cand.cid, family=cand.family, metrics=cand.metrics,
            L_block=Lb, L_test_intrinsic=Lt,
            fitted_test_at_Lblock=ext_e, L_validated=Lval,
            L_block_med=int(round(float(np.median(Lb)))),
            L_test_med=int(round(float(np.median(Lt)))),
            ext_med=float(np.median(ext_e)),
            seeds=per_seed,
        )
        print(f'  {cand.cid:36s}  S/Sgeo={cand.metrics["S_over_Sgeo"]:.3f}  '
              f'L_block={Lb}  L_test={Lt}  ext@Lblock={np.median(ext_e):.3e}  '
              f'L_val={Lval}')
        results.append(row)

    geo = next(r for r in results if r['cid'] == 'geometry')
    L = geo['L_block_med']
    accuracy_resolved = all(
        v is not None for v in geo['L_validated']) and geo['ext_med'] <= PRIMARY_EPS
    # matched-accuracy table rows
    matched = []
    gpu = env.get('gpu_mem_GiB', 0.0)
    for r in results:
        Lp = r['L_block_med']
        s = r['metrics']['s']
        cons = predict_cost(L, max(Lp, 1), s, data.n_src, data.n_tgt,
                            data.n_fft_base, GAMMA_CONS, BETA_CONS, W_CONS)
        opt = predict_cost(L, max(Lp, 1), s, data.n_src, data.n_tgt,
                           data.n_fft_base, GAMMA_OPT, BETA_OPT, W_OPT)
        mem = memory_ok(max(Lp, 1), data.n_src, data.n_tgt, cons['Nprime'],
                        gpu, data.n_coils)
        spd_c = 1.0 / max(cons['T_new_over_T_base'], 1e-12)
        spd_o = 1.0 / max(opt['T_new_over_T_base'], 1e-12)
        kappa = cons['T_new_over_T_base'] * L / max(Lp, 1)
        block_improved = Lp < L and r['cid'] != 'geometry'
        gate = False
        reasons = []
        if not accuracy_resolved:
            reasons.append('baseline factors do not meet held-out ε=1e-3')
        if r['ext_med'] > PRIMARY_EPS:
            reasons.append(f'candidate extension {r["ext_med"]:.3e} > 1e-3')
        if not (Lp < L):
            reasons.append(f"L'={Lp} not < L={L}")
        if spd_c < 1.5:
            reasons.append(f'conservative speedup {spd_c:.2f}x')
        if not mem['ok']:
            reasons.append('memory')
        gate = accuracy_resolved and r['ext_med'] <= PRIMARY_EPS and Lp < L \
            and spd_c >= 1.5 and mem['ok']
        matched.append(dict(
            cid=r['cid'], family=r['family'], L=L, Lp=Lp,
            S_over_Sgeo=r['metrics']['S_over_Sgeo'],
            Nprime_cons=cons['Nprime'], held_ext=r['ext_med'],
            speedup_cons=spd_c, speedup_opt=spd_o, kappa_cons=kappa,
            mem_ok=mem['ok'], peak_GiB=mem['peak_GiB'],
            gate=gate, reasons=reasons, block_improved=block_improved,
        ))
    any_gate = any(m['gate'] for m in matched)
    any_block = any(m['block_improved'] for m in matched)
    print(f'  accuracy_resolved={accuracy_resolved}  any_gate={any_gate}  '
          f'block_improved={any_block}')
    return dict(
        results=results, matched=matched, L_block_geo=L,
        accuracy_resolved=accuracy_resolved, any_gate=any_gate,
        any_block_improvement=any_block,
    )


# ---------------------------------------------------------------------------
# Plots + report
# ---------------------------------------------------------------------------
def plot_all(data, A, B, C, out: Path):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for r in A['rows']:
        if r.get('method') == 'svd_factors':
            ax.scatter(r['l'], r['exact_tail'], marker='o', c='C0', s=18)
            ax.scatter(r['l'], r['fitted_train'], marker='x', c='C1', s=18)
            ax.scatter(r['l'], r['fitted_test'], marker='+', c='C2', s=28)
    ax.axhline(PRIMARY_EPS, ls='--', c='k', lw=0.8)
    ax.set(xlabel='rank l', ylabel='relative Frobenius error', yscale='log',
           title=f'{data.name} SVD tail vs fitted train/test')
    ax.grid(True, alpha=0.3, which='both')
    fig.savefig(out / 'svd_tails.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
    shown = 0
    for cid, tr in B.get('traces', {}).items():
        if not tr or 'continue' not in cid and shown > 8:
            continue
        if shown > 10:
            break
        ax[0].semilogy([t['step'] for t in tr], [t['loss'] for t in tr],
                       lw=1, label=cid[:28])
        ax[1].plot([t['step'] for t in tr],
                   [t['S'] / B['S_geo'] for t in tr], lw=1)
        shown += 1
    ax[0].set(xlabel='update', ylabel=r'$\mathcal{J}_l$', title='opt. tail loss')
    ax[1].set(xlabel='update', ylabel=r'$S/S_{\mathrm{geo}}$', title='bandwidth')
    ax[0].legend(fontsize=5)
    ax[0].grid(True, alpha=0.3)
    ax[1].grid(True, alpha=0.3)
    fig.savefig(out / 'opt_traces.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    markers = dict(geometry='*', phase_svd='o', spatial='s', temporal='^',
                   joint='D', general='P')
    for c in B['candidates']:
        m = c.metrics
        ax.scatter(m['S_over_Sgeo'], m.get('val_tail', m.get('train_tail', 1)),
                   marker=markers.get(c.family, 'x'), s=32, label=c.family)
    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7)
    ax.set(xlabel=r'achieved $S/S_{\mathrm{geo}}$', ylabel='val SVD tail',
           yscale='log', xscale='log',
           title=f'{data.name} residual tail vs bandwidth')
    ax.grid(True, alpha=0.3, which='both')
    fig.savefig(out / 'residual_vs_bandwidth.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    xs = [g['ns'] * g['nt'] for g in A['growth']]
    for e, c in zip(('1e-02', '1e-03', '1e-04'), ('C0', 'C1', 'C2')):
        ax.plot(xs, [g['L'][e] for g in A['growth']], marker='o', c=c, label=e)
    ax.set(xlabel='|I| |J|', ylabel=r'$L_\epsilon^{\mathrm{block}}$',
           title=f'{data.name} rank vs sample size', xscale='log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out / 'val_vs_samples.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for m in C['matched']:
        ax.scatter(m['S_over_Sgeo'], m['speedup_cons'], s=40)
        ax.annotate(m['cid'][:22], (m['S_over_Sgeo'], m['speedup_cons']),
                    fontsize=6)
    ax.axhline(1.5, ls='--', c='k', lw=0.8, label='1.5× gate')
    ax.set(xlabel=r'$S/S_{\mathrm{geo}}$', ylabel='predicted conservative speedup',
           title=f'{data.name} predicted speedup (unvalidated if ext fails)',
           xscale='log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.savefig(out / 'speedup.png')
    plt.close(fig)


def _label(A, B, C) -> str:
    if not A['dense_controls_pass'] or not B['grad_check']['ok']:
        return 'Inconclusive'
    if C['any_gate']:
        return 'Promising'
    if C['accuracy_resolved'] and not C['any_gate']:
        return 'Negative for tested families'
    if C['any_block_improvement'] and not C['accuracy_resolved']:
        return 'Block-level improvement only'
    if not C['accuracy_resolved']:
        return 'Inconclusive'
    return 'Negative for tested families'


AUDIT_MD = r'''
## Audit of run `20260910T071914Z`

Git hash `a229ac6` is shared with earlier reports; the type-3 experiment code was uncommitted, so a commit is not provenance. File SHA-256 prefixes are recorded in `provenance.json`.

**What the reported ranks meant.** $L$ and $L'$ were exact economy SVD ranks of a stratified $1024\times 2048$ residual kernel $E_R=\exp(-i2\pi\mathcal{R})$ in **complex64**, using relative Frobenius tail $\epsilon_l=\sqrt{\sum_{j>l}\sigma_j^2}/\sqrt{\sum\sigma_j^2}$. They were **not** randomized estimates and **not** fitted-factor ranks. A rank cap was not binding at $\epsilon=10^{-3}$. Those block ranks were then used as $L,L'$ even when factor-extension error was $0.26$–$0.69$. That is why a listed $L=2$ or $6$ is not a validated rank at tolerance $10^{-3}$.

**How factors were extended.** Training-block SVD factors $U_L,\Sigma_L,V_L^*$ defined a column-space projector $U_L U_L^*$ (new times, same sources) and a row-space projector $V_L V_L^*$ (new sources, same times). Both axes were independently held out. The gate used the worse of those two one-sided relative errors at the **block** rank. Two-sided Nyström of the held-out block was not used. This is subspace generalization of a local SVD, not a globally evaluated HOFFT factorization.

**What `mix_temp_*` was.** Names are `mix_{mode}_{dir}_e{eta}` with `mode ∈ {spat,temp,joint}`. Shortlisted rows were **temporal-only**: $F=R$ (spatial mix $U=0$) and $Q=K+CV$, with $V$ a PCA/leading-basis direction **scaled by binary search** so that full-data $S(F,Q)$ met the requested $\eta$. Coefficients were **not** optimized against residual-exponential error. Spatial-only and joint scale-to-$\eta$ starts were generated but lost the per-$\eta$ shortlist to temporal-only because they had worse train-block $L$. An Adam refine against a frozen truncated-SVD target existed ($40$ inner steps $\times 3$ outer), but it increased residual rank into the hundreds and was not shortlisted. Phase-preserving rotations cleaned the enclosing box only.

**`remove_linear_terms`.** For each high-order spatial basis, a masked least-squares fit onto $\{1,x,y\}$ (ESPIRiT mask as $0/1$ row weights) is subtracted from $\Phi$. The fitted coefficients times $\alpha(t)$ are folded into the $k$-space trajectory (`trj_term`); the constant-in-space piece (`zeroth`) is discarded by the loader. That code path did not change between the two supplied reports. The in-vivo high-order count is the number of bases with mean$|\phi|\,\mathrm{mean}|\alpha|>10^{-6}$ **after** linear removal. A near-threshold extra column explains $K=22$ vs $K=23$ under the same git hash.

**Nine Stage A checks.** QR vs dense SVD (synthetic and a $40\times 50$ phase block); separable-offset identity; factor recentering; HO / total phase convention vs $\Phi C^T$ and $RK^T+\Phi C^T$; geometry residual equals HO phase (and matching block ranks); zero-HO kernel is identically $1$ with rank $1$. These are identities. They do not validate compression or factor extension.

**Sampling bug in the prior ranks.** `stratified_indices` concatenated random/energy/extrema indices, then `torch.unique` (which sorts) and took `[:n_pick]`. That keeps the *smallest* voxel/time indices, i.e. a single corner patch, and collapses all seeds. Reported $L=2$ (coco) and $L=6$ (in-vivo) were that patch, not a full-support block. This follow-up samples extrema, high-energy points and random points with order-preserving unique so seeds and spatial coverage are real.
'''


def write_report(run_id, env, per, out_root: Path):
    lines = [
        '# Follow-up: residual ranks and joint 2D phase mixing',
        '',
        'Implements `math_docs/t3n_followup.md` on **coco_spiral** and '
        '**tilt_spi_invivo** (requested name `tilted_spi_invivo`). '
        'Physical imaging and latent coordinates stay 2D. '
        'No type-3 adapter is implemented.',
        '',
        f'**Run:** `{run_id}`  **git:** `{env.get("git")}`  '
        f'**GPU:** {env.get("gpu", "cpu")} ({env.get("gpu_mem_GiB", 0):.1f} GiB)  '
        f'**torch:** {env.get("torch")}',
        '',
        'Primary tolerance $\\epsilon=10^{-3}$. Phase in cycles, '
        '$(Ax)_m=\\sum_n x_n e^{-i2\\pi\\Psi_{nm}}$. '
        'Dense SVD controls use complex128; optimization uses complex64 SVD tails. '
        'A bandwidth budget is an upper bound, not a target to inflate $S$.',
        AUDIT_MD,
        '## Verdict',
        '',
        '| dataset | dense A | grad-check | block $L$ | validated $L$ | label | type-3 |',
        '|---|---|---|---:|---|---|---|',
    ]
    for name, blob in per.items():
        A, B, C = blob['A'], blob['B'], blob['C']
        lab = blob['label']
        geo = next((r for r in C['results'] if r['cid'] == 'geometry'), None)
        Lval = geo['L_validated'] if geo else None
        Lval_s = 'unresolved' if (not geo or all(v is None for v in Lval)) else str(Lval)
        lines.append(
            f"| {name} | {'PASS' if A['dense_controls_pass'] else 'FAIL'} | "
            f"{'PASS' if B['grad_check']['ok'] else 'FAIL'} | "
            f"{C['L_block_geo']} | {Lval_s} | **{lab}** | skip |"
        )
    lines += [
        '',
        'Validated $L$ is the smallest SVD-factor rank whose **two-sided '
        'least-squares extension** meets $\\epsilon=10^{-3}$ on an independent '
        'test block. Block $L$ is the exact dense-SVD tail rank only. '
        'Predicted speedups that use unvalidated block ranks are not a go/no-go.',
        '',
    ]
    for name, blob in per.items():
        A, B, C, pre = blob['A'], blob['B'], blob['C'], blob['pre']
        lab = blob['label']
        lines += [
            f'## {name}',
            '',
            f'Loader alias: requested `{pre["alias_requested"]}`, '
            f'resolved `{pre["alias_resolved"]}`. '
            f'HO bases raw {pre["n_high_order_raw"]}, kept '
            f'{pre["n_high_order_kept"]} (thresh {pre["energy_thresh"]}). '
            f'Dropped energies: {pre["dropped_energies"]}.',
            '',
            f'Dense SVD controls: **{"PASS" if A["dense_controls_pass"] else "FAIL"}**. '
            f'Sample-growth $L(\\epsilon=10^{{-3}})$ = {A["L_growth"]}, '
            f'stable={A["growth_stable"]}. '
            f'Opt block {B["opt_ns"]}×{B["opt_nt"]}, trial ranks {B["trial_ranks"]}. '
            f'Starts per family: {B["n_starts"]}.',
            '',
            f'**Label:** {lab}.',
            '',
            '### Rank audit (seed 0, $\\epsilon=10^{-3}$ block)',
            '',
            '| method | rank/l | train err | test ext | intrinsic val $L$ |',
            '|---|---:|---:|---:|---:|',
        ]
        for r in A['rows']:
            if r.get('method') == 'dense_svd_block' and r.get('eps') == PRIMARY_EPS and r.get('seed') == 0:
                lines.append(
                    f"| dense SVD | {r['L']} | {r['fitted_train']:.3e} | — | — |"
                )
            if r.get('method') == 'svd_factors' and r.get('seed') == 0:
                lines.append(
                    f"| SVD $ZT^T$ | {r['l']} | {r['fitted_train']:.3e} | "
                    f"{r['fitted_test']:.3e} | {r.get('val_intrinsic_L', '')} |"
                )
            if r.get('method') == 'extension_scan' and r.get('seed') == 0:
                lines.append(
                    f"| extension scan | {r['L_label']} | — | "
                    f"{r['best_test_err']:.3e} | — |"
                )
        lines += [
            '',
            '### Search (optimized families, feasible only, shortlist)',
            '',
            '| candidate | family | $\\eta_{\\max}$ | $S/S_{\\mathrm{geo}}$ | '
            'trial $l$ | train $L$ | val $L$ | updates |',
            '|---|---|---:|---:|---:|---:|---:|---:|',
        ]
        for c in B['shortlist']:
            m = c.metrics
            lines.append(
                f"| `{c.cid}` | {c.family} | {m.get('eta_max', '')} | "
                f"{m['S_over_Sgeo']:.3f} | {m.get('trial_rank', '')} | "
                f"{m.get('train_L', '')} | {m.get('val_L', '')} | "
                f"{m.get('n_updates', 0)} |"
            )
        lines += [
            '',
            '### Matched-accuracy / cost screen',
            '',
            '| candidate | $L$ | $L\'$ | $S/S_{\\mathrm{geo}}$ | ext. err | '
            'speedup cons/opt | gate |',
            '|---|---:|---:|---:|---:|---:|---|',
        ]
        for m in C['matched']:
            lines.append(
                f"| `{m['cid']}` | {m['L']} | {m['Lp']} | {m['S_over_Sgeo']:.3f} | "
                f"{m['held_ext']:.3e} | {m['speedup_cons']:.2f}× / {m['speedup_opt']:.2f}× | "
                f"{'pass' if m['gate'] else 'no'} |"
            )
        ds_out = f'results/type3_followup/{name}/{run_id}'
        lines += [
            '',
            f'Figures: `{ds_out}/svd_tails.png`, `opt_traces.png`, '
            f'`residual_vs_bandwidth.png`, `val_vs_samples.png`, `speedup.png`.',
            '',
        ]
    lines += [
        '## Interpretation',
        '',
        'Block SVD rank is a diagnostic of a bounded kernel tile. '
        'It becomes a residual HOFFT rank only after the same tolerance is met '
        'by extensible factors on held-out rows **and** columns. '
        'Scale-to-bandwidth perturbations are controls; the search minimizes '
        '$\\mathcal{J}_l=\\sum_{j>l}\\sigma_j(E_{R,IJ})^2/(|I||J|)$ '
        'subject to $S(F,Q)\\le\\eta_{\\max}S_{\\mathrm{geo}}$.',
        '',
        'A genuine validated $L=2$ would leave little type-3 headroom because '
        "$L'\\ge 1$, but that conclusion is not available from an unvalidated "
        'block rank 2.',
        '',
    ]
    text = '\n'.join(lines) + '\n'
    (HERE / 'REPORT.md').write_text(text)
    (out_root / 'REPORT.md').write_text(text)
    print(f'\nWrote {HERE / "REPORT.md"}')


def run_dataset(name, device, env, out_root, run_id):
    print('\n' + '=' * 72)
    print(f'DATASET {name}')
    print('=' * 72)
    data = load_phase_data(name, device, repo=REPO)
    pre = audit_preprocessing(name, device, data)
    print(f'  alias {pre["alias_requested"]} -> {pre["alias_resolved"]}  '
          f'K raw={pre["n_high_order_raw"]} kept={pre["n_high_order_kept"]}  '
          f'dropped={pre["dropped_energies"]}')
    cores = dict(raw=qr_core_svd(data.B, data.D, center=False),
                 offset=qr_core_svd(data.B, data.D, center=True))
    A = experiment_a(data, device)
    if not A['dense_controls_pass']:
        print('  dense controls failed; search still runs as a block diagnostic.')
    B = experiment_b(data, cores, A, device)
    C = experiment_c(data, B, env)
    label = _label(A, B, C)
    print(f'  LABEL: {label}')
    ds_out = out_root / name / run_id
    ds_out.mkdir(parents=True, exist_ok=True)
    plot_all(data, A, B, C, ds_out)
    with open(ds_out / 'rank_audit.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in A['rows'] for k in r}))
        w.writeheader()
        for r in A['rows']:
            w.writerow({k: r.get(k, '') for k in w.fieldnames})
    with open(ds_out / 'search.csv', 'w', newline='') as f:
        rows = B['search_rows']
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(ds_out / 'matched.csv', 'w', newline='') as f:
        if C['matched']:
            w = csv.DictWriter(f, fieldnames=list(C['matched'][0].keys()))
            w.writeheader()
            w.writerows([{k: _jsonable(v) for k, v in r.items()}
                         for r in C['matched']])
    _dump(ds_out / 'summary.json', dict(
        pre=pre, A={k: v for k, v in A.items() if k != 'rows'},
        label=label, n_starts=B['n_starts'],
        shortlist=[c.cid for c in B['shortlist']],
        C={k: v for k, v in C.items() if k != 'results'},
    ))
    # traces are large; store a compact subset
    _dump(ds_out / 'traces_compact.json',
          {k: v[:: max(1, len(v) // 40)] for k, v in B['traces'].items() if v})
    return dict(A=A, B=B, C=C, pre=pre, label=label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+',
                   default=['coco_spiral', 'tilt_spi_invivo'])
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = _env(device)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_root = REPO / 'results' / 'type3_followup'
    out_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps(_jsonable(env), indent=2))
    print(f'run_id={run_id}  short protocol: '
          f'opt {OPT_NS}x{OPT_NT}, {MAX_UPDATES} updates, '
          f'eta={list(ETA_MAX)}, seeds={list(AUDIT_SEEDS)}')
    _dump(out_root / f'provenance_{run_id}.json', env)
    per = {}
    for name in args.datasets:
        if name == 'tilted_spi_invivo':
            name = 'tilt_spi_invivo'
        if name not in DATASETS:
            raise KeyError(name)
        per[name] = run_dataset(name, device, env, out_root, run_id)
        # write partial report after each dataset
        write_report(run_id, env, per, out_root)
    print('Done.')


if __name__ == '__main__':
    main()
