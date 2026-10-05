"""
Stage B of the block-partitioned SVD feasibility study (math_docs/Qblock_svd_feas.md
Sec. 4): tilt_spi_invivo only.

Scores three arms on the same dense phase matrix

    P(t, r) = exp(-2j pi [ r . k_dev(t) + phi(r) . alpha(t) ])

in relative Frobenius norm, then plots error against the Sec. 1.3 predicted wall-clock
at the measured cufinufft R=2 rho (not against S).

Arms
----
1. global SVD at rank L
2. HOFFT (dense ALS, 32 sweeps, best-scoring sweep)
3. block SVD: spatial {m_q b_{l,q}}, temporal h_{l,q}(t) * zeroth_q(t) * n_w(k_dev,q(t))
   with the analytic KB stencil (splitting, not HOFFT)

Gate B: the block curve sits at least 2x below SVD over a factor-of-2 runtime range,
and within 2x of HOFFT.

Run with:
    paper_experiments/qblock_feas/run.sh stageB_dense.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import einsum

from hofft.decomp import build_kern_bases
from hofft.phase_coeffs import trj_dev_to_phis_alphas
from hofft.utils import gen_grd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks import (  # noqa: E402
    allocate_ranks,
    block_singular_values,
    cost_ratios,
    fit_block_affine,
    make_blocks,
    phase_matrix,
)
from common import load_problem, select_anchors  # noqa: E402
from stageA_block_diagnostic import _factor_uniform  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
DATASET = 'tilt_spi_invivo'
RHO = 0.178          # cufinufft W3, extra R=2
W = 3
N_ALS = 32
M_ANCHORS = 256
Q_TOTS = (4, 8, 16)
HOFFT_LS = (5, 10, 20, 30, 40)
SVD_LS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 70, 90, 107)


def svd_init(P, L):
    U, S, Vh = torch.linalg.svd(P, full_matrices=False)
    return (S[:L, None] * Vh[:L]).conj().contiguous()


def als_dense(P, Kw, g_init, os, rs, n_iter, lamda=1e-4):
    """Dense HOFFT ALS; scores every sweep and keeps the best (non-monotone)."""
    M, N = P.shape
    K = Kw.shape[0]
    L = g_init.shape[0]
    torch_dev = P.device
    g = g_init.clone()
    nrm = P.norm()
    best = None
    mbs = max(1, min(M, int(2 ** 26 / max(N * L * K, 1))))
    for _ in range(n_iter):
        H = torch.empty((L, M, K), dtype=torch.complex64, device=torch_dev)
        for m1 in range(0, M, mbs):
            m2 = min(m1 + mbs, M)
            E = (g[None, :, None, :] * Kw[None, None, :, :]).expand(m2 - m1, L, K, N)
            E = E.reshape((m2 - m1, L * K, N))
            A = einsum(E.conj(), E, 'M P N, M Q N -> M P Q')
            b = einsum(E.conj(), P[m1:m2], 'M P N, M N -> M P')
            ridge = lamda * A.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)
            A = A + ridge[:, None, None] * torch.eye(L * K, device=torch_dev)
            H[:, m1:m2] = torch.linalg.solve(A, b).reshape(
                (m2 - m1, L, K)).permute(1, 0, 2)
            del E, A, b
        S = einsum(H, Kw, 'L M K, K N -> L M N')
        A = einsum(S.conj(), S, 'L M N, Q M N -> N L Q')
        b = einsum(S.conj(), P, 'L M N, M N -> N L')
        ridge = lamda * A.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)
        A = A + ridge[:, None, None] * torch.eye(L, device=torch_dev)
        g = torch.linalg.solve(A, b).T.contiguous()
        err = float((P - einsum(g, S, 'L N, L M N -> M N')).norm() / nrm)
        best = err if best is None else min(best, err)
        del S, A, b
    return best


def build_P(prob, n_anchors=M_ANCHORS, seed=0):
    """Dense P on the study grid, including sub-grid k_dev, Stage-A-style anchors."""
    d = prob.d
    torch_dev = prob.phis.device
    phis_dev, alphas_dev = trj_dev_to_phis_alphas(prob.trj.float(), prob.im_size, prob.os)
    # trj_dev_to_phis_alphas builds phis_dev on im_size; alphas_dev is (d, M)
    B = prob.phis.shape[0]
    phis_all = torch.cat([phis_dev.double(), prob.phis], dim=0)
    alphas_all = torch.cat([alphas_dev.reshape((d, -1)).double(),
                            prob.alphas.reshape((B, -1))], dim=0)

    t_idx = select_anchors(prob.alphas, n_anchors, pool=max(20_000, 4 * n_anchors),
                           seed=seed)
    a_field = prob.alphas[:, t_idx].contiguous()
    a_all = alphas_all[:, t_idx].contiguous()
    trj_a = prob.trj.reshape((-1, d))[t_idx].contiguous()

    sel = torch.argwhere(prob.mask.reshape(-1))[:, 0]
    phis_flt = phis_all.reshape((phis_all.shape[0], -1))[:, sel]
    P = torch.exp(-2j * np.pi * (a_all.T @ phis_flt)).type(torch.complex64)

    Kw = build_kern_bases((W,) * d, prob.im_size, prob.os).to(torch_dev)
    Kw = Kw.reshape((Kw.shape[0], -1))[:, sel].type(torch.complex64)
    rs = gen_grd(prob.im_size).to(torch_dev).reshape((-1, d))[sel]
    return dict(P=P, Kw=Kw, rs=rs, sel=sel, t_idx=t_idx, a_field=a_field,
                trj_a=trj_a, n_mask=int(sel.numel()))


def svd_curve(P, Ls):
    s = torch.linalg.svdvals(P)
    tot = float(s.square().sum())
    out = {}
    for L in Ls:
        L = min(int(L), s.numel() - 1)
        out[L] = float((s[L:].square().sum() / tot).sqrt())
    return out


def block_approx(prob, P, sel, a_field, trj_a, rs, Q_tot, S_list):
    """
    Rank-S splitting reconstruction of P at several total ranks.

    Per block: residual SVD of exp(-2j pi phi_res . alpha), times the exact
    absorbed affine / k_dev phase ``exp(-2j pi r . (k_dev + C_q^T alpha)) * zeroth``.
    The analytic KB stencil is the NUFFT *implementation* of that phase (test 4);
    the dense harness evaluates it exactly, matching Sec. 2's "global rank-S
    splitting model".
    """
    d = prob.d
    bs = make_blocks(prob.mask, _factor_uniform(Q_tot, d), style='uniform')
    C, chat, res = fit_block_affine(prob.phis, bs)
    svals = block_singular_values(res, a_field)
    nrm = float(P.norm())
    os = float(prob.os)
    k_dev = trj_a - (trj_a * os).round() / os

    pos = torch.full((int(np.prod(prob.im_size)),), -1, dtype=torch.long,
                     device=sel.device)
    pos[sel] = torch.arange(sel.numel(), device=sel.device)

    factors = []
    for q, pr in enumerate(res):
        Pq = phase_matrix(pr, a_field)
        U, s, Vh = torch.linalg.svd(Pq, full_matrices=False)
        factors.append((U, s, Vh, pos[bs.idx[q]]))

    rows = []
    for S in S_list:
        S = max(int(S), bs.Q_kept)
        L_q, _ = allocate_ranks(svals, S, floor=1)
        P_hat = torch.zeros_like(P)
        for q, (U, s, Vh, cols) in enumerate(factors):
            Lq = int(L_q[q])
            recon = (U[:, :Lq] * s[:Lq]) @ Vh[:Lq]
            zeroth = torch.exp(-2j * np.pi * (chat[q] @ a_field))
            klin = k_dev + einsum(C[q], a_field, 'B d, B M -> M d')
            linph = torch.exp(-2j * np.pi * (rs[cols].double() @ klin.T))
            P_hat[:, cols] = (zeroth[:, None] * recon * linph.T).type(P.dtype)
        err = float((P - P_hat).norm() / nrm)
        cost = cost_ratios(1.0, bs, os, RHO, im_size_cost=prob.im_size_full)
        t = float(S) * (cost['lam'] / cost['Q_eff'] + RHO)
        rows.append(dict(Q_tot=Q_tot, S=int(S), Lq=L_q.tolist(), err=err,
                         t=t, lam=cost['lam'], Q_eff=cost['Q_eff'],
                         Q_kept=bs.Q_kept, V=list(bs.V)))
        del P_hat
    return rows, bs


def runtime_svd(L):
    return float(L) * (1.0 + RHO)


def _interp_log(x, y, xq):
    x, y = np.asarray(x, float), np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    xq = np.asarray(xq, float)
    return np.exp(np.interp(np.log(xq), np.log(np.maximum(x, 1e-12)),
                            np.log(np.maximum(y, 1e-16))))


def gate_b_verdict(svd, hofft, blocks):
    """
    Gate B: over some interval [t, 2t] inside the overlap of the three curves,
    err_block <= err_svd / 2  and  err_block <= 2 * err_hofft.
    """
    t_s = np.array([runtime_svd(L) for L in svd])
    e_s = np.array([svd[L] for L in svd])
    t_h = np.array([r['t'] for r in hofft])
    e_h = np.array([r['err'] for r in hofft])
    t_b = np.array([r['t'] for r in blocks])
    e_b = np.array([r['err'] for r in blocks])

    lo = max(t_s.min(), t_h.min(), t_b.min())
    hi = min(t_s.max(), t_h.max(), t_b.max())
    if hi < 2 * lo:
        return dict(passed=False, reason='no factor-of-2 overlap in predicted runtime',
                    lo=float(lo), hi=float(hi), n_windows=0, passing=[])

    # Slide a [t, 2t] window across the overlap
    t0s = np.geomspace(lo, hi / 2, 12)
    passing = []
    for t0 in t0s:
        ts = np.geomspace(t0, min(2 * t0, hi), 8)
        es, eh, eb = _interp_log(t_s, e_s, ts), _interp_log(t_h, e_h, ts), _interp_log(t_b, e_b, ts)
        vs_svd = float(np.median(es / eb))
        vs_h = float(np.median(eb / eh))
        ok = vs_svd >= 2.0 and vs_h <= 2.0
        passing.append(dict(t0=float(t0), vs_svd=vs_svd, vs_hofft=vs_h, ok=ok))
    n_ok = sum(1 for p in passing if p['ok'])
    return dict(passed=n_ok >= 1, n_windows=len(passing), n_passing=n_ok,
                lo=float(lo), hi=float(hi), passing=passing,
                reason=None if n_ok else 'no [t,2t] window meets both bars')


def make_figure(svd, hofft, block_by_Q, verdict):
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.6))
    Ls = sorted(svd)
    t_s = [runtime_svd(L) for L in Ls]
    ax[0].plot(Ls, [svd[L] for L in Ls], 'k-o', ms=4, label='global SVD')
    ax[0].plot([r['L'] for r in hofft], [r['err'] for r in hofft], 's-',
               color='tab:blue', label='HOFFT')
    for Q, rows in block_by_Q.items():
        ax[0].plot([r['S'] for r in rows], [r['err'] for r in rows], '^-',
                   label=f'block Q={Q}')
    ax[0].set(xlabel='rank  $L$ or $S$', ylabel=r'$\|E\|_F/\|P\|_F$',
              yscale='log', title='error vs rank')
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].plot(t_s, [svd[L] for L in Ls], 'k-o', ms=4, label='global SVD')
    ax[1].plot([r['t'] for r in hofft], [r['err'] for r in hofft], 's-',
               color='tab:blue', label='HOFFT')
    for Q, rows in block_by_Q.items():
        ax[1].plot([r['t'] for r in rows], [r['err'] for r in rows], '^-',
                   label=f'block Q={Q}')
    ax[1].set(xlabel=r'predicted wall-clock  (cufinufft $\rho=0.18$)',
              ylabel=r'$\|E\|_F/\|P\|_F$', yscale='log', xscale='log',
              title='error vs predicted runtime')
    ax[1].grid(alpha=0.3, which='both')
    ax[1].legend(fontsize=8)
    status = 'PASS' if verdict['passed'] else 'FAIL'
    fig.suptitle(f'Stage B  tilt_spi_invivo  Gate B: {status}', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'stageB_tilt.png', dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-als', type=int, default=N_ALS)
    ap.add_argument('--skip-hofft', action='store_true')
    args, _ = ap.parse_known_args()

    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}  rho={RHO}  W={W}')
    OUT.mkdir(exist_ok=True)

    prob = load_problem(DATASET, torch_dev)
    built = build_P(prob)
    P, Kw, rs = built['P'], built['Kw'], built['rs']
    print(f'P {tuple(P.shape)}  K={Kw.shape[0]}  grid {prob.im_size}  '
          f'native {prob.im_size_full}  os={prob.os:.4f}')

    svd = svd_curve(P, SVD_LS)
    print('SVD:', '  '.join(f'L={L}:{svd[L]:.3e}' for L in sorted(svd)))

    hofft = []
    prev = OUT / 'stageB_tilt.json'
    if args.skip_hofft and prev.exists():
        hofft = json.load(open(prev)).get('hofft', [])
        print(f'reusing {len(hofft)} HOFFT points from {prev}')
    elif not args.skip_hofft:
        for L in HOFFT_LS:
            g0 = svd_init(P, L)
            err = als_dense(P, Kw, g0, prob.os, rs, args.n_als)
            hofft.append(dict(L=L, err=err, t=runtime_svd(L)))
            print(f'  HOFFT L={L:<3d} {err:.4e}  t={runtime_svd(L):.2f}')
            del g0
            torch.cuda.empty_cache()

    # Block: S grid covers the SVD ranks that matter, times a modest rank penalty
    S_grid = [8, 12, 16, 24, 32, 48, 64, 80, 107, 140, 179, 220]
    block_by_Q = {}
    all_block = []
    for Q in Q_TOTS:
        rows, bs = block_approx(prob, P, built['sel'], built['a_field'],
                                built['trj_a'], rs, Q, S_grid)
        # Fill r = S / L_svd(err) using the SVD curve
        Ls = np.array(sorted(svd))
        es = np.array([svd[int(L)] for L in Ls])
        for row in rows:
            L_eq = float(np.interp(row['err'], es[::-1], Ls[::-1]))
            row['L_eq'] = L_eq
            row['r'] = row['S'] / max(L_eq, 1.0)
            print(f'  block Q={Q:<3d} S={row["S"]:<4d} err={row["err"]:.4e}  '
                  f't={row["t"]:.2f}  r~{row["r"]:.2f}')
        block_by_Q[Q] = rows
        all_block.extend(rows)
        del bs
        torch.cuda.empty_cache()

    # Gate B judged on the Q=8 curve (the Gate A winner)
    verdict = gate_b_verdict(svd, hofft, block_by_Q[8]) if hofft else dict(
        passed=False, reason='HOFFT skipped')
    print('\n================ GATE B ================')
    print(f'  [{("PASS" if verdict["passed"] else "FAIL")}] tilt_spi_invivo  '
          f'{verdict.get("reason") or ""}')
    if verdict.get('passing'):
        for p in verdict['passing']:
            flag = 'ok' if p['ok'] else '  '
            print(f'    [{flag}] t0={p["t0"]:.2f}  SVD/block={p["vs_svd"]:.2f}x  '
                  f'block/HOFFT={p["vs_hofft"]:.2f}x')

    make_figure(svd, hofft, block_by_Q, verdict)
    out = dict(dataset=DATASET, rho=RHO, W=W, svd={str(k): v for k, v in svd.items()},
               hofft=hofft, block=block_by_Q, verdict=verdict,
               im_size=list(prob.im_size), M=int(P.shape[0]), N=int(P.shape[1]))
    with open(OUT / 'stageB_tilt.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'\nwrote {OUT / "stageB_tilt.json"}')


if __name__ == '__main__':
    main()
