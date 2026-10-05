"""
Stage 2 of the temporally sparse factor study (math_docs/qtemp_block.md Sec. 4).

Combined ``(Q_N, S)`` diagnostic: reuse the qblock partitioner, absorb per-block affine
phase, then sparsify *within* each block. Measurement only -- no linop.

    S_tot  = sum_q L_q,sparse
    r*nu   = S_tot / L_svd
    S_eff  = min(S, mean L_q)

    FFT ratio    = r*nu * lam / Q_N
    gather ratio = Q_N * S_eff / L_svd
    speedup      = (1 + rho) / (FFT ratio + gather ratio * rho)

Evaluated at the three measured rhos ``{0.32, 0.60, 1.87}``. Reports the Pareto
surface, the argmax ``(Q_N, S)`` at each rho, ``S / L_q`` (saturation), and the
gather floor ``Q_N / L_svd``.

Run with:
    paper_experiments/qtemp_feas/run.sh stage2_combined.py
    paper_experiments/qtemp_feas/run.sh stage2_combined.py --datasets tilt_spi_invivo
"""
import argparse
import json
import sys

from pathlib import Path

import numpy as np
import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from hofft.shear import alpha_second_moment
from hofft.sparse_temporal import (
    anchors_arclength,
    anchors_fps,
    fit_fixed_support,
    init_factors,
    phi_gram,
    rel_error,
    support_consecutive,
    support_nearest,
    whitened_alphas,
    whitening_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATASETS,
    RHO,
    RHO_SWEEP,
    _load_qblock,
    build_dense,
    rank_at_error,
    svd_error_curve,
)

blocks = _load_qblock('blocks')


def _factor_uniform(Q_tot, d):
    a = int(round(np.log2(Q_tot)))
    per = [a // d] * d
    for i in range(a - sum(per)):
        per[i] += 1
    return tuple(2 ** p for p in per)


def _build_partitions(prob, Q_tot, jac_axis):
    mask = prob.mask
    if Q_tot == 1:
        return [blocks.make_blocks(mask, (1,) * prob.d, style='global')]
    out = [blocks.make_blocks(mask, _factor_uniform(Q_tot, prob.d), style='uniform')]
    for ax in range(prob.d):
        bs = blocks.make_slabs(mask, Q_tot, ax)
        bs.style = f'slab_ax{ax}' + (' [jac]' if ax == jac_axis else '')
        out.append(bs)
    return out

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

Q_TOTS = (1, 2, 4, 8, 16, 32, 64)
SS = (1, 2, 3, 4, 5, 6)
TARGET = 1e-2
N_ALS = 8
# Best Stage-A style per dataset. tilt: uniform (Gate A pass). 3D: z-slabs
# (d_eff=1). coco: uniform (nothing passed; still the default).
STYLE = {
    'coco_spiral':     'uniform',
    'tilt_spi_invivo': 'uniform',
    'coco_7t_spi':     'slab',
}


def _L_grid(L_hi):
    hi = max(int(L_hi), 8)
    raw = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128]
    return [L for L in raw if L <= hi] or [hi]


def _sparse_err(P, phis_q, alphas, Sigma_sqrt, L, S, one_d):
    """ALS error of an S-sparse rank-L model on one block."""
    S = min(S, L)
    aw = whitened_alphas(alphas, Sigma_sqrt)
    if one_d:
        betas_w, s_beta = anchors_arclength(aw, L)
        support = support_consecutive(aw, s_beta, S)
        betas = torch.linalg.solve(Sigma_sqrt, betas_w)
    else:
        betas_w = anchors_fps(aw, L, seed=0)
        support = support_nearest(aw, betas_w, S)
        betas = torch.linalg.solve(Sigma_sqrt, betas_w)
    H0, _ = init_factors(phis_q, alphas, betas, support, Sigma_sqrt)
    fit = fit_fixed_support(P, support, H0, n_iter=N_ALS, lamda=1e-12)
    return fit['err']


def _L_sparse_block(P, phis_q, alphas, Sigma_sqrt, S, target, one_d, L_hi):
    """Smallest L whose S-sparse ALS error on this block is <= target."""
    last = None
    for L in _L_grid(L_hi):
        e = _sparse_err(P, phis_q, alphas, Sigma_sqrt, L, S, one_d)
        last = (L, e)
        if e <= target:
            return L, e
    return last


def _pick_style(prob, Q_tot, jac_axis, want):
    parts = _build_partitions(prob, Q_tot, jac_axis)
    if want == 'uniform':
        for bs in parts:
            if bs.style.startswith('uniform') or bs.style == 'global':
                return bs
    if want == 'slab':
        for bs in parts:
            if '[jac]' in bs.style or bs.style.startswith(f'slab_ax{jac_axis}'):
                return bs
        for bs in parts:
            if bs.style.startswith('slab'):
                return bs
    return parts[0]


def run_dataset(name, torch_dev, n_anchors=256):
    print(f'\n================ {name} ================')
    d = build_dense(name, torch_dev, n_anchors=n_anchors, anchor_mode='fps')
    prob = d['prob']
    alphas = d['alphas']
    phis_g = d['phis']
    P_g = d['P']
    L_ax, err_g = svd_error_curve(P_g)
    L_svd = rank_at_error(L_ax, err_g, TARGET)
    print(f'  P {tuple(P_g.shape)}  L_svd@{TARGET:.0e} = {L_svd}')

    _, sig_sqrt = alpha_second_moment(prob.alphas.float())
    jac_ev, jac_vec = blocks.jacobian_principal_axes(prob.phis, prob.mask, sig_sqrt)
    jac_axis = int(jac_vec[:, 0].abs().argmax())
    d_eff = float(1.0 / jac_ev.square().sum())
    one_d = d_eff < 1.5
    print(f'  d_eff={d_eff:.2f}  jac_axis={jac_axis}  style={STYLE[name]}  '
          f'1-D anchors={one_d}')

    cells = []
    for Q_tot in Q_TOTS:
        bs = _pick_style(prob, Q_tot, jac_axis, STYLE[name])
        C, chat, phi_res = blocks.fit_block_affine(prob.phis, bs)
        Q = bs.Q_kept
        # Per-block SVD ranks at the global target (for S/L_q saturation)
        svals = blocks.block_singular_values(phi_res, alphas)
        # allocate so the *pooled* block-SVD error hits TARGET
        S_budget = int(np.ceil(L_svd)) if L_svd else 8
        # walk the pooled curve to the target
        S_ax, err_b = blocks.pooled_error_curve(svals, float(P_g.numel()))
        S_block = blocks.rank_at_error(S_ax, err_b, TARGET)
        Lq_svd, _ = blocks.allocate_ranks(svals, int(np.ceil(S_block or S_budget)))
        lam = blocks.cost_ratios(1.0, bs, prob.os, rho=1.0,
                                 im_size_cost=prob.im_size_full)['lam']
        Q_eff = blocks.cost_ratios(1.0, bs, prob.os, rho=1.0,
                                   im_size_cost=prob.im_size_full)['Q_eff']

        print(f'\n  Q_tot={Q_tot} Q_kept={Q} style={bs.style}  '
              f'Lq_svd mean={Lq_svd.mean():.2f}  S_block={S_block}  '
              f'lam={lam:.3f} Q_eff={Q_eff:.2f}')

        for S in SS:
            Lq_sp = np.zeros(Q, dtype=int)
            err_q = np.zeros(Q)
            for q in range(Q):
                Pq = blocks.phase_matrix(phi_res[q], alphas)
                Sig_q = phi_gram(phi_res[q].reshape((phi_res[q].shape[0], -1)))
                # residual of a near-affine block can be rank-deficient
                ev = torch.linalg.eigvalsh(Sig_q).clamp(min=0)
                if float(ev.max()) < 1e-18:
                    Lq_sp[q] = 1
                    err_q[q] = 0.0
                    continue
                Wq = whitening_matrix(Sig_q, eps=1e-8)
                L_hi = max(int(Lq_svd[q]) * 4, S + 4, 8)
                L_hi = min(L_hi, Pq.shape[0], Pq.shape[1])
                Lsp, e = _L_sparse_block(
                    Pq, phi_res[q], alphas, Wq, S, TARGET, one_d, L_hi)
                Lq_sp[q] = int(Lsp)
                err_q[q] = e
            S_tot = int(Lq_sp.sum())
            rnu = S_tot / L_svd if L_svd else None
            Lq_mean = float(Lq_sp.mean())
            S_eff = float(np.mean(np.minimum(S, Lq_sp)))
            sat = S / Lq_mean if Lq_mean else None
            gfloor = Q / L_svd if L_svd else None
            speeds = {}
            for rho in RHO_SWEEP:
                fft_r = rnu * lam / max(Q_eff, 1e-12)
                gath_r = Q_eff * S_eff / L_svd
                speeds[str(rho)] = (1 + rho) / (fft_r + gath_r * rho)
            print(f'    S={S}  S_tot={S_tot:4d}  r*nu={rnu:5.2f}  '
                  f'Lq={Lq_mean:5.2f}  S/Lq={sat:4.2f}  '
                  f'speed@0.32={speeds["0.32"]:.2f}  '
                  f'@0.60={speeds["0.60"]:.2f}  @1.87={speeds["1.87"]:.2f}')
            cells.append(dict(
                Q_tot=Q_tot, Q_kept=Q, style=bs.style, S=S,
                S_tot=S_tot, rnu=rnu, Lq_mean=Lq_mean, Lq=Lq_sp.tolist(),
                S_eff=S_eff, sat=sat, gfloor=gfloor, lam=lam, Q_eff=Q_eff,
                Lq_svd=Lq_svd.tolist(), Lq_svd_mean=float(Lq_svd.mean()),
                err_q_max=float(err_q.max()), speeds=speeds,
            ))
        torch.cuda.empty_cache()

    # argmax speedup at each rho
    argmax = {}
    for rho in RHO_SWEEP:
        best = max(cells, key=lambda c: c['speeds'][str(rho)])
        argmax[str(rho)] = dict(Q_tot=best['Q_tot'], S=best['S'],
                                speedup=best['speeds'][str(rho)],
                                rnu=best['rnu'], sat=best['sat'])
        print(f'  argmax rho={rho}: Q_N={best["Q_tot"]} S={best["S"]}  '
              f'speedup={best["speeds"][str(rho)]:.2f}x  r*nu={best["rnu"]:.2f}')

    return dict(name=name, L_svd=L_svd, d_eff=d_eff, jac_axis=jac_axis,
                style=STYLE[name], cells=cells, argmax=argmax,
                rho_measured=RHO.get(name))


def make_figures(out):
    rhos = list(RHO_SWEEP)
    fig, axes = plt.subplots(len(out), len(rhos),
                             figsize=(4.4 * len(rhos), 3.8 * len(out)),
                             squeeze=False)
    for i, name in enumerate(out):
        res = out[name]
        Qs = sorted({c['Q_tot'] for c in res['cells']})
        Ss = sorted({c['S'] for c in res['cells']})
        for j, rho in enumerate(rhos):
            ax = axes[i, j]
            Z = np.full((len(Ss), len(Qs)), np.nan)
            for c in res['cells']:
                Z[Ss.index(c['S']), Qs.index(c['Q_tot'])] = c['speeds'][str(rho)]
            im = ax.imshow(Z, origin='lower', aspect='auto', cmap='viridis')
            ax.set_xticks(range(len(Qs)))
            ax.set_xticklabels(Qs)
            ax.set_yticks(range(len(Ss)))
            ax.set_yticklabels(Ss)
            am = res['argmax'][str(rho)]
            ax.set_title(f'{name}  rho={rho}\nargmax Q={am["Q_tot"]} S={am["S"]}  '
                         f'{am["speedup"]:.2f}x')
            ax.set_xlabel('Q_N')
            ax.set_ylabel('S')
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = OUT / 'stage2_pareto.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--anchors', type=int, default=256)
    args, _ = ap.parse_known_args()

    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')
    print('Stage 2 -- combined (Q_N, S) diagnostic')
    print(f'error target {TARGET:.0e}; rhos {RHO_SWEEP}')

    out = {}
    for name in args.datasets:
        out[name] = run_dataset(name, torch_dev, n_anchors=args.anchors)
        torch.cuda.empty_cache()

    print('\n================ STAGE 2 ARGMAX ================')
    Qs = {str(rho): [] for rho in RHO_SWEEP}
    for name, res in out.items():
        for rho in RHO_SWEEP:
            am = res['argmax'][str(rho)]
            Qs[str(rho)].append(am['Q_tot'])
            print(f'  {name:<16s} rho={rho:<4}  Q_N={am["Q_tot"]:<3} S={am["S"]}  '
                  f'{am["speedup"]:.2f}x  r*nu={am["rnu"]:.2f}  S/Lq={am["sat"]:.2f}')
    # Flat-in-rho: same (Q,S) or Q shifting as predicted (4 at high rho, 8-16 at low)
    print('\n  Q_N vs rho:')
    for rho in RHO_SWEEP:
        print(f'    rho={rho}: Q_N = {Qs[str(rho)]}  mean {np.mean(Qs[str(rho)]):.1f}')

    make_figures(out)
    torch.save(out, OUT / 'stage2.pt')
    slim = {n: {k: v for k, v in r.items() if k != 'cells'} | dict(cells=r['cells'])
            for n, r in out.items()}
    (OUT / 'stage2.json').write_text(json.dumps(slim, indent=2, default=str))
    print(f'wrote {OUT / "stage2.pt"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
