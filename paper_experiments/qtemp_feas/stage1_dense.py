"""
Stage 1 of the temporally sparse factor study (math_docs/qtemp_block.md Sec. 4).

Same reduced grid / mask / anchors / float64 as Stage 0. Adds a ``sparse_temporal``
arm that, for each (L, S, support variant), measures the relative Frobenius error of

    P ~= H B     H row-sparse with nnz(row) = S

with and without the fixed-support ALS of Sec. 3.4.

Reports ``nu(S) = L_sparse / L_svd`` at errors 1e-2 and 1e-3, the error-vs-S curve
against Sec. 2's exponential, the structured-vs-OMP gap, and the run-length penalty.

Gate 1: measured nu <= 1.3 at S <= 5 on both 2D datasets and the 3D dataset.

Run with:
    paper_experiments/qtemp_feas/run.sh stage1_dense.py
    paper_experiments/qtemp_feas/run.sh stage1_dense.py --datasets coco_spiral
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

from hofft.sparse_temporal import (
    anchors_arclength,
    anchors_fps,
    fit_fixed_support,
    init_factors,
    phi_gram,
    rel_error,
    runify_support,
    support_consecutive,
    support_nearest,
    support_omp,
    support_stats,
    whitened_alphas,
    whitening_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATASETS,
    build_dense,
    err_at_rank,
    rank_at_error,
    svd_error_curve,
)

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

TARGET_ERRS = (1e-2, 1e-3)
SS = (1, 2, 3, 4, 5, 6, 8)
VARIANTS = ('consecutive', 'nearest', 'omp')
NU_GATE = 1.3
S_GATE = 5
N_ALS = 12
RUN_LENS = (1, 64, 128, 256, 512)

# L grids: wide enough that L_sparse at 1e-3 is inside the sweep for each dataset
LS = {
    'coco_spiral':     (5, 8, 10, 12, 16, 20, 24, 32, 40, 48),
    'tilt_spi_invivo': (16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 200),
    'coco_7t_spi':     (12, 16, 20, 24, 32, 40, 48, 64, 80, 96),
}


def _unwhiten(Sigma_sqrt, betas_w):
    return torch.linalg.solve(Sigma_sqrt, betas_w)


def _place(d, L):
    """Arclength anchors plus FPS anchors. Consecutive uses the former."""
    aw = d['aw']
    if aw.shape[1] < 2:
        raise ValueError('need at least 2 samples')
    betas_w_arc, s_beta = anchors_arclength(aw, L)
    betas_w_fps = anchors_fps(aw, L, seed=0)
    return dict(arc=(betas_w_arc, s_beta), fps=betas_w_fps)


def _support(d, placed, L, S, variant, P, Bmat_fps):
    S = min(S, L)
    if variant == 'consecutive':
        return support_consecutive(d['aw'], placed['arc'][1], S)
    if variant == 'nearest':
        return support_nearest(d['aw'], placed['fps'], S)
    if variant == 'omp':
        return support_omp(P, Bmat_fps, S)
    raise ValueError(variant)


def _betas_for(d, placed, variant):
    betas_w = placed['arc'][0] if variant == 'consecutive' else placed['fps']
    return _unwhiten(d['Sigma_sqrt'], betas_w), betas_w


def fit_one(d, L, S, variant, do_als=True):
    P = d['P']
    placed = _place(d, L)
    betas, betas_w = _betas_for(d, placed, variant)
    # Dictionary for OMP = analytic anchor phases (same B init as the other arms)
    B_dict = torch.exp(-2j * np.pi * (betas.T @ d['phis'].double()))
    support = _support(d, placed, L, S, variant, P, B_dict)
    H0, B0 = init_factors(d['phis'], d['alphas'], betas, support, d['Sigma_sqrt'])
    err0 = rel_error(P, H0, B0)
    rec = dict(L=L, S=min(S, L), variant=variant, err_init=err0,
               err_als=None, n_patterns=None, monotone=None,
               stats=support_stats(support, L))
    if do_als:
        fit = fit_fixed_support(P, support, H0, n_iter=N_ALS, lamda=1e-12)
        rec['err_als'] = fit['err']
        rec['n_patterns'] = fit['n_patterns']
        rec['monotone'] = fit['monotone']
    return rec


def nu_from_curve(rows, L_svd, target, which='err_als'):
    """Smallest L whose error is <= target, then nu = L / L_svd. None if never."""
    if L_svd is None:
        return None
    by_L = sorted((r['L'], r[which]) for r in rows if r[which] is not None)
    for L, e in by_L:
        if e <= target:
            return dict(L_sparse=L, nu=L / L_svd, err=e)
    return dict(L_sparse=None, nu=None, err=by_L[-1][1] if by_L else None)


def run_dataset(name, torch_dev, n_anchors=256):
    print(f'\n================ {name} ================')
    d = build_dense(name, torch_dev, n_anchors=n_anchors, anchor_mode='fps')
    Sig = phi_gram(d['prob'].phis, d['prob'].mask)
    d['Sigma_sqrt'] = whitening_matrix(Sig)
    d['aw'] = whitened_alphas(d['alphas'], d['Sigma_sqrt'])

    L_ax, err_svd = svd_error_curve(d['P'])
    L_svd = {f'{e:.0e}': rank_at_error(L_ax, err_svd, e) for e in TARGET_ERRS}
    print(f'  P {tuple(d["P"].shape)}  B={d["alphas"].shape[0]}  '
          f'L_svd @ 1e-2 = {L_svd["1e-02"]}  @ 1e-3 = {L_svd["1e-03"]}')

    Ls = LS[name]
    rows = []
    for L in Ls:
        placed = _place(d, L)
        for S in list(SS) + [L]:
            if S > L:
                continue
            # S = L is the dense arm: one full-support fit, not three copies of OMP
            variants = ('consecutive',) if S == L else VARIANTS
            for variant in variants:
                rec = fit_one(d, L, S, variant, do_als=True)
                if S == L:
                    rec['variant'] = 'dense'
                rows.append(rec)
                e = rec['err_als']
                print(f'    L={L:<3d} S={rec["S"]:<3d} {variant:<12s} '
                      f'init {rec["err_init"]:.3e}  als {e:.3e}  '
                      f'pats {rec["n_patterns"]}')
                del rec
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    # Run-length penalty at a representative operating point
    L_rl = min(Ls, key=lambda L: abs(L - 1.5 * (L_svd['1e-02'] or 20)))
    S_rl = 4
    print(f'\n  run-length sweep  L={L_rl} S={S_rl} nearest')
    placed = _place(d, L_rl)
    betas, betas_w = _betas_for(d, placed, 'nearest')
    sup0 = support_nearest(d['aw'], betas_w, S_rl)
    run_rows = []
    for rl in RUN_LENS:
        sup = runify_support(sup0, rl)
        H0, _ = init_factors(d['phis'], d['alphas'], betas, sup, d['Sigma_sqrt'])
        fit = fit_fixed_support(d['P'], sup, H0, n_iter=N_ALS, lamda=1e-12)
        run_rows.append(dict(run_len=rl, err=fit['err'], n_patterns=fit['n_patterns']))
        print(f'    run={rl:<5d} err {fit["err"]:.4e}  pats {fit["n_patterns"]}')
    base = run_rows[0]['err']
    pen256 = next(r['err'] / base - 1 for r in run_rows if r['run_len'] == 256)

    # nu(S) tables
    nu_tab = {}
    print(f'\n  nu(S)  (als)')
    print(f'  {"S":>3} {"var":<12} {"nu@1e-2":>8} {"Lsp":>5} {"nu@1e-3":>8} {"Lsp":>5}')
    for variant in VARIANTS:
        nu_tab[variant] = {}
        for S in SS:
            sub = [r for r in rows if r['variant'] == variant and r['S'] == S]
            cell = {}
            for tag, target in (('1e-02', 1e-2), ('1e-03', 1e-3)):
                cell[tag] = nu_from_curve(sub, L_svd[tag], target)
            nu_tab[variant][S] = cell
            a, b = cell['1e-02'], cell['1e-03']
            def _fmt(c):
                return (f'{c["nu"]:8.2f} {c["L_sparse"]:5d}'
                        if c and c['nu'] is not None else f'{"n/a":>8} {"--":>5}')
            print(f'  {S:3d} {variant:<12} {_fmt(a)} {_fmt(b)}')

    # Gate 1 for this dataset: some variant with nu@1e-2 <= 1.3 at S<=5
    passing = []
    for variant, byS in nu_tab.items():
        for S, cell in byS.items():
            if S > S_GATE:
                continue
            c = cell['1e-02']
            if c and c['nu'] is not None and c['nu'] <= NU_GATE:
                passing.append(dict(variant=variant, S=S, **c))
    verdict = dict(passed=bool(passing), n_passing=len(passing),
                   pen256=pen256, L_svd=L_svd)
    tag = 'PASS' if verdict['passed'] else 'FAIL'
    print(f'  Gate 1 [{tag}] {len(passing)} (variant, S<=5) cells with nu@1e-2 <= {NU_GATE}')
    if passing:
        best = min(passing, key=lambda c: c['nu'])
        print(f'         best nu={best["nu"]:.3f}  {best["variant"]} S={best["S"]} '
              f'L_sparse={best["L_sparse"]}')
    print(f'  run-256 penalty {pen256:+.2%}')

    return dict(name=name, L_svd=L_svd, Ls=Ls, rows=rows, nu=nu_tab,
                run=run_rows, pen256=pen256, verdict=verdict,
                svd_curve=dict(L=L_ax.tolist(), err=err_svd.tolist()),
                P_shape=tuple(d['P'].shape))


def make_figures(out):
    datasets = list(out)
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.2 * len(datasets), 8.4),
                             squeeze=False)
    colors = dict(consecutive='C0', nearest='C1', omp='C2')
    for j, name in enumerate(datasets):
        res = out[name]
        ax = axes[0, j]
        for variant in VARIANTS:
            Ss, nus = [], []
            for S, cell in res['nu'][variant].items():
                c = cell['1e-02']
                if c and c['nu'] is not None:
                    Ss.append(S)
                    nus.append(c['nu'])
            if Ss:
                ax.plot(Ss, nus, 'o-', color=colors[variant], label=variant)
        ax.axhline(NU_GATE, color='k', ls='--', lw=0.8, label=f'Gate 1 ({NU_GATE})')
        ax.axhline(1.0, color='k', ls=':', lw=0.8)
        ax.set_title(f'{name}\nL_svd@1e-2 = {res["L_svd"]["1e-02"]}')
        ax.set_xlabel('S')
        ax.set_ylabel(r'$\nu = L_\mathrm{sparse}/L_\mathrm{svd}$ at 1e-2')
        ax.set_ylim(0.8, max(3.5, ax.get_ylim()[1]))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, j]
        # error vs S at fixed L ~ 1.5 L_svd, nearest + consecutive, vs exp prediction
        L_fix = min(res['Ls'], key=lambda L: abs(L - 1.5 * (res['L_svd']['1e-02'] or 20)))
        for variant in ('consecutive', 'nearest', 'omp'):
            sub = [r for r in res['rows']
                   if r['variant'] == variant and r['L'] == L_fix and r['S'] < r['L']]
            sub = sorted(sub, key=lambda r: r['S'])
            if sub:
                ax.semilogy([r['S'] for r in sub], [r['err_als'] for r in sub],
                            'o-', color=colors[variant], label=f'{variant} ALS')
                ax.semilogy([r['S'] for r in sub], [r['err_init'] for r in sub],
                            'o--', color=colors[variant], alpha=0.4, label=f'{variant} init')
        # Sec. 2 exponential, scaled to match S=3 nearest ALS if present
        Ss = np.arange(1, 9)
        pred = np.exp(-np.pi * Ss * np.sqrt(1 - 1 / 1.25 ** 2))
        ref = [r for r in res['rows']
               if r['variant'] == 'nearest' and r['L'] == L_fix and r['S'] == 3]
        if ref and ref[0]['err_als']:
            pred = pred * (ref[0]['err_als'] / pred[2])
            ax.semilogy(Ss, pred, 'k:', lw=1.2, label=r'Sec. 2 $\propto e^{-\pi S\sqrt{1-1/\sigma^2}}$')
        ax.axhline(1e-2, color='gray', ls='--', lw=0.7)
        ax.axhline(1e-3, color='gray', ls=':', lw=0.7)
        ax.set_title(f'error vs S at L={L_fix}')
        ax.set_xlabel('S')
        ax.set_ylabel('rel. Frobenius error')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    path = OUT / 'stage1_nu_and_error.png'
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
    print('Stage 1 -- measured nu(S) on dense P')
    print(f'Gate 1: nu <= {NU_GATE} at S <= {S_GATE} on every dataset')

    out = {}
    for name in args.datasets:
        out[name] = run_dataset(name, torch_dev, n_anchors=args.anchors)
        torch.cuda.empty_cache()

    print('\n================ GATE 1 ================')
    all_pass = True
    for name, res in out.items():
        v = res['verdict']
        tag = 'PASS' if v['passed'] else 'FAIL'
        all_pass &= v['passed']
        print(f'  [{tag}] {name:<16s} L_svd@1e-2={res["L_svd"]["1e-02"]}  '
              f'{v["n_passing"]} cells  run-256 {v["pen256"]:+.1%}')
    print()
    if all_pass and set(out) >= set(DATASETS):
        print(f'Gate 1 PASS: nu <= {NU_GATE} at S <= {S_GATE} on all three datasets.')
    elif all_pass:
        print(f'Gate 1 partial: every requested dataset passed; run the rest to close the gate.')
    else:
        failed = [n for n, r in out.items() if not r['verdict']['passed']]
        print(f'Gate 1 FAIL on {failed}. Do not start Stage 3/4.')

    make_figures(out)

    # drop bulky svd arrays from the printed summary; keep them in the pt
    torch.save(out, OUT / 'stage1.pt')
    slim = {}
    for n, r in out.items():
        slim[n] = dict(L_svd=r['L_svd'], nu=r['nu'], run=r['run'],
                       pen256=r['pen256'], verdict=r['verdict'],
                       P_shape=r['P_shape'],
                       rows=[{k: rec[k] for k in rec if k != 'stats'} | dict(stats=rec['stats'])
                             for rec in r['rows']])
    (OUT / 'stage1.json').write_text(json.dumps(slim, indent=2, default=str))
    print(f'wrote {OUT / "stage1.pt"} and {OUT / "stage1.json"}')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
