"""
Validation for the temporal-sparsity machinery (tests 1-7 of math_docs/qtemp_block.md).

Test 3 runs first by construction: it validates the kernel scaling and deapodization
conventions in one dimension with no confounds, so every later number rests on it.

Run with:
    paper_experiments/qtemp_feas/run.sh test_sparse_temporal.py
    paper_experiments/qtemp_feas/run.sh test_sparse_temporal.py --only 3
"""
import argparse
import sys

from pathlib import Path

import numpy as np
import torch

from hofft.sparse_temporal import (
    analytic_gridding_1d,
    arc_length,
    b_update,
    fit_fixed_support,
    h_update,
    h_update_dense,
    init_factors,
    phi_gram,
    rel_error,
    runify_support,
    support_nearest,
    support_stats,
    whitened_alphas,
    whitening_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_dense, svd_error_curve  # noqa: E402

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))
    return ok


# ---------------------------------------------------------------------------
def test_3_analytic_scaling(torch_dev):
    """
    Test 3 -- run this before anything else.

    ``exp(-2j pi phi alpha)`` gridded along alpha with a KB kernel must lose accuracy as
    ``exp(-pi S sqrt(1 - 1/sigma^2))``, up to a constant. Fitting ``log(err)`` against
    ``S`` recovers that exponent only if the kernel width, the anchor spacing and the
    spatial deapodization all use consistent conventions.
    """
    print('\n=== Test 3: analytic gridding follows exp(-pi S sqrt(1 - 1/sigma^2)) ===')
    N, M = 1501, 2001
    Dphi, Dalpha = 1.0, 64.0
    # Offset phi so the centring path is exercised rather than assumed
    phi = torch.linspace(0.3, 0.3 + Dphi, N, dtype=torch.float64, device=torch_dev)
    alpha = torch.linspace(0.0, Dalpha, M, dtype=torch.float64, device=torch_dev)
    P = torch.exp(-2j * np.pi * alpha[:, None] * phi[None, :])

    ok_all = True
    rows = []
    for mode in ('beatty', 'ideal'):
        for sigma in (1.25, 1.5, 2.0):
            pred_slope = -np.pi * np.sqrt(1 - 1 / sigma ** 2)
            Ss, errs = [], []
            for S in range(2, 7):
                g = analytic_gridding_1d(phi, alpha, sigma, S, beta_mode=mode)
                e = rel_error(P, g['H'], g['Bmat'])
                Ss.append(S)
                errs.append(e)
                rows.append((mode, sigma, S, g['L'], g['max_deapod'], e))
            slope = float(np.polyfit(Ss, np.log(errs), 1)[0])
            ratio = slope / pred_slope
            ok = 0.8 <= ratio <= 1.25
            if mode == 'beatty':
                ok_all &= ok
            print(f'    {mode:<7} sigma={sigma:<5} slope {slope:7.3f}  '
                  f'predicted {pred_slope:7.3f}  ratio {ratio:.3f}  '
                  f'err {errs[0]:.2e} -> {errs[-1]:.2e}')
    print(f'    {"mode":>8} {"sigma":>6} {"S":>3} {"L":>5} {"deapod":>8} {"err":>10}')
    for mode, sigma, S, L, dp, e in rows:
        print(f'    {mode:>8} {sigma:>6} {S:>3} {L:>5} {dp:>8.2f} {e:>10.3e}')
    return check('3. analytic gridding matches the predicted exponent', ok_all,
                 'slope ratio in [0.8, 1.25] for all three sigma (beatty beta)')


# ---------------------------------------------------------------------------
def _small_problem(torch_dev, name='coco_spiral', M=256, anchor_mode='fps'):
    d = build_dense(name, torch_dev, n_anchors=M, anchor_mode=anchor_mode)
    Sig = phi_gram(d['prob'].phis, d['prob'].mask)
    d['Sigma_sqrt'] = whitening_matrix(Sig)
    d['aw'] = whitened_alphas(d['alphas'], d['Sigma_sqrt'])
    return d


def test_1_full_support_is_svd(d):
    """S = L must reproduce the SVD arm to machine precision."""
    print('\n=== Test 1: S = L reproduces the SVD ===')
    P = d['P']
    L = 12
    U, s, Vh = torch.linalg.svd(P, full_matrices=False)
    H0 = (U[:, :L] * s[:L]).contiguous()
    Bmat0 = Vh[:L].contiguous()
    err_svd = rel_error(P, H0, Bmat0)

    support = torch.arange(L, device=P.device)[None, :].expand(P.shape[0], L).contiguous()
    fit = fit_fixed_support(P, support, H0, n_iter=6, lamda=0.0)
    rel = abs(fit['err'] - err_svd) / err_svd
    print(f'    SVD err {err_svd:.12e}   fixed-support err {fit["err"]:.12e}')
    return check('1. S = L reproduces the SVD', rel < 1e-10,
                 f'relative difference {rel:.2e}')


def test_2_s1_is_alpha_seg(d):
    """S = 1 with nearest-anchor supports must reproduce alpha segmentation."""
    print('\n=== Test 2: S = 1 reproduces alpha-segmentation ===')
    P, phis, alphas = d['P'], d['phis'], d['alphas']
    L = 16
    from hofft.sparse_temporal import anchors_fps
    betas_w = anchors_fps(d['aw'], L, seed=0)
    # Anchors back in unwhitened alpha space
    betas = torch.linalg.solve(d['Sigma_sqrt'], betas_w)
    support = support_nearest(d['aw'], betas_w, 1)

    # alpha_seg_init: b_l(r) = exp(-2j pi phi(r).beta_l), fixed; H solved
    Bmat = torch.exp(-2j * np.pi * (betas.T @ phis.double()))
    H, _ = h_update(P, Bmat, support, lamda=0.0)

    # Closed form: project each sample onto its single assigned atom
    l_of_m = support[:, 0]
    atoms = Bmat[l_of_m]
    coef = (atoms.conj() * P).sum(dim=-1) / (atoms.abs() ** 2).sum(dim=-1)
    H_ref = torch.zeros_like(H)
    H_ref[torch.arange(P.shape[0], device=P.device), l_of_m] = coef

    diff = float((H - H_ref).norm() / H_ref.norm())
    print(f'    hard-segmented err {rel_error(P, H_ref, Bmat):.6e}, '
          f'L={L}, distinct segments {int(torch.unique(l_of_m).numel())}')
    return check('2. S = 1 equals hard nearest-anchor segmentation', diff < 1e-10,
                 f'||H - H_ref|| / ||H_ref|| = {diff:.2e}')


def test_4_monotone(d):
    """Fixed-support ALS must be monotone: both half-steps are exact least squares."""
    print('\n=== Test 4: fixed-support ALS is monotone ===')
    P = d['P']
    from hofft.sparse_temporal import anchors_fps
    worst = 0.0
    for L, S in ((20, 3), (40, 4), (60, 5)):
        betas_w = anchors_fps(d['aw'], L, seed=0)
        betas = torch.linalg.solve(d['Sigma_sqrt'], betas_w)
        support = support_nearest(d['aw'], betas_w, S)
        H0, _ = init_factors(d['phis'], d['alphas'], betas, support, d['Sigma_sqrt'])
        fit = fit_fixed_support(P, support, H0, n_iter=25, lamda=1e-12)
        rise = fit['monotone'] / max(fit['errs'][0], 1e-30)
        worst = max(worst, rise)
        print(f'    L={L:<3d} S={S}  err {fit["errs"][0]:.4e} -> {fit["err"]:.4e}  '
              f'({len(fit["errs"])} sweeps)  worst rise {rise:.2e}')
    return check('4. ALS is monotone', worst < 1e-10,
                 f'largest relative increase {worst:.2e}')


def test_5_gram_submatrix(d):
    """H update via Gram submatrix must match a dense masked least squares solve."""
    print('\n=== Test 5: Gram-submatrix H update == dense masked least squares ===')
    P = d['P']
    from hofft.sparse_temporal import anchors_fps
    L, S = 24, 4
    betas_w = anchors_fps(d['aw'], L, seed=0)
    betas = torch.linalg.solve(d['Sigma_sqrt'], betas_w)
    support = support_nearest(d['aw'], betas_w, S)
    Bmat = torch.exp(-2j * np.pi * (betas.T @ d['phis'].double()))
    H_fast, n_pat = h_update(P, Bmat, support, lamda=1e-12)
    H_ref = h_update_dense(P, Bmat, support, lamda=1e-12)
    diff = float((H_fast - H_ref).norm() / H_ref.norm())
    print(f'    {n_pat} distinct support patterns over {P.shape[0]} samples')
    return check('5. Gram submatrix matches dense masked LS', diff < 1e-9,
                 f'relative difference {diff:.2e}')


def test_6_run_structure(torch_dev):
    """Run-structured supports vs per-sample: accuracy penalty < 10% at run length 256."""
    print('\n=== Test 6: run-structured supports cost < 10% at run length 256 ===')
    # Two things have to be right for this to mean anything: rows must be consecutive
    # *acquired* samples, and the anchors must be spread over the whole trajectory as
    # they are in deployment. Placing L anchors inside the chunk instead would make one
    # run span several anchor spacings and overstate the penalty.
    d = _small_problem(torch_dev, M=4096, anchor_mode='contiguous')
    P = d['P']
    from hofft.sparse_temporal import anchors_fps
    L, S = 40, 4
    aw_all = whitened_alphas(d['prob'].alphas, d['Sigma_sqrt'])
    betas_w = anchors_fps(aw_all, L, seed=0)
    sup_g = support_nearest(d['aw'], betas_w, S)

    # Keep only the anchors this chunk actually reaches, and relabel
    used = torch.unique(sup_g)
    remap = torch.full((L,), -1, dtype=torch.long, device=P.device)
    remap[used] = torch.arange(used.numel(), device=P.device)
    sup = remap[sup_g]
    betas_w = betas_w[:, used]
    betas = torch.linalg.solve(d['Sigma_sqrt'], betas_w)
    spacing = d['prob'].alphas.shape[1] / L
    print(f'    {L} anchors over {d["prob"].alphas.shape[1]} samples '
          f'({spacing:.0f} samples/anchor); chunk reaches {used.numel()} of them')

    def _fit(support):
        H0, _ = init_factors(d['phis'], d['alphas'], betas, support, d['Sigma_sqrt'])
        return fit_fixed_support(P, support, H0, n_iter=20, lamda=1e-12)

    base = _fit(sup)
    rows = []
    ok = True
    for run_len in (64, 128, 256, 512):
        r = _fit(runify_support(sup, run_len))
        pen = r['err'] / base['err'] - 1
        rows.append((run_len, r['err'], pen, r['n_patterns']))
        print(f'    run={run_len:<5d} err {r["err"]:.4e}  penalty {pen:+7.2%}  '
              f'{r["n_patterns"]} patterns')
        if run_len == 256:
            ok = pen < 0.10
    print(f'    per-sample err {base["err"]:.4e}  ({base["n_patterns"]} patterns)')
    pen256 = [p for rl, _, p, _ in rows if rl == 256][0]
    return check('6. run length 256 costs < 10% accuracy', ok,
                 f'penalty {pen256:+.2%}')


def test_7_sparsity_accounting(d):
    """sum_l nnz(h_l) == sum_m |Omega_m|, and the tap count is S*M."""
    print('\n=== Test 7: sparsity accounting ===')
    from hofft.sparse_temporal import anchors_fps
    L, S = 32, 4
    betas_w = anchors_fps(d['aw'], L, seed=0)
    betas = torch.linalg.solve(d['Sigma_sqrt'], betas_w)
    support = support_nearest(d['aw'], betas_w, S)
    H0, _ = init_factors(d['phis'], d['alphas'], betas, support, d['Sigma_sqrt'])
    fit = fit_fixed_support(d['P'], support, H0, n_iter=8, lamda=1e-12)

    st = support_stats(support, L)
    nnz_H = int((fit['H'].abs() > 0).sum())
    M = d['P'].shape[0]
    ok = (nnz_H == st['taps']) and (st['taps'] == S * M) and st['dead_factors'] == 0
    print(f'    S*M = {S * M}, sum_m |Omega_m| = {st["taps"]}, nnz(H) = {nnz_H}')
    print(f'    nnz per factor: mean {st["nnz_mean"]:.1f}, max {st["nnz_max"]:.0f}, '
          f'{st["dead_factors"]} dead')
    return check('7. tap count matches S*M and H is not silently dense', ok)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=int, default=None)
    args, _ = ap.parse_known_args()

    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')

    if args.only == 3 or args.only is None:
        ok3 = test_3_analytic_scaling(torch_dev)
        if not ok3:
            print('\nTest 3 failed -- the kernel/deapodization conventions are wrong. '
                  'Everything downstream would be measuring that bug, so stopping here.')
            return 1
    if args.only == 3:
        return 0

    d = _small_problem(torch_dev)
    print(f'\nproblem: P {tuple(d["P"].shape)}, B={d["alphas"].shape[0]}, '
          f'{d["n_mask"]} masked voxels')
    L_ax, err = svd_error_curve(d['P'])
    print(f'SVD: L=10 -> {err[10]:.3e}, L=30 -> {err[30]:.3e}, L=60 -> {err[60]:.3e}')

    for fn in (test_1_full_support_is_svd, test_2_s1_is_alpha_seg,
               test_4_monotone, test_5_gram_submatrix):
        if args.only is None or fn.__name__.startswith(f'test_{args.only}_'):
            fn(d)
    if args.only is None or args.only == 6:
        test_6_run_structure(torch_dev)
    if args.only is None or args.only == 7:
        test_7_sparsity_accounting(d)

    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f'\n================ {n_ok}/{len(RESULTS)} PASS ================')
    for name, ok, detail in RESULTS:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
