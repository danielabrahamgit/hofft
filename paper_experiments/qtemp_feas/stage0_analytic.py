"""
Stage 0 of the temporally sparse factor study (math_docs/qtemp_block.md Sec. 4).

No fitting. On each of coco_spiral, tilt_spi_invivo, coco_7t_spi report:

  1. Sigma_phi, its condition number, and the whitened SBP_b per basis
  2. B_effective = # axes with whitened SBP_b >= 1
  3. Effective rank of the whitened alpha cloud (is the manifold 1-D?)
  4. Predicted L_b, prod_b L_b, and nu = L_pred / L_svd, for
     sigma_beta in {1.25, 1.5, 2.0} and S in {2,3,4,5,6}
  5. max_r 1/psihat(phi(r))

Gate 0: predicted nu <= 1.3 at S <= 5 on at least one dataset, with prod_b L_b
within ~2x of L_svd. If prod_b L_b blows up because B_effective is large, that is
the type-3 failure mode -- report it and stop.

Run with:
    paper_experiments/qtemp_feas/run.sh stage0_analytic.py
"""
import json
import sys

from pathlib import Path

import numpy as np
import torch

from hofft.phase_coeffs import rescale_phis_alphas, whiten_phis_alphas
from hofft.sparse_temporal import (
    _inv_psihat,
    arc_length,
    effective_rank,
    kb_beta,
    phi_gram,
    whitened_alphas,
    whitening_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATASETS,
    build_dense,
    rank_at_error,
    svd_error_curve,
)

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

SIGMAS = (1.25, 1.5, 2.0)
SS = (2, 3, 4, 5, 6)
TARGET_ERRS = (1e-2, 1e-3)
SBP_CUT = 1.0
NU_GATE = 1.3
L_RATIO_GATE = 2.0
DEAPOD_WARN = 10.0


def _range(x: torch.Tensor) -> float:
    return float(x.max() - x.min())


def _sigma_phi_axes(phis_flt: torch.Tensor, alphas: torch.Tensor, ev_floor: float = 1e-4):
    """
    Eigenbasis of Sigma_phi, then per-axis ranges of the dual pair

        phi_w = Lambda^{-1/2} V^T phi,   alpha_w = Lambda^{1/2} V^T alpha

    so that phi . alpha = phi_w . alpha_w and D(alpha, beta) is Euclidean in alpha_w.
    Axes with ``ev < ev_floor * ev_max`` are numerical nulls of Sigma_phi -- they are
    reported but excluded from B_effective / prod L_b, because whitening them by
    ``Lambda^{-1/2}`` turns leftover bits into a fake SBP.
    """
    Sigma = (phis_flt @ phis_flt.T) / max(phis_flt.shape[1], 1)
    ev, V = torch.linalg.eigh(Sigma)
    ev = ev.flip(0).clamp(min=0)
    V = V.flip(1)
    live = ev > ev_floor * float(ev.max().clamp(min=1e-30))
    ev_use = ev.clone()
    ev_use[~live] = float(ev[live].min()) if live.any() else 1.0
    phi_e = V.T @ phis_flt
    alpha_e = V.T @ alphas
    phi_w = phi_e / ev_use.sqrt()[:, None]
    alpha_w = alpha_e * ev_use.sqrt()[:, None]
    Dphi = torch.tensor([_range(phi_w[b]) for b in range(phi_w.shape[0])])
    Dalpha = torch.tensor([_range(alpha_w[b]) for b in range(alpha_w.shape[0])])
    SBP = Dphi * Dalpha
    SBP[~live] = 0.0
    live_ev = ev[live]
    cond = float(live_ev.max() / live_ev.min()) if live_ev.numel() else float('inf')
    return dict(Sigma=Sigma, ev=ev, V=V, Dphi=Dphi, Dalpha=Dalpha, SBP=SBP,
                phi_w=phi_w, alpha_w=alpha_w, cond=cond, live=live)


def _joint_svd_axes(phis: torch.Tensor, alphas: torch.Tensor, mask: torch.Tensor):
    """
    Thin SVD of the total phase, matching ``whiten_phis_alphas``. SBP of mode b is
    the product of ranges of the scale-invariant pair (V_b, S_b U_b).
    """
    # rescale_phis_alphas quantiles must match the input dtype
    phis_n, _, alphas_n, _ = rescale_phis_alphas(
        phis.float(), alphas.float(), mask=mask.float())
    phis_n = phis_n.double() * mask
    alphas_n = alphas_n.double()
    B = phis_n.shape[0]
    phis_w, alphas_w, S = whiten_phis_alphas(
        phis_n, alphas_n, B_compressed=B, return_singular_values=True)
    sel = mask.reshape(-1) > 0
    Pw = phis_w.reshape((B, -1))[:, sel]
    Aw = alphas_w.reshape((B, -1))
    Dphi = torch.tensor([_range(Pw[b]) for b in range(B)])
    Dalpha = torch.tensor([_range(S[b] * Aw[b]) for b in range(B)])
    return dict(S=S.detach().cpu(), Dphi=Dphi, Dalpha=Dalpha, SBP=Dphi * Dalpha,
                Pw=Pw, Aw=Aw)


def _max_deapod(phi_w: torch.Tensor, Dphi: torch.Tensor, sigma: float, S: int) -> float:
    """
    max_r prod_b 1/psihat(phi_w,b(r) * h_beta,b), with h_beta,b = 1 / (sigma * Dphi_b).
    Only axes with Dphi_b > 0 enter. Normalized so the value is 1 at phi = 0.
    """
    beta = kb_beta(sigma, S, 'beatty')
    deap = torch.ones(phi_w.shape[1], dtype=torch.float64, device=phi_w.device)
    z = torch.zeros(1, dtype=torch.float64, device=phi_w.device)
    for b in range(phi_w.shape[0]):
        Dp = float(Dphi[b])
        if Dp < 1e-12:
            continue
        h = 1.0 / (sigma * Dp)
        phi_c = phi_w[b] - 0.5 * (phi_w[b].max() + phi_w[b].min())
        num = _inv_psihat(phi_c * h, beta, S)
        den = _inv_psihat(z, beta, S)
        deap = deap * (num / den)
    return float(deap.max())


def _pred_L(SBP: torch.Tensor, sigma: float, S: int, axes=None) -> dict:
    sbp = SBP.cpu().numpy()
    if axes is None:
        axes = np.arange(len(sbp))
    Lb = np.array([int(np.ceil(sigma * sbp[b]) + S) for b in axes], dtype=int)
    return dict(L_b=Lb.tolist(), L_prod=int(np.prod(Lb)) if len(Lb) else S)


def analyze(name: str, torch_dev: torch.device) -> dict:
    print(f'\n================ {name} ================')
    d = build_dense(name, torch_dev, n_anchors=256, anchor_mode='fps')
    phis, alphas, mask = d['prob'].phis, d['prob'].alphas, d['prob'].mask
    B = phis.shape[0]
    phis_flt = d['phis']                          # (B, N_mask), already masked
    # Full-M alphas for the manifold / SBP (not the 256-anchor subsample)
    alphas_full = alphas.reshape((B, -1))

    # --- Sigma_phi (Sec. 3.1) ---
    Sig = phi_gram(phis, mask)
    ax = _sigma_phi_axes(phis_flt, alphas_full)
    print(f'  B = {B}, N_mask = {d["n_mask"]}, M = {alphas_full.shape[1]}')
    n_live = int(ax['live'].sum())
    print(f'  Sigma_phi cond (live axes) = {ax["cond"]:.3e}   '
          f'{n_live}/{B} axes above 1e-4 * ev_max')
    print(f'  Sigma_phi evals (desc): {np.array2string(ax["ev"].cpu().numpy(), precision=3)}')

    sbp = ax['SBP'].cpu().numpy()
    print(f'  whitened SBP_b (Sigma_phi): {np.array2string(sbp, precision=3)}')
    print(f'  Dphi_w:  {np.array2string(ax["Dphi"].cpu().numpy(), precision=3)}')
    print(f'  Dalpha_w:{np.array2string(ax["Dalpha"].cpu().numpy(), precision=3)}')
    B_eff = int((ax['SBP'] >= SBP_CUT).sum())
    axes_eff = [b for b in range(B) if float(ax['SBP'][b]) >= SBP_CUT]
    print(f'  B_effective (SBP_b >= {SBP_CUT}) = {B_eff}  (of B = {B})  axes {axes_eff}')
    # 1-D manifold prediction: grid the curve, not the B-dimensional tensor product
    sbp_lead = float(ax['SBP'][0]) if B else 0.0

    # --- whitened alpha manifold ---
    Sig_sqrt = whitening_matrix(Sig)
    aw = whitened_alphas(alphas_full, Sig_sqrt)
    d_eff, spec = effective_rank(aw)
    s = arc_length(aw)
    print(f'  whitened-alpha d_eff = {d_eff:.3f}   spectrum {np.array2string(spec.cpu().numpy(), precision=3)}')
    print(f'  arc length = {float(s[-1]):.3f}   (1-D manifold: {d_eff < 1.5})')

    # --- joint SVD, the pipeline's own whitening ---
    jnt = _joint_svd_axes(phis, alphas_full, mask)
    print(f'  joint-SVD S:     {np.array2string(jnt["S"].numpy(), precision=3)}')
    print(f'  joint-SVD SBP_b: {np.array2string(jnt["SBP"].cpu().numpy(), precision=3)}')

    # --- L_svd on the same dense P the later stages use ---
    L_ax, err = svd_error_curve(d['P'])
    L_svd = {f'{e:.0e}': rank_at_error(L_ax, err, e) for e in TARGET_ERRS}
    print(f'  L_svd @ 1e-2 = {L_svd["1e-02"]}   @ 1e-3 = {L_svd["1e-03"]}   '
          f'(P {tuple(d["P"].shape)})')

    # --- predicted L and nu ---
    rows = []
    print(f'  {"sigma":>6} {"S":>3} {"L_eff":>7} {"L_all":>7} '
          f'{"nu@1e-2":>8} {"nu@1e-3":>8} {"deapod":>8}')
    for sigma in SIGMAS:
        live_idx = [b for b in range(B) if bool(ax['live'][b])]
        deap = _max_deapod(ax['phi_w'][live_idx], ax['Dphi'][live_idx], sigma, S=5) if live_idx else 1.0
        for S in SS:
            pred_eff = _pred_L(ax['SBP'], sigma, S, axes_eff)
            pred_all = _pred_L(ax['SBP'], sigma, S, None)
            pred_1d = _pred_L(ax['SBP'], sigma, S, [0])
            # Use the 1-D formula whenever the whitened alpha cloud is a curve;
            # the tensor product is the type-3 diagnostic, not the algorithm.
            L_used = pred_1d['L_prod'] if d_eff < 1.5 else pred_eff['L_prod']
            nu = {}
            for tag, Lv in L_svd.items():
                nu[tag] = (L_used / Lv) if Lv else None
            print(f'  {sigma:6.2f} {S:3d} {L_used:7d} {pred_all["L_prod"]:7d} '
                  f'{nu["1e-02"] if nu["1e-02"] else float("nan"):8.2f} '
                  f'{nu["1e-03"] if nu["1e-03"] else float("nan"):8.2f} '
                  f'{deap:8.2f}')
            rows.append(dict(sigma=sigma, S=S, L_b_eff=pred_eff['L_b'],
                             L_pred=L_used, L_pred_1d=pred_1d['L_prod'],
                             L_pred_all=pred_all['L_prod'],
                             L_b_all=pred_all['L_b'], nu=nu, max_deapod=deap))

    # Gate-0 cells: S <= 5, any sigma, nu@1e-2 <= 1.3 and L_pred within 2x of L_svd
    L2 = L_svd['1e-02']
    passing = []
    for r in rows:
        if r['S'] > 5 or L2 is None:
            continue
        nu = r['nu']['1e-02']
        ratio = r['L_pred'] / L2 if L2 else None
        close = ratio is not None and (1.0 / L_RATIO_GATE) <= ratio <= L_RATIO_GATE
        if nu is not None and nu <= NU_GATE and close:
            passing.append(r)

    verdict = dict(
        passed=bool(passing),
        n_passing=len(passing),
        B_eff=B_eff, d_eff=d_eff, blown=B_eff >= 3,
        note=('nu <= 1.3 at S<=5 with L_1d within 2x of L_svd'
              if passing else
              ('B_effective >= 3 and the 1-D formula is not within 2x of L_svd'
               if B_eff >= 3 else
               'no (sigma, S<=5) cell has nu<=1.3 and L_pred/L_svd within 2x')),
    )
    tag = 'PASS' if verdict['passed'] else 'FAIL'
    print(f'  Gate 0 [{tag}] {verdict["note"]}')
    if passing:
        best = min(passing, key=lambda r: r['nu']['1e-02'])
        print(f'         best nu={best["nu"]["1e-02"]:.3f} at sigma={best["sigma"]} '
              f'S={best["S"]} L_pred={best["L_pred"]} L_svd={L2}')

    return dict(
        name=name, B=B, N_mask=d['n_mask'], M=int(alphas_full.shape[1]),
        Sigma_cond=ax['cond'], Sigma_evals=ax['ev'].cpu().tolist(),
        SBP=sbp.tolist(), Dphi=ax['Dphi'].cpu().tolist(),
        Dalpha=ax['Dalpha'].cpu().tolist(),
        B_eff=B_eff, axes_eff=axes_eff,
        d_eff=d_eff, alpha_spectrum=spec.cpu().tolist(),
        arc_length=float(s[-1]),
        joint_S=jnt['S'].tolist(), joint_SBP=jnt['SBP'].cpu().tolist(),
        L_svd=L_svd, rows=rows, verdict=verdict,
        P_shape=tuple(d['P'].shape),
    )


def main():
    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')
    print('Stage 0 -- analytic prediction, no fitting')
    print(f'Gate 0: nu <= {NU_GATE} at S <= 5 on at least one dataset, '
          f'with L_pred / L_svd <= {L_RATIO_GATE}')

    out = {}
    for name in DATASETS:
        out[name] = analyze(name, torch_dev)
        torch.cuda.empty_cache()

    print('\n================ GATE 0 ================')
    any_pass = False
    for name, res in out.items():
        v = res['verdict']
        tag = 'PASS' if v['passed'] else 'FAIL'
        print(f'  [{tag}] {name:<16s} B={res["B"]} B_eff={res["B_eff"]} '
              f'd_eff={res["d_eff"]:.2f}  L_svd@1e-2={res["L_svd"]["1e-02"]}  '
              f'{v["note"]}')
        any_pass |= v['passed']
    print()
    if any_pass:
        print(f'Gate 0 PASS: at least one dataset has predicted nu <= {NU_GATE} '
              f'at S <= 5 with L_pred within {L_RATIO_GATE}x of L_svd.')
    else:
        blown = [n for n, r in out.items() if r['verdict']['blown']]
        if blown:
            print(f'Gate 0 FAIL: prod_b L_b blows up on {blown} '
                  f'(B_effective >= 3). Same failure mode as the type-3 analysis. Stop.')
        else:
            print('Gate 0 FAIL: no dataset meets nu <= 1.3 at S <= 5. Stop.')

    payload = {n: {k: v for k, v in r.items() if k != 'rows'} | dict(rows=r['rows'])
               for n, r in out.items()}
    # tensors already converted
    (OUT / 'stage0.json').write_text(json.dumps(payload, indent=2, default=str))
    print(f'\nwrote {OUT / "stage0.json"}')
    return 0 if any_pass else 1


if __name__ == '__main__':
    sys.exit(main())
