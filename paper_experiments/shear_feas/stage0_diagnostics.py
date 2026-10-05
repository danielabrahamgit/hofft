"""
Stage 0 of the per-factor trajectory shear feasibility study (math_docs/feas_test.md).

Geometry and cost diagnostics only -- no ALS, no reconstruction. Sweeps the cluster
length scale ``ell`` and the shear cap, and reports for each ``L``:

  * residual phase range after the per-factor affine fit, versus the ell=0 no-shear
    baseline (the number to beat)
  * residual gradient reach relative to the W/2 stencil half-width
  * per-factor grid growth ``rho_l`` and the effective factor count ``L_eff``

Gate 0: at matched L, some (ell, cap) must cut the residual range by >= 2x relative to
baseline while keeping ``L_eff <= 1.3 * L``.

Run with:
    PYTHONPATH=src python paper_experiments/shear_feas/stage0_diagnostics.py
"""
import json
import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

from hofft.shear import (
    alpha_second_moment,
    spatial_jacobian,
    shear_features,
    weighted_kmeans,
    kaffine_clustering,
    init_label_candidates,
    fit_affine_per_cluster,
    shear_trajectory,
    grid_growth,
    residual_diagnostics,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_dataset, subsample_times, DATASETS  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

# Feature-clustering length scales from the doc, extended upward because ell <= 1 turns
# out to be dominated by the value term, plus 'kaffine' which optimizes the post-fit
# residual directly and so gives the shear its best possible partition.
ELLS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 'kaffine']
CAPS = [0.05, 0.1, 0.25, 0.5, 1.0, None]
N_TIME_SUB = 1024
GATE0_RANGE_FACTOR = 2.0
GATE0_LEFF_FACTOR = 1.3


def run_dataset(name: str, torch_dev: torch.device) -> dict:
    ds = load_dataset(name, torch_dev)
    d = ds.trj.shape[-1]
    B = ds.phis.shape[0]
    print(f'\n===== {name} =====')
    print(f'  phis {tuple(ds.phis.shape)}  alphas {tuple(ds.alphas.shape)}  '
          f'trj {tuple(ds.trj.shape)}')
    print(f'  im_size(reduced) {ds.im_size}  im_size(full) {ds.im_size_full}  '
          f'os {ds.os:.4f}  kern {ds.kern_size}')

    # ---- 1. Temporal metric -------------------------------------------------
    sigma, sigma_sqrt = alpha_second_moment(ds.alphas)
    evals_sig = torch.linalg.eigvalsh(sigma.double())
    cond = float(evals_sig.max() / evals_sig.clamp(min=0).min().clamp(min=1e-30))
    print(f'  Sigma_alpha: B={B}  cond={cond:.3e}  '
          f'trace={float(sigma.diagonal().sum()):.4g}')

    t_idx = subsample_times(ds.alphas, sigma_sqrt, N_TIME_SUB)
    alphas_sub = ds.alphas.reshape((B, -1))[:, t_idx].contiguous()

    jac = spatial_jacobian(ds.phis)
    print(f'  |J| max {float(jac.abs().amax()):.4g}  '
          f'residual-free grad reach os*max|J^T a|_inf = '
          f'{ds.os * float((jac.reshape(B, d, -1).permute(2, 1, 0) @ alphas_sub).abs().amax()):.2f}')

    # ---- 2/3. Sweep L, ell, cap --------------------------------------------
    rows = []
    baseline = {}
    feats_cache = {}
    for L in ds.Ls:
        # Baseline: constant-only model, partition optimized by the same Lloyd solver
        # from the same pool of inits. Anything weaker would flatter the shear.
        inits = init_label_candidates(ds.phis, jac, ds.weights, sigma_sqrt, L)
        labels0, consts0, zero_grads, _ = kaffine_clustering(
            ds.phis, ds.weights, sigma_sqrt, L, inits, include_linear=False)
        rng0, rngw0, gmax0, gsamp0 = residual_diagnostics(
            ds.phis, jac, alphas_sub, labels0, ds.weights, consts0, zero_grads)
        kappa0, _, _ = shear_trajectory(ds.trj, ds.alphas, zero_grads, ds.os)
        rho0, leff0, nominal = grid_growth(ds.trj, kappa0, ds.os, ds.kern_size, ds.im_size_full)
        baseline[L] = dict(
            range_max=float(rng0.max()), range_mean=float(rng0.mean()),
            range_w=float(rngw0.mean()), grad_max=float(gmax0.max()),
            L_eff=leff0, grad_samples=gsamp0.cpu(),
        )
        print(f'  [baseline] L={L:<3d} range_ptp={rng0.max():8.3f} '
              f'range_rms={rngw0.mean():8.4f} L_eff={leff0:6.2f} '
              f'(nominal grid growth {nominal:.4f})')

        for ell in ELLS:
            key = (L, ell)
            if key not in feats_cache:
                if ell == 'kaffine':
                    labels, consts, grads, _ = kaffine_clustering(
                        ds.phis, ds.weights, sigma_sqrt, L, inits + [labels0],
                        include_linear=True)
                else:
                    feats = shear_features(ds.phis, jac, sigma_sqrt, ell)
                    labels = weighted_kmeans(feats, ds.weights, L)
                    consts, grads = fit_affine_per_cluster(
                        ds.phis, labels, ds.weights, L)
                    del feats
                feats_cache[key] = (labels, consts, grads)
            labels, consts, grads = feats_cache[key]

            for cap in CAPS:
                kappa, delta, tau = shear_trajectory(
                    ds.trj, ds.alphas, grads, ds.os, cap=cap, im_size=ds.im_size_full)
                rho, leff, _ = grid_growth(ds.trj, kappa, ds.os, ds.kern_size, ds.im_size_full)
                grads_eff = grads * tau[:, None, None]
                rng, rngw, gmax, gsamp = residual_diagnostics(
                    ds.phis, jac, alphas_sub, labels, ds.weights, consts, grads_eff)
                shear_pk = float((kappa - ds.trj[None]).abs().amax())
                dflt = delta.reshape((delta.shape[0], -1, d))
                n_delta = max(int(torch.unique(dflt[l], dim=0).shape[0])
                              for l in range(dflt.shape[0]))
                rows.append(dict(
                    L=L, ell=ell, cap=(-1.0 if cap is None else cap),
                    range_max=float(rng.max()), range_mean=float(rng.mean()),
                    range_w=float(rngw.mean()), grad_max=float(gmax.max()),
                    L_eff=leff, rho_max=float(rho.max()), tau_min=float(tau.min()),
                    shear_peak=shear_pk, n_delta_max=n_delta,
                    gain=baseline[L]['range_max'] / max(float(rng.max()), 1e-12),
                ))
                del kappa, delta, gsamp
        best_L = max([r for r in rows if r['L'] == L], key=lambda r: r['gain'])
        print(f'             L={L:<3d} best ell={best_L["ell"]!s:<7} cap={best_L["cap"]:<5g} '
              f'range_ptp={best_L["range_max"]:8.3f} gain={best_L["gain"]:5.2f}x '
              f'L_eff={best_L["L_eff"]:6.2f} shear_pk={best_L["shear_peak"]:7.1f} '
              f'n_delta={best_L["n_delta_max"]}')
        torch.cuda.empty_cache()

    # ---- Gate 0 -------------------------------------------------------------
    verdict = []
    for L in ds.Ls:
        cands = [r for r in rows if r['L'] == L
                 and r['L_eff'] <= GATE0_LEFF_FACTOR * baseline[L]['L_eff']]
        # Ties in gain are common once the cap stops binding; prefer the cheaper grid
        best = min(cands, key=lambda r: (-r['gain'], r['L_eff'])) if cands else None
        passed = best is not None and best['gain'] >= GATE0_RANGE_FACTOR
        verdict.append(dict(L=L, passed=bool(passed),
                            best=best, baseline_range=baseline[L]['range_max'],
                            baseline_L_eff=baseline[L]['L_eff']))
        if best is not None:
            print(f'  [gate0]    L={L:<3d} best gain={best["gain"]:6.2f}x at '
                  f'ell={best["ell"]!s:<7} cap={best["cap"]:<5g} '
                  f'L_eff={best["L_eff"]:6.2f} ({best["L_eff"]/L:.2f}L)  '
                  f'{"PASS" if passed else "fail"}')

    # ---- Stencil reach figure (best config at the largest L) ----------------
    # Show the best-effort shear (k-affine partition), not whichever tie-broken config
    # happens to top the gain table with a near-zero shear
    L_fig = ds.Ls[-1]
    best_fig = max([r for r in rows if r['L'] == L_fig and r['ell'] == 'kaffine'],
                   key=lambda r: r['gain'])
    ell_b, cap_b = best_fig['ell'], (None if best_fig['cap'] < 0 else best_fig['cap'])
    labels, consts, grads = feats_cache[(L_fig, ell_b)]
    _, _, tau = shear_trajectory(ds.trj, ds.alphas, grads, ds.os,
                                 cap=cap_b, im_size=ds.im_size_full)
    _, _, _, gsamp_shear = residual_diagnostics(
        ds.phis, jac, alphas_sub, labels, ds.weights, consts,
        grads * tau[:, None, None])
    gsamp_base = baseline[L_fig]['grad_samples'].to(torch_dev)

    sel = ds.weights > 0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = 80
    for a, (vals, ttl) in zip(ax, [
        ((ds.os * gsamp_base[sel]).cpu(), f'no shear (L={L_fig})'),
        ((ds.os * gsamp_shear[sel]).cpu(), f'shear ell={ell_b}, cap={cap_b}'),
    ]):
        a.hist(vals.numpy(), bins=bins, color='steelblue')
        a.axvline(ds.kern_size[0] / 2, color='crimson', ls='--',
                  label=f'W/2 = {ds.kern_size[0] / 2:g}')
        a.set_yscale('log')
        a.set_xlabel(r'$\sigma\,\max_t\,|(J-G_l)^T\alpha|_\infty$  [grid units]')
        a.set_title(ttl)
        a.legend()
    fig.suptitle(f'{name}: stencil reach of the residual gradient')
    fig.tight_layout()
    fig.savefig(OUT / f'stage0_stencil_reach_{name}.png', dpi=140)
    plt.close(fig)

    # ---- L-curve: L_eff vs residual range -----------------------------------
    fig, ax = plt.subplots(1, len(ds.Ls), figsize=(3.2 * len(ds.Ls), 3.4), squeeze=False)
    for j, L in enumerate(ds.Ls):
        a = ax[0, j]
        for ell in ELLS:
            pts = sorted([r for r in rows if r['L'] == L and r['ell'] == ell],
                         key=lambda r: r['L_eff'])
            a.plot([p['L_eff'] / L for p in pts], [p['range_max'] for p in pts],
                   marker='o', ms=3, label=f'{ell}')
        a.axhline(baseline[L]['range_max'], color='k', ls='--', lw=1, label='baseline')
        a.axhline(baseline[L]['range_max'] / GATE0_RANGE_FACTOR, color='crimson',
                  ls=':', lw=1, label='gate (2x)')
        a.axvline(GATE0_LEFF_FACTOR, color='gray', ls=':', lw=1)
        a.set_yscale('log')
        a.set_xlabel(r'$L_{eff}/L$')
        a.set_title(f'L = {L}')
        if j == 0:
            a.set_ylabel('residual phase range [cycles]')
            a.legend(fontsize=6, title=r'$\ell$', ncol=2)
    fig.suptitle(f'{name}: shear-cap L-curve (Stage 0)')
    fig.tight_layout()
    fig.savefig(OUT / f'stage0_lcurve_{name}.png', dpi=140)
    plt.close(fig)

    return dict(
        dataset=name, rows=rows,
        baseline={L: {k: v for k, v in b.items() if k != 'grad_samples'}
                  for L, b in baseline.items()},
        verdict=verdict, sigma_cond=cond, B=B,
        os=ds.os, kern_size=list(ds.kern_size),
        im_size=list(ds.im_size), im_size_full=list(ds.im_size_full),
    )


def main() -> None:
    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')

    out = {}
    for name in DATASETS:
        out[name] = run_dataset(name, torch_dev)
        torch.cuda.empty_cache()

    torch.save(out, OUT / 'stage0_results.pt')
    with open(OUT / 'stage0_results.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)

    print('\n================ GATE 0 SUMMARY ================')
    for name, res in out.items():
        n_pass = sum(v['passed'] for v in res['verdict'])
        print(f'{name}: {n_pass}/{len(res["verdict"])} L values pass Gate 0')
        for v in res['verdict']:
            b = v['best']
            if b is None:
                print(f'   L={v["L"]:<3d} no config within L_eff budget')
                continue
            print(f'   L={v["L"]:<3d} range {v["baseline_range"]:9.3f} -> '
                  f'{b["range_max"]:9.3f} ({b["gain"]:5.2f}x)  '
                  f'L_eff {b["L_eff"]:6.2f}  ell={b["ell"]!s:<7} cap={b["cap"]:<5g}  '
                  f'{"PASS" if v["passed"] else "FAIL"}')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
