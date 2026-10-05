"""
Stage A of the block-partitioned SVD feasibility study (math_docs/Qblock_svd_feas.md
Sec. 3): partition, absorb per-block affine phase, and measure the rank penalty

    r = S / L        S = sum_q L_q (block),   L = global SVD rank at the same error

against the FFT saving it buys. No SVD reconstruction and no new linop -- the block model
is exactly a global rank-S splitting model, so the whole accuracy question is linear
algebra on the anchor phase matrix.

Ranks are allocated by a single pooled threshold over all blocks' singular values, which
is the Frobenius-optimal split of a budget because the blocks are disjoint and
``sum_q m_q = 1`` (Sec. 3.2 step 3).

Run with:
    paper_experiments/qblock_feas/run.sh stageA_block_diagnostic.py
"""
import argparse
import json
import sys

import numpy as np
import torch

from pathlib import Path
from typing import Optional

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks import (  # noqa: E402
    block_singular_values,
    cost_ratios,
    fit_block_affine,
    global_error_curve,
    jacobian_principal_axes,
    make_blocks,
    make_slabs,
    phase_matrix,
    rank_at_error,
    storage_terms,
)
from common import REAL, SYNTH, Problem, load_problem, select_anchors  # noqa: E402
from hofft.shear import alpha_second_moment  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'

M_ANCHORS = 256
Q_TOTS = [1, 4, 8, 16, 32, 64]
ERR_TARGETS = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
GATE_A_ERRS = [1e-1, 3e-2, 1e-2]     # the band Gate A is judged in
RHO_ASSUMED = 0.15                   # the value Sec. 1.3 works its examples at


# ---------------------------------------------------------------------------
# Partition styles
# ---------------------------------------------------------------------------
def _factor_uniform(Q_tot: int, d: int) -> tuple:
    """Split ``Q_tot`` into per-axis powers of two, as square as possible (Sec. 3.1)."""
    a = int(round(np.log2(Q_tot)))
    per = [a // d] * d
    for i in range(a - sum(per)):
        per[i] += 1
    return tuple(2 ** p for p in per)


def build_partitions(prob: Problem, Q_tot: int, jac_axis: int) -> list:
    """
    All partition styles at one ``Q_tot``: uniform blocks, plus a slab along each axis.

    The slab along ``jac_axis`` is the ``d_eff = 1``-aware partition of Sec. 1.4 and is
    flagged as such so the outcome can be checked against that prediction.
    """
    mask = prob.mask
    out = []
    if Q_tot == 1:
        return [make_blocks(mask, (1,) * prob.d, style='global')]
    out.append(make_blocks(mask, _factor_uniform(Q_tot, prob.d), style='uniform'))
    for ax in range(prob.d):
        bs = make_slabs(mask, Q_tot, ax)
        bs.style = f'slab_ax{ax}' + (' [jac]' if ax == jac_axis else '')
        out.append(bs)
    return out


# ---------------------------------------------------------------------------
# Stage A per dataset
# ---------------------------------------------------------------------------
def run_problem(name: str,
                torch_dev: torch.device,
                rho: float,
                n_anchors: int = M_ANCHORS,
                q_tots: Optional[list] = None,
                styles: Optional[list] = None) -> dict:
    prob = load_problem(name, torch_dev)
    d, im_size = prob.d, prob.im_size
    sel = torch.argwhere(prob.mask.reshape(-1))[:, 0]
    n_mask = sel.numel()

    t_idx = select_anchors(prob.alphas, n_anchors, pool=max(20_000, 4 * n_anchors))
    a_anchor = prob.alphas[:, t_idx].contiguous()
    M = a_anchor.shape[1]
    norm_sq = float(M * n_mask)          # ||P||_F^2, identical for every arm

    # d_eff read-out: the eigenvalue spread of the pooled whitened Jacobian
    _, sig_sqrt = alpha_second_moment(prob.alphas.float())
    jac_ev, jac_vec = jacobian_principal_axes(prob.phis, prob.mask, sig_sqrt)
    lead = jac_vec[:, 0].abs()
    jac_axis = int(lead.argmax())
    d_eff = float(1.0 / (jac_ev.square().sum()))         # participation ratio

    print(f'\n===== {name} =====')
    print(f'  grid {im_size} (native {prob.im_size_full}), masked {n_mask} voxels, '
          f'B={prob.phis.shape[0]}, M={M} anchors, os={prob.os:.4f}')
    print(f'  Jacobian principal spectrum {[round(float(v), 3) for v in jac_ev]}  '
          f'-> d_eff={d_eff:.2f}, leading axis {jac_axis} '
          f'(|u|={[round(float(v), 3) for v in lead]})')

    # ---- global reference curve (L vs error) ------------------------------
    phis_flt = prob.phis.reshape((prob.phis.shape[0], -1))[:, sel].contiguous()
    s_glob = torch.linalg.svdvals(phase_matrix(phis_flt, a_anchor)).cpu()
    L_ax, err_glob = global_error_curve(s_glob, norm_sq)
    L_at = {e: rank_at_error(L_ax, err_glob, e) for e in ERR_TARGETS}
    print('  global SVD rank at error: ' +
          ', '.join(f'{e:.0e}->L={L_at[e]}' for e in ERR_TARGETS))

    rows, curves = [], {}
    for Q_tot in (q_tots or Q_TOTS):
        for bs in build_partitions(prob, Q_tot, jac_axis):
            if styles and not any(s in bs.style for s in styles):
                continue
            C, chat, res = fit_block_affine(prob.phis, bs)
            svals = block_singular_values(res, a_anchor)
            from blocks import pooled_error_curve
            S_ax, err_blk = pooled_error_curve(svals, norm_sq, floor=1)
            curves[(Q_tot, bs.style)] = (S_ax, err_blk)

            summ = bs.summary()
            for e in ERR_TARGETS:
                S = rank_at_error(S_ax, err_blk, e)
                L = L_at[e]
                if S is None or L is None:
                    rows.append(dict(**summ, err=e, S=None, L=L, r=None,
                                     speedup=None, lam=None))
                    continue
                r = S / L
                cost = cost_ratios(r, bs, prob.os, rho, im_size_cost=prob.im_size_full)
                # Also at the rho Sec. 1.3 assumed, so a failure caused by the platform's
                # gather cost stays distinguishable from a failure of the block idea.
                cost_ref = cost_ratios(r, bs, prob.os, RHO_ASSUMED,
                                       im_size_cost=prob.im_size_full)
                stor = storage_terms(int(S), bs, int(prob.alphas.shape[1]), prob.W,
                                     im_size_cost=prob.im_size_full)
                # L_q hits its floor of 1 when the field is already affine inside a
                # block. Past that point extra blocks buy no accuracy and only add
                # gather work, so S = Q_kept is where the speedup starts falling.
                rows.append(dict(**summ, err=e, S=float(S), L=float(L), r=float(r),
                                 Lq_mean=float(S) / summ['Q_kept'],
                                 floor_bound=bool(float(S) <= summ['Q_kept'] + 1e-9),
                                 **{k: cost[k] for k in
                                    ('lam', 'Q_eff', 'fft_ratio', 'speedup')},
                                 speedup_ref=cost_ref['speedup'],
                                 storage_ratio=stor['block_per_factor'] /
                                 stor['hofft_per_factor'] * r))

            # Residual range after absorption, a cheap sanity read-out
            rng = max(float(r_.abs().max()) for r_ in res)
            print(f'  Q_tot={Q_tot:<3d} {bs.style:<14s} Q_kept={summ["Q_kept"]:<3d}'
                  f'/{summ["Q_tot"]:<3d} V={str(summ["V"]):<10s} '
                  f'occ={summ["occ_frac_mean"]:.2f} max|res|={rng:7.3f}  ' +
                  '  '.join(
                      f'{e:.0e}: S={_fmt(_get(rows, e, summ))}' for e in GATE_A_ERRS))

    return dict(dataset=name, kind=prob.kind, im_size=list(im_size),
                im_size_full=list(prob.im_size_full), n_mask=int(n_mask), M=int(M),
                os=prob.os, B=int(prob.phis.shape[0]), rho=rho,
                jac_evals=[float(v) for v in jac_ev], d_eff=d_eff, jac_axis=jac_axis,
                L_at={f'{e:.0e}': L_at[e] for e in ERR_TARGETS}, rows=rows,
                curves={f'{q}|{s}': (list(map(int, c[0])), list(map(float, c[1])))
                        for (q, s), c in curves.items()})


def _get(rows, e, summ):
    for r in reversed(rows):
        if r['err'] == e and r['style'] == summ['style'] and r['Q_tot'] == summ['Q_tot']:
            return r
    return None


def _fmt(row):
    if row is None or row['S'] is None:
        return '  --      '
    return f'{row["S"]:>4.0f} r={row["r"]:5.2f} x{row["speedup"]:5.2f}'


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def gate_a_verdict(res: dict) -> dict:
    """
    Gate A: some ``(Q_tot >= 8, style)`` with ``r <= 1.5`` and predicted speedup ``>= 2x``.

    The ``affine`` control is excluded from the speedup bar. There the global baseline has
    deliberately *not* had ``remove_linear_terms`` applied (it would zero the field), so
    its ``L`` is not a fair comparator and ``r`` comes out below one. That control is
    judged instead by exactness: ``L_q = 1`` for every block, i.e. ``S == Q_kept``.
    """
    if res['dataset'] == 'affine':
        chk = [r for r in res['rows'] if r['S'] is not None]
        exact = all(r['S'] == r['Q_kept'] for r in chk)
        return dict(kind='correctness', passed=exact, n_passing=len(chk),
                    note=f'S == Q_kept at all {len(chk)} operating points'
                         if exact else 'S != Q_kept somewhere',
                    best_speedup=None, passing=[])

    best, passing, passing_ref = None, [], []
    for row in res['rows']:
        if row['Q_tot'] < 8 or row['r'] is None or row['err'] not in GATE_A_ERRS:
            continue
        if row['r'] <= 1.5 and row['speedup'] >= 2.0:
            passing.append(row)
        if row['r'] <= 1.5 and row['speedup_ref'] >= 2.0:
            passing_ref.append(row)
        if best is None or row['speedup'] > best['speedup']:
            best = row
    best_r = min((row for row in res['rows']
                  if row['Q_tot'] >= 8 and row['r'] is not None
                  and row['err'] in GATE_A_ERRS),
                 key=lambda row: row['r'], default=None)
    return dict(kind='speedup', passed=bool(passing), n_passing=len(passing),
                n_passing_ref=len(passing_ref), passed_ref=bool(passing_ref),
                best_speedup=best, best_r=best_r, passing=passing[:6],
                passing_ref=passing_ref[:6])


def apply_rho(res: dict, rho: float) -> dict:
    """
    Re-score a saved Stage A result at a new ``rho``.

    Rank numbers ``S``, ``L``, ``r``, ``lam``, ``Q_eff`` are properties of the field and
    the partition; only the Sec. 1.3 wall-clock prediction depends on the NUFFT backend
    and the sample count.
    """
    res = dict(res)
    res['rho'] = float(rho)
    rows = []
    for row in res['rows']:
        row = dict(row)
        if row.get('r') is not None and row.get('fft_ratio') is not None:
            row['speedup'] = float(
                (1.0 + rho) / (row['fft_ratio'] + row['r'] * rho))
        elif row.get('r') is not None and row.get('lam') is not None:
            fft_ratio = row['r'] * row['lam'] / row['Q_eff']
            row['fft_ratio'] = float(fft_ratio)
            row['speedup'] = float((1.0 + rho) / (fft_ratio + row['r'] * rho))
        rows.append(row)
    res['rows'] = rows
    return res


def rhos_from_cufinufft(path: Path, tag: str = 'W3') -> dict:
    """Per-dataset W≈3 cufinufft rhos, plus a median for the synthetic controls."""
    raw = json.load(open(path))
    rhos = {}
    for key, rec in raw.items():
        if not key.endswith(f'_{tag}'):
            continue
        name = key[len('cufi_'):-len(f'_{tag}')]
        rhos[name] = float(rec['rho'])
    if not rhos:
        raise ValueError(f'no *_{tag} entries in {path}')
    twod = [v for k, v in rhos.items() if k in ('coco_spiral', 'tilt_spi_invivo')]
    rhos['_median'] = float(np.median(twod if twod else list(rhos.values())))
    return rhos


def speedup_ceiling(res: dict, rho: float) -> list:
    """
    The best speedup the block model could reach at each ``Q_tot``, and the ``r`` needed
    to clear Gate A's 2x bar.

    Gather cost never decreases (Sec. 1.3), so at ``r = 1`` -- a partition with no rank
    penalty at all -- the speedup is already capped at
    ``(1 + rho) / (lam/Q_eff + rho)``, rising to ``(1 + rho) / rho`` as ``Q_tot -> inf``.
    Inverting the same expression at 2x gives the ``r`` Gate A actually demands, which is
    the useful number: if it comes out well below 1.5, the two halves of Gate A are
    inconsistent on this platform and the rank bar was never the binding one.
    """
    seen, out = set(), []
    for row in res['rows']:
        key = (row['Q_tot'], row['style'])
        if key in seen or row.get('lam') is None:
            continue
        seen.add(key)
        base = row['lam'] / row['Q_eff'] + rho
        out.append(dict(Q_tot=row['Q_tot'], style=row['style'], lam=row['lam'],
                        Q_eff=row['Q_eff'], ceiling=(1 + rho) / base,
                        r_needed=(1 + rho) / (2 * base)))
    return out


def format_table(res: dict) -> str:
    lines = [f'--- {res["dataset"]}: Sec. 3.4 table (rho={res["rho"]:.3f}, '
             f'grid {res["im_size"]}, cost grid {res["im_size_full"]}) ---',
             f'    d_eff={res["d_eff"]:.2f}  jac spectrum '
             f'{[round(v, 3) for v in res["jac_evals"]]}  leading axis {res["jac_axis"]}']
    hdr = (f'{"err":>7s} {"Q_tot":>5s} {"style":<14s} {"Q_kept":>6s} {"occ":>5s} '
           f'{"V":>10s} {"S":>5s} {"L":>5s} {"r":>6s} {"Lq":>5s} {"lam":>5s} '
           f'{"Q_eff":>6s} {"spdup":>6s} {"@.15":>6s} {"stor":>6s}')
    lines += [hdr, '-' * len(hdr)]
    for row in res['rows']:
        if row['err'] not in GATE_A_ERRS:
            continue
        head = (f'{row["err"]:>7.0e} {row["Q_tot"]:>5d} {row["style"]:<14s} '
                f'{row["Q_kept"]:>6d} {row["occ_frac_mean"]:>5.2f} '
                f'{str(tuple(row["V"])):>10s}')
        if row['S'] is None:
            lines.append(head + f' {"--":>5s} {"--":>5s}   (error unreachable)')
            continue
        lines.append(
            head + f' {row["S"]:>5.0f} {row["L"]:>5.0f} {row["r"]:>6.2f} '
                   f'{row["Lq_mean"]:>5.2f} {row["lam"]:>5.3f} {row["Q_eff"]:>6.1f} '
                   f'{row["speedup"]:>6.2f} {row["speedup_ref"]:>6.2f} '
                   f'{row["storage_ratio"]:>6.3f}'
                   + ('  [Lq floor]' if row['floor_bound'] else ''))
    return '\n'.join(lines)


STYLE_COLORS = {'uniform': 'tab:blue', 'slab_ax0': 'tab:orange',
                'slab_ax1': 'tab:green'}


def _style_key(style: str) -> str:
    return style.replace(' [jac]', '')


def make_figures(out: dict, rho: float = None, err: float = 1e-2,
                 title_note: str = '') -> None:
    """
    Three rows, one column per problem, all read at a single error level:

    1. error vs total rank ``S`` -- the rank penalty, as a horizontal gap from the global
       SVD curve.
    2. ``r`` vs ``Q_tot`` against the ``r >= 1`` theorem floor and Gate A's ``r <= 1.5``.
    3. predicted speedup vs ``Q_tot`` against Gate A's 2x bar and the ``r = 1`` ceiling,
       which is the speedup a partition with no rank penalty at all would get.
    """
    names = list(out)
    fig, ax = plt.subplots(3, len(names), figsize=(3.4 * len(names), 9.4), squeeze=False)
    for j, name in enumerate(names):
        res = out[name]

        a = ax[0, j]
        gl = res['curves'].get('1|global')
        if gl:
            a.plot(gl[0][1:], gl[1][1:], 'k-', lw=2.0, label='global SVD')
        for key, (S, e) in res['curves'].items():
            q, style = key.split('|')
            if q == '1' or int(q) not in (8, 64):
                continue
            a.plot(S, e, '-' if int(q) == 8 else '--', lw=1.1,
                   color=STYLE_COLORS.get(_style_key(style), 'gray'),
                   label=f'Q={q} {_style_key(style)}')
        a.axhline(err, color='r', ls=':', lw=0.8)
        a.set(xscale='log', yscale='log', xlabel='total rank $S$',
              ylabel=r'$\|E\|_F / \|P\|_F$', ylim=(1e-4, 1.5))
        a.set_title(f'{name}\n$d_{{eff}}$={res["d_eff"]:.2f}', fontsize=9)
        a.grid(alpha=0.3, which='both')
        a.legend(fontsize=5.5)

        b = ax[1, j]
        for style in sorted({_style_key(r['style']) for r in res['rows']} - {'global'}):
            pts = sorted((r['Q_tot'], r['r']) for r in res['rows']
                         if _style_key(r['style']) == style and r['err'] == err
                         and r['r'] is not None)
            if pts:
                b.plot(*zip(*pts), 'o-', ms=4, lw=1.2,
                       color=STYLE_COLORS.get(style, 'gray'), label=style)
        b.axhline(1.0, color='k', lw=1.0, label='$r\\geq1$ (theorem)')
        b.axhline(1.5, color='r', ls=':', lw=1.2, label='Gate A: $r\\leq1.5$')
        b.set(xscale='log', yscale='log', xlabel=r'$Q_{tot}$',
              ylabel=f'rank penalty $r$ @ err={err:.0e}')
        b.grid(alpha=0.3, which='both')
        b.legend(fontsize=5.5)

        c = ax[2, j]
        rho_j = float(res.get('rho', rho if rho is not None else 0.15))
        for style in sorted({_style_key(r['style']) for r in res['rows']} - {'global'}):
            pts = sorted((r['Q_tot'], r['speedup']) for r in res['rows']
                         if _style_key(r['style']) == style and r['err'] == err
                         and r['speedup'] is not None)
            if pts:
                c.plot(*zip(*pts), 'o-', ms=4, lw=1.2,
                       color=STYLE_COLORS.get(style, 'gray'), label=style)
        ceil = [(x['Q_tot'], x['ceiling']) for x in speedup_ceiling(res, rho_j)
                if 'uniform' in x['style']]
        if ceil:
            c.plot(*zip(*sorted(ceil)), 'k--', lw=1.2, label='ceiling at $r=1$')
        c.axhline(2.0, color='r', ls=':', lw=1.2, label='Gate A: $\\geq2\\times$')
        c.axhline(1.0, color='k', lw=0.6)
        c.axhline((1 + rho_j) / rho_j, color='0.5', ls='-.', lw=1.0,
                  label=f'$(1+\\rho)/\\rho$={((1 + rho_j) / rho_j):.2f}')
        c.set(xscale='log', yscale='log', xlabel=r'$Q_{tot}$',
              ylabel=f'predicted speedup @ err={err:.0e}')
        c.grid(alpha=0.3, which='both')
        c.legend(fontsize=5.5)

    rhos = [float(out[n].get('rho', np.nan)) for n in names]
    rho_note = (f'per-dataset $\\rho$={min(rhos):.2f}–{max(rhos):.2f}'
                if len(set(round(r, 3) for r in rhos)) > 1
                else f'measured $\\rho$={rhos[0]:.2f}')
    fig.suptitle('Stage A: rank penalty vs FFT saving'
                 f' ({title_note + ", " if title_note else ""}{rho_note})',
                 fontsize=11)
    fig.tight_layout()
    dest = OUT / ('stageA_rank_and_speedup.png'
                  if not title_note else
                  f'stageA_rank_and_speedup_{title_note.replace(" ", "_")}.png')
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*',
                    default=[n for n in REAL if n != 'coco_7t_spi'] + list(SYNTH))
    ap.add_argument('--rho', type=float, default=None)
    ap.add_argument('--anchors', type=int, default=M_ANCHORS)
    ap.add_argument('--qtots', type=int, nargs='*', default=None)
    ap.add_argument('--styles', nargs='*', default=None)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUT.mkdir(exist_ok=True)
    print(f'device: {torch_dev}')

    rho = args.rho
    if rho is None:
        prof = OUT / 'rho_profile.json'
        if prof.exists():
            vals = [v['rho'] for v in json.load(open(prof)).values()]
            rho = float(np.median(vals))
            print(f'using measured rho={rho:.3f} (median of {len(vals)} profiles)')
        else:
            rho = RHO_ASSUMED
            print(f'no rho_profile.json; using rho={rho:.3f} from Sec. 1.3')

    out = {}
    for name in args.datasets:
        out[name] = run_problem(name, torch_dev, rho, n_anchors=args.anchors,
                                q_tots=args.qtots, styles=args.styles)
        torch.cuda.empty_cache()

    report = ['\n'.join(format_table(out[n]) for n in out)]

    report.append(f'\n--- Speedup ceiling at r = 1 (rho={rho:.3f}) ---')
    report.append(f'{"dataset":<16s} {"Q_tot":>5s} {"style":<14s} {"lam":>5s} '
                  f'{"Q_eff":>6s} {"ceil":>6s} {"r for 2x":>9s}')
    for name, res in out.items():
        for c in speedup_ceiling(res, rho):
            if 'uniform' not in c['style'] and c['style'] != 'global':
                continue
            report.append(f'{name:<16s} {c["Q_tot"]:>5d} {c["style"]:<14s} '
                          f'{c["lam"]:>5.3f} {c["Q_eff"]:>6.1f} '
                          f'{c["ceiling"]:>6.2f} {c["r_needed"]:>9.2f}')
    report.append(f'  asymptote as Q_tot -> inf: {(1 + rho) / rho:.2f}x '
                  f'(gather cost is untouched by any partition)')

    report.append('\n================ GATE A ================')
    report.append('Bar: some (Q_tot >= 8, style) with r <= 1.5 AND predicted speedup')
    report.append(f'     >= 2x, judged at errors {[f"{e:.0e}" for e in GATE_A_ERRS]}, '
                  f'rho={rho:.3f}.\n')
    verdicts = {}
    for name, res in out.items():
        v = gate_a_verdict(res)
        verdicts[name] = v
        status = 'PASS' if v['passed'] else 'FAIL'
        line = (f'  [{status}] {name:<16s} d_eff={res["d_eff"]:.2f}  ')
        if v['kind'] == 'correctness':
            line += f'correctness control: {v["note"]}'
            report.append(line)
            continue
        line += f'{v["n_passing"]} qualifying (Q_tot, style, err) points'
        b, br = v['best_speedup'], v['best_r']
        if b:
            line += (f'; best speedup {b["speedup"]:.2f}x at Q_tot={b["Q_tot"]} '
                     f'{b["style"]} err={b["err"]:.0e} (r={b["r"]:.2f})')
        report.append(line)
        if br:
            report.append(f'         best r: {br["r"]:.2f} at Q_tot={br["Q_tot"]} '
                          f'{br["style"]} err={br["err"]:.0e} '
                          f'-> speedup {br["speedup"]:.2f}x')
        report.append(f'         at the rho={RHO_ASSUMED} Sec. 1.3 assumes: '
                      f'{"PASS" if v["passed_ref"] else "FAIL"}, '
                      f'{v["n_passing_ref"]} qualifying points')
        for p in v['passing'] or v['passing_ref']:
            report.append(f'         pass: Q_tot={p["Q_tot"]} {p["style"]} '
                          f'err={p["err"]:.0e} r={p["r"]:.2f} '
                          f'speedup={p["speedup"]:.2f}x '
                          f'(@rho={RHO_ASSUMED}: {p["speedup_ref"]:.2f}x)')
    text = '\n'.join(report)
    print(text)

    tag = f'_{args.tag}' if args.tag else ''
    torch.save(out, OUT / f'stageA_results{tag}.pt')
    slim = {n: {k: v for k, v in r_.items() if k != 'curves'} for n, r_ in out.items()}
    with open(OUT / f'stageA_results{tag}.json', 'w') as f:
        json.dump({'rho': rho, 'results': slim, 'verdicts': verdicts}, f, indent=1,
                  default=float)
    with open(OUT / f'stageA_tables{tag}.txt', 'w') as f:
        f.write(text + '\n')
    if not args.tag:
        make_figures(out, rho)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
