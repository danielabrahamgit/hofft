"""
Regenerate the Stage A tables, the speedup ceiling and the Gate A verdict from saved
results, without re-running the sweep.

Also folds in the anchor-count robustness runs, which matter for the verdict: the global
rank ``L`` is capped by the anchor count, and on ``tilt_spi_invivo`` ``L`` is a sizeable
fraction of ``M``, so ``r = S/L`` could be biased upward by too few anchors.

Re-score at a new NUFFT backend / sample count with ``--rho-json`` -- rank numbers stay
put, only the Sec. 1.3 speedup prediction changes.

Run with:
    paper_experiments/qblock_feas/run.sh summarize.py
    paper_experiments/qblock_feas/run.sh summarize.py --rho-json results/rho_cufinufft_R2.json --tag cufi_R2
"""
import argparse
import json
import sys

import numpy as np
import torch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stageA_block_diagnostic import (  # noqa: E402
    GATE_A_ERRS,
    RHO_ASSUMED,
    apply_rho,
    format_table,
    gate_a_verdict,
    make_figures,
    rhos_from_cufinufft,
    speedup_ceiling,
)

OUT = Path(__file__).resolve().parent / 'results'


def anchor_convergence(rho: float) -> list:
    """``r`` at Q_tot = 4, 8, 16 as the anchor count grows, to check it has converged."""
    runs = [('256', OUT / 'stageA_results.pt'),
            ('512', OUT / 'stageA_results_M512.pt'),
            ('1024', OUT / 'stageA_results_M1024.pt')]
    have = [(m, torch.load(p, weights_only=False)) for m, p in runs if p.exists()]
    if not have:
        return []

    lines = ['\n--- Anchor-count convergence of r (uniform blocks, err=1e-2) ---',
             f'{"dataset":<16s} {"Q_tot":>5s} ' +
             ' '.join(f'{"M=" + m:>12s}' for m, _ in have)]
    names = [n for n in have[0][1] if n in ('coco_spiral', 'tilt_spi_invivo')]
    for name in names:
        for Q in (4, 8, 16):
            cells = []
            for _, res in have:
                hit = [r for r in res.get(name, {}).get('rows', [])
                       if r['Q_tot'] == Q and 'uniform' in r['style']
                       and r['err'] == 1e-2 and r['S'] is not None]
                cells.append(f'r={hit[0]["r"]:.2f} L={hit[0]["L"]:.0f}'
                             if hit else f'{"--":>12s}')
            lines.append(f'{name:<16s} {Q:>5d} ' + ' '.join(f'{c:>12s}' for c in cells))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rho-json', default=None,
                    help='cufinufft (or compatible) json; W3 entries become per-dataset rho')
    ap.add_argument('--tag', default='',
                    help='Suffix for written tables / json / figure')
    ap.add_argument('--note', default='',
                    help='Figure title note, e.g. "cufinufft R=2"')
    args, _ = ap.parse_known_args()

    res_path = OUT / 'stageA_results.pt'
    out = torch.load(res_path, weights_only=False)

    if args.rho_json:
        rho_path = Path(args.rho_json)
        if not rho_path.exists():
            rho_path = OUT / Path(args.rho_json).name
        rhos = rhos_from_cufinufft(rho_path)
        print('per-dataset rho from', args.rho_json)
        for k, v in rhos.items():
            print(f'  {k}: {v:.3f}')
        out = {n: apply_rho(r_, rhos.get(n, rhos['_median'])) for n, r_ in out.items()}
        rho = float(np.median([v['rho'] for n, v in out.items() if n != 'affine']))
    else:
        rho = float(np.median([v['rho'] for v in out.values()]))

    report = ['\n'.join(format_table(out[n]) for n in out)]

    report.append('\n--- Speedup ceiling at r = 1 (per-dataset rho) ---')
    report.append('The block model cannot beat this even with a perfect partition, '
                  'because gather cost')
    report.append('is untouched by any partition. "r for 2x" is what Gate A actually '
                  'demands.')
    report.append(f'{"dataset":<16s} {"rho":>6s} {"Q_tot":>5s} {"lam":>5s} {"Q_eff":>6s} '
                  f'{"ceil":>6s} {"r for 2x":>9s} {"asymp":>6s}')
    for name, r_ in out.items():
        rho_n = float(r_['rho'])
        for c in speedup_ceiling(r_, rho_n):
            if 'uniform' not in c['style']:
                continue
            report.append(f'{name:<16s} {rho_n:>6.3f} {c["Q_tot"]:>5d} {c["lam"]:>5.3f} '
                          f'{c["Q_eff"]:>6.1f} {c["ceiling"]:>6.2f} '
                          f'{c["r_needed"]:>9.2f} {(1 + rho_n) / rho_n:>6.2f}')

    report += anchor_convergence(rho)

    report.append('\n================ GATE A ================')
    report.append('Bar: some (Q_tot >= 8, style) with r <= 1.5 AND predicted speedup')
    report.append(f'     >= 2x, judged at errors {[f"{e:.0e}" for e in GATE_A_ERRS]}.')
    report.append('     rho is per-dataset (cufinufft W3 when --rho-json is set).\n')
    for name, r_ in out.items():
        v = gate_a_verdict(r_)
        status = 'PASS' if v['passed'] else 'FAIL'
        line = f'  [{status}] {name:<16s} d_eff={r_["d_eff"]:.2f}  '
        if v['kind'] == 'correctness':
            report.append(line + f'correctness control: {v["note"]}')
            continue
        b, br = v['best_speedup'], v['best_r']
        line += f'{v["n_passing"]} qualifying (Q_tot, style, err) points'
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

    tag = f'_{args.tag}' if args.tag else ''
    make_figures(out, rho, title_note=args.note or args.tag)

    # The per-Q_tot error curves run to S = Q_tot * M points and dominate the JSON; they
    # live in the .pt for plotting, so keep the JSON to the numbers people read.
    slim = {n: {k: v for k, v in r_.items() if k != 'curves'} for n, r_ in out.items()}
    dest_json = OUT / f'stageA_results{tag}.json'
    dest_txt = OUT / f'stageA_tables{tag}.txt'
    with open(dest_json, 'w') as f:
        json.dump({'rho': rho, 'rho_per_dataset': {n: r_['rho'] for n, r_ in out.items()},
                   'results': slim,
                   'verdicts': {n: gate_a_verdict(r_) for n, r_ in out.items()}},
                  f, indent=1, default=float)

    text = '\n'.join(report)
    print(text)
    with open(dest_txt, 'w') as f:
        f.write(text + '\n')
    print(f'\nwrote {dest_txt}')


if __name__ == '__main__':
    main()
