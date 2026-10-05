"""
Stage 1 -- local rank atlas (feas.md Sec. 10) and tolerance calibration.

Two halves:

**1a, the atlas.** Local rank ``C_q(Q, eps)`` over a sweep of block scales, weightings
and tolerances, from Gram matrices rather than per-block SVDs (see ``tree.py``).

**1b, the calibration.** feas.md Failure 9 warns that a small SVD residual on the
sensitivity maps does not imply a small forward-operator error, and it matters here
because essentially all the local rank reduction lives at ``eps = 1e-2`` while the study
targets ``eps_A ~ 1e-3``. Gate 1 is meaningless until we know which tolerance column is
admissible evidence, so we measure the map ``eps -> eps_A`` directly: block-truncate the
sensitivity field, ``s_hat = sum_q 1_{R_q} U_q U_q^H s``, and push it through the
*global* NUFFT. That is exact -- no hierarchical operator needed -- and costs two
Baseline-A applications per point. It also yields the per-coil (Sec. 18.2), per-|k|
(Sec. 18.3) and normal-operator (Sec. 18.5) errors for free.

GATE 1: median(C_q) <~ 0.5 C' at the tolerance that actually delivers eps_A ~ 1e-3.

Run with:
    JOBID=<n> ./block_comp/run.sh stage1_rank_atlas.py --datasets sos2d hr2d sos3d
"""
import argparse
import sys
import warnings
import numpy as np
import torch

from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt                                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import refop                                                           # noqa: E402
import tree as T                                                       # noqa: E402
from common import (DATASETS, crop_center, gate, load_dataset,         # noqa: E402
                    pad_center, rel_err, save_results)

OUT = Path(__file__).resolve().parent / 'results'
CPRIMES = (16, 30)            # feas.md: the method's value grows with C'
TOLS = T.FROB_TOLS            # 1e-2, 1e-3, 1e-4
ENERGIES = T.ENERGY_FRACS     # 0.99, 0.999, 0.9999
LEVELS = {2: (0, 1, 2, 3, 4, 5), 3: (0, 1, 2, 3, 4)}
WEIGHTINGS = ('mask', 'maskf', 'img', 'none')
CAL_WEIGHTINGS = ('mask', 'maskf', 'none')   # swept in the calibration half
MASK_FLOOR = 1e-2             # background weight; see weight_field
EPS_A_TARGET = 1e-3           # feas.md Sec. 22 operator-accuracy target
GATE1_RATIO = 0.5             # feas.md Sec. 10.5 / Sec. 22
GATE_PROBE = 'masked'         # object-supported broadband probe; see test_images


# ---------------------------------------------------------------------------
def weight_field(ds, kind):
    """
    feas.md Sec. 10.1 voxel weights.

    ``maskf`` exists because a hard mask leaves the local basis **completely
    unconstrained outside the object**: those voxels contribute nothing to the Gram, so
    ``U_q`` is arbitrary there and truncation can destroy the maps in the background
    entirely. That does not matter for object-supported images, but it makes the
    forward error on a full-FOV random image saturate (~0.2 here) and stop responding
    to the tolerance. A small floor re-couples the background at negligible cost and
    is the honest default for an operator that will see arbitrary iterates.
    """
    if kind == 'none':
        return None
    if kind == 'mask':
        return ds.mask.float()
    if kind == 'maskf':
        return ds.mask.float().clamp(min=MASK_FLOOR)
    if kind == 'img':
        return (ds.img_ref.abs() / ds.img_ref.abs().max()).clamp(min=MASK_FLOOR)
    raise ValueError(kind)


def atlas(ds, cprime, torch_dev):
    """1a: per-level rank statistics for every weighting and tolerance."""
    dsc = ds.compress(cprime)
    S = pad_center(dsc.mps, ds.im_size_pad)
    rows, spectra, maps = [], {}, {}
    for wk in WEIGHTINGS:
        w = weight_field(dsc, wk)
        wp = None if w is None else pad_center(w[None], ds.im_size_pad)[0]
        for lvl in LEVELS[ds.d]:
            G = T.block_grams(S, lvl, wp)
            eigs = T.gram_eigs(G)
            occ = T.occupied(G)
            blk = T.block_shape(ds.im_size_pad, lvl)
            for tol in TOLS:
                r = T.rank_from_eigs(eigs, tol=tol)
                rows.append(dict(C=cprime, weight=wk, level=lvl, Q=2 ** (ds.d * lvl),
                                 block=blk, crit='frob', tol=tol,
                                 **T.rank_stats(r, occ)))
            for en in ENERGIES:
                r = T.rank_from_eigs(eigs, energy=en)
                rows.append(dict(C=cprime, weight=wk, level=lvl, Q=2 ** (ds.d * lvl),
                                 block=blk, crit='energy', tol=1 - en,
                                 **T.rank_stats(r, occ)))
            if wk == 'mask':
                maps[lvl] = T.rank_map(T.rank_from_eigs(eigs, tol=1e-3), lvl, ds.d)
                spectra[lvl] = _representative_spectra(eigs, occ, lvl, ds.d)
            del G, eigs
            torch.cuda.empty_cache()
    return rows, spectra, maps


def _representative_spectra(eigs, occ, lvl, d):
    """Singular-value decay for centre / edge / corner / body-boundary blocks."""
    s = 1 << lvl
    if s == 1:
        return {'global': eigs[0].sqrt().cpu().numpy()}
    occ_g = occ.reshape((s,) * d)
    out = {}
    mid = s // 2
    picks = {'center': (mid,) * d, 'corner': (0,) * d,
             'edge': (0,) + (mid,) * (d - 1)}
    for label, ijk in picks.items():
        q = int(np.ravel_multi_index(ijk, (s,) * d))
        out[label] = eigs[q].sqrt().cpu().numpy()
    # A body-boundary block: occupied, with at least one empty neighbour
    idx = torch.argwhere(occ_g)
    for row in idx:
        ijk = tuple(int(v) for v in row)
        nb = [tuple(min(max(ijk[a] + (dl if a == ax else 0), 0), s - 1)
                    for a in range(d)) for ax in range(d) for dl in (-1, 1)]
        if any(not bool(occ_g[n]) for n in nb):
            out['boundary'] = eigs[int(np.ravel_multi_index(ijk, (s,) * d))].sqrt().cpu().numpy()
            break
    return out


# ---------------------------------------------------------------------------
def test_images(ds, torch_dev, seed=0):
    """
    feas.md Sec. 18.1 probe images.

    ``masked`` -- broadband but object-supported -- is the one the gate uses. It is the
    strictest *fair* probe: ``gaussian`` also excites the background, where a
    mask-weighted basis makes no claim at all, and ``image`` is spectrally narrow enough
    to flatter the approximation. ``gaussian`` is retained as a diagnostic of exactly
    that background leakage.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    sh = ds.im_size
    rnd = torch.randn(sh, generator=g, dtype=torch.complex64).to(torch_dev)
    smooth = torch.fft.ifftn(torch.fft.fftn(rnd) * _lowpass(sh, torch_dev))
    return {'gaussian': rnd,
            'smooth': smooth / smooth.abs().max(),
            'masked': rnd * ds.mask,
            'image': ds.img_ref}


def _lowpass(sh, dev, frac=0.1):
    ks = [torch.fft.fftfreq(n, device=dev) for n in sh]
    r2 = sum(k.reshape([-1 if i == j else 1 for j in range(len(sh))]) ** 2
             for i, k in enumerate(ks))
    return torch.exp(-r2 / (2 * frac ** 2))


def calibrate(ds, cprime, torch_dev, args):
    """1b: the eps -> eps_A map, plus per-coil, per-|k| and normal-operator errors."""
    dsc = ds.compress(cprime)
    op = refop.SenseOp(dsc.mps, dsc.trj, oversamp=args.os, width=args.width)
    imgs = test_images(dsc, torch_dev)
    ref = {k: op.forward(v) for k, v in imgs.items()}
    ref_n = op.normal(imgs['image'])
    kmag = dsc.trj.norm(dim=-1)

    S = pad_center(dsc.mps, ds.im_size_pad)
    rows, ksp_curves = [], {}
    print(f'    {"cfg":<26} {"medC":>5} {"Rleaf":>7} {"eps_map":>9} '
          + ' '.join(f'{"eA:" + k:>10}' for k in imgs) + f' {"eps_AHA":>9}')
    for wk in CAL_WEIGHTINGS:
        w = weight_field(dsc, wk)
        wp = None if w is None else pad_center(w[None], ds.im_size_pad)[0]
        for lvl in LEVELS[ds.d]:
            G = T.block_grams(S, lvl, wp)
            eigs, occ = T.gram_eigs(G), T.occupied(G)
            for tol in TOLS:
                r = T.rank_from_eigs(eigs, tol=tol)
                mps_h = crop_center(T.truncate_field(S, lvl, r, G), ds.im_size)
                oph = op.with_maps(mps_h.contiguous())
                fwd = {k: oph.forward(v) for k, v in imgs.items()}
                errs = {k: rel_err(fwd[k], ref[k]) for k in imgs}
                e_map = rel_err(mps_h * dsc.mask, dsc.mps * dsc.mask)
                per_coil = ((fwd['image'] - ref['image']).norm(dim=-1)
                            / ref['image'].norm())
                e_norm = rel_err(oph.adjoint(fwd['image']), ref_n)
                st = T.rank_stats(r, occ)
                rows.append(dict(C=cprime, weight=wk, level=lvl,
                                 Q=2 ** (ds.d * lvl), tol=tol, eps_map=e_map,
                                 eps_normal=e_norm, eps_A=errs,
                                 per_coil_max=float(per_coil.max()), **st))
                if tol == 1e-2 and wk == 'maskf':
                    d_ksp = (fwd['image'] - ref['image']).abs().sum(0)
                    ksp_curves[lvl] = _bin_by_k(kmag, d_ksp, ref['image'].abs().sum(0))
                cfg = f'{wk} L{lvl} Q={2 ** (ds.d * lvl)} tol={tol:.0e}'
                print(f'    {cfg:<26} {st["median"]:5.0f} {st["sum"]:7.0f} '
                      f'{e_map:9.2e} '
                      + ' '.join(f'{errs[k]:10.2e}' for k in imgs)
                      + f' {e_norm:9.2e}')
                del mps_h, oph, fwd
                torch.cuda.empty_cache()
            del G, eigs
    return rows, ksp_curves


def _bin_by_k(kmag, err, ref, nbin=40):
    """feas.md Sec. 18.3: error against |k|."""
    edges = torch.linspace(0, float(kmag.max()) * 1.001, nbin + 1, device=kmag.device)
    b = torch.bucketize(kmag, edges) - 1
    num = torch.zeros(nbin, device=kmag.device).index_add_(0, b.clamp(0, nbin - 1), err)
    den = torch.zeros(nbin, device=kmag.device).index_add_(0, b.clamp(0, nbin - 1), ref)
    return (0.5 * (edges[:-1] + edges[1:])).cpu().numpy(), \
           (num / den.clamp(min=1e-30)).cpu().numpy()


# ---------------------------------------------------------------------------
def figures(name, rows, cal, spectra, maps, ksp_curves, cprime, d, tag):
    sel = [r for r in rows if r['C'] == cprime and r['crit'] == 'frob']

    # 1. rank vs block size
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for wk, ls in zip(WEIGHTINGS, ('-', '--', ':')):
        for tol, c in zip(TOLS, ('C0', 'C1', 'C2')):
            rs = sorted([r for r in sel if r['weight'] == wk and r['tol'] == tol],
                        key=lambda r: r['Q'])
            ax[0].plot([r['Q'] for r in rs], [r['median'] for r in rs], ls, color=c,
                       marker='o', ms=3, label=f'{wk} {tol:.0e}')
            ax[1].plot([r['Q'] for r in rs], [r['sum'] for r in rs], ls, color=c,
                       marker='o', ms=3)
    ax[0].axhline(GATE1_RATIO * cprime, color='k', lw=1, ls='-.',
                  label=f'gate: {GATE1_RATIO}C\'')
    ax[0].set(xscale='log', xlabel='Q', ylabel='median $C_q$',
              title=f'{name}: local rank vs block count (C\'={cprime})')
    ax[1].set(xscale='log', yscale='log', xlabel='Q',
              ylabel='$R_{leaf}=\\sum_q C_q$', title='total leaf rank')
    ax[1].axhline(cprime, color='k', lw=1, ls='-.')
    ax[0].legend(fontsize=6, ncol=3)
    for a in ax:
        a.grid(alpha=.3)
    _save(fig, f'rank_vs_block_size_{name}{tag}.png')

    # 2. singular value decay for representative blocks
    lv = sorted(spectra)[len(spectra) // 2]
    fig, ax = plt.subplots(figsize=(5, 4))
    for lab, sv in spectra[lv].items():
        ax.semilogy(sv / max(sv[0], 1e-30), marker='o', ms=3, label=lab)
    ax.set(xlabel='index', ylabel='$\\sigma_i/\\sigma_0$',
           title=f'{name}: block spectra at level {lv}')
    ax.grid(alpha=.3), ax.legend(fontsize=7)
    _save(fig, f'singular_value_examples_{name}{tag}.png')

    # 3. spatial rank map (mid-slice in 3D)
    lv = sorted(maps)[-2] if len(maps) > 1 else sorted(maps)[0]
    m = maps[lv]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(m if d == 2 else m[:, :, m.shape[2] // 2], cmap='viridis')
    fig.colorbar(im, ax=ax, label='$C_q$ at $\\epsilon=10^{-3}$')
    ax.set(title=f'{name}: rank map, level {lv} (C\'={cprime})')
    _save(fig, f'rank_map_L{lv}_{name}{tag}.png')

    # 4. rank histograms per scale
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for lvl in sorted(maps):
        ax.hist(maps[lvl].ravel(), bins=np.arange(cprime + 2) - .5, histtype='step',
                density=True, label=f'L{lvl} (Q={2 ** (d * lvl)})')
    ax.set(xlabel='$C_q$ at $\\epsilon=10^{-3}$', ylabel='density',
           title=f'{name}: rank histograms')
    ax.legend(fontsize=7), ax.grid(alpha=.3)
    _save(fig, f'rank_histograms_{name}{tag}.png')

    # 5. the calibration: rank tolerance vs actual operator error.
    # Left is the plot the gate is read off: rank reduction achieved (x) against the
    # operator error it actually costs (y). The gate needs a point in the lower-left.
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for wk, mk in zip(CAL_WEIGHTINGS, ('o', 's', '^')):
        for tol, c in zip(TOLS, ('C0', 'C1', 'C2')):
            rs = sorted([r for r in cal if r['tol'] == tol and r['weight'] == wk],
                        key=lambda r: r['Q'])
            if not rs:
                continue
            ax[0].plot([r['median'] / cprime for r in rs],
                       [r['eps_A'][GATE_PROBE] for r in rs], marker=mk, ls='-',
                       color=c, ms=4, label=f'{wk} $\\epsilon$={tol:.0e}')
            ax[1].plot([r['Q'] for r in rs], [r['eps_A'][GATE_PROBE] for r in rs],
                       marker=mk, ls='-', color=c, ms=4)
    rs = sorted([r for r in cal if r['weight'] == 'maskf'], key=lambda r: r['Q'])
    for probe, c in zip(('image', 'masked', 'gaussian'), ('C0', 'C1', 'C3')):
        for tol, ls in zip(TOLS, ('-', '--', ':')):
            sub = [r for r in rs if r['tol'] == tol]
            ax[2].plot([r['Q'] for r in sub], [r['eps_A'][probe] for r in sub],
                       ls, color=c, marker='.', label=f'{probe} {tol:.0e}')
    ax[0].axhline(EPS_A_TARGET, color='k', ls='-.', lw=1, label='target $10^{-3}$')
    ax[0].axvline(GATE1_RATIO, color='r', ls=':', lw=1, label='gate $0.5C\'$')
    ax[0].set(yscale='log', xlabel='median $C_q/C\'$',
              ylabel=f'$\\epsilon_A$ ({GATE_PROBE})',
              title=f'{name}: what rank reduction actually costs')
    ax[1].axhline(EPS_A_TARGET, color='k', ls='-.', lw=1)
    ax[1].set(xscale='log', yscale='log', xlabel='Q',
              ylabel=f'$\\epsilon_A$ ({GATE_PROBE})', title='operator error vs Q')
    ax[2].axhline(EPS_A_TARGET, color='k', ls='-.', lw=1)
    ax[2].set(xscale='log', yscale='log', xlabel='Q', ylabel='$\\epsilon_A$',
              title='probe sensitivity (weight=maskf)')
    for a in ax:
        a.grid(alpha=.3), a.legend(fontsize=5, ncol=2)
    _save(fig, f'tolerance_calibration_{name}{tag}.png')

    # 6. error vs |k|
    if ksp_curves:
        fig, ax = plt.subplots(figsize=(5, 4))
        for lvl, (k, e) in sorted(ksp_curves.items()):
            ax.semilogy(k, e, label=f'L{lvl}')
        ax.set(xlabel='$|k|$ (cycles/FOV)', ylabel='relative error',
               title=f'{name}: error vs $|k|$ ($\\epsilon=10^{{-2}}$)')
        ax.grid(alpha=.3), ax.legend(fontsize=7)
        _save(fig, f'error_vs_kmag_{name}{tag}.png')


def _save(fig, fname):
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def verdict(cal, cprime):
    """
    Gate 1, judged at the tolerance that actually delivers ``eps_A ~ 1e-3``.

    Searches every (weighting, level, tolerance) configuration and reports the deepest
    rank reduction among those whose measured operator error on the object-supported
    broadband probe meets the target -- giving the idea its best shot rather than
    judging it at one arbitrary tolerance.
    """
    ok = [r for r in cal if r['eps_A'][GATE_PROBE] <= EPS_A_TARGET and r['Q'] > 1]
    if not ok:
        best_any = min(cal, key=lambda r: r['eps_A'][GATE_PROBE]) if cal else None
        m = (f'no configuration reaches eps_A({GATE_PROBE}) <= {EPS_A_TARGET:.0e}'
             + (f'; best was {best_any["eps_A"][GATE_PROBE]:.2e} at '
                f'{best_any["weight"]} Q={best_any["Q"]} tol={best_any["tol"]:.0e}'
                if best_any else ''))
        return False, None, m
    best = min(ok, key=lambda r: r['median'])
    ratio = best['median'] / cprime
    return (ratio <= GATE1_RATIO), best, (
        f'best admissible: {best["weight"]} Q={best["Q"]} tol={best["tol"]:.0e} '
        f'median C_q={best["median"]:.0f}/{cprime} = {ratio:.2f} C\', '
        f'R_leaf={best["sum"]:.0f}, eps_A({GATE_PROBE})={best["eps_A"][GATE_PROBE]:.2e}')


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=list(DATASETS))
    ap.add_argument('--cprimes', nargs='*', type=int, default=list(CPRIMES))
    ap.add_argument('--os', type=float, default=refop.OS_DEFAULT)
    ap.add_argument('--width', type=int, default=refop.W_DEFAULT)
    ap.add_argument('--skip-calibration', action='store_true')
    ap.add_argument('--tag', type=str, default='')
    args, _ = ap.parse_known_args()

    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')
    OUT.mkdir(exist_ok=True)

    blob, verdicts = {}, []
    for name in args.datasets:
        ds = load_dataset(name, torch_dev)
        print(f'\n=== {name} ===  ' + ', '.join(f'{k}={v}' for k, v in ds.summary().items()))
        rows, spectra, maps, cals = [], {}, {}, {}
        for cp in args.cprimes:
            if cp > ds.C:
                continue
            print(f'  -- C\'={cp} atlas')
            r, sp, mp = atlas(ds, cp, torch_dev)
            rows += r
            if cp == args.cprimes[0]:
                spectra, maps = sp, mp
            if not args.skip_calibration:
                print(f'  -- C\'={cp} calibration (eps -> eps_A)')
                cals[cp], ksp = calibrate(ds, cp, torch_dev, args)
            else:
                cals[cp], ksp = [], {}
        cp0 = args.cprimes[0]
        figures(name, rows, cals.get(cp0, []), spectra, maps, ksp, cp0, ds.d, args.tag)
        passed, best, msg = verdict(cals.get(cp0, []), cp0)
        verdicts.append((name, passed, msg))
        blob[name] = dict(atlas=rows, calibration=cals, summary=ds.summary())
        del ds
        torch.cuda.empty_cache()

    detail = '\n'.join(f'  {n:>6}  {m}  [{"PASS" if p else "FAIL"}]'
                       for n, p, m in verdicts)
    passed = gate('GATE 1', all(p for _, p, _ in verdicts),
                  f'bar: median(C_q) <= {GATE1_RATIO} C\' at a tolerance delivering '
                  f'eps_A <= {EPS_A_TARGET:.0e}\n{detail}')
    save_results(OUT, f'stage1_rank_atlas{args.tag}', blob,
                 slim={k: dict(atlas=v['atlas'], calibration=v['calibration'],
                               summary=v['summary']) for k, v in blob.items()})
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
