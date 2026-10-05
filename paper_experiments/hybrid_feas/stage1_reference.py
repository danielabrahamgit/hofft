"""
Priority 1 of math_docs/hybrid_feas.md.

Converge a dense-rank / cuFINUFFT-eps reference, freeze τ, and pick L_0 as the
fastest validated dense configuration with image NRMSE ≤ τ. Same solver,
mask, and scaling for every rank. One volume per dataset: within-dataset
feasibility, not held-out validation.

Run with:
    paper_experiments/hybrid_feas/run.sh stage1_reference.py
    paper_experiments/hybrid_feas/run.sh stage1_reference.py --datasets coco_spiral
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from mr_recon.recons import CG_SENSE_recon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, load_dataset, nrmse  # noqa: E402
from cufi_op import factor_phase_cur, make_dense_linop  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

TAU = 0.01                 # engineering default 1%
TAU_FRAC = 0.1             # successive-refinement and NUFFT-eps budget
CG_ITERS = 20
CG_TOL = 1e-8
EPS_LOOSE = 1e-4
EPS_TIGHT = 1e-6

# Coarse rank grids; refined only if the frontier sits at an edge.
LS = {
    'coco_spiral':     (4, 8, 12, 16, 24, 32),
    # First pass at 80 was not 0.1τ-stable (Δ(64,80)=2.45%). Higher ranks needed.
    'tilt_spi_invivo': (80, 112, 144, 176, 208, 256),
}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _recon(ds, spatial, temporal, eps, max_eigen=1.0, nufft_out=None):
    A, nft = make_dense_linop(ds, spatial, temporal, eps=eps)
    img = CG_SENSE_recon(
        A, ds.ksp, max_iter=CG_ITERS, max_eigen=max_eigen,
        tolerance=CG_TOL, clear_gpu_mem=False, verbose=False)
    if nufft_out is not None:
        nufft_out.append(nft)
    else:
        nft.clear_plans()
    return img


def _time_recon(ds, spatial, temporal, eps, max_eigen, n_reps=3):
    A, nft = make_dense_linop(ds, spatial, temporal, eps=eps)
    # warmup
    _ = CG_SENSE_recon(A, ds.ksp, max_iter=2, max_eigen=max_eigen,
                       tolerance=CG_TOL, clear_gpu_mem=False, verbose=False)
    times = []
    img = None
    for _ in range(n_reps):
        _sync()
        t0 = perf_counter()
        img = CG_SENSE_recon(
            A, ds.ksp, max_iter=CG_ITERS, max_eigen=max_eigen,
            tolerance=CG_TOL, clear_gpu_mem=False, verbose=False)
        _sync()
        times.append(perf_counter() - t0)
    nft.clear_plans()
    arr = np.array(times)
    return img, dict(median=float(np.median(arr)), p10=float(np.percentile(arr, 10)),
                     p90=float(np.percentile(arr, 90)), n=n_reps, iters=CG_ITERS)


def run_dataset(name, torch_dev, n_time_reps=3, ls=None):
    print(f'\n================ {name} ================', flush=True)
    ds = load_dataset(name, torch_dev)
    mask = ds.mask
    print(f'  grid {ds.im_size}  C={ds.C}  M={ds.M}  B={ds.B}  '
          f'(within-dataset, one volume)', flush=True)

    Ls = tuple(ls) if ls is not None else LS[name]
    print('  factorizing ranks', Ls)
    factors = {}
    t_setup = {}
    for L in Ls:
        _sync()
        t0 = perf_counter()
        factors[L] = factor_phase_cur(ds, L)
        _sync()
        t_setup[L] = perf_counter() - t0
        print(f'    CUR+SVD L={L:<3d}  setup {t_setup[L]:.2f}s', flush=True)

    # Reference: largest L at tight eps. Then check L-1 and loose eps.
    L_hi = Ls[-1]
    print(f'  building reference at L={L_hi} eps={EPS_TIGHT}', flush=True)
    img_ref = _recon(ds, *factors[L_hi], EPS_TIGHT)
    den = float((img_ref * mask).norm())
    if den <= 0:
        raise RuntimeError('reference norm is empty')

    img_looser_L = _recon(ds, *factors[Ls[-2]], EPS_TIGHT)
    img_looser_eps = _recon(ds, *factors[L_hi], EPS_LOOSE)
    dL = nrmse(img_looser_L, img_ref, mask)
    dE = nrmse(img_looser_eps, img_ref, mask)
    print(f'  ref stability: Δ(L={Ls[-2]} vs {L_hi}) = {dL:.4f}   '
          f'Δ(eps {EPS_LOOSE} vs {EPS_TIGHT}) = {dE:.4f}   '
          f'budget 0.1τ = {TAU_FRAC * TAU:.4f}', flush=True)
    ref_ok = (dL <= TAU_FRAC * TAU) and (dE <= TAU_FRAC * TAU)
    if not ref_ok:
        print('  WARNING: reference not yet stable at 0.1τ. '
              'Do not loosen τ. Sweep will still run so we can see the curve.',
              flush=True)

    # Dense rank sweep at the operating eps (loose is the stock MRI default;
    # we time it because it is the candidate baseline). Tight-eps used only
    # for the reference.
    rows = []
    print(f'\n  dense sweep at eps={EPS_LOOSE}, τ={TAU}')
    print(f'  {"L":>4} {"E":>8} {"|E|":>8} {"t_med":>8} {"t_p10":>8} {"t_p90":>8} {"setup":>7}')
    for L in Ls:
        img, t = _time_recon(ds, *factors[L], EPS_LOOSE, max_eigen=1.0,
                             n_reps=n_time_reps)
        E = nrmse(img, img_ref, mask)
        Emag = nrmse(img.abs(), img_ref.abs(), mask)
        rows.append(dict(L=L, E=E, E_mag=Emag, time=t, setup=t_setup[L],
                         pass_tau=E <= TAU))
        print(f'  {L:4d} {E:8.4f} {Emag:8.4f} {t["median"]:8.3f} '
              f'{t["p10"]:8.3f} {t["p90"]:8.3f} {t_setup[L]:7.2f}', flush=True)
        # keep the image only for the passing L_0 / last
        if L == L_hi:
            img_hi = img

    passing = [r for r in rows if r['pass_tau']]
    if passing:
        L0 = min(passing, key=lambda r: r['time']['median'])
        print(f'  L_0 = {L0["L"]}  E={L0["E"]:.4f}  t={L0["time"]["median"]:.3f}s')
    else:
        L0 = None
        print('  no dense rank reached τ. Resolve accuracy before timing candidates.')

    # Also score vs the existing expanded-encoding image if present (diagnostic)
    E_ee = None
    if ds.img_ee is not None:
        E_ee = nrmse(img_ref, ds.img_ee, mask)
        print(f'  ||x_ref - img_ee|| / ||img_ee|| = {E_ee:.4f}  (diagnostic)')

    # Save images for the report
    sl = ds.im_size[0] // 2
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
    ax[0].imshow(img_ref.abs().cpu(), cmap='gray')
    ax[0].set_title(f'ref L={L_hi} eps={EPS_TIGHT}')
    ax[1].imshow(img_hi.abs().cpu(), cmap='gray')
    ax[1].set_title(f'L={L_hi} eps={EPS_LOOSE}')
    err = ((img_hi - img_ref) * mask).abs().cpu()
    im = ax[2].imshow(err, cmap='magma')
    ax[2].set_title(f'|Δ|  E={rows[-1]["E"]:.3f}')
    for a in ax:
        a.axis('off')
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / f'stage1_{name}_images.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot([r['time']['median'] for r in rows], [r['E'] for r in rows], 'o-')
    ax.axhline(TAU, color='k', ls='--', label=f'τ={TAU}')
    if L0:
        ax.scatter([L0['time']['median']], [L0['E']], s=80, zorder=3, label=f'L_0={L0["L"]}')
    ax.set_xlabel('reconstruction time [s]')
    ax.set_ylabel('complex image NRMSE vs ref')
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f'stage1_{name}_nrmse_vs_time.png', dpi=130)
    plt.close(fig)

    torch.save(dict(img_ref=img_ref.cpu(), mask=mask.cpu()),
               OUT / f'stage1_{name}_ref.pt')

    return dict(
        name=name, im_size=ds.im_size, C=ds.C, M=ds.M, B=ds.B,
        tau=TAU, L_hi=L_hi, dL=dL, dE=dE, ref_ok=ref_ok,
        rows=rows, L0=L0, E_ee=E_ee,
        cg=dict(max_iter=CG_ITERS, tol=CG_TOL, max_eigen=1.0),
        eps_ref=EPS_TIGHT, eps_op=EPS_LOOSE,
        within_dataset=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--ls', type=int, nargs='+', default=None,
                    help='Override the rank grid for every requested dataset.')
    args, _ = ap.parse_known_args()

    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')
    print(f'Priority 1 — dense reference, τ={TAU}')
    if torch_dev.type != 'cuda':
        print('BLOCKER: no GPU.')
        return 2

    dest = OUT / 'stage1.json'
    prev = {}
    if dest.exists():
        try:
            prev = json.loads(dest.read_text())
        except json.JSONDecodeError:
            prev = {}

    out = dict(prev)
    all_ok = True
    for name in args.datasets:
        out[name] = run_dataset(name, torch_dev, n_time_reps=args.reps, ls=args.ls)
        all_ok &= bool(out[name]['L0']) and bool(out[name]['ref_ok'])
        torch.cuda.empty_cache()

    print('\n================ PRIORITY 1 ================')
    for name, res in out.items():
        tag = 'PASS' if res['L0'] else 'FAIL'
        ref = 'stable' if res['ref_ok'] else 'UNSTABLE'
        L0 = res['L0']
        line = f'  [{tag}] {name:<16s} ref {ref}  ΔL={res["dL"]:.4f}  Δeps={res["dE"]:.4f}'
        if L0:
            line += (f'  L_0={L0["L"]}  E={L0["E"]:.4f}  '
                     f't={L0["time"]["median"]:.3f}s')
        print(line)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f'wrote {dest}')
    if not all_ok:
        print('Priority 1: reference unstable or no dense config with E≤τ. '
              'Resolve accuracy before timing candidates.')
        return 1
    print('Priority 1 PASS on requested datasets.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
