"""
(Q, s, L) alpha-seg sweep vs global SVD baseline (math_docs/feast_test_alpha.md).

Both arms: cuFINUFFT, same preprocessing as run_sweep.py, CG max_iter=10.
Baseline uses one reused plan and the largest n_trans that fits.
NRMSE is vs a global SVD reference (coco L=32; tilt L=160 — not 4× L_svd_max).

    paper_experiments/alpha_seg/run.sh run_alpha_seg_sweep.py
    paper_experiments/alpha_seg/run.sh run_alpha_seg_sweep.py --datasets coco_spiral
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from mr_recon.recons import CG_SENSE_recon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'hybrid_feas'))
from common import load_dataset, nrmse  # noqa: E402
from cufi_op import CufiNUFFT, factor_phase_cur  # noqa: E402

from hofft.alpha_seg import decompose, q_layout
from hofft.alpha_seg_linop import AlphaSegLinop

OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)

CG_ITERS = 10          # match paper_experiments/run_sweep.py
CG_TOL = 1e-8
EPS = 1e-4
# Sec. 2 rho from hybrid stage2 (gpu_spreadinterponly / FFT on the real grid).
RHO = {'coco_spiral': 0.289, 'tilt_spi_invivo': 0.164}
OS = 2.0

BASELINE_LS = {
    'coco_spiral': (4, 8, 12, 16, 24, 32),
    'tilt_spi_invivo': (20, 40, 80, 112, 160),
}
REF_L = {
    'coco_spiral': 32,
    'tilt_spi_invivo': 160,
}
ASEG_GRID = {
    'coco_spiral': dict(
        Q=(1, 2, 4, 8, 16),
        s=(0.25, 0.5, 1.0),
        L=(2, 4, 6, 8, 12, 16),
    ),
    'tilt_spi_invivo': dict(
        Q=(1, 2, 4, 8),
        s=(0.25, 0.5, 1.0),
        L=(8, 16, 32, 48),
    ),
}
N_RECON = {'coco_spiral': 2, 'tilt_spi_invivo': 1}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_mib():
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / 2**20


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _stats(xs):
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return dict(median=float('nan'), p10=float('nan'), p90=float('nan'), n=0)
    return dict(median=float(np.median(arr)),
                p10=float(np.percentile(arr, 10)),
                p90=float(np.percentile(arr, 90)), n=int(arr.size))


def _max_n_trans(im_size, mem_bytes=12 * (1 << 30)) -> int:
    os_n = int(np.prod([int(OS * n) for n in im_size]))
    per = 16 * os_n  # complex64 + workspace
    return max(1, int(mem_bytes // per))


def predicted_speedup(L, Q, s, L_svd, N, rho):
    """Sec. 2. N is voxel count."""
    lam = math.log(OS ** 2 * N / max(Q, 1)) / math.log(OS ** 2 * N)
    den = L * lam / L_svd + rho * s * Q * L / L_svd
    return (1 + rho) / max(den, 1e-12), lam


class BaselineDenseLinop:
    """Global SVD sense: one trajectory, largest feasible n_trans, plan reuse."""

    def __init__(self, ds, spatial, temporal, eps=EPS):
        self.ds = ds
        self.spatial = spatial
        self.temporal = temporal.reshape(spatial.shape[0], -1)
        self.mps = ds.mps
        self.dcf = ds.dcf
        self.L = spatial.shape[0]
        self.C = ds.C
        self.im_size = ds.im_size
        self.ishape = ds.im_size
        self.oshape = (ds.C, *ds.trj_size)
        self.M = ds.M
        self.nft = CufiNUFFT(ds.im_size, eps=eps)
        self.trj_pi = self.nft.rescale_trajectory(ds.trj)[None].contiguous()
        want = self.L * self.C
        self.n_trans = min(want, _max_n_trans(ds.im_size))
        self.n_trans = max(1, self.n_trans)
        self.plan_setup_s = 0.0
        # Touch plans once so setup is not inside the timed recon.
        t0 = perf_counter()
        dummy = torch.zeros((self.n_trans, *ds.im_size), device=ds.trj.device,
                            dtype=torch.complex64)
        _ = self.nft.forward(dummy, self.trj_pi)
        dummy_k = torch.zeros((self.n_trans, *ds.trj_size), device=ds.trj.device,
                              dtype=torch.complex64)
        _ = self.nft.adjoint(dummy_k, self.trj_pi)
        _sync()
        self.plan_setup_s = perf_counter() - t0

    def forward(self, img):
        L, C = self.L, self.C
        ksp = torch.zeros(self.oshape, device=img.device, dtype=img.dtype)
        ksp_f = ksp.reshape(C, self.M)
        pairs = [(c, l) for c in range(C) for l in range(L)]
        nt = self.n_trans
        src = torch.empty((nt, *self.im_size), device=img.device, dtype=img.dtype)
        for i0 in range(0, len(pairs), nt):
            chunk = pairs[i0:i0 + nt]
            n = len(chunk)
            for b, (c, l) in enumerate(chunk):
                src[b] = self.mps[c] * self.spatial[l] * img
            if n < nt:
                src[n:].zero_()
            y = self.nft.forward(src, self.trj_pi).reshape(nt, self.M)
            for b, (c, l) in enumerate(chunk):
                ksp_f[c] += y[b] * self.temporal[l]
        return ksp

    def adjoint(self, ksp):
        L, C = self.L, self.C
        img = torch.zeros(self.im_size, device=ksp.device, dtype=ksp.dtype)
        zf = (ksp * self.dcf).reshape(C, self.M)
        pairs = [(c, l) for c in range(C) for l in range(L)]
        nt = self.n_trans
        ck = torch.empty((nt, self.M), device=ksp.device, dtype=ksp.dtype)
        for i0 in range(0, len(pairs), nt):
            chunk = pairs[i0:i0 + nt]
            n = len(chunk)
            ck.zero_()
            for b, (c, l) in enumerate(chunk):
                ck[b] = zf[c] * self.temporal[l].conj()
            x = self.nft.adjoint(ck.reshape(nt, *self.ds.trj_size), self.trj_pi)
            x = x.reshape(nt, *self.im_size)
            for b, (c, l) in enumerate(chunk):
                img += x[b] * self.mps[c].conj() * self.spatial[l].conj()
        return img

    def normal(self, img):
        return self.adjoint(self.forward(img))

    def clear(self):
        self.nft.clear_plans()


def _recon(A, ksp, n_reps, warmup=True):
    if warmup:
        _ = CG_SENSE_recon(A, ksp, max_iter=2, max_eigen=1.0,
                           tolerance=CG_TOL, clear_gpu_mem=False, verbose=False)
    times, img = [], None
    for _ in range(n_reps):
        _sync()
        t0 = perf_counter()
        img = CG_SENSE_recon(A, ksp, max_iter=CG_ITERS, max_eigen=1.0,
                             tolerance=CG_TOL, clear_gpu_mem=False, verbose=False)
        _sync()
        times.append(perf_counter() - t0)
    return img, _stats(times)


def _save(path, obj):
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def run_dataset(name, torch_dev, skip_baseline=False, skip_aseg=False, quick=False):
    print(f'\n================ {name} ================', flush=True)
    ds = load_dataset(name, torch_dev)
    print(f'  grid {ds.im_size}  C={ds.C}  M={ds.M}  B={ds.B}  R={ds.R}',
          flush=True)
    out = dict(name=name, im_size=list(ds.im_size), C=ds.C, M=ds.M, B=ds.B,
               R=ds.R, cg_iters=CG_ITERS, eps=EPS, ref_L=REF_L[name],
               baseline=[], aseg=[], notes=[])
    out_path = OUT / f'sweep_{name}.json'
    prev = _load_json(out_path)
    done_base = {r['L'] for r in (prev or {}).get('baseline', [])}
    done_aseg = {(r['Q'], r['s'], r['L']) for r in (prev or {}).get('aseg', [])}
    if prev:
        out['baseline'] = prev.get('baseline', [])
        out['aseg'] = prev.get('aseg', [])
        out['notes'] = prev.get('notes', [])
        print(f'  resume: {len(done_base)} baseline, {len(done_aseg)} aseg',
              flush=True)

    ref_path = OUT / f'ref_{name}.pt'
    if ref_path.exists():
        img_ref = torch.load(ref_path, map_location=torch_dev, weights_only=True)
        print(f'  loaded reference {ref_path}', flush=True)
    else:
        Lref = REF_L[name]
        print(f'  factorizing reference SVD L={Lref}', flush=True)
        _sync()
        t0 = perf_counter()
        spat, temp = factor_phase_cur(ds, Lref)
        t_fac = perf_counter() - t0
        A = BaselineDenseLinop(ds, spat, temp, eps=EPS)
        img_ref, st = _recon(A, ds.ksp, n_reps=1)
        A.clear()
        torch.save(img_ref, ref_path)
        out['notes'].append(
            f'reference = global CUR+SVD L={Lref} (not dense matvec; '
            f'tilt 4x L_svd_max is too expensive). factor={t_fac:.2f}s '
            f'recon={st["median"]:.2f}s n_trans={A.n_trans}')
        print(f'  ref done  factor={t_fac:.2f}s  recon={st["median"]:.2f}s',
              flush=True)
        _save(out_path, out)

    mask = ds.mask
    Ls = list(BASELINE_LS[name])
    if quick:
        Ls = Ls[::2]
    if not skip_baseline:
        for L in Ls:
            if L in done_base:
                print(f'  skip baseline L={L}', flush=True)
                continue
            print(f'  baseline L_svd={L}', flush=True)
            try:
                _reset_peak()
                _sync()
                t0 = perf_counter()
                spat, temp = factor_phase_cur(ds, L)
                t_fac = perf_counter() - t0
                A = BaselineDenseLinop(ds, spat, temp, eps=EPS)
                t_plan = A.plan_setup_s
                img, st = _recon(A, ds.ksp, n_reps=N_RECON[name])
                e = nrmse(img, img_ref, mask)
                em = nrmse(img.abs(), img_ref.abs(), mask)
                rec = dict(
                    arm='baseline', L=L, S_tot=L, Q=1, s=1.0, k=L,
                    nrmse=e, nrmse_mag=em,
                    t_decomp=t_fac + t_plan,
                    t_decomp_itemized=dict(cur_svd=t_fac, plan_setup=t_plan),
                    t_recon=st, t_total=t_fac + t_plan + st['median'],
                    t_per_iter=st['median'] / CG_ITERS,
                    peak_MiB=_peak_mib(),
                    n_trans=A.n_trans,
                    storage=dict(spatial=L * ds.N, temporal=L * ds.M),
                )
                A.clear()
                del A, spat, temp, img
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                out['baseline'].append(rec)
                _save(out_path, out)
                print(f'    E={e:.4f}  recon={st["median"]:.3f}s  '
                      f'total={rec["t_total"]:.3f}s', flush=True)
            except Exception as exc:
                print(f'    FAIL L={L}: {exc}', flush=True)
                out['notes'].append(f'baseline L={L} failed: {exc}')
                _save(out_path, out)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    grid = ASEG_GRID[name]
    Qs, ss, Ls_a = grid['Q'], grid['s'], grid['L']
    if quick:
        Qs, ss, Ls_a = Qs[:3], (0.5, 1.0), Ls_a[::2]
    L_svd_max = max(BASELINE_LS[name])
    if skip_aseg:
        _save(out_path, out)
        return out

    for Q in Qs:
        for s in ss:
            for L in Ls_a:
                k = min(L, max(1, math.ceil(s * L)))
                if s < 1.0 and k >= L:
                    continue
                if Q * L > 4 * L_svd_max:
                    continue
                key = (Q, float(s), L)
                if key in done_aseg or (Q, s, L) in done_aseg:
                    print(f'  skip aseg Q={Q} s={s} L={L}', flush=True)
                    continue
                print(f'  aseg Q={Q} s={s} L={L}  S_tot={Q*L} k={k}', flush=True)
                try:
                    _reset_peak()
                    _sync()
                    t0 = perf_counter()
                    model = decompose(
                        ds.phis, ds.alphas, ds.mask, ds.trj,
                        q_layout(Q), L, s=s,
                        reduced_im_size=ds.reduced_im_size, seed=0)
                    t_decomp = perf_counter() - t0
                    A = AlphaSegLinop(model, ds.mps, ds.dcf, ds.trj_size, eps=EPS)
                    model.times['plan_setup'] = A.plan_setup_s
                    model.times['decomp'] = t_decomp
                    t_all_decomp = t_decomp + A.plan_setup_s
                    img, st = _recon(A, ds.ksp, n_reps=N_RECON[name])
                    e = nrmse(img, img_ref, mask)
                    em = nrmse(img.abs(), img_ref.abs(), mask)
                    pred, lam = predicted_speedup(
                        L, model.Q, s, L_svd_max, ds.N, RHO[name])
                    rec = dict(
                        arm='alpha_seg', Q=model.Q, Q_req=Q, s=s, L=L, k=model.k,
                        S_tot=model.S_tot, V=list(model.V),
                        live_axes=model.live_axes,
                        n_patterns=model.n_patterns, mean_run=model.mean_run,
                        nrmse=e, nrmse_mag=em,
                        t_decomp=t_all_decomp,
                        t_decomp_itemized=dict(model.times),
                        t_recon=st, t_total=t_all_decomp + st['median'],
                        t_per_iter=st['median'] / CG_ITERS,
                        peak_MiB=_peak_mib(),
                        n_cufi_samples=A.n_cufi_samples,
                        expect_samples=model.Q * model.k * ds.M,
                        storage=dict(
                            atoms=model.S_tot * int(np.prod(model.V)),
                            temporal=model.Q * model.k * ds.M),
                        pred_speedup_vs_Lmax=pred, lam=lam,
                        fwd_split=getattr(A, 'last_fwd_split', {}),
                    )
                    A.clear()
                    del A, model, img
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    out['aseg'].append(rec)
                    _save(out_path, out)
                    print(f'    E={e:.4f}  decomp={t_all_decomp:.2f}s  '
                          f'recon={st["median"]:.3f}s  S_tot={rec["S_tot"]}',
                          flush=True)
                except Exception as exc:
                    print(f'    FAIL Q={Q} s={s} L={L}: {exc}', flush=True)
                    out['notes'].append(f'aseg Q={Q} s={s} L={L} failed: {exc}')
                    _save(out_path, out)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    _save(out_path, out)
    return out


def _pareto(xs, ys):
    """Lower y is better; return sorted undominated points."""
    pts = sorted(zip(xs, ys))
    front, best = [], float('inf')
    for x, y in pts:
        if y < best:
            front.append((x, y))
            best = y
    return front


def make_plots(names):
    rows = []
    for name in names:
        p = OUT / f'sweep_{name}.json'
        if p.exists():
            rows.append(json.loads(p.read_text()))
    if not rows:
        print('no sweep json to plot')
        return

    # 1. NRMSE vs total time
    fig, axes = plt.subplots(1, len(rows), figsize=(6.2 * len(rows), 5.0),
                             squeeze=False)
    cmap = {1: 'C0', 2: 'C1', 4: 'C2', 8: 'C3', 16: 'C4'}
    mark = {0.125: 'o', 0.25: 's', 0.5: 'D', 1.0: '^'}
    for ax, rec in zip(axes[0], rows):
        b = rec['baseline']
        if b:
            bt = [r['t_total'] for r in b]
            be = [r['nrmse'] for r in b]
            ax.loglog(bt, be, 'k-o', label='SVD baseline', ms=5)
        for r in rec['aseg']:
            ax.loglog(r['t_total'], r['nrmse'],
                      color=cmap.get(r['Q'], 'C5'),
                      marker=mark.get(r['s'], 'x'),
                      ms=6, linestyle='none',
                      label=None)
        xs = [r['t_total'] for r in rec['aseg']] + [r['t_total'] for r in b]
        ys = [r['nrmse'] for r in rec['aseg']] + [r['nrmse'] for r in b]
        if xs:
            front = _pareto(xs, ys)
            ax.loglog([p[0] for p in front], [p[1] for p in front],
                      'k--', lw=0.8, alpha=0.6, label='Pareto')
        ax.set_xlabel('total time [s]')
        ax.set_ylabel('NRMSE')
        ax.set_title(rec['name'])
        ax.grid(True, which='both', ls=':', alpha=0.5)
    # legend proxies
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color='k', marker='o', label='SVD')]
    for Q, c in cmap.items():
        handles.append(Line2D([0], [0], color=c, marker='o', linestyle='none',
                              label=f'Q={Q}'))
    for s, m in mark.items():
        handles.append(Line2D([0], [0], color='k', marker=m, linestyle='none',
                              label=f's={s}'))
    axes[0, -1].legend(handles=handles, fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(OUT / 'pareto.png', dpi=140)
    fig.savefig(OUT / 'nrmse_vs_total.png', dpi=140)
    plt.close(fig)

    # 2. NRMSE vs recon time
    fig, axes = plt.subplots(1, len(rows), figsize=(6.2 * len(rows), 5.0),
                             squeeze=False)
    for ax, rec in zip(axes[0], rows):
        b = rec['baseline']
        if b:
            ax.loglog([r['t_recon']['median'] for r in b],
                      [r['nrmse'] for r in b], 'k-o', label='SVD', ms=5)
        for r in rec['aseg']:
            ax.loglog(r['t_recon']['median'], r['nrmse'],
                      color=cmap.get(r['Q'], 'C5'),
                      marker=mark.get(r['s'], 'x'),
                      ms=6, linestyle='none')
        ax.set_xlabel('recon time [s]')
        ax.set_ylabel('NRMSE')
        ax.set_title(rec['name'])
        ax.grid(True, which='both', ls=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / 'nrmse_vs_recon.png', dpi=140)
    plt.close(fig)

    # 3. stacked decomp vs recon at three matched-NRMSE points per dataset
    fig, axes = plt.subplots(1, len(rows), figsize=(6.4 * len(rows), 4.6),
                             squeeze=False)
    targets = (0.05, 0.02, 0.01)
    for ax, rec in zip(axes[0], rows):
        labels, decomp, recon = [], [], []
        for tau in targets:
            bb = _nearest(rec['baseline'], tau)
            aa = _nearest(rec['aseg'], tau)
            if bb:
                labels.append(f'SVD L={bb["L"]}\n@~{tau}')
                decomp.append(bb['t_decomp'])
                recon.append(bb['t_recon']['median'])
            if aa:
                labels.append(f'Q={aa["Q"]} s={aa["s"]} L={aa["L"]}\n@~{tau}')
                decomp.append(aa['t_decomp'])
                recon.append(aa['t_recon']['median'])
        if not labels:
            ax.set_title(rec['name'] + ' (no points)')
            continue
        x = np.arange(len(labels))
        ax.bar(x, decomp, label='decomp')
        ax.bar(x, recon, bottom=decomp, label='recon')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('time [s]')
        ax.set_title(rec['name'])
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'stacked_decomp_recon.png', dpi=140)
    plt.close(fig)

    # 4. S_tot and predicted vs measured speedup
    fig, axes = plt.subplots(1, len(rows), figsize=(6.2 * len(rows), 5.0),
                             squeeze=False)
    for ax, rec in zip(axes[0], rows):
        if not rec['baseline'] or not rec['aseg']:
            continue
        for r in rec['aseg']:
            bb = _nearest(rec['baseline'], r['nrmse'])
            if not bb or bb['t_recon']['median'] <= 0:
                continue
            meas = bb['t_recon']['median'] / r['t_recon']['median']
            pred, _ = predicted_speedup(
                r['L'], r['Q'], r['s'], bb['L'],
                int(np.prod(rec['im_size'])), RHO.get(rec['name'], 0.2))
            ax.scatter(r['S_tot'], meas, c=cmap.get(r['Q'], 'C5'),
                       marker=mark.get(r['s'], 'x'), s=40)
            ax.scatter(r['S_tot'], pred, facecolors='none',
                       edgecolors=cmap.get(r['Q'], 'C5'),
                       marker=mark.get(r['s'], 'x'), s=40)
        ax.axhline(1.0, color='k', ls=':', lw=0.8)
        ax.set_xlabel(r'$S_{\mathrm{tot}}=Q L$')
        ax.set_ylabel('recon speedup vs nearest SVD')
        ax.set_title(rec['name'] + '  filled=meas, open=pred')
        ax.set_xscale('log')
        ax.grid(True, which='both', ls=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / 'stot_speedup.png', dpi=140)
    plt.close(fig)
    print('wrote', OUT / 'pareto.png', OUT / 'nrmse_vs_recon.png',
          OUT / 'stacked_decomp_recon.png', OUT / 'stot_speedup.png')


def _nearest(rows, tau):
    if not rows:
        return None
    return min(rows, key=lambda r: abs(math.log(max(r['nrmse'], 1e-12))
                                       - math.log(max(tau, 1e-12))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['coco_spiral', 'tilt_spi_invivo'])
    ap.add_argument('--skip-baseline', action='store_true')
    ap.add_argument('--skip-aseg', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    if args.plot_only:
        make_plots(args.datasets)
        return
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={torch_dev}  CG={CG_ITERS}  eps={EPS}', flush=True)
    for name in args.datasets:
        run_dataset(name, torch_dev, skip_baseline=args.skip_baseline,
                    skip_aseg=args.skip_aseg, quick=args.quick)
    make_plots(args.datasets)
    print('done', flush=True)


if __name__ == '__main__':
    main()
