"""
Priority 2 of math_docs/hybrid_feas.md.

Profile the dense L_0 operator: per-basis FFT time F, interpolation/spread
time G (gpu_spreadinterponly on the actual upsampfac=2 grid), leftover H,
full forward/adjoint, and reconstruction. Measure smaller-grid FFTs for the
spatial-block model. Rank branches by optimistic reconstruction speedup
after unchanged work. Stop further optimization if even a free NUFFT cannot
reach 1.2× end-to-end.

F is the FFT on the same oversampled grid the interpolator reads; G is not
"total minus an unrelated FFT".

Run with:
    paper_experiments/hybrid_feas/run.sh stage2_profile.py
    paper_experiments/hybrid_feas/run.sh stage2_profile.py --datasets coco_spiral
"""
import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import cufinufft
import numpy as np
import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from mr_recon.recons import CG_SENSE_recon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, load_dataset  # noqa: E402
from cufi_op import CufiNUFFT, factor_phase_cur, make_dense_linop  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

SPEED_GATE = 1.2
EPS = 1e-4
CG_ITERS = 20
CG_TOL = 1e-8
N_OP = 20          # operator / kernel repeats (doc: ≥20)
N_RECON = 3
FIELD_BATCH = 4
QS = (2, 4, 8)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    try:
        import cupy as cp
        cp.cuda.Device().synchronize()
    except Exception:
        pass


def _cuda_s(fn):
    _sync()
    t0 = perf_counter()
    out = fn()
    _sync()
    return perf_counter() - t0, out


def _stats(xs):
    arr = np.asarray(xs, dtype=float)
    return dict(median=float(np.median(arr)),
                p10=float(np.percentile(arr, 10)),
                p90=float(np.percentile(arr, 90)),
                n=int(arr.size))


def _repeat(fn, n, warmup=2):
    for _ in range(warmup):
        fn()
        _sync()
    times = []
    for _ in range(n):
        t, _ = _cuda_s(fn)
        times.append(t)
    return _stats(times)


def _n_os(im_size, upsampfac=2.0):
    return [int(round(upsampfac * n)) for n in im_size]


def _nspread(eps):
    if eps >= 1e-1:
        return 2
    if eps >= 1e-2:
        return 3
    if eps >= 1e-3:
        return 4
    if eps >= 1e-4:
        return 5
    if eps >= 1e-6:
        return 7
    return 9


def _pts_from_pi(trj_pi):
    d = trj_pi.shape[-1]
    pts = trj_pi.reshape(-1, d).T.contiguous()
    return [pts[i] for i in range(d)], pts.shape[1]


def _stage_plan(nufft_type, n_os, n_trans, eps, args, isign):
    """Interpolate- or spread-only plan on the oversampled grid."""
    kwargs = dict(
        n_trans=n_trans, eps=eps, isign=isign, dtype='complex64',
        gpu_spreadinterponly=1, gpu_method=1, gpu_sort=1,
        gpu_kerevalmeth=1, upsampfac=2.0, modeord=0,
    )
    try:
        plan = cufinufft.Plan(nufft_type, tuple(n_os), **kwargs)
    except RuntimeError:
        kwargs['upsampfac'] = 1.0
        kwargs['gpu_kerevalmeth'] = 0
        plan = cufinufft.Plan(nufft_type, tuple(n_os), **kwargs)
        kwargs['_fallback'] = 'upsampfac=1,kerevalmeth=0'
    plan.setpts(*args)
    return plan, kwargs


def _time_spread(n_os, n_trans, eps, args, ksp, n_reps):
    """Type-1 spread-only. Large (n_os, n_trans) can fail on this cuFINUFFT."""
    for nt in (n_trans, 8, 1):
        if nt > n_trans:
            continue
        torch.cuda.empty_cache()
        try:
            plan, kw = _stage_plan(1, n_os, nt, eps, args, isign=+1)
            t = _repeat(lambda p=plan, x=ksp[:nt].contiguous(): p.execute(x), n_reps)
            del plan
            scale = n_trans / nt
            if scale != 1:
                t = {k: (v * scale if k != 'n' else v) for k, v in t.items()}
                kw['scaled_from_n_trans'] = nt
            return t, kw
        except Exception as e:
            print(f'  spread-only n_trans={nt} n_os={n_os} failed: {e}')
    return None, {'failed': True}


def profile_kernels(trj_pi, im_size, n_trans, eps, n_reps=N_OP):
    """
    Split one batched NUFFT into FFT (F), interpolate/spread (G), leftover.

    Grid is the upsampfac=2 grid the stock CufiNUFFT plan actually uses.
    G is gpu_spreadinterponly; F is torch.fft on that same grid.
    """
    d = len(im_size)
    n_os = _n_os(im_size)
    args, M = _pts_from_pi(trj_pi)
    img = torch.randn((n_trans, *im_size), device=trj_pi.device, dtype=torch.complex64)
    ksp = torch.randn((n_trans, M), device=trj_pi.device, dtype=torch.complex64)

    nft = CufiNUFFT(im_size, eps=eps)
    # Match sense_linop: leading 1 on trj so y.reshape uses trj.shape[1:-1].
    trj_b = trj_pi[None]
    trj_size = tuple(trj_pi.shape[:-1])
    kimg = ksp.reshape(n_trans, *trj_size)
    t_fwd = _repeat(lambda: nft.forward(img, trj_b), n_reps)
    t_adj = _repeat(lambda: nft.adjoint(kimg, trj_b), n_reps)

    def _fft():
        return torch.fft.fftn(img, s=n_os, dim=tuple(range(-d, 0))).contiguous()

    def _ifft():
        grid = torch.fft.fftn(img, s=n_os, dim=tuple(range(-d, 0)))
        return torch.fft.ifftn(grid, s=n_os, dim=tuple(range(-d, 0))).contiguous()

    t_fft = _repeat(_fft, n_reps)
    grid = _fft()
    t_ifft_only = _repeat(
        lambda: torch.fft.ifftn(grid, s=n_os, dim=tuple(range(-d, 0))).contiguous(),
        n_reps)

    plan2, kw2 = _stage_plan(2, n_os, n_trans, eps, args, isign=-1)
    t_interp = _repeat(lambda: plan2.execute(grid), n_reps)
    del plan2

    t_spread, kw1 = _time_spread(n_os, n_trans, eps, args, ksp, n_reps)

    F_fwd = t_fft['median']
    G_fwd = t_interp['median']
    T_fwd = t_fwd['median']
    F_adj = t_ifft_only['median']
    T_adj = t_adj['median']
    if t_spread is None:
        # Last resort: same gather fraction as the type-2 split.
        G_adj = T_adj * G_fwd / T_fwd if T_fwd > 0 else 0.0
        t_spread = dict(median=G_adj, p10=G_adj, p90=G_adj, n=0,
                        estimated='fwd_fraction')
        print(f'  spread-only unavailable; G_adj estimated from fwd fraction '
              f'({G_adj*1e3:.2f} ms)')
    else:
        G_adj = t_spread['median']
    H_fwd = T_fwd - F_fwd - G_fwd
    H_adj = T_adj - F_adj - G_adj

    nft.clear_plans()

    return dict(
        n_os=n_os, M=M, n_trans=n_trans, nspread=_nspread(eps),
        upsampfac=2.0, plan_interp=kw2, plan_spread=kw1,
        t_nufft_fwd=t_fwd, t_nufft_adj=t_adj,
        t_fft=t_fft, t_ifft=t_ifft_only, t_interp=t_interp, t_spread=t_spread,
        F_fwd=F_fwd, G_fwd=G_fwd, H_fwd=H_fwd,
        F_adj=F_adj, G_adj=G_adj, H_adj=H_adj,
        rho_fwd=G_fwd / F_fwd if F_fwd > 0 else None,
        rho_adj=G_adj / F_adj if F_adj > 0 else None,
    )


def profile_block_ffts(im_size, n_trans, qs=QS, n_reps=N_OP, device='cuda'):
    """α_q = T_FFT(cropped grid) / T_FFT(full upsampfac=2 grid)."""
    d = len(im_size)
    n_os = _n_os(im_size)
    img_full = torch.randn((n_trans, *im_size), device=device, dtype=torch.complex64)
    t_full = _repeat(lambda: torch.fft.fftn(img_full, s=n_os, dim=tuple(range(-d, 0))), n_reps)
    rows = []
    for Q in qs:
        # Equal rectangular crops; FFT-friendly even sizes.
        side = [max(8, 2 * int(math.ceil(n / (Q ** (1 / d)) / 2))) for n in im_size]
        n_os_q = _n_os(side)
        img_q = torch.randn((n_trans, *side), device=device, dtype=torch.complex64)
        t_q = _repeat(lambda im=img_q, s=n_os_q: torch.fft.fftn(im, s=s, dim=tuple(range(-d, 0))),
                      n_reps)
        alpha = t_q['median'] / t_full['median'] if t_full['median'] > 0 else None
        pred = (1.0 / Q) * math.log(max(np.prod(n_os_q), 2)) / math.log(max(np.prod(n_os), 2))
        rows.append(dict(
            Q=Q, crop=side, n_os=n_os_q,
            t_fft=t_q, alpha=alpha, alpha_log_pred=pred,
        ))
    return dict(n_os_full=n_os, t_fft_full=t_full, blocks=rows)


def _launch_ntrans(C, L, cb, fb):
    return [(min(c0 + cb, C) - c0) * (min(l0 + fb, L) - l0)
            for c0 in range(0, C, cb)
            for l0 in range(0, L, fb)]


def _fit_fg(F, G, T_nufft, T_op, trans_scale):
    """
    Map one-launch kernel F,G onto the measured operator.

    1. Scale F+G down to the measured full-NUFFT time (fused plan is faster
       than isolated FFT + interpolate-only).
    2. Scale by the true Σ n_trans / n_trans_profiled.
    3. Cap F+G at the measured operator time so fractions stay ≤ 1.
    """
    denom = F + G
    if denom <= 0 or T_nufft <= 0:
        return 0.0, 0.0
    F1, G1 = T_nufft * F / denom, T_nufft * G / denom
    F_op, G_op = F1 * trans_scale, G1 * trans_scale
    tot = F_op + G_op
    if tot > T_op > 0:
        F_op *= T_op / tot
        G_op *= T_op / tot
    return F_op, G_op


def _optimistic_bounds(T0, n_fwd, n_adj, T_fwd, T_adj, kern, launches, L0, block_ffts):
    """
    Reconstruction-level Amdahl bounds. Unchanged work stays in the residual.

    T_fwd / T_adj already include every coil/basis launch. Kernel F,G are
    rescaled so they partition the measured NUFFT, not an isolated sum.
    """
    n_trans = kern['n_trans']
    trans_scale = sum(launches) / max(n_trans, 1)
    F_fwd_op, G_fwd_op = _fit_fg(
        kern['F_fwd'], kern['G_fwd'], kern['t_nufft_fwd']['median'], T_fwd, trans_scale)
    F_adj_op, G_adj_op = _fit_fg(
        kern['F_adj'], kern['G_adj'], kern['t_nufft_adj']['median'], T_adj, trans_scale)
    H_fwd_op = max(T_fwd - F_fwd_op - G_fwd_op, 0.0)
    H_adj_op = max(T_adj - F_adj_op - G_adj_op, 0.0)
    T_nufft_full_op = n_fwd * T_fwd + n_adj * T_adj
    T_other = max(T0 - T_nufft_full_op, 0.0)

    def _speed(T_new):
        T_new = max(T_new, 1e-9)
        return T0 / T_new, T_new

    S_free, T_free = _speed(T_other)

    # Sparse optimistic: D = L0 (same FFTs and H), k̄ → 0 (no gather).
    T_s = T_other + n_fwd * (H_fwd_op + F_fwd_op) + n_adj * (H_adj_op + F_adj_op)
    S_sparse, T_s = _speed(T_s)

    # Block optimistic: Q=8, L_b=1, measured α_q, γ_q=1 (every block still all M).
    blk8 = next((b for b in block_ffts['blocks'] if b['Q'] == 8), None)
    alpha = blk8['alpha'] if blk8 and blk8['alpha'] else 1.0 / 8
    Q, Lb = 8, 1
    fft_scale = (Q * Lb * alpha) / max(L0, 1)
    gath_scale = (Q * Lb) / max(L0, 1)
    T_b = (T_other
           + n_fwd * (H_fwd_op + fft_scale * F_fwd_op + gath_scale * G_fwd_op)
           + n_adj * (H_adj_op + fft_scale * F_adj_op + gath_scale * G_adj_op))
    S_block, T_b = _speed(T_b)

    # Hybrid optimistic: one atom per block, no gather.
    T_h = (T_other
           + n_fwd * (H_fwd_op + fft_scale * F_fwd_op)
           + n_adj * (H_adj_op + fft_scale * F_adj_op))
    S_hybrid, T_h = _speed(T_h)

    fft_recon = n_fwd * F_fwd_op + n_adj * F_adj_op
    gath_recon = n_fwd * G_fwd_op + n_adj * G_adj_op
    return dict(
        T0=T0, T_other=T_other, T_nufft_op=T_nufft_full_op,
        nufft_frac=T_nufft_full_op / T0 if T0 > 0 else None,
        gather_frac=gath_recon / T0 if T0 > 0 else None,
        fft_frac=fft_recon / T0 if T0 > 0 else None,
        F_fwd_op=F_fwd_op, G_fwd_op=G_fwd_op, H_fwd_op=H_fwd_op,
        F_adj_op=F_adj_op, G_adj_op=G_adj_op, H_adj_op=H_adj_op,
        n_launch=len(launches), trans_scale=trans_scale, L0=L0,
        Q=Q, Lb=Lb, alpha_Q8=alpha,
        fft_scale=fft_scale, gath_scale=gath_scale,
        T_free=T_free, S_free=S_free,
        T_sparse=T_s, S_sparse=S_sparse,
        T_block=T_b, S_block=S_block,
        T_hybrid=T_h, S_hybrid=S_hybrid,
        gate=SPEED_GATE,
        pass_any=max(S_sparse, S_block, S_hybrid) >= SPEED_GATE,
        pass_free=S_free >= SPEED_GATE,
        pass_sparse=S_sparse >= SPEED_GATE,
        pass_block=S_block >= SPEED_GATE,
        pass_hybrid=S_hybrid >= SPEED_GATE,
    )


def run_dataset(name, torch_dev, stage1):
    print(f'\n================ {name} ================')
    L0 = int(stage1['L0']['L'])
    T0_s1 = float(stage1['L0']['time']['median'])
    ds = load_dataset(name, torch_dev)
    C, L = ds.C, L0
    cb = max(C // 2, 1)
    launches = _launch_ntrans(C, L, cb, FIELD_BATCH)
    n_launch = len(launches)
    n_trans = max(launches)
    print(f'  L_0={L0}  C={C}  coil_batch={cb}  field_batch={FIELD_BATCH}  '
          f'launches/op={n_launch} {launches}  n_trans={n_trans}')

    spatial, temporal = factor_phase_cur(ds, L0)
    A, nft = make_dense_linop(ds, spatial, temporal, eps=EPS)
    x = torch.randn(ds.im_size, device=torch_dev, dtype=torch.complex64)
    z = ds.ksp

    # Warm plans
    _ = A.forward(x)
    _ = A.adjoint(z)
    _sync()

    t_fwd = _repeat(lambda: A.forward(x), N_OP, warmup=2)
    t_adj = _repeat(lambda: A.adjoint(z), N_OP, warmup=2)
    print(f'  A.forward  {t_fwd["median"]*1e3:8.2f} ms   '
          f'[{t_fwd["p10"]*1e3:.2f}, {t_fwd["p90"]*1e3:.2f}]')
    print(f'  A.adjoint  {t_adj["median"]*1e3:8.2f} ms   '
          f'[{t_adj["p10"]*1e3:.2f}, {t_adj["p90"]*1e3:.2f}]')

    torch.cuda.empty_cache()
    kern = profile_kernels(A.trj, ds.im_size, n_trans, EPS)
    print(f'  kernel n_os={kern["n_os"]}  nspread~{kern["nspread"]}')
    print(f'  F_fwd={kern["F_fwd"]*1e3:.2f} ms  G_fwd={kern["G_fwd"]*1e3:.2f} ms  '
          f'H_fwd={kern["H_fwd"]*1e3:.2f} ms  ρ_fwd={kern["rho_fwd"]:.3f}')
    print(f'  F_adj={kern["F_adj"]*1e3:.2f} ms  G_adj={kern["G_adj"]*1e3:.2f} ms  '
          f'H_adj={kern["H_adj"]*1e3:.2f} ms  ρ_adj={kern["rho_adj"]:.3f}')
    print(f'  full NUFFT fwd={kern["t_nufft_fwd"]["median"]*1e3:.2f} ms  '
          f'adj={kern["t_nufft_adj"]["median"]*1e3:.2f} ms')

    block_ffts = profile_block_ffts(ds.im_size, n_trans, device=torch_dev)
    print('  smaller-grid FFT (α_q = T_q / T_full):')
    for b in block_ffts['blocks']:
        print(f'    Q={b["Q"]}  crop={b["crop"]}  n_os={b["n_os"]}  '
              f'α={b["alpha"]:.3f}  log-pred={b["alpha_log_pred"]:.3f}  '
              f't={b["t_fft"]["median"]*1e3:.2f} ms')

    # Reconstruction timing at L_0 (same solver as Priority 1).
    _ = CG_SENSE_recon(A, ds.ksp, max_iter=2, max_eigen=1.0,
                       tolerance=CG_TOL, clear_gpu_mem=False, verbose=False)
    recon_times = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for _ in range(N_RECON):
        _sync()
        t0 = perf_counter()
        _ = CG_SENSE_recon(A, ds.ksp, max_iter=CG_ITERS, max_eigen=1.0,
                           tolerance=CG_TOL, clear_gpu_mem=False, verbose=False)
        _sync()
        recon_times.append(perf_counter() - t0)
    t_recon = _stats(recon_times)
    peak_mib = (torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available() else None)
    T0 = t_recon['median']
    print(f'  recon T_0={T0:.3f}s  (stage1 {T0_s1:.3f}s)  peak={peak_mib:.0f} MiB'
          if peak_mib is not None else f'  recon T_0={T0:.3f}s')

    # CG: 1 adjoint (AHb) + N normals = N fwd + (N+1) adj.
    n_fwd, n_adj = CG_ITERS, CG_ITERS + 1
    bounds = _optimistic_bounds(
        T0, n_fwd, n_adj, t_fwd['median'], t_adj['median'],
        kern, launches, L0, block_ffts)

    print(f'  NUFFT fraction of recon ≈ {bounds["nufft_frac"]:.2f}  '
          f'(FFT {bounds["fft_frac"]:.2f}, gather {bounds["gather_frac"]:.2f})')
    print(f'  unchanged work T_other={bounds["T_other"]:.3f}s')
    print(f'  optimistic S:  free-NUFFT {bounds["S_free"]:.2f}x   '
          f'sparse {bounds["S_sparse"]:.2f}x   '
          f'block {bounds["S_block"]:.2f}x   '
          f'hybrid {bounds["S_hybrid"]:.2f}x   '
          f'gate {SPEED_GATE}x')

    nft.clear_plans()
    return dict(
        name=name, L0=L0, C=C, M=ds.M, im_size=ds.im_size,
        coil_batch=cb, field_batch=FIELD_BATCH, n_launch=n_launch, n_trans=n_trans,
        t_fwd=t_fwd, t_adj=t_adj, t_recon=t_recon, T0_stage1=T0_s1,
        peak_MiB=peak_mib, kern=kern, block_ffts=block_ffts, bounds=bounds,
        cg=dict(max_iter=CG_ITERS, n_fwd=n_fwd, n_adj=n_adj),
        within_dataset=True,
    )


def _plot(results):
    names = list(results)
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.0))
    labels = ['free NUFFT', 'sparse', 'block Q=8', 'hybrid']
    keys = ['S_free', 'S_sparse', 'S_block', 'S_hybrid']
    x = np.arange(len(labels))
    width = 0.35 if len(names) > 1 else 0.5
    for i, name in enumerate(names):
        S = [results[name]['bounds'][k] for k in keys]
        ax[0].bar(x + (i - 0.5 * (len(names) - 1)) * width, S, width, label=name)
    ax[0].axhline(SPEED_GATE, color='k', ls='--', label=f'{SPEED_GATE}× gate')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=15)
    ax[0].set_ylabel('optimistic recon speedup')
    ax[0].legend(fontsize=8)
    ax[0].grid(True, axis='y', alpha=0.3)

    for name in names:
        b = results[name]['bounds']
        ax[1].barh(name, b['fft_frac'], label='FFT' if name == names[0] else None)
        ax[1].barh(name, b['gather_frac'], left=b['fft_frac'],
                   label='gather' if name == names[0] else None)
        left = b['fft_frac'] + b['gather_frac']
        ax[1].barh(name, max(1.0 - left, 0.0), left=left,
                   label='other' if name == names[0] else None)
    ax[1].set_xlabel('fraction of reconstruction time')
    ax[1].set_xlim(0, 1)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'stage2_bounds.png', dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args, _ = ap.parse_known_args()

    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}  cufinufft {cufinufft.__version__}')
    print(f'Priority 2 — profile F/G/H, optimistic {SPEED_GATE}× recon gate')
    if torch_dev.type != 'cuda':
        print('BLOCKER: no GPU.')
        return 2

    s1_path = OUT / 'stage1.json'
    if not s1_path.exists():
        print('BLOCKER: results/stage1.json missing. Run stage1_reference.py first.')
        return 2
    stage1 = json.loads(s1_path.read_text())

    dest = OUT / 'stage2.json'
    prev = {}
    if dest.exists():
        try:
            prev = json.loads(dest.read_text())
        except json.JSONDecodeError:
            prev = {}

    out = dict(prev)
    all_pass = True
    for name in args.datasets:
        if name not in stage1 or not stage1[name].get('L0'):
            print(f'BLOCKER: {name} has no L_0 in stage1. Resolve Priority 1.')
            all_pass = False
            continue
        out[name] = run_dataset(name, torch_dev, stage1[name])
        all_pass &= bool(out[name]['bounds']['pass_any'])
        torch.cuda.empty_cache()

    if any(n in out for n in args.datasets):
        _plot({k: out[k] for k in args.datasets if k in out})

    print('\n================ PRIORITY 2 ================')
    for name in args.datasets:
        if name not in out:
            print(f'  [SKIP] {name}')
            continue
        b = out[name]['bounds']
        tag = 'PASS' if b['pass_any'] else 'STOP'
        rank = sorted(
            [('sparse', b['S_sparse'], b['pass_sparse']),
             ('block', b['S_block'], b['pass_block']),
             ('hybrid', b['S_hybrid'], b['pass_hybrid'])],
            key=lambda t: -t[1])
        print(f'  [{tag}] {name:<16s}  T_0={b["T0"]:.3f}s  '
              f'NUFFT={b["nufft_frac"]:.0%}  other={b["T_other"]:.3f}s')
        print(f'        ρ_fwd={out[name]["kern"]["rho_fwd"]:.3f}  '
              f'ρ_adj={out[name]["kern"]["rho_adj"]:.3f}')
        print(f'        optimistic  free={b["S_free"]:.2f}x  '
              f'sparse={b["S_sparse"]:.2f}x  block={b["S_block"]:.2f}x  '
              f'hybrid={b["S_hybrid"]:.2f}x')
        print(f'        rank branches: ' +
              ', '.join(f'{n} {s:.2f}x{"*" if p else ""}' for n, s, p in rank))
        if not b['pass_any']:
            print('        STOP: even optimistic savings cannot reach '
                  f'{SPEED_GATE}× total reconstruction. No P3–P7.')
        elif b['S_block'] >= b['S_sparse']:
            print('        FFT-heavier: run Priority 5 (blocks) before 3–4.')
        else:
            print('        Gather-heavier: run Priority 3–4 (sparse) first.')

    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f'wrote {dest}')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
