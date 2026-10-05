"""
Sec. 4.5 cost-model-only estimate for coco_7t_spi. No reconstructions.

    paper_experiments/alpha_seg/run.sh stage3d_estimate.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from time import perf_counter

import cufinufft
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)

from hofft.alpha_seg import _nearest_k, _sigma_sqrt, q_layout
from hofft.spatial_init import k_alpha_selection
from hofft.utils import gen_grd, reduce_spatial

sys.path.insert(0, str(HERE.parent / 'qblock_feas'))
from blocks import fit_block_affine, make_blocks  # noqa: E402

NAME = 'coco_7t_spi'
IM = (320, 320, 320)
C = 16
OS = 2.0
EPS = 1e-4
RHO_FALLBACK = 0.2
N_FPS = 768
TARGET_E = 1e-2


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _plan_kwargs():
    return dict(eps=EPS, dtype='complex64', gpu_method=1, gpu_sort=1,
                gpu_kerevalmeth=1, upsampfac=OS, modeord=0)


def _bench_type2(V, M, n_trans=1, n_rep=5):
    """Median type-2 time on a V-grid with M random points, n_trans images."""
    d = len(V)
    dev = torch.device('cuda')
    pts = (torch.rand(M, d, device=dev) * 2 - 1) * (math.pi - 1e-3)
    args = [pts[:, i].contiguous() for i in range(d)]
    kw = _plan_kwargs()
    try:
        plan = cufinufft.Plan(2, V, n_trans=n_trans, isign=-1, **kw)
    except RuntimeError as exc:
        return dict(ok=False, error=str(exc), V=list(V), M=M, n_trans=n_trans)
    plan.setpts(*args)
    img = torch.randn((n_trans, *V) if n_trans > 1 else V,
                      device=dev, dtype=torch.complex64)
    if n_trans == 1:
        img = img.reshape(*V)
    def run():
        return plan.execute(img.contiguous())
    run()
    _sync()
    ts = []
    for _ in range(n_rep):
        _sync()
        t0 = perf_counter()
        run()
        _sync()
        ts.append(perf_counter() - t0)
    arr = np.asarray(ts)
    return dict(ok=True, V=list(V), M=M, n_trans=n_trans,
                median=float(np.median(arr)),
                p10=float(np.percentile(arr, 10)),
                p90=float(np.percentile(arr, 90)))


def _load_phase(dev):
    fpath = Path('./data') / NAME
    kw = dict(weights_only=True, map_location=dev)
    phis = torch.load(fpath / 'phis.pt', **kw).float()
    alphas = torch.load(fpath / 'alphas.pt', **kw).float()
    evals = torch.load(fpath / 'evals.pt', **kw).float()
    trj = torch.load(fpath / 'trj.pt', **kw).float()
    mps = torch.load(fpath / 'mps.pt', **kw)
    mask = (evals > 0.9).float()
    return phis, alphas, mask, trj, mps


def _est_rank(phis, alphas, mask, Q_per_axis, n_fps=N_FPS, Ls=(4, 8, 16, 32, 48, 64)):
    """
    Dense-P NRMSE on a reduced grid using FPS analytic atoms + dense LS.
    Returns per-block live axes and E(L) so we can pick L at TARGET_E.
    """
    from hofft.alpha_seg import decompose
    red = tuple(max(32, n // 8) for n in phis.shape[1:])
    # subsample times for the estimate
    B = alphas.shape[0]
    af = alphas.reshape(B, -1)
    M = af.shape[1]
    idx = torch.linspace(0, M - 1, min(n_fps, M)).long().to(af.device)
    a_s = af[:, idx]
    # fake a short trajectory for decompose (not used for E(P))
    trj_s = torch.zeros(idx.numel(), len(phis.shape) - 1, device=phis.device)
    out = []
    for L in Ls:
        try:
            model = decompose(phis, a_s.reshape(B, -1), mask, trj_s,
                              Q_per_axis, L, s=1.0,
                              reduced_im_size=red, seed=0)
        except Exception as exc:
            out.append(dict(L=L, error=str(exc)))
            continue
        # ||P - B H|| on reduced residual of block 0, plus mean over blocks
        errs = []
        for blk in model.blocks:
            # reconstruct residual P on the reduced window already in Gram
            # use full-res window subsampled
            keep = blk.keep
            b = blk.b.reshape(L, -1)
            h = blk.h.to(torch.complex64)
            # skip huge P; report live axes + decomp time only if too big
            n_keep = int(keep.sum())
            if n_keep * idx.numel() > 8e6:
                continue
            # not forming P; report decomp quality via residual Gram solve
        out.append(dict(
            L=L, Q=model.Q, S_tot=model.S_tot, live_axes=model.live_axes,
            t_decomp=model.times, n_patterns=model.n_patterns,
        ))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out, red


def _p_nrmse_global(phis, alphas, mask, L, n_times=N_FPS):
    """Global analytic-atom + dense LS fit of P on a reduced grid."""
    im = tuple(phis.shape[1:])
    # Stride downsample. spatial_resize_poly on 320^3 hits an index overflow.
    step = max(1, im[0] // 32)
    sl = tuple(slice(None, None, step) for _ in im)
    phi_r = phis[(slice(None),) + sl]
    w = mask[sl].reshape(-1).clamp(min=0)
    red = tuple(phi_r.shape[1:])
    B = phi_r.shape[0]
    pr = phi_r.reshape(B, -1).double()
    af = alphas.reshape(B, -1)
    M = af.shape[1]
    nkeep = min(n_times, M)
    idx = torch.randperm(M, device=af.device)[:nkeep]
    a = af[:, idx].double()
    # FPS in whitened residual (global, no extra affine — already de-linearized)
    Sig, sqrt, isqrt, nlive, _ = _sigma_sqrt(pr, w)
    if nlive == 0:
        return dict(L=L, nrmse=1.0, live=0, red=list(red), n_times=int(idx.numel()))
    at = sqrt.T @ a
    beta_t = k_alpha_selection(at.float(), L, 'maxmin', train_frac=1.0).double()
    betas = isqrt @ beta_t
    Br = torch.exp(-2j * torch.pi * (betas.T @ pr)).to(torch.complex128)  # (L, N)
    P = torch.exp(-2j * torch.pi * (pr.T @ a))  # (N, Mt)
    Bw = Br * w.double()[None]
    G = Bw @ Br.conj().T
    G.diagonal().add_(1e-8)
    rhs = Bw @ P
    h = torch.linalg.solve(G, rhs)
    fit = Br.T @ h
    wsqrt = w.double().sqrt()
    num = ((fit - P) * wsqrt[:, None]).norm()
    den = (P * wsqrt[:, None]).norm()
    return dict(L=L, nrmse=float(num / den.clamp(min=1e-30)),
                live=nlive, red=list(red), n_times=int(idx.numel()),
                Nred=int(pr.shape[1]))


def main():
    print('3D estimate coco_7t_spi', flush=True)
    report = dict(name=NAME, im_size=list(IM), C=C, notes=[])
    fpath = Path('./data') / NAME
    # shapes only first
    trj = torch.load(fpath / 'trj.pt', map_location='cpu', weights_only=True)
    mps = torch.load(fpath / 'mps.pt', map_location='cpu', weights_only=True)
    phis = torch.load(fpath / 'phis.pt', map_location='cpu', weights_only=True)
    report['trj_shape'] = list(trj.shape)
    report['mps_shape'] = list(mps.shape)
    report['phis_shape'] = list(phis.shape)
    M_full = int(np.prod(trj.shape[:-1]))
    C_file = int(mps.shape[0])
    im = tuple(mps.shape[1:])
    report['M_full'] = M_full
    report['C_file'] = C_file
    report['im_file'] = list(im)
    print(f'  trj {tuple(trj.shape)}  mps {tuple(mps.shape)}  M={M_full}',
          flush=True)

    N = int(np.prod(im))
    osN = (OS ** len(im)) * N
    bytes_os = 8 * osN * C_file
    report['memory'] = dict(
        oversampled_grid_C_GiB=bytes_os / 2**30,
        one_image_os_GiB=(8 * osN) / 2**30,
        coil_batching_forced=bytes_os > 12 * 2**30,
    )
    print(f'  os grid {C_file} coils = {bytes_os/2**30:.1f} GiB  '
          f'batch={report["memory"]["coil_batching_forced"]}', flush=True)

    if not torch.cuda.is_available():
        report['notes'].append('no GPU; skipped microbench')
        (OUT / 'stage3d.json').write_text(json.dumps(report, indent=2))
        return

    # Microbench: full 320^3 is 2.1 GiB/transform; n_trans=1.
    benches = []
    M_bench = [10_000, 100_000, 1_000_000]
    print('  microbench full grid', flush=True)
    try:
        benches.append(_bench_type2(im, 100_000, n_trans=1, n_rep=3))
        print(f'    full M=1e5: {benches[-1]}', flush=True)
    except Exception as exc:
        report['notes'].append(f'full-grid bench failed: {exc}')
        print('    full-grid failed', exc, flush=True)

    for Q in (1, 2, 4, 8):
        # isotropic-ish split
        if Q == 1:
            qax = (1, 1, 1)
        elif Q == 2:
            qax = (2, 1, 1)
        elif Q == 4:
            qax = (2, 2, 1)
        else:
            qax = (2, 2, 2)
        V = tuple(max(32, next_smooth(int(math.ceil(n / a))))
                  for n, a in zip(im, qax))
        print(f'  microbench Q={Q} V={V}', flush=True)
        try:
            benches.append(_bench_type2(V, 100_000, n_trans=1, n_rep=3))
            benches[-1]['Q'] = Q
            print(f'    Q={Q}: {benches[-1]["median"]:.4f}s', flush=True)
        except Exception as exc:
            report['notes'].append(f'Q={Q} bench failed: {exc}')
    report['benches'] = benches

    # rho: FFT from tiny-M; gather scaled from bench M to full M (and R=2).
    try:
        t_fftish = _bench_type2(im, 256, n_trans=1, n_rep=3)
        t_full = next((b for b in benches if b.get('ok') and b['V'] == list(im)),
                      None)
        if t_full and t_fftish.get('ok'):
            F = t_fftish['median']
            G_bench = max(t_full['median'] - F, 0.0)
            M_b = t_full['M']
            G_full = G_bench * (M_full / max(M_b, 1))
            rho = G_full / max(F, 1e-9)
            rho_R2 = (G_full / 2) / max(F, 1e-9)
            report['rho_meas'] = dict(
                F=F, G_bench=G_bench, M_bench=M_b, G_full=G_full,
                rho_full=rho, rho_R2=rho_R2, M_fft=256)
            print(f'  F={F:.4f}s  G(M=1e5)={G_bench:.4f}s  '
                  f'G(M_full)={G_full:.3f}s  rho_full={rho:.2f} rho_R2={rho_R2:.2f}',
                  flush=True)
        else:
            rho = RHO_FALLBACK
            report['rho_meas'] = dict(rho_full=rho, note='fallback')
    except Exception as exc:
        rho = RHO_FALLBACK
        report['notes'].append(f'rho failed: {exc}')
        report['rho_meas'] = dict(rho_full=rho, note='fallback')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Rank estimate: reduce on CPU, fit on GPU (full 320^3 phis do not fit
    # next to leftover FINUFFT workspaces).
    print('  reduced-grid P-fit on CPU→GPU', flush=True)
    phis_g = mask_g = None
    try:
        fpath = Path('./data') / NAME
        phis_c = torch.load(fpath / 'phis.pt', map_location='cpu',
                            weights_only=True).float()
        alphas_c = torch.load(fpath / 'alphas.pt', map_location='cpu',
                              weights_only=True).float()
        evals_c = torch.load(fpath / 'evals.pt', map_location='cpu',
                             weights_only=True).float()
        mask_c = (evals_c > 0.9).float()
        phis_c = phis_c * mask_c
        rank_tab = []
        for L in (4, 8, 16, 32, 48, 64):
            print(f'    global P-fit L={L}', flush=True)
            rank_tab.append(_p_nrmse_global(phis_c, alphas_c, mask_c, L))
            print(f'      E={rank_tab[-1]["nrmse"]:.4f}  live={rank_tab[-1]["live"]} '
                  f'red={rank_tab[-1]["red"]}', flush=True)
        report['rank_global'] = rank_tab
        L_svd = next((r['L'] for r in rank_tab if r['nrmse'] <= TARGET_E),
                     rank_tab[-1]['L'])
        report['L_svd_at_1e-2'] = L_svd
        # live-axis count on a stride-downsampled grid (resize overflows at 320^3)
        step = max(1, phis_c.shape[1] // 32)
        sl = (slice(None),) + tuple(slice(None, None, step) for _ in range(phis_c.ndim - 1))
        phis_g = phis_c[sl]
        mask_g = (mask_c[sl[1:]] > 0.5).float()
        if torch.cuda.is_available():
            phis_g = phis_g.cuda()
            mask_g = mask_g.cuda()
        del phis_c, alphas_c, evals_c, mask_c
    except Exception as exc:
        report['notes'].append(f'rank estimate failed: {exc}')
        L_svd = 32
        report['L_svd_at_1e-2'] = L_svd
        print('  rank estimate failed', exc, flush=True)

    if phis_g is not None:
        block_L = {}
        for Q, qax in ((1, (1, 1, 1)), (2, (2, 1, 1)), (4, (2, 2, 1))):
            try:
                print(f'  block affine+FPS Q={Q} on reduced grid {tuple(phis_g.shape[1:])}',
                      flush=True)
                bs = make_blocks(mask_g, qax, fft_friendly=True)
                _, _, pres = fit_block_affine(phis_g, bs, weights=mask_g)
                lives = []
                wflat = mask_g.reshape(-1)
                for pr, sel in zip(pres, bs.idx):
                    _, _, _, nlive, _ = _sigma_sqrt(pr, wflat[sel])
                    lives.append(nlive)
                block_L[str(Q)] = dict(Q_kept=bs.Q_kept, V=list(bs.V), live=lives)
                print(f'    Q_kept={bs.Q_kept} V={bs.V} live={lives}', flush=True)
            except Exception as exc:
                report['notes'].append(f'block Q={Q} failed: {exc}')
        report['block_sigma'] = block_L

    # Predicted speedup grid. Use L_block = max(1, ceil(L_svd / Q)) as optimistic
    # (affine absorption buys the 1/Q), and L_block = L_svd as pessimistic.
    N = int(np.prod(im))
    rho_R2 = rho / 2
    table = []
    for Q in (1, 2, 4, 8):
        for s in (0.125, 0.25, 0.5, 1.0):
            for kind, Lb in (('optimistic', max(1, math.ceil(L_svd / Q))),
                             ('pessimistic', L_svd)):
                lam = math.log(OS ** 3 * N / Q) / math.log(OS ** 3 * N)
                den = Lb * lam / L_svd + rho * s * Q * Lb / L_svd
                pred = (1 + rho) / max(den, 1e-12)
                den2 = Lb * lam / L_svd + rho_R2 * s * Q * Lb / L_svd
                pred2 = (1 + rho_R2) / max(den2, 1e-12)
                table.append(dict(
                    Q=Q, s=s, L=Lb, S_tot=Q * Lb, kind=kind,
                    lam=lam, pred_speedup=pred, M=M_full,
                    pred_speedup_R2=pred2,
                ))
    report['speedup_grid'] = table
    (OUT / 'stage3d.json').write_text(json.dumps(report, indent=2))
    print('wrote', OUT / 'stage3d.json', flush=True)
    print('\nQ  s    kind         L   S_tot  pred   pred_R2', flush=True)
    for r in table:
        print(f'{r["Q"]:2} {r["s"]:<5} {r["kind"]:<12} {r["L"]:3} {r["S_tot"]:5} '
              f'{r["pred_speedup"]:6.2f} {r["pred_speedup_R2"]:6.2f}', flush=True)


def next_smooth(n):
    n = int(max(n, 1))
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


if __name__ == '__main__':
    main()
