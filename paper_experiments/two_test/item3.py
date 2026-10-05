"""
Item 3 of math_docs/two_test_follow.md: n_trans packing in _QBlockSenseLinop.

No new linop. Times legacy (n_trans=coil_batch) vs packed (n_trans=S_tot*coil_batch
or L_q*coil_batch + streams).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from hofft.decomp import hofft_params
from hofft.pipelines import qblock_svd_decomp_linop, svd_decomp_linop
from mr_recon.linops import batching_params
from mr_recon.recons import CG_SENSE_recon

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)


def _load_hybrid():
    path = HERE.parent / 'hybrid_feas' / 'common.py'
    spec = importlib.util.spec_from_file_location('hybrid_feas_common', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hybrid = _load_hybrid()
CG_KW = dict(max_iter=10, max_eigen=1.0, verbose=False)
N_WARM = 2
N_REP = 5


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median_s(fn, warm=N_WARM, rep=N_REP):
    for _ in range(warm):
        fn()
    _sync()
    ts = []
    for _ in range(rep):
        _sync()
        t0 = perf_counter()
        fn()
        _sync()
        ts.append(perf_counter() - t0)
    return float(np.median(ts))


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _hparams(ds, L):
    return hofft_params(
        (3, 3), ds.os, int(L),
        reduced_im_size=ds.reduced_im_size,
        time_reduction_factor=ds.time_reduction_factor,
        num_compressed_bases=ds.num_compressed_bases,
        cur_rank=ds.cur_rank,
        normalize_coeffs=True,
        verbose=False,
    )


def _adjoint_rel(A, im_size, ksp_shape):
    x = torch.randn(im_size, dtype=torch.complex64, device=A.mps.device)
    y = torch.randn(ksp_shape, dtype=torch.complex64, device=A.mps.device)
    lhs = torch.vdot(A.forward(x).reshape(-1), (A.dcf * y).reshape(-1))
    rhs = torch.vdot(x.reshape(-1), A.adjoint(y).reshape(-1))
    return float(abs(lhs - rhs) / abs(lhs).clamp_min(1e-30))


def _inspect(A):
    nfts = A.nfts
    kw = getattr(nfts[0], 'plan_kwargs', {}) if nfts else {}
    return dict(
        n_plans=len(nfts),
        n_streams=len(A._streams),
        pack=bool(getattr(A, 'pack_n_trans', None)),
        n_trans_plan=list(getattr(A, 'n_trans_plan', [])),
        n_trans_wanted=list(getattr(A, 'n_trans_wanted', [])),
        gpu_stream=kw.get('gpu_stream', 'UNSET'),
        gpu_maxbatchsize=kw.get('gpu_maxbatchsize', 'UNSET'),
        gpu_method=kw.get('gpu_method', 'UNSET'),
        coil_batch=int(A.coil_batch),
        S_tot=int(A.S_tot),
        L_q=list(A.L_q),
        V=list(A.V),
    )


def nrmse_mag(img, ref, mask=None):
    if mask is not None:
        img, ref = img * mask, ref * mask
    return float((img.abs() - ref.abs()).norm() / ref.abs().norm().clamp_min(1e-30))


def main():
    torch_dev = torch.device('cuda')
    ds = hybrid.load_dataset('tilt_spi_invivo', torch_dev)
    mps, ksp, mask = ds.mps, ds.ksp, ds.mask
    C = mps.shape[0]
    x = ds.img_ee.to(torch.complex64)
    bparams = batching_params(coil_batch_size=C // 2, field_batch_size=1)
    print(f'tilt {ds.im_size} C={C} coil_batch={C//2}', flush=True)

    rows = []

    # ---- SVD baseline ----
    print('\n---- SVD L=80 ----', flush=True)
    hp = _hparams(ds, 80)
    A_s = svd_decomp_linop(
        ds.phis, ds.alphas, mps, ds.trj, hp,
        svd_method='cur', spatial_mask=mask, dcf=ds.dcf, bparams=bparams,
    )
    A_s.bparams.field_batch_size = 80
    t_fwd = _median_s(lambda: A_s.forward(x))
    _sync()
    t0 = perf_counter()
    img = CG_SENSE_recon(A_s, ksp, **CG_KW)
    _sync()
    t_re = perf_counter() - t0
    e = nrmse_mag(img, ds.img_ee, mask)
    svd_fwd = t_fwd / 80
    svd_rec = t_re / 80
    print(f'  fwd={t_fwd*1e3:.1f} ms  recon={t_re:.2f}s  |E|={e:.4f}  '
          f'per-factor fwd={svd_fwd*1e3:.2f} ms  recon={svd_rec*1e3:.1f} ms',
          flush=True)
    rows.append(dict(
        tag='svd', L=80, S_tot=80, fwd_s=t_fwd, recon_s=t_re, nrmse=e,
        per_fwd_s=svd_fwd, per_recon_s=svd_rec,
    ))
    del A_s, img
    torch.cuda.empty_cache()

    S = 80
    for Q in (4, 8, 16):
        for shear in (True, False):
            print(f'\n---- Q={Q} shear={shear} S={S} ----', flush=True)
            hp = _hparams(ds, S)
            A = qblock_svd_decomp_linop(
                ds.phis, ds.alphas, mps, ds.trj, hp,
                Q=Q, shear=shear, overlap=0.05,
                svd_method='cur', spatial_mask=mask, dcf=ds.dcf,
                bparams=bparams, pack_n_trans=False,
            )
            S_tot = int(A.S_tot)

            # legacy
            info0 = _inspect(A)
            adj0 = _adjoint_rel(A, ds.im_size, ksp.shape)
            y0 = A.forward(x)
            t0 = _median_s(lambda: A.forward(x))
            print(f'  legacy n_trans={info0["n_trans_plan"]}  '
                  f'streams={info0["n_streams"]}  adj={adj0:.2e}  '
                  f'fwd={t0*1e3:.1f} ms  per={t0/S_tot*1e3:.2f} ms', flush=True)

            # packed
            A.repack(True)
            info1 = _inspect(A)
            adj1 = _adjoint_rel(A, ds.im_size, ksp.shape)
            y1 = A.forward(x)
            match = _rel(y1, y0)
            t1 = _median_s(lambda: A.forward(x))
            print(f'  packed n_trans={info1["n_trans_plan"]}  '
                  f'streams={info1["n_streams"]}  gpu_stream={info1["gpu_stream"]}  '
                  f'adj={adj1:.2e}  match={match:.3e}  '
                  f'fwd={t1*1e3:.1f} ms  per={t1/S_tot*1e3:.2f} ms', flush=True)

            rec = dict(
                tag=f'Q{Q}_shear{int(bool(shear))}',
                Q=Q, shear=bool(shear), S_tot=S_tot, L_q=list(A.L_q), V=list(A.V),
                legacy=dict(info=info0, adj=adj0, fwd_s=t0, per_fwd_s=t0 / S_tot),
                packed=dict(info=info1, adj=adj1, fwd_s=t1, per_fwd_s=t1 / S_tot,
                            match=match),
                vs_svd_legacy=(svd_fwd / (t0 / S_tot)) if t0 > 0 else None,
                vs_svd_packed=(svd_fwd / (t1 / S_tot)) if t1 > 0 else None,
                vs_pred_legacy=(t0 / S_tot) / (svd_fwd / Q) if svd_fwd > 0 else None,
                vs_pred_packed=(t1 / S_tot) / (svd_fwd / Q) if svd_fwd > 0 else None,
                frac_of_svd_packed=(t1 / S_tot) / svd_fwd if svd_fwd > 0 else None,
            )
            rows.append(rec)

            # recon only for Q=8 (gate + matched-error r)
            if Q == 8:
                _sync()
                t_re0 = perf_counter()
                # still packed; retime packed recon, then legacy
                img1 = CG_SENSE_recon(A, ksp, **CG_KW)
                _sync()
                t_rp = perf_counter() - t_re0
                e1 = nrmse_mag(img1, ds.img_ee, mask)
                A.repack(False)
                _sync()
                t_re0 = perf_counter()
                img0 = CG_SENSE_recon(A, ksp, **CG_KW)
                _sync()
                t_rl = perf_counter() - t_re0
                e0 = nrmse_mag(img0, ds.img_ee, mask)
                rec['legacy']['recon_s'] = t_rl
                rec['legacy']['nrmse'] = e0
                rec['legacy']['per_recon_s'] = t_rl / S_tot
                rec['packed']['recon_s'] = t_rp
                rec['packed']['nrmse'] = e1
                rec['packed']['per_recon_s'] = t_rp / S_tot
                rec['r_matched_svd80'] = S_tot / 80
                print(f'  recon legacy={t_rl:.2f}s |E|={e0:.4f}  '
                      f'packed={t_rp:.2f}s |E|={e1:.4f}', flush=True)
                del img0, img1

            if hasattr(A, 'clear_plans'):
                A.clear_plans()
            del A, y0, y1
            torch.cuda.empty_cache()
            (OUT / 'item3.json').write_text(json.dumps({
                'rows': rows, 'svd_per_fwd_s': svd_fwd, 'svd_per_recon_s': svd_rec,
            }, indent=2, default=str))

    # Gate
    q8 = [r for r in rows if r.get('Q') == 8]
    fracs = [r['frac_of_svd_packed'] for r in q8 if r.get('frac_of_svd_packed')]
    gate = None
    if fracs:
        frac = float(min(fracs))
        gate = dict(
            q8_min_frac_of_svd=frac,
            pass_below_30pct=frac < 0.30,
            verdict=(
                f'Q=8 packed per-factor is {frac:.2%} of a global factor; '
                + ('packing recovered the cost-model 9x path'
                   if frac < 0.30 else
                   'still above 30% — blocking capped at ~2x, stop')
            ),
        )
        print(f'\nGATE: {gate["verdict"]}', flush=True)

    (OUT / 'item3.json').write_text(json.dumps({
        'rows': rows, 'svd_per_fwd_s': svd_fwd, 'svd_per_recon_s': svd_rec,
        'gate': gate,
    }, indent=2, default=str))
    print(f'wrote {OUT / "item3.json"}', flush=True)


if __name__ == '__main__':
    main()
