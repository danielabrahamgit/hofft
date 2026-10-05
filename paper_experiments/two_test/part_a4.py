"""
Part A.4 of math_docs/two_test.md: shear=False single-plan Q-block on tilt.

Only meaningful if A.3 gave eta > 0.7. Uses the existing qblock linop;
does not add a new one. Confirms whether opts.gpu_stream is set.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from time import perf_counter

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


def nrmse_mag(img, ref, mask=None):
    if mask is not None:
        img, ref = img * mask, ref * mask
    return float((img.abs() - ref.abs()).norm() / ref.abs().norm().clamp_min(1e-30))


def _gpu_timer(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return perf_counter() - t0, out


def _hparams(ds, L):
    return hofft_params(
        (3, 3), ds.os, int(L),
        reduced_im_size=ds.reduced_im_size,
        time_reduction_factor=ds.time_reduction_factor,
        num_compressed_bases=ds.num_compressed_bases,
        cur_rank=ds.cur_rank,
        normalize_coeffs=True,
        verbose=True,
    )


def _inspect_streams(A, tag):
    nfts = getattr(A, 'nfts', None)
    streams = getattr(A, '_streams', None)
    n_plans = len(nfts) if nfts is not None else None
    gpu_stream = None
    if nfts:
        kw = getattr(nfts[0], 'plan_kwargs', {})
        gpu_stream = kw.get('gpu_stream', 'UNSET')
    print(f'  {tag}: n_plans={n_plans}  _streams={streams}  '
          f'plan_kwargs.gpu_stream={gpu_stream}  shear={getattr(A, "shear", None)}',
          flush=True)
    return dict(n_plans=n_plans, streams=list(streams) if streams else [],
                gpu_stream=str(gpu_stream), shear=str(getattr(A, 'shear', None)))


def main():
    torch_dev = torch.device('cuda')
    ds = hybrid.load_dataset('tilt_spi_invivo', torch_dev)
    mps, ksp, mask = ds.mps, ds.ksp, ds.mask
    C = mps.shape[0]
    print(f'tilt {ds.im_size} C={C}', flush=True)

    rows = []
    bparams = batching_params(coil_batch_size=C // 2, field_batch_size=1)

    def go(tag, decomp, L, extra=None):
        hp = _hparams(ds, L)
        t_de, A = _gpu_timer(decomp)
        info = _inspect_streams(A, tag)
        S = int(getattr(A, 'S_tot', L))
        Lq = getattr(A, 'L_q', [L])
        A.bparams.field_batch_size = max(Lq) if isinstance(Lq, list) else L
        t_re, img = _gpu_timer(lambda: CG_SENSE_recon(A, ksp, **CG_KW))
        e = nrmse_mag(img, ds.img_ee, mask)
        per = t_re / max(S, 1)
        rec = dict(tag=tag, L=int(L), S_tot=S, L_q=list(Lq) if isinstance(Lq, list) else [int(L)],
                   nrmse=e, decomp_s=t_de, recon_s=t_re, per_factor_s=per)
        rec.update(info)
        if extra:
            rec.update(extra)
        print(f'  {tag} L={L} S={S}  |E|={e:.4f}  recon={t_re:.2f}s  '
              f'per-factor={per*1e3:.2f} ms', flush=True)
        rows.append(rec)
        if hasattr(A, 'clear_plans'):
            A.clear_plans()
        del A, img
        torch.cuda.empty_cache()

    for L in (40, 80):
        go('svd', lambda L=L: svd_decomp_linop(
            ds.phis, ds.alphas, mps, ds.trj, _hparams(ds, L),
            svd_method='cur', spatial_mask=mask, dcf=ds.dcf, bparams=bparams,
        ), L)

    for Q in (4, 8):
        for shear in (True, False):
            for S in (80, 160):
                tag = f'qblock_Q{Q}_shear{int(bool(shear))}'
                go(tag, lambda Q=Q, shear=shear, S=S: qblock_svd_decomp_linop(
                    ds.phis, ds.alphas, mps, ds.trj, _hparams(ds, S),
                    Q=Q, shear=shear, overlap=0.05,
                    svd_method='cur', spatial_mask=mask, dcf=ds.dcf,
                    bparams=bparams,
                ), S, extra=dict(Q=Q, shear=shear))

    # Matched-error r: S_qblock / L_svd at the closest NRMSE
    svd = [r for r in rows if r['tag'] == 'svd']
    for r in rows:
        if r['tag'] == 'svd':
            continue
        # nearest SVD NRMSE
        best = min(svd, key=lambda s: abs(s['nrmse'] - r['nrmse']))
        r['r_matched'] = r['S_tot'] / best['L'] if best['L'] else None
        r['svd_match_L'] = best['L']
        r['svd_match_nrmse'] = best['nrmse']

    (OUT / 'part_a4.json').write_text(json.dumps(rows, indent=2))
    print(f'wrote {OUT / "part_a4.json"}', flush=True)


if __name__ == '__main__':
    main()
