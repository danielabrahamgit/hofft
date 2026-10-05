"""
Calibrate ofs / origin / signs with the same pipeline as sanity.py:

  crds = gen_grd(im_size) * fov + origin     # offsets before phis
  phis = coco + spha (with signs)
  phis, trj, zero = remove_linear_terms(...) # recon-frame linear/0th
  ksp *= exp(2πj zero)
  B0 via b0_to_phis_alphas + a few time segments (no HO SVD)

HO coco/spha only enter through their linear+0th projection on the mask.

Run from repo root:
  srun --jobid=7996 --overlap env PYTHONUNBUFFERED=1 PYTHONPATH=src \\
    python data/coco_7t_spi/calibrate_params.py
"""
from __future__ import annotations

import gc
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from mr_recon.fourier import cufi_nufft
from mr_recon.imperfections.field import alpha_segementation
from mr_recon.linops import batching_params, sense_linop
from mr_recon.multi_coil.calib import calc_coil_subspace
from mr_recon.recons import CG_SENSE_recon
from mr_recon.utils import gen_grd

from hofft.phase_coeffs import (
    b0_to_phis_alphas,
    coco_bases,
    remove_linear_terms,
    sph_bases,
)
from hofft.utils import reduce_spatial

ROOT = Path(__file__).resolve().parent
FOV = 0.24
OUT = ROOT / 'calibrate_results.json'

M = 4
ROS = slice(None, 15_000, M)
N_GROUP = 6
TR_STRIDE = 8
IM_CAP = 100
CG_ITERS = 3
C_COMP = 8
L_B0 = 5


def tenengrad(img: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    x = img.abs()
    if x.ndim == 3:
        x = x[..., x.shape[-1] // 2]
    kx = x[1:, :] - x[:-1, :]
    ky = x[:, 1:] - x[:, :-1]
    g = kx[:, :-1].square() + ky[:-1, :].square()
    if mask is not None:
        m = mask[..., mask.shape[-1] // 2] if mask.ndim == 3 else mask
        m = m[:-1, :-1] > 0.5
        if bool(m.any()):
            g = g[m]
    return float(g.mean())


def default_signs(device):
    crds = torch.ones(3, device=device)
    coco = torch.ones(4, device=device)
    spha = torch.ones(16, device=device)
    spha[0] = -1
    spha[4::2] = -1
    return crds, coco, spha


def build_phis(im_size, origin, sgns_crds, sgns_coco, sgns_spha, device):
    # Same order as sanity.py: offsets on crds, then axis signs, then bases.
    crds = gen_grd(im_size).to(device) * FOV + origin.to(device)
    crds = crds * sgns_crds.to(device)
    phis_coco = coco_bases(crds[..., 0], crds[..., 1], crds[..., 2])
    phis_coco = phis_coco * sgns_coco.to(device)[:, None, None, None]
    phis_spha = sph_bases(crds[..., 0], crds[..., 1], crds[..., 2])
    phis_spha = phis_spha * sgns_spha.to(device)[:, None, None, None]
    return torch.cat([phis_coco, phis_spha], dim=0)


def make_A(trj, mps, dcf, spatial_funcs, temporal_funcs):
    im_size = tuple(mps.shape[1:])
    os = 2 * round(1.25 * im_size[0] / 2) / im_size[0]
    nft = cufi_nufft(im_size, oversamp=os, width=3)
    nft.plan(trj[None] if trj.ndim == len(im_size) + 1 else trj)
    A = sense_linop(
        trj, mps, dcf, nufft=nft,
        spatial_funcs=spatial_funcs,
        temporal_funcs=temporal_funcs,
        bparams=batching_params(coil_batch_size=mps.shape[0], field_batch_size=1),
    )
    return A, nft


def score(trj, zero, mps, ksp, dcf, mask, b0, dt, device):
    ksp_u = ksp * torch.exp(2j * torch.pi * zero)
    phis_b0, alphas_b0 = b0_to_phis_alphas(
        -b0, tuple(dcf.shape), ro_dim=0, dt=dt, repeat_empty_dims=False)
    spat, temp, _ = alpha_segementation(
        phis_b0, alphas_b0, L=L_B0, interp_type='zero',
        method='maxmin', verbose=False)
    A, nft = make_A(trj, mps, dcf, spat, temp)
    img = CG_SENSE_recon(A, ksp_u, max_iter=CG_ITERS, max_eigen=1.0, verbose=False)
    pred = A.forward(img)
    rel = float((pred - ksp_u).norm() / ksp_u.norm().clamp(min=1e-30))
    sharp = tenengrad(img, mask)
    if hasattr(A, 'clear_plans'):
        A.clear_plans()
    del A, nft, pred, ksp_u, spat, temp, phis_b0, alphas_b0, img
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return rel, sharp


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    trj_full = torch.load(ROOT / 'trj.pt', weights_only=True, map_location=device)
    dcf_full = torch.load(ROOT / 'dcf.pt', weights_only=True, map_location=device)
    ksp = torch.load(ROOT / 'ksp.pt', weights_only=True, map_location=device)
    evals = torch.load(ROOT / 'evals.pt', weights_only=True, map_location=device)
    mps = torch.load(ROOT / 'mps.pt', weights_only=True, map_location=device)
    b0 = torch.load(ROOT / 'b0.pt', weights_only=True, map_location=device)
    pred_0 = torch.load(ROOT / 'pred_0.pt', weights_only=True, map_location=device)
    coco_term = torch.load(ROOT / 'coco_term.pt', weights_only=True, map_location=device)
    alphas_mm = torch.load(ROOT / 'alphas.pt', weights_only=True, mmap=True, map_location='cpu')
    T_full = trj_full.shape[0]
    print(f'trj {tuple(trj_full.shape)}  alphas {tuple(alphas_mm.shape)}  '
          f'max_ofs={alphas_mm.shape[1] - T_full}', flush=True)

    mask = (evals > 0.95).float()
    _, ksp, mps = calc_coil_subspace(ksp[:, :10_000:4, :, ::10], C_COMP, ksp, mps)

    # Match sanity: scanner 0th-order, then readout decimation, first 6 groups.
    ksp = ksp * torch.exp(1j * pred_0.to(ksp.device))
    ksp = ksp * torch.exp(1j * coco_term.to(ksp.device))
    ksp = ksp[:, ROS, :N_GROUP].contiguous()
    trj_nom = trj_full[ROS, :N_GROUP].contiguous()
    dcf = dcf_full[ROS, :N_GROUP].contiguous()
    del pred_0, coco_term, trj_full, dcf_full, evals

    kmax = float(trj_nom.abs().max())
    N_new = int(round(kmax) * 2)
    im_size = (min(N_new, IM_CAP),) * 3
    mps = reduce_spatial(mps, im_size)
    mask = reduce_spatial(mask, im_size)
    b0 = reduce_spatial(b0, im_size)
    mask = (mask > 0.5).float()

    # Extra TR stride so the sweep fits; encoding is unchanged.
    ksp = ksp[..., ::TR_STRIDE].contiguous()
    dcf = dcf[..., ::TR_STRIDE].contiguous()
    dt = M * 1e-6
    print(f'subset ksp {tuple(ksp.shape)}  im {im_size}  C={mps.shape[0]}  '
          f'N_new_full={N_new}', flush=True)

    def alphas_at(ofs):
        sl = alphas_mm[:, ofs:ofs + T_full, :N_GROUP]
        sl = sl[:, ROS, :, ::TR_STRIDE]
        return sl.to(device=device, dtype=torch.float32).contiguous()

    sc, sco, ss = default_signs(device)
    origin0 = torch.zeros(3, device=device)
    scores = []

    def eval_cfg(ofs, origin, sgns_crds, sgns_coco, sgns_spha, tag):
        phis = build_phis(im_size, origin, sgns_crds, sgns_coco, sgns_spha, device)
        al = alphas_at(ofs)
        _, trj_lin, zero = remove_linear_terms(phis, al, mask)
        rel, sharp = score(trj_lin, zero, mps, ksp, dcf, mask, b0, dt, device)
        print(f'  {tag:<48s}  ||Ax-b||={rel:.4f}  sharp={sharp:.4e}', flush=True)
        rec = dict(
            tag=tag, ofs=int(ofs),
            origin=tuple(float(v) for v in origin.tolist()),
            sgns_crds=tuple(float(v) for v in sgns_crds.tolist()),
            sgns_coco=tuple(float(v) for v in sgns_coco.tolist()),
            sgns_spha=tuple(float(v) for v in sgns_spha.tolist()),
            rel=rel, sharp=sharp,
        )
        scores.append(rec)
        del phis, al, trj_lin, zero
        return rec

    print('\nbaselines (sanity pipeline, B0 time-seg, no HO SVD)', flush=True)
    eval_cfg(46, origin0, sc, sco, ss, 'sanity ofs=46 origin=0')
    eval_cfg(46, torch.tensor([0.0, 0.01, 0.02], device=device),
             sc, sco, ss, 'sanity ofs=46 origin=(0, 1cm, 2cm)')

    print('\nstage 1: ofs × xyz signs  (origin=0, full coco+spha → lin/0th)',
          flush=True)
    lin_pats = list(product((-1.0, 1.0), repeat=3))
    ofs_grid = list(range(36, 57, 2))
    if 46 not in ofs_grid:
        ofs_grid.append(46)
    best_sharp = -1.0
    best_ofs, best_lin = 46, (1.0, 1.0, 1.0)
    for ofs in sorted(ofs_grid):
        for sgn in lin_pats:
            ss_t = ss.clone()
            ss_t[1:4] = torch.tensor(sgn, device=device, dtype=ss.dtype)
            rec = eval_cfg(ofs, origin0, sc, sco, ss_t,
                           f'ofs={ofs} xyz={sgn}')
            if rec['sharp'] > best_sharp:
                best_sharp = rec['sharp']
                best_ofs, best_lin = ofs, sgn
                print(f'  new best sharp={best_sharp:.4e}', flush=True)
    ss[1:4] = torch.tensor(best_lin, device=device, dtype=ss.dtype)
    print(f'locked ofs={best_ofs} xyz={best_lin}', flush=True)

    print('\nstage 1b: refine ofs ±6', flush=True)
    for ofs in range(max(0, best_ofs - 6), best_ofs + 7):
        rec = eval_cfg(ofs, origin0, sc, sco, ss, f'ofs={ofs} refine')
        if rec['sharp'] > best_sharp:
            best_sharp, best_ofs = rec['sharp'], ofs
    print(f'locked ofs={best_ofs}', flush=True)

    print('\nstage 2: origin on crds (before phis)', flush=True)
    origin = origin0.clone()
    for ax, name in enumerate('xyz'):
        axis_best, axis_sharp = float(origin[ax]), -1.0
        for val in np.arange(-0.04, 0.041, 0.01):
            o = origin.clone()
            o[ax] = float(val)
            rec = eval_cfg(best_ofs, o, sc, sco, ss,
                           f'origin {name}={val:+.3f}')
            if rec['sharp'] > axis_sharp:
                axis_sharp, axis_best = rec['sharp'], float(val)
        origin[ax] = axis_best
        print(f'  locked {name}0={axis_best:+.3f} m  sharp={axis_sharp:.4e}',
              flush=True)
    best_sharp = max(s['sharp'] for s in scores
                     if s['ofs'] == best_ofs
                     and np.allclose(s['origin'], origin.tolist(), atol=1e-6))

    print('\nstage 3: greedy sign flips', flush=True)
    flip_groups = [
        ('crds', sc, list(range(3))),
        ('coco', sco, list(range(4))),
        ('spha', ss, [0] + list(range(4, 16))),
    ]
    improved = True
    while improved:
        improved = False
        for gname, vec, idxs in flip_groups:
            for i in idxs:
                vec[i] *= -1
                rec = eval_cfg(best_ofs, origin, sc, sco, ss,
                               f'flip {gname}[{i}] -> {float(vec[i]):+.0f}')
                if rec['sharp'] > best_sharp * 1.005:
                    best_sharp = rec['sharp']
                    improved = True
                    print(f'  KEEP {gname}[{i}]  sharp={best_sharp:.4e}',
                          flush=True)
                else:
                    vec[i] *= -1

    winner_s = max(scores, key=lambda s: s['sharp'])
    winner_r = min(scores, key=lambda s: s['rel'])
    payload = dict(
        scores=scores,
        winner_sharp=winner_s,
        winner_residual=winner_r,
        locked_ofs=best_ofs,
        locked_origin=tuple(float(v) for v in origin.tolist()),
        locked_sgns_crds=tuple(float(v) for v in sc.tolist()),
        locked_sgns_coco=tuple(float(v) for v in sco.tolist()),
        locked_sgns_spha=tuple(float(v) for v in ss.tolist()),
    )
    OUT.write_text(json.dumps(payload, indent=2))
    print('\n======== max sharpness ========', flush=True)
    print(json.dumps(winner_s, indent=2), flush=True)
    print('\n======== min ||Ax-b|| ========', flush=True)
    print(json.dumps(winner_r, indent=2), flush=True)
    print(f'wrote {OUT}', flush=True)


def main_pred0():
    """Freeze ofs/origin/signs; sweep a readout roll of pred_0 vs ksp."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  pred_0 readout-shift sweep', flush=True)

    trj_full = torch.load(ROOT / 'trj.pt', weights_only=True, map_location=device)
    dcf_full = torch.load(ROOT / 'dcf.pt', weights_only=True, map_location=device)
    ksp_raw = torch.load(ROOT / 'ksp.pt', weights_only=True, map_location=device)
    evals = torch.load(ROOT / 'evals.pt', weights_only=True, map_location=device)
    mps = torch.load(ROOT / 'mps.pt', weights_only=True, map_location=device)
    b0 = torch.load(ROOT / 'b0.pt', weights_only=True, map_location=device)
    pred_0 = torch.load(ROOT / 'pred_0.pt', weights_only=True, map_location=device)
    coco_term = torch.load(ROOT / 'coco_term.pt', weights_only=True, map_location=device)
    alphas_mm = torch.load(ROOT / 'alphas.pt', weights_only=True, mmap=True, map_location='cpu')
    T_full = trj_full.shape[0]
    print(f'pred_0 {tuple(pred_0.shape)}  ksp {tuple(ksp_raw.shape)}', flush=True)

    mask = (evals > 0.95).float()
    _, ksp_raw, mps = calc_coil_subspace(
        ksp_raw[:, :10_000:4, :, ::10], C_COMP, ksp_raw, mps)
    pred_0 = pred_0.to(device=ksp_raw.device, dtype=ksp_raw.real.dtype)
    coco_term = coco_term.to(device=ksp_raw.device, dtype=ksp_raw.real.dtype)

    ksp_nom = ksp_raw[:, ROS, :N_GROUP]
    dcf = dcf_full[ROS, :N_GROUP].contiguous()
    kmax = float(trj_full[ROS, :N_GROUP].abs().max())
    N_new = int(round(kmax) * 2)
    im_size = (min(N_new, IM_CAP),) * 3
    mps = reduce_spatial(mps, im_size)
    mask = (reduce_spatial(mask, im_size) > 0.5).float()
    b0 = reduce_spatial(b0, im_size)
    dcf = dcf[..., ::TR_STRIDE].contiguous()
    dt = M * 1e-6

    ofs = 46
    origin = torch.zeros(3, device=device)
    sc, sco, ss = default_signs(device)
    sl = alphas_mm[:, ofs:ofs + T_full, :N_GROUP][:, ROS, :, ::TR_STRIDE]
    al = sl.to(device=device, dtype=torch.float32).contiguous()
    phis = build_phis(im_size, origin, sc, sco, ss, device)
    _, trj_lin, zero = remove_linear_terms(phis, al, mask)
    print(f'frozen ofs={ofs} origin=0 xyz+++  im={im_size}  '
          f'ksp subset will be {tuple(ksp_nom[..., ::TR_STRIDE].shape)}', flush=True)

    phis_b0, alphas_b0 = b0_to_phis_alphas(
        -b0, tuple(dcf.shape), ro_dim=0, dt=dt, repeat_empty_dims=False)
    spat, temp, _ = alpha_segementation(
        phis_b0, alphas_b0, L=L_B0, interp_type='zero',
        method='maxmin', verbose=False)
    A, nft = make_A(trj_lin, mps, dcf, spat, temp)

    def ksp_for(shift, sign=1.0, use_pred=True):
        ksp = ksp_raw.clone()
        if use_pred:
            p = torch.roll(pred_0, int(shift), dims=0)
            p = p - p[:1]
            ksp = ksp * torch.exp(1j * sign * p)
        ksp = ksp * torch.exp(1j * coco_term)
        ksp = ksp[:, ROS, :N_GROUP, ::TR_STRIDE].contiguous()
        return ksp * torch.exp(2j * torch.pi * zero)

    def eval_ksp(ksp_u, tag):
        img = CG_SENSE_recon(A, ksp_u, max_iter=CG_ITERS, max_eigen=1.0, verbose=False)
        pred = A.forward(img)
        rel = float((pred - ksp_u).norm() / ksp_u.norm().clamp(min=1e-30))
        sharp = tenengrad(img, mask)
        print(f'  {tag:<40s}  ||Ax-b||={rel:.4f}  sharp={sharp:.4e}', flush=True)
        del img, pred
        return dict(tag=tag, rel=rel, sharp=sharp)

    scores = []
    rec = eval_ksp(ksp_for(0, use_pred=False), 'no pred_0 (coco_term only)')
    rec['shift'] = None
    rec['sign'] = 0.0
    scores.append(rec)
    rec = eval_ksp(ksp_for(0, sign=-1.0), 'shift=0  sign=-1')
    rec.update(shift=0, sign=-1.0)
    scores.append(rec)

    coarse = list(range(-80, 81, 4))
    if 0 not in coarse:
        coarse.append(0)
    best_sharp, best_shift = -1.0, 0
    print('\ncoarse readout roll of pred_0 (samples)', flush=True)
    for sh in sorted(coarse):
        rec = eval_ksp(ksp_for(sh), f'shift={sh:+d}')
        rec.update(shift=sh, sign=1.0)
        scores.append(rec)
        if rec['sharp'] > best_sharp:
            best_sharp, best_shift = rec['sharp'], sh
            print(f'  new best shift={sh:+d}  sharp={best_sharp:.4e}', flush=True)

    print(f'\nrefine ±8 around {best_shift:+d}', flush=True)
    for sh in range(best_shift - 8, best_shift + 9):
        if any(s.get('shift') == sh and s.get('sign') == 1.0 for s in scores):
            continue
        rec = eval_ksp(ksp_for(sh), f'shift={sh:+d}')
        rec.update(shift=sh, sign=1.0)
        scores.append(rec)
        if rec['sharp'] > best_sharp:
            best_sharp, best_shift = rec['sharp'], sh

    winner_s = max(scores, key=lambda s: s['sharp'])
    winner_r = min(scores, key=lambda s: s['rel'])
    out = ROOT / 'calibrate_pred0.json'
    out.write_text(json.dumps(dict(
        scores=scores, winner_sharp=winner_s, winner_residual=winner_r,
        best_shift=best_shift,
    ), indent=2))
    print('\n======== max sharpness ========', flush=True)
    print(json.dumps(winner_s, indent=2), flush=True)
    print('\n======== min ||Ax-b|| ========', flush=True)
    print(json.dumps(winner_r, indent=2), flush=True)
    print(f'wrote {out}', flush=True)
    if hasattr(A, 'clear_plans'):
        A.clear_plans()


if __name__ == '__main__':
    import sys
    if '--pred0' in sys.argv:
        main_pred0()
    else:
        main()
