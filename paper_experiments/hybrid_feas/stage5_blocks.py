"""
Priority 5 of math_docs/hybrid_feas.md.

Disjoint rectangular blocks, cropped cuFINUFFT (not a full-size zero-mask),
per-block CUR+SVD rank. Every block still writes all M samples. Same frozen
CG and τ as Priority 1. Gate: E≤τ and median recon speedup ≥1.2×.

Run with:
    paper_experiments/hybrid_feas/run.sh stage5_blocks.py
    paper_experiments/hybrid_feas/run.sh stage5_blocks.py --datasets coco_spiral
"""
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

from mr_recon.utils import batch_iterator
from mr_recon.recons import CG_SENSE_recon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATASETS, load_dataset, nrmse  # noqa: E402
from cufi_op import CufiNUFFT  # noqa: E402
sys.path.append(str(HERE.parent / 'qblock_feas'))
from blocks import make_blocks  # noqa: E402
from hofft.pipelines import _process_phase_coefficients
from hofft.cur_ops import build_cur_factors
from hofft.utils import expand_spatial, expand_temporal, reduce_spatial, reduce_temporal

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

TAU = 0.01
SPEED_GATE = 1.2
EPS = 1e-4
CG_ITERS = 20
CG_TOL = 1e-8
FIELD_BATCH = 4
N_RECON = 2

# Q layouts and per-block ranks. Coarse; refine only near the frontier.
LAYOUTS = {
    'coco_spiral': (
        ((2, 1), (2, 3, 4, 6, 8, 10, 12)),
        ((2, 2), (2, 3, 4, 6, 8)),
        ((2, 4), (1, 2, 3, 4)),
    ),
    'tilt_spi_invivo': (
        ((2, 1), (20, 40, 80, 96, 104, 112)),
        ((2, 2), (16, 32, 48, 56, 64, 72)),
        ((2, 4), (8, 16, 24, 32, 40)),
    ),
}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats(xs):
    arr = np.asarray(xs, dtype=float)
    return dict(median=float(np.median(arr)),
                p10=float(np.percentile(arr, 10)),
                p90=float(np.percentile(arr, 90)), n=int(arr.size))


def factor_block(ds, L, slc, seed=0):
    """Rank-L CUR+SVD of the residual phase on one spatial crop."""
    phis = ds.phis[(slice(None),) + slc]
    mask = ds.mask[slc]
    phis_n, alphas_n, spat, temp = _process_phase_coefficients(
        phis, ds.alphas, normalize_coeffs=True,
        num_compressed_bases=(min(ds.num_compressed_bases, phis.shape[0])
                              if ds.num_compressed_bases else None))
    crop = tuple(phis.shape[1:])
    if ds.reduced_im_size is not None:
        red = tuple(max(16, int(round(r * c / n)))
                    for r, c, n in zip(ds.reduced_im_size, crop, ds.im_size))
        red = tuple(max(16, e + (e % 2)) for e in red)
        phis_n = reduce_spatial(phis_n, im_size_low=red, order=3)
        mask_r = reduce_spatial(mask.float(), im_size_low=red, order=3) > 0.5
        phis_n = phis_n * mask_r
    else:
        phis_n = phis_n * (mask > 0)
    if ds.time_reduction_factor is not None:
        n_low = max(round(alphas_n.shape[1] / ds.time_reduction_factor), 2)
        alphas_n = reduce_temporal(alphas_n, num_time_low=n_low, dim=1, order=3)
    B = phis_n.shape[0]
    n_q = int(np.prod(phis_n.shape[1:]))
    rank = min(ds.cur_rank, n_q, alphas_n.reshape(B, -1).shape[1], 500)
    rank = max(rank, L)
    Rcur, Ccur = build_cur_factors(
        phis_n.reshape((B, -1)), alphas_n.reshape((B, -1)),
        rank=rank, normalize_method='svd', cluster_method='maxmin', seed=seed)
    Qc, Tc = torch.linalg.qr(Ccur.T, mode='reduced')
    Qr, Tr = torch.linalg.qr(Rcur.T, mode='reduced')
    Um, S, Vm = torch.svd_lowrank(Tc @ Tr.T, q=L)
    Vm = Vm.conj()
    U = Qc @ Um
    V = Qr @ Vm
    spatial = (V[:, :L] * (S[:L] ** 0.5)).T
    temporal = (U[:, :L] * (S[:L] ** 0.5)).T
    spatial = spatial.reshape((L, *phis_n.shape[1:]))
    temporal = temporal.reshape((L, *alphas_n.shape[1:]))
    if ds.time_reduction_factor is not None:
        temporal = expand_temporal(temporal, num_time_high=ds.alphas.shape[1],
                                   dim=1, order=3)
    if ds.reduced_im_size is not None:
        spatial = expand_spatial(spatial, crop, order=3)
    spatial = spatial * spat
    temporal = temporal * temp
    return spatial.contiguous(), temporal.contiguous()


class BlockSense:
    """
    Dense local-rank sense operator. Each block is a cropped type-2 / type-1
    cuFINUFFT at the original voxel spacing, plus the origin phase ramp.
    """

    def __init__(self, ds, blockset, factors, eps=EPS):
        self.ds = ds
        self.blockset = blockset
        self.factors = factors
        self.mps = ds.mps
        self.dcf = ds.dcf
        self.C = ds.C
        self.im_size = ds.im_size
        self.trj_size = ds.trj_size
        self.ishape = ds.im_size
        self.oshape = (ds.C, *ds.trj_size)
        self.cb = max(ds.C // 2, 1)
        self.fb = FIELD_BATCH
        d = len(ds.im_size)
        N = torch.tensor(ds.im_size, device=ds.trj.device, dtype=ds.trj.dtype)
        V = torch.tensor(blockset.V, device=ds.trj.device, dtype=ds.trj.dtype)
        D = V / N
        # Crop NUFFT divides by √V; match the full-grid 1/√N convention.
        self.scale = float(np.prod(blockset.V) / np.prod(ds.im_size)) ** 0.5
        # Native cycles/FOV trj, same as load_dataset (not π-rescaled).
        trj = ds.trj
        self.trj_q = (trj * D).contiguous()
        # ramp_m = exp(-2jπ k·c), c in fractional FOV.
        c = blockset.centers.to(device=trj.device, dtype=trj.dtype)
        kdotc = (trj.reshape(-1, d) @ c.T).T  # (Q, M)
        self.ramps = torch.exp(-2j * torch.pi * kdotc).reshape(
            len(factors), *ds.trj_size)
        self.nfts = []
        self.trj_pi = []
        self.slices = []
        self.labels = blockset.labels.to(ds.trj.device)
        for q, (spatial, temporal) in enumerate(factors):
            nft = CufiNUFFT(tuple(blockset.V), eps=eps)
            self.nfts.append(nft)
            # Leading 1 so CufiNUFFT reshape uses trj.shape[1:-1] = trj_size.
            self.trj_pi.append(nft.rescale_trajectory(self.trj_q)[None].contiguous())
            lo = blockset.win_lo[q].tolist()
            self.slices.append(tuple(slice(int(lo[i]), int(lo[i]) + blockset.V[i])
                                     for i in range(d)))

    def forward(self, img):
        ksp = torch.zeros(self.oshape, device=img.device, dtype=img.dtype)
        for q, (spatial, temporal) in enumerate(self.factors):
            slc = self.slices[q]
            keep = self.labels[slc] == q
            xq = img[slc] * keep
            mps_q = self.mps[(slice(None),) + slc]
            L = spatial.shape[0]
            trj_pi = self.trj_pi[q]
            ramp = self.ramps[q]
            nft = self.nfts[q]
            for c1, c2 in batch_iterator(self.C, self.cb):
                Sx = mps_q[c1:c2] * xq
                for l1, l2 in batch_iterator(L, self.fb):
                    Bx = Sx[:, None] * spatial[l1:l2]
                    y = nft.forward(Bx, trj_pi) * self.scale
                    ksp[c1:c2] += (y * temporal[l1:l2] * ramp).sum(dim=1)
        return ksp

    def adjoint(self, ksp):
        img = torch.zeros(self.im_size, device=ksp.device, dtype=ksp.dtype)
        for q, (spatial, temporal) in enumerate(self.factors):
            slc = self.slices[q]
            mps_q = self.mps[(slice(None),) + slc]
            L = spatial.shape[0]
            trj_pi = self.trj_pi[q]
            ramp = self.ramps[q].conj()
            nft = self.nfts[q]
            acc = torch.zeros(spatial.shape[1:], device=ksp.device, dtype=ksp.dtype)
            keep = self.labels[slc] == q
            for c1, c2 in batch_iterator(self.C, self.cb):
                wy = ksp[c1:c2] * self.dcf
                for l1, l2 in batch_iterator(L, self.fb):
                    Hy = wy[:, None] * temporal[l1:l2].conj() * ramp
                    xq = nft.adjoint(Hy, trj_pi) * self.scale
                    acc += (xq * mps_q[c1:c2, None].conj()
                            * spatial[l1:l2].conj()).sum(dim=(0, 1))
            img[slc] += acc * keep
        return img

    def normal(self, img):
        return self.adjoint(self.forward(img))

    def clear(self):
        for nft in self.nfts:
            nft.clear_plans()


def _adjoint_check(A, img_shape, ksp_shape, device, n=2):
    rels = []
    for _ in range(n):
        x = torch.randn(img_shape, device=device, dtype=torch.complex64)
        z = torch.randn(ksp_shape, device=device, dtype=torch.complex64)
        Ax = A.forward(x)
        # DCF is inside A.adjoint, so the identity is ⟨Ax, Wz⟩ = ⟨x, A*z⟩
        # with W already applied in A*. Test ⟨Ax, z⟩ vs ⟨x, A*(z / dcf * dcf)⟩
        # i.e. just ⟨Ax, z⟩ vs ⟨x, A*z⟩ is wrong. Use the same check as P0:
        # ⟨A x, W z⟩ = ⟨x, A* z⟩ where A* includes W=dcf.
        Wz = z  # A.adjoint multiplies dcf; pair with unweighted z on the left
        # Left: ⟨Ax, z⟩  Right: ⟨x, A* (z / dcf)⟩ if we want unweighted...
        # P0 used ⟨Ax, Wz⟩ = ⟨x, A*z⟩. Here W is inside A*.
        lhs = torch.vdot(Ax.reshape(-1), (z * A.dcf).reshape(-1))
        rhs = torch.vdot(x.reshape(-1), A.adjoint(z).reshape(-1))
        rels.append(float(abs(lhs - rhs) / (abs(lhs) + 1e-30)))
    return float(np.max(rels))


def _time_recon(A, ksp, n_reps=N_RECON):
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


def run_dataset(name, torch_dev, stage1, skip_q1=False):
    print(f'\n================ {name} ================', flush=True)
    L0 = int(stage1['L0']['L'])
    T0 = float(stage1['L0']['time']['median'])
    ds = load_dataset(name, torch_dev)
    ref = torch.load(OUT / f'stage1_{name}_ref.pt',
                     map_location=torch_dev, weights_only=False)
    img_ref, mask = ref['img_ref'].to(torch_dev), ref['mask'].to(torch_dev)
    print(f'  L_0={L0}  T_0={T0:.3f}s  τ={TAU}', flush=True)

    d = len(ds.im_size)
    E1 = None
    if not skip_q1:
        b1 = make_blocks(ds.mask, (1,) * d, fft_friendly=False)
        print(f'  Q=1 sanity  V={b1.V}  (full grid {ds.im_size})', flush=True)
        slc1 = tuple(slice(int(b1.win_lo[0, i]), int(b1.win_lo[0, i]) + b1.V[i])
                     for i in range(d))
        A1 = BlockSense(ds, b1, [factor_block(ds, L0, slc1)], eps=EPS)
        rel1 = _adjoint_check(A1, ds.im_size, ds.ksp.shape, torch_dev)
        img1 = CG_SENSE_recon(A1, ds.ksp, max_iter=CG_ITERS, max_eigen=1.0,
                              tolerance=CG_TOL, clear_gpu_mem=False, verbose=False)
        E1 = nrmse(img1, img_ref, mask)
        print(f'  Q=1 L={L0}  adjoint={rel1:.2e}  E vs dense ref={E1:.4f}', flush=True)
        A1.clear()
        if E1 > 0.05:
            print('  BLOCKER: one-block operator does not match the dense model. '
                  'Fix geometry before the Q-sweep.', flush=True)
            return dict(name=name, L0=L0, T0=T0, tau=TAU, rows=[], best=None,
                        Q1_E=E1, within_dataset=True)

    rows = []
    for Qax, Lbs in LAYOUTS[name]:
        bset = make_blocks(ds.mask, Qax, fft_friendly=True)
        Q = bset.Q_kept
        print(f'\n  layout {Qax}  Q_kept={Q}  V={bset.V}  stride={bset.stride}',
              flush=True)
        for Lb in Lbs:
            t0 = perf_counter()
            factors = []
            for q in range(Q):
                lo = bset.win_lo[q].tolist()
                slc = tuple(slice(int(lo[i]), int(lo[i]) + bset.V[i])
                            for i in range(len(ds.im_size)))
                factors.append(factor_block(ds, Lb, slc))
            t_setup = perf_counter() - t0
            A = BlockSense(ds, bset, factors, eps=EPS)
            rel = _adjoint_check(A, ds.im_size, ds.ksp.shape, torch_dev)
            print(f'    L_b={Lb:<3d}  setup {t_setup:.2f}s  adjoint rel={rel:.2e}',
                  flush=True)
            if rel > 1e-4:
                print('      SKIP: adjoint identity failed', flush=True)
                A.clear()
                rows.append(dict(Qax=Qax, Q=Q, Lb=Lb, V=list(bset.V),
                                 setup=t_setup, adjoint_rel=rel, skip=True))
                continue
            img, t = _time_recon(A, ds.ksp)
            E = nrmse(img, img_ref, mask)
            Emag = nrmse(img.abs(), img_ref.abs(), mask)
            S = T0 / t['median'] if t['median'] > 0 else 0.0
            ok = (E <= TAU) and (S >= SPEED_GATE)
            print(f'      E={E:.4f}  |E|={Emag:.4f}  t={t["median"]:.3f}s  '
                  f'S={S:.2f}x  {"PASS" if ok else ""}', flush=True)
            rows.append(dict(
                Qax=list(Qax), Q=Q, Lb=Lb, V=list(bset.V),
                setup=t_setup, adjoint_rel=rel, E=E, E_mag=Emag,
                time=t, speedup=S, pass_gate=ok, skip=False,
            ))
            A.clear()
            torch.cuda.empty_cache()

    passing = [r for r in rows if r.get('pass_gate')]
    best = min(passing, key=lambda r: r['time']['median']) if passing else None
    if best:
        print(f'  best block: Q={best["Q"]} L_b={best["Lb"]}  '
              f'E={best["E"]:.4f}  t={best["time"]["median"]:.3f}s  '
              f'S={best["speedup"]:.2f}x', flush=True)
    else:
        print('  no (Q, L_b) cleared E≤τ and 1.2×. Stop this branch.', flush=True)
    return dict(name=name, L0=L0, T0=T0, tau=TAU, rows=rows, best=best,
                within_dataset=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--skip-q1', action='store_true')
    args, _ = ap.parse_known_args()

    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')
    print(f'Priority 5 — spatial blocks, τ={TAU}, speed gate {SPEED_GATE}×')
    if torch_dev.type != 'cuda':
        print('BLOCKER: no GPU.')
        return 2

    s1 = json.loads((OUT / 'stage1.json').read_text())
    dest = OUT / 'stage5.json'
    prev = json.loads(dest.read_text()) if dest.exists() else {}
    out = dict(prev)
    any_pass = False
    for name in args.datasets:
        if name not in s1 or not s1[name].get('L0'):
            print(f'BLOCKER: {name} has no L_0')
            continue
        fresh = run_dataset(name, torch_dev, s1[name], skip_q1=args.skip_q1)
        if name in prev and prev[name].get('rows'):
            fresh['rows'] = list(prev[name]['rows']) + list(fresh['rows'])
            passing = [r for r in fresh['rows'] if r.get('pass_gate')]
            fresh['best'] = (min(passing, key=lambda r: r['time']['median'])
                             if passing else None)
        out[name] = fresh
        any_pass |= bool(out[name]['best'])
        torch.cuda.empty_cache()

    print('\n================ PRIORITY 5 ================')
    for name in args.datasets:
        if name not in out:
            continue
        b = out[name]['best']
        tag = 'PASS' if b else 'FAIL'
        line = f'  [{tag}] {name}'
        if b:
            line += (f'  Q={b["Q"]} L_b={b["Lb"]}  E={b["E"]:.4f}  '
                     f'S={b["speedup"]:.2f}x  t={b["time"]["median"]:.3f}s')
        print(line)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f'wrote {dest}')
    return 0 if any_pass else 1


if __name__ == '__main__':
    sys.exit(main())
