"""
Q=1 unconstrained K-SVD vs SVD (math_docs/ksvd_feas.md).

    paper_experiments/ksvd_feas/run.sh run.py
    paper_experiments/ksvd_feas/run.sh run.py --datasets coco_spiral
"""
from __future__ import annotations

import argparse
import importlib.util
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

from hofft.cur_ops import _pivot_indices, _prepare_cur_space
from hofft.sparse_temporal import ksvd_temporal
from hofft.utils import reduce_spatial, reduce_temporal

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)

DATASETS = ('coco_spiral', 'tilt_spi_invivo')
KS = (1, 3, 5, 8, 10)
LS = {
    'coco_spiral':     (4, 6, 8, 10, 12, 16, 20, 24, 32, 48),
    'tilt_spi_invivo': (16, 24, 32, 48, 64, 80, 112, 160),
}
TARGET_ERRS = (1e-1, 1e-2)
CUR_RANK = 500
N_ITER = 8
N_TIME_REPS = 3


def _load_mod(folder: str, filename: str = 'common.py'):
    path = HERE.parent / folder / filename
    key = f'{folder}_{filename}'
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


hybrid = _load_mod('hybrid_feas')


class GPUTimer:
    def __init__(self, torch_dev: torch.device):
        self.is_cuda = torch_dev.type == 'cuda'

    def __enter__(self):
        if self.is_cuda:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
            self._start_evt.record()
        else:
            self._t0 = perf_counter()
        return self

    def __exit__(self, *exc_info):
        if self.is_cuda:
            self._end_evt.record()
            torch.cuda.synchronize()
            self.elapsed = self._start_evt.elapsed_time(self._end_evt) / 1000
        else:
            self.elapsed = perf_counter() - self._t0
        return False


def time_median(fn, torch_dev, n_reps=N_TIME_REPS, warmup=True):
    if warmup:
        fn()
        if torch_dev.type == 'cuda':
            torch.cuda.synchronize()
    times = []
    for _ in range(n_reps):
        with GPUTimer(torch_dev) as t:
            fn()
        times.append(t.elapsed)
    return float(torch.tensor(times).median())


def load_reduced(name: str, torch_dev: torch.device) -> dict:
    """run_sweep spatial/time reductions, then flatten masked voxels."""
    ds = hybrid.load_dataset(name, torch_dev)
    phis, alphas, mask = ds.phis, ds.alphas, ds.mask
    if ds.reduced_im_size is not None:
        mask_r = reduce_spatial(mask, ds.reduced_im_size, order=3) > 0.5
        phis = reduce_spatial(phis, ds.reduced_im_size, order=3) * mask_r
        mask = mask_r
    if ds.time_reduction_factor is not None:
        nlow = max(int(round(alphas.shape[1] / ds.time_reduction_factor)), 1)
        alphas = reduce_temporal(alphas, nlow, dim=1, order=3)
    B = phis.shape[0]
    sel = torch.argwhere(mask.reshape(-1) > 0)[:, 0]
    phis_flt = phis.reshape(B, -1)[:, sel].contiguous()
    alphas_flt = alphas.reshape(B, -1).contiguous()
    return dict(
        name=name, phis=phis_flt, alphas=alphas_flt, mask=mask,
        reduced_im_size=ds.reduced_im_size,
        time_reduction_factor=ds.time_reduction_factor,
        cur_rank=int(ds.cur_rank),
        n_mask=int(sel.numel()),
        M_all=int(alphas_flt.shape[1]),
    )


def build_cur(d: dict, rank: int = CUR_RANK, seed: int = 0) -> torch.Tensor:
    """All reduced times × spatial CUR pivots (no extra time FPS)."""
    phis, alphas = d['phis'], d['alphas']
    Kp = min(int(rank), phis.shape[1])
    phis_full, alphas_full, phis_pivot, _, _, _ = _prepare_cur_space(
        phis, alphas, 'svd')
    phi_idxs = _pivot_indices(phis_pivot, Kp, 'maxmin', seed=seed)
    return torch.exp(-2j * math.pi * (
        alphas_full.T @ phis_full[:, phi_idxs]
    )).cfloat()


def svd_nrmse_curve(P: torch.Tensor, Ls: list[int]) -> list[float]:
    tot = P.norm().square().clamp_min(1e-30)
    s = torch.linalg.svdvals(P)
    c2 = torch.cumsum(s.square(), 0)
    n = int(s.numel())
    out = []
    for L in Ls:
        Lq = max(min(int(L), n), 1)
        out.append(float(torch.sqrt((tot - c2[Lq - 1]).clamp_min(0) / tot)))
    return out


def rank_at_error(Ls, errs, target: float):
    for L, e in zip(Ls, errs):
        if e is not None and e <= target:
            return int(L), float(e)
    return None, None


def nu_table(Ls, err_svd, err_by_k, targets=TARGET_ERRS) -> dict:
    out = {}
    for eps in targets:
        Ls_svd, _ = rank_at_error(Ls, err_svd, eps)
        row = {}
        for k, errs in err_by_k.items():
            L_k, e_k = rank_at_error(Ls, errs, eps)
            nu = (float(L_k) / Ls_svd) if (L_k and Ls_svd) else None
            row[str(k)] = dict(L_svd=Ls_svd, L_ksvd=L_k, nu=nu, err=e_k)
        out[f'{eps:.0e}'] = row
    return out


def plot_nrmse(name, kind, Ls, err_svd, err_by_k, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.semilogy(Ls, err_svd, 'k-o', label='SVD', ms=4)
    for k in sorted(err_by_k, key=int):
        ys = [np.nan if e is None else e for e in err_by_k[k]]
        ax.semilogy(Ls, ys, '-o', label=f'K-SVD k={k}', ms=4)
    ax.set_xlabel('L (rank / atoms)')
    ax.set_ylabel('phase NRMSE  ||P - P_L||_F / ||P||_F')
    ax.set_title(f'{name}  {kind}')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', ls=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_nu(name, kind, nu, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ks = None
    for eps, row in nu.items():
        ks = [int(k) for k in row]
        ys = [row[str(k)]['nu'] if row[str(k)]['nu'] is not None else np.nan
              for k in ks]
        ax.plot(ks, ys, '-o', label=f'ε={eps}')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, label='nu=1 (SVD)')
    ax.axhline(1.3, color='0.5', ls=':', lw=0.8, label='linop gate 1.3')
    ax.set_xlabel('k (nnz per time sample)')
    ax.set_ylabel('nu(k) = L_ksvd / L_svd')
    ax.set_title(f'{name}  {kind}  matched-error nu')
    ax.legend(fontsize=8)
    ax.grid(True, ls=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_matrix(P: torch.Tensor, Ls, ks, torch_dev, tag: str) -> dict:
    Ls = [int(L) for L in Ls if L <= min(P.shape)]
    print(f'    SVD curve {tuple(P.shape)} L={Ls}', flush=True)
    err_svd = svd_nrmse_curve(P, Ls)
    for L, e in zip(Ls, err_svd):
        print(f'      SVD L={L:4d}  nrmse={e:.4e}', flush=True)

    err_by_k = {str(k): [None] * len(Ls) for k in ks}
    for k in ks:
        for i, L in enumerate(Ls):
            if k >= L:
                continue
            fit = ksvd_temporal(P, n_atoms=L, k_sparse=k, n_iter=N_ITER)
            err_by_k[str(k)][i] = fit['err']
            print(f'      K-SVD k={k} L={L:4d}  nrmse={fit["err"]:.4e}  '
                  f'live={fit["n_live"]}/{L}', flush=True)
            del fit
            if err_by_k[str(k)][i] <= 1e-2:
                break

    L_max = Ls[-1]
    print(f'    timing L_max={L_max}', flush=True)
    t_svd = time_median(lambda: torch.svd_lowrank(P, q=L_max), torch_dev)
    print(f'      time SVD={t_svd:.3f}s', flush=True)
    t_ksvd = {}
    for k in ks:
        if k >= L_max:
            continue
        t_ksvd[str(k)] = time_median(
            lambda k=k: ksvd_temporal(P, n_atoms=L_max, k_sparse=k, n_iter=N_ITER),
            torch_dev,
        )
        print(f'      time K-SVD k={k}: {t_ksvd[str(k)]:.3f}s', flush=True)

    nu = nu_table(Ls, err_svd, err_by_k)
    return dict(
        shape=list(P.shape),
        Ls=Ls,
        err_svd=err_svd,
        err_ksvd=err_by_k,
        nu=nu,
        time=dict(svd_s=t_svd, ksvd_s=t_ksvd, L_max=L_max, n_iter=N_ITER),
        tag=tag,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    args = p.parse_args()
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={torch_dev}', flush=True)

    report = {}
    for name in args.datasets:
        print(f'\n======== {name} ========', flush=True)
        d = load_reduced(name, torch_dev)
        print(f'  reduced {d["reduced_im_size"]}  time/={d["time_reduction_factor"]}  '
              f'M_all={d["M_all"]}  N_mask={d["n_mask"]}', flush=True)
        rec = dict(meta=dict(
            reduced_im_size=list(d['reduced_im_size']) if d['reduced_im_size'] else None,
            time_reduction_factor=d['time_reduction_factor'],
            cur_rank=d['cur_rank'], M_all=d['M_all'], n_mask=d['n_mask'],
        ))
        print('  -- cur (all times x spatial pivots) --', flush=True)
        P = build_cur(d, rank=d['cur_rank'] or CUR_RANK)
        rec['cur'] = run_matrix(P, list(LS[name]), KS, torch_dev, 'cur')
        plot_nrmse(name, 'cur', rec['cur']['Ls'], rec['cur']['err_svd'],
                   rec['cur']['err_ksvd'],
                   OUT / f'nrmse_{name}.png')
        plot_nu(name, 'cur', rec['cur']['nu'],
                OUT / f'nu_{name}.png')
        del P
        if torch_dev.type == 'cuda':
            torch.cuda.empty_cache()
        report[name] = rec

    out_json = OUT / 'q1.json'
    out_json.write_text(json.dumps(report, indent=2))
    print(f'\nwrote {out_json}', flush=True)


if __name__ == '__main__':
    main()
