"""
Shared problem setup for the temporally sparse factor study.

Datasets, preprocessing and anchor selection are pulled straight from
``paper_experiments/qblock_feas/common.py`` so that ``L_svd`` here is the *same* number as
``L`` in the block study's Stage A tables. That is what lets Sec. 4's Stage 2 multiply the
two penalties (``r`` from blocking, ``nu`` from sparsity) instead of re-deriving a
baseline.

The dense target is the field phase only,

    P(t, r) = exp(-2j pi phi(r) . alpha(t))

matching ``qblock_feas.blocks.phase_matrix``. The sub-grid ``k_dev`` term is carried by the
NUFFT itself and is not part of the rank being sparsified.
"""
import importlib.util
import sys

from pathlib import Path

import numpy as np
import torch

_QBLOCK = Path(__file__).resolve().parent.parent / 'qblock_feas'
if str(_QBLOCK) not in sys.path:
    sys.path.append(str(_QBLOCK))


def _load_qblock(name):
    """Import a qblock_feas module under a prefixed name (both folders ship common.py)."""
    key = f'qblock_{name}'
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _QBLOCK / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


_qb_common = _load_qblock('common')
blocks = _load_qblock('blocks')

REAL = _qb_common.REAL
SYNTH = _qb_common.SYNTH
Problem = _qb_common.Problem
load_problem = _qb_common.load_problem
select_anchors = _qb_common.select_anchors
phase_matrix = blocks.phase_matrix

# rho measured with cufinufft at W~3, extra R=2 (qblock_feas/results/rho_cufinufft_R2.json)
RHO = {'coco_spiral': 0.638, 'tilt_spi_invivo': 0.178, 'coco_7t_spi': 0.898}
RHO_SWEEP = (0.32, 0.60, 1.87)      # the three values Sec. 4 Stage 2 asks for
DATASETS = ('coco_spiral', 'tilt_spi_invivo', 'coco_7t_spi')


def build_dense(name: str,
                torch_dev: torch.device,
                n_anchors: int = 256,
                anchor_mode: str = 'fps',
                seed: int = 0) -> dict:
    """
    Dense phase matrix plus everything the sparse arms need.

    Args
    ----
    name : str
        Dataset key in ``REAL`` or ``SYNTH``
    torch_dev : torch.device
        Device
    n_anchors : int
        Number of time samples in ``P``
    anchor_mode : str
        ``'fps'`` farthest-point in the whitened alpha cloud (matches Stage A),
        ``'uniform'`` evenly strided in acquisition order, or ``'contiguous'`` a single
        unbroken block of samples. Run structure (Sec. 3.5) is a statement about
        *consecutive* samples, so only ``'contiguous'`` measures it honestly -- under
        ``'uniform'`` a run of 256 rows spans ``256 * stride`` real samples.
    seed : int
        RNG seed

    Returns
    -------
    dict
        ``P`` (M, N) complex128, ``phis`` (B, N), ``alphas`` (B, M), ``prob``, ``t_idx``
    """
    prob = load_problem(name, torch_dev)
    sel = torch.argwhere(prob.mask.reshape(-1))[:, 0]
    M_all = prob.alphas.shape[1]

    if anchor_mode == 'fps':
        t_idx = select_anchors(prob.alphas, n_anchors,
                               pool=max(20_000, 4 * n_anchors), seed=seed)
        t_idx, _ = t_idx.sort()          # keep acquisition order for arc length
    elif anchor_mode == 'uniform':
        stride = max(M_all // n_anchors, 1)
        t_idx = torch.arange(0, stride * min(n_anchors, M_all // stride), stride,
                             device=prob.alphas.device)
    elif anchor_mode == 'contiguous':
        M = min(n_anchors, M_all)
        start = max((M_all - M) // 2, 0)          # mid-readout, away from both edges
        t_idx = torch.arange(start, start + M, device=prob.alphas.device)
    else:
        raise ValueError(anchor_mode)

    alphas = prob.alphas[:, t_idx].contiguous()
    phis_flt = prob.phis.reshape((prob.phis.shape[0], -1))[:, sel].contiguous()
    P = phase_matrix(phis_flt, alphas)
    return dict(P=P, phis=phis_flt, alphas=alphas, prob=prob, t_idx=t_idx,
                sel=sel, n_mask=int(sel.numel()), anchor_mode=anchor_mode)


def svd_error_curve(P: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Relative Frobenius error of the optimal rank-L approximation, for every L."""
    s = torch.linalg.svdvals(P).double().cpu()
    tot = float(s.square().sum())
    tail = tot - torch.cumsum(s.square(), dim=0)
    tail = np.concatenate([[tot], tail.clamp(min=0).numpy()])
    return np.arange(len(tail)), np.sqrt(np.maximum(tail, 0) / tot)


def rank_at_error(L_ax: np.ndarray, err: np.ndarray, target: float):
    """Smallest rank reaching ``target``; None if the curve never gets there."""
    ok = np.argwhere(err <= target)[:, 0]
    return float(L_ax[ok[0]]) if ok.size else None


def err_at_rank(L_ax: np.ndarray, err: np.ndarray, L: int) -> float:
    """Error of the optimal rank-``L`` approximation."""
    return float(err[min(int(L), len(err) - 1)])
