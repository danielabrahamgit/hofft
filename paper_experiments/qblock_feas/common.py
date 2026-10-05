"""
Shared problem setup for the block-partitioned SVD feasibility study.

Real-dataset preprocessing mirrors ``paper_experiments/run_sweep.py`` exactly -- same
``R``, same ESPIRiT mask threshold, same global ``remove_linear_terms`` -- so that the
per-block affine fit is removing *residual local* affine content rather than content the
baseline never saw. Everything is returned in float64 (Sec. 7).

The synthetic controls of Sec. 5 are built here too, so all five problems flow through one
code path in Stage A.
"""
import numpy as np
import torch

from dataclasses import dataclass
from typing import Optional

from hofft.phase_coeffs import remove_linear_terms
from hofft.shear import alpha_second_moment
from hofft.utils import gen_grd, maxmin_indices, reduce_spatial

OS_NOMINAL = 1.25
MASK_THRESH = 0.9
W_NOMINAL = 3

REAL = {
    'coco_spiral':     dict(R=3, im_size_s=(100, 100)),
    'tilt_spi_invivo': dict(R=2, im_size_s=(200, 200)),
    # Concomitant-only field matching trj.pt. Extra sample_r=2 is the R=2 operating
    # point the cufinufft rho was measured at. Spatial grid is reduced; cost uses 320^3.
    'coco_7t_spi':     dict(R=1, sample_r=2, im_size_s=(64, 64, 64),
                            phis='phis_coco.pt', alphas='alphas_coco.pt'),
}
SYNTH = ('deff1_quad_x', 'deff2_quad_xy', 'affine')


@dataclass
class Problem:
    """One feasibility problem: a phase field, a trajectory, and a support mask."""
    name: str
    kind: str                    # 'real' or 'synthetic'
    phis: torch.Tensor           # (B, *im_size) float64, global linear terms removed
    alphas: torch.Tensor         # (B, M) float64
    trj: torch.Tensor            # (M, d) float64, cycles / FOV
    mask: torch.Tensor           # (*im_size) bool
    im_size: tuple               # grid the study runs on
    im_size_full: tuple          # native grid, used for cost accounting
    os: float
    W: int = W_NOMINAL

    @property
    def d(self) -> int:
        return len(self.im_size)

    @property
    def n_mask(self) -> int:
        return int(self.mask.sum())


def _snap_os(os_nominal: float, n: int) -> float:
    """Even oversampled grid, as ``run_sweep`` does."""
    return 2 * round(os_nominal * n / 2) / n


def select_anchors(alphas: torch.Tensor,
                   M: int,
                   pool: int = 20_000,
                   seed: int = 0) -> torch.Tensor:
    """
    Farthest-point anchor times in the whitened coefficient space.

    Maximizing model error over time is really a maximization over the convex hull of the
    alpha cloud, and farthest-point selection picks hull vertices first. Indices (rather
    than the centroids ``k_alpha_selection`` returns) are needed so the matching ``trj``
    rows can be pulled for Stage B.

    Args
    ----
    alphas : torch.Tensor
        Temporal coefficients with shape (B, M_all)
    M : int
        Number of anchors
    pool : int
        Random candidate pool size
    seed : int
        RNG seed

    Returns
    -------
    idx : torch.Tensor
        Flat time indices with shape (min(M, pool),)
    """
    B = alphas.shape[0]
    a = alphas.reshape((B, -1)).float()
    _, sig_sqrt = alpha_second_moment(a)
    gen = torch.Generator(device=a.device)
    gen.manual_seed(seed)
    cand = (torch.randperm(a.shape[1], generator=gen, device=a.device)[:pool]
            if a.shape[1] > pool else torch.arange(a.shape[1], device=a.device))
    aw = (sig_sqrt @ a[:, cand]).T.contiguous()
    return cand[maxmin_indices(aw, min(M, cand.numel()), seed=seed)]


def load_real(name: str, torch_dev: torch.device) -> Problem:
    """Load one real dataset with ``run_sweep`` preprocessing applied."""
    cfg = REAL[name]
    fpath = f'./data/{name}'
    kw = {'weights_only': True, 'map_location': torch_dev}

    trj = torch.load(f'{fpath}/trj.pt', **kw).float()
    evals = torch.load(f'{fpath}/evals.pt', **kw).float()
    phis = torch.load(f'{fpath}/{cfg.get("phis", "phis.pt")}', **kw).float()
    alphas = torch.load(f'{fpath}/{cfg.get("alphas", "alphas.pt")}', **kw).float()
    im_size_full = tuple(evals.shape)
    d = trj.shape[-1]

    # 2D run_sweep convention: stride the interleave axis. 3D / extra R=2: stride
    # flattened samples (or the readout axis) so M halves regardless of layout.
    R = cfg['R']
    sample_r = int(cfg.get('sample_r', 1))
    if trj.ndim >= 3 and R > 1:
        trj = trj[:, ::R].contiguous()
        # alphas is (B, *trj_size); stride the same axis as trj's interleave
        alphas = alphas[:, :, ::R].contiguous()
    if sample_r > 1:
        trj = trj.reshape(-1, d)[::sample_r].contiguous()
        alphas = alphas.reshape(alphas.shape[0], -1)[:, ::sample_r].contiguous()
    mask = (evals > MASK_THRESH).float()

    # Fold the global linear phase into the trajectory (the run_sweep baseline)
    phis, trj_term, _ = remove_linear_terms(phis, alphas, mask=mask)
    trj = trj + trj_term
    phis = phis * mask

    # Drop bases that carry no energy
    B = phis.shape[0]
    energy = (phis.reshape((B, -1)).abs().mean(dim=1)
              * alphas.reshape((B, -1)).abs().mean(dim=1))
    keep = torch.argwhere(energy > 1e-6)[:, 0]
    phis, alphas = phis[keep].contiguous(), alphas[keep].contiguous()

    # Reduce to the study grid
    im_size = tuple(cfg['im_size_s'])
    mask_r = reduce_spatial(mask, im_size_low=im_size, order=3) > 0.5
    phis_r = reduce_spatial(phis, im_size_low=im_size, order=3) * mask_r

    return Problem(
        name=name, kind='real', phis=phis_r.double(),
        alphas=alphas.reshape((phis_r.shape[0], -1)).double(),
        trj=trj.reshape((-1, d)).double(), mask=mask_r,
        im_size=im_size, im_size_full=im_size_full,
        os=_snap_os(OS_NOMINAL, im_size_full[0]),
    )


def _shepp_logan_mask(im_size: tuple, torch_dev: torch.device) -> torch.Tensor:
    """Shepp-Logan support, or an inscribed ellipse if sigpy is unavailable."""
    try:
        import sigpy as sp
        img = torch.as_tensor(np.asarray(sp.shepp_logan(im_size)), device=torch_dev)
        return img.abs() > 1e-6
    except Exception:
        rs = gen_grd(im_size).to(torch_dev)
        return (rs[..., 0].square() / 0.42 ** 2 + rs[..., 1].square() / 0.34 ** 2) < 1.0


def make_synthetic(kind: str,
                   torch_dev: torch.device,
                   im_size: tuple = (128, 128),
                   M: int = 20_000,
                   wraps: float = 5.0,
                   seed: int = 0) -> Problem:
    """
    One of the Sec. 5 synthetic controls.

    Args
    ----
    kind : str
        ``deff1_quad_x`` (``phi = r_x^2``), ``deff2_quad_xy`` (``phi = r_x^2 + r_y^2``),
        or ``affine`` (``phi = a + b.r``)
    torch_dev : torch.device
        Device
    im_size : tuple
        Grid size
    M : int
        Number of time points
    wraps : float
        Peak phase in cycles over the support
    seed : int
        RNG seed

    Returns
    -------
    Problem
    """
    d = len(im_size)
    rs = gen_grd(im_size).to(torch_dev).double()
    mask = _shepp_logan_mask(im_size, torch_dev)

    if kind == 'deff1_quad_x':
        phis = rs[..., 0].square()[None]
    elif kind == 'deff2_quad_xy':
        phis = (rs[..., 0].square() + rs[..., 1].square())[None]
    elif kind == 'affine':
        phis = (0.3 + 0.7 * rs[..., 0] - 0.4 * rs[..., 1])[None]
    else:
        raise ValueError(kind)

    # alpha(t) = t, scaled so the peak phase over the support is ``wraps`` cycles
    gen = torch.Generator(device=torch_dev)
    gen.manual_seed(seed)
    t = torch.linspace(0, 1, M, device=torch_dev, dtype=torch.float64)
    span = float(phis[0][mask].abs().max())
    alphas = (wraps / max(span, 1e-12)) * t[None]

    # A spiral-ish trajectory, only used by the block-gather geometry in Stage B
    ang = 32 * np.pi * t
    rad = (im_size[0] / 2) * t
    trj = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)]
                      + [torch.zeros_like(t)] * (d - 2), dim=-1)

    # The exactly-affine control must keep its affine content, otherwise the global
    # ``remove_linear_terms`` deletes the very thing the per-block fit is meant to absorb.
    if kind != 'affine':
        phis, trj_term, _ = remove_linear_terms(
            phis.float(), alphas.float()[..., None].squeeze(-1), mask=mask.float())
        phis = phis.double()
        trj = trj + trj_term.double()
    phis = phis * mask

    return Problem(
        name=kind, kind='synthetic', phis=phis.double(), alphas=alphas,
        trj=trj.double(), mask=mask, im_size=tuple(im_size),
        im_size_full=tuple(im_size), os=_snap_os(OS_NOMINAL, im_size[0]),
    )


def load_problem(name: str, torch_dev: torch.device, **kw) -> Problem:
    """Dispatch to ``load_real`` or ``make_synthetic``."""
    return load_real(name, torch_dev) if name in REAL else make_synthetic(
        name, torch_dev, **kw)
