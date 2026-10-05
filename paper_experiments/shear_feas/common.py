"""
Shared data loading for the per-factor trajectory shear feasibility study.

Mirrors the preprocessing in ``paper_experiments/run_sweep.py`` so the shear results are
directly comparable to the published HOFFT / SVD sweeps: same undersampling, same mask,
and the same ``remove_linear_terms`` step that already folds the *global* linear phase
into the trajectory. Per-factor shear therefore has to earn its keep on top of a
baseline that has already absorbed the global linear term.
"""
import torch

from dataclasses import dataclass
from typing import Optional

from hofft.phase_coeffs import remove_linear_terms
from hofft.utils import reduce_spatial


@dataclass
class ShearDataset:
    name: str
    phis: torch.Tensor          # (B, *im_size) reduced grid
    alphas: torch.Tensor        # (B, *trj_size)
    trj: torch.Tensor           # (*trj_size, d)
    mask: torch.Tensor          # (*im_size) reduced grid
    weights: torch.Tensor       # (N,) flattened clustering weights
    im_size: tuple              # reduced image shape
    im_size_full: tuple         # native image shape
    os: float
    kern_size: tuple
    Ls: list


DATASETS = {
    'coco_spiral': dict(
        R=3,
        reduced_im_size=(100, 100),
        Ls=[1, 2, 4, 8, 16],
        Ls_svd=[1, 5, 10, 15],
        Ls_hofft=[1, 3, 5, 7, 9],
    ),
    'tilt_spi_invivo': dict(
        R=2,
        reduced_im_size=(200, 200),
        Ls=[2, 4, 8, 16, 32],
        Ls_svd=[30, 40, 50, 60, 70, 80],
        Ls_hofft=[10, 15, 20, 25, 30],
    ),
}


def load_dataset(name: str,
                 torch_dev: torch.device,
                 os_nominal: float = 1.25,
                 W: int = 3,
                 mask_thresh: float = 0.9,
                 reduce: bool = True,
                 signal_weight: bool = False) -> ShearDataset:
    """
    Load one dataset with run_sweep preprocessing applied.

    Args
    ----
    name : str
        Dataset folder under ``./data``
    torch_dev : torch.device
        Device to load onto
    os_nominal : float
        Nominal grid oversampling, snapped to an even grid as in run_sweep
    W : int
        Interpolation stencil width
    mask_thresh : float
        ESPIRiT eigenvalue threshold for the support mask
    reduce : bool
        Downsample phis/mask to ``reduced_im_size`` for the decomposition
    signal_weight : bool
        Weight clustering by ``|img_gt|^2`` instead of the binary mask

    Returns
    -------
    ShearDataset
    """
    cfg = DATASETS[name]
    fpath = f'./data/{name}'
    load_kw = {'weights_only': True, 'map_location': torch_dev}

    trj = torch.load(f'{fpath}/trj.pt', **load_kw).type(torch.float32)
    mps = torch.load(f'{fpath}/mps.pt', **load_kw).type(torch.complex64)
    evals = torch.load(f'{fpath}/evals.pt', **load_kw).type(torch.float32)
    phis = torch.load(f'{fpath}/phis.pt', **load_kw).type(torch.float32)
    alphas = torch.load(f'{fpath}/alphas.pt', **load_kw).type(torch.float32)
    img_gt = torch.load(f'{fpath}/img_gt.pt', **load_kw).type(torch.complex64)

    im_size_full = mps.shape[1:]
    d = trj.shape[-1]
    del mps

    # Undersample exactly as run_sweep does
    R = cfg['R']
    trj = trj[:, ::R].contiguous()
    alphas = alphas[:, :, ::R].contiguous()

    mask = (evals > mask_thresh).float()

    # Fold global linear phase into the trajectory (run_sweep baseline)
    phis, trj_term, _ = remove_linear_terms(phis, alphas, mask=mask)
    trj = trj + trj_term
    phis = phis * mask

    # Drop bases that carry no energy
    B = phis.shape[0]
    energy = (phis.reshape((B, -1)).abs().mean(dim=1)
              * alphas.reshape((B, -1)).abs().mean(dim=1))
    idxs = torch.argwhere(energy > 1e-6)[:, 0]
    phis, alphas = phis[idxs].contiguous(), alphas[idxs].contiguous()

    weights_full = mask * img_gt.abs().square() if signal_weight else mask

    if reduce:
        im_size = cfg['reduced_im_size']
        phis_r = reduce_spatial(phis, im_size_low=im_size, order=3)
        mask_r = reduce_spatial(mask, im_size_low=im_size, order=3)
        weights_r = reduce_spatial(weights_full, im_size_low=im_size, order=3)
        mask_r = (mask_r > 0.5).float()
        weights_r = (weights_r * mask_r).clamp(min=0.0)
        phis_r = phis_r * mask_r
    else:
        im_size = im_size_full
        phis_r, mask_r, weights_r = phis, mask, weights_full

    os = 2 * round(os_nominal * im_size_full[0] / 2) / im_size_full[0]

    return ShearDataset(
        name=name,
        phis=phis_r,
        alphas=alphas,
        trj=trj,
        mask=mask_r,
        weights=weights_r.reshape(-1),
        im_size=tuple(im_size),
        im_size_full=tuple(im_size_full),
        os=os,
        kern_size=(W,) * d,
        Ls=cfg['Ls'],
    )


def subsample_times(alphas: torch.Tensor,
                    sigma_sqrt: torch.Tensor,
                    n_sub: int,
                    pool: int = 30_000,
                    seed: int = 0) -> torch.Tensor:
    """
    Farthest-point subsample of time points in whitened alpha space.

    Maximizing over time is really a maximization over the convex hull of the alpha
    cloud, and farthest-point selection picks hull vertices first.

    Args
    ----
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    sigma_sqrt : torch.Tensor
        Whitening matrix with shape (B, B)
    n_sub : int
        Number of time points to keep
    pool : int
        Random candidate pool size
    seed : int
        RNG seed

    Returns
    -------
    idxs : torch.Tensor
        Flat time indices with shape (min(n_sub, pool),)
    """
    from hofft.utils import maxmin_indices

    B = alphas.shape[0]
    a = alphas.reshape((B, -1))
    M = a.shape[1]
    gen = torch.Generator(device=a.device)
    gen.manual_seed(seed)
    if M > pool:
        cand = torch.randperm(M, generator=gen, device=a.device)[:pool]
    else:
        cand = torch.arange(M, device=a.device)
    aw = (sigma_sqrt @ a[:, cand]).T.contiguous()
    n_sub = min(n_sub, cand.numel())
    return cand[maxmin_indices(aw, n_sub, seed=seed)]
