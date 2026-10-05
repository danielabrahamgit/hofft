"""
Shared data loading for the hybrid feasibility study (math_docs/hybrid_feas.md).

Preprocessing mirrors ``paper_experiments/run_sweep.py``: same R, ESPIRiT mask,
``remove_linear_terms``, density compensation, and the same reduced grid used only
for the phase factorization. Reconstruction stays on the native coil-map grid.
Each dataset is inventoried from its files; nothing is inferred from the name.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from hofft.phase_coeffs import remove_linear_terms
from mr_recon.algs import density_compensation

MASK_THRESH = 0.9
OS_NOMINAL = 1.25

# Reconstruction preprocessing used by run_sweep. Factorization reductions are
# disclosed; they do not change the image grid.
DATASETS = {
    'coco_spiral': dict(
        R=3, reduced_im_size=(100, 100), time_reduction_factor=None,
        cur_rank=500, num_compressed_bases=4,
    ),
    'tilt_spi_invivo': dict(
        R=2, reduced_im_size=(200, 200), time_reduction_factor=10,
        cur_rank=500, num_compressed_bases=10,
    ),
}


def _snap_os(os_nominal: float, n: int) -> float:
    return 2 * round(os_nominal * n / 2) / n


@dataclass
class Dataset:
    """One measured volume plus the phase model, after run_sweep preprocessing."""
    name: str
    ksp: torch.Tensor          # (C, *trj_size) complex64
    mps: torch.Tensor          # (C, *im_size) complex64, masked
    trj: torch.Tensor          # (*trj_size, d) float32, cycles/FOV, linear terms folded in
    dcf: torch.Tensor          # (*trj_size,)
    phis: torch.Tensor         # (B, *im_size) float32, linear terms removed, masked
    alphas: torch.Tensor       # (B, *trj_size) float32
    mask: torch.Tensor         # (*im_size,) float
    im_size: tuple
    trj_size: tuple
    os: float
    R: int
    reduced_im_size: Optional[tuple]
    time_reduction_factor: Optional[int]
    cur_rank: int
    num_compressed_bases: Optional[int]
    img_gt: Optional[torch.Tensor]
    img_ee: Optional[torch.Tensor]

    @property
    def C(self) -> int:
        return int(self.mps.shape[0])

    @property
    def B(self) -> int:
        return int(self.phis.shape[0])

    @property
    def d(self) -> int:
        return len(self.im_size)

    @property
    def N(self) -> int:
        return int(np.prod(self.im_size))

    @property
    def M(self) -> int:
        return int(np.prod(self.trj_size))

    @property
    def n_mask(self) -> int:
        return int((self.mask > 0).sum())


def load_dataset(name: str, torch_dev: torch.device) -> Dataset:
    if name not in DATASETS:
        raise KeyError(f'unknown dataset {name}; files decide the rest')
    cfg = DATASETS[name]
    fpath = Path('./data') / name
    kw = {'weights_only': True, 'map_location': torch_dev}

    trj = torch.load(fpath / 'trj.pt', **kw).float()
    dcf = torch.load(fpath / 'dcf.pt', **kw).float()
    mps = torch.load(fpath / 'mps.pt', **kw).type(torch.complex64)
    ksp = torch.load(fpath / 'ksp.pt', **kw).type(torch.complex64)
    evals = torch.load(fpath / 'evals.pt', **kw).float()
    phis = torch.load(fpath / 'phis.pt', **kw).float()
    alphas = torch.load(fpath / 'alphas.pt', **kw).float()
    img_gt = None
    if (fpath / 'img_gt.pt').exists():
        img_gt = torch.load(fpath / 'img_gt.pt', **kw).type(torch.complex64)

    R = cfg['R']
    # Undersample the same axis run_sweep does (interleave).
    if trj.ndim >= 3 and R > 1:
        trj = trj[:, ::R].contiguous()
        ksp = ksp[:, :, ::R].contiguous()
        alphas = alphas[:, :, ::R].contiguous()

    im_size = tuple(mps.shape[1:])
    dcf = density_compensation(trj, im_size)
    mask = (evals > MASK_THRESH).float()
    mps = mps * mask

    phis_new, trj_term, zeroth = remove_linear_terms(phis, alphas, mask=mask)
    trj = trj + trj_term
    ksp = ksp * torch.exp(2j * torch.pi * zeroth)
    phis = phis_new * mask

    B = phis.shape[0]
    energy = (phis.reshape((B, -1)).abs().mean(dim=1)
              * alphas.reshape((B, -1)).abs().mean(dim=1))
    keep = torch.argwhere(energy > 1e-6)[:, 0]
    phis, alphas = phis[keep].contiguous(), alphas[keep].contiguous()

    img_ee = None
    ee_path = fpath / f'img_ee_R{R}.pt'
    if ee_path.exists():
        img_ee = torch.load(ee_path, map_location=torch_dev).type(torch.complex64)

    return Dataset(
        name=name, ksp=ksp, mps=mps, trj=trj, dcf=dcf, phis=phis, alphas=alphas,
        mask=mask, im_size=im_size, trj_size=tuple(trj.shape[:-1]),
        os=_snap_os(OS_NOMINAL, im_size[0]), R=R,
        reduced_im_size=cfg['reduced_im_size'],
        time_reduction_factor=cfg['time_reduction_factor'],
        cur_rank=cfg['cur_rank'],
        num_compressed_bases=cfg['num_compressed_bases'],
        img_gt=img_gt, img_ee=img_ee,
    )


def memory_estimate(ds: Dataset, L: int = 32, bytes_c64: int = 8) -> dict:
    """Bytes that would be touched; flags an infeasible dense P."""
    N, M, C, B = ds.N, ds.M, ds.C, ds.B
    p_dense = N * M * bytes_c64
    factors = L * (N + M) * bytes_c64
    ksp = C * M * bytes_c64
    mps = C * N * bytes_c64
    nufft_batch = C * L * N * bytes_c64   # one (C,L) NUFFT batch
    return dict(
        N=N, M=M, C=C, B=B, L=L,
        P_dense_GiB=p_dense / 2**30,
        factors_MiB=factors / 2**20,
        ksp_MiB=ksp / 2**20,
        mps_MiB=mps / 2**20,
        nufft_batch_MiB=nufft_batch / 2**20,
        P_infeasible=p_dense > 4 * 2**30,
    )


def nrmse(img, ref, mask=None):
    if mask is not None:
        img, ref = img * mask, ref * mask
    den = torch.linalg.norm(ref)
    if float(den) <= 0:
        raise ValueError('reference norm is empty/near-zero')
    return float(torch.linalg.norm(img - ref) / den)
