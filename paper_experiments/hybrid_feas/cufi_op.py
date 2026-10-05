"""
Stock cuFINUFFT operators for the hybrid study.

``CufiNUFFT`` implements ``mr_recon.fourier.NUFFT`` with reusable Plans so the
dense multi-basis model can go through ``sense_linop`` / ``CG_SENSE_recon``
without substituting another NUFFT library. Coordinates stay GPU-resident.

Normalization matches the existing ``cufi_nufft`` wrapper (1/sqrt(N)) so the
adjoint identity and CG scaling stay consistent with the rest of the stack.
"""
from typing import Optional

import numpy as np
import torch

import cufinufft

from hofft.cur_ops import build_cur_factors
from hofft.pipelines import _process_phase_coefficients
from hofft.utils import expand_spatial, expand_temporal, reduce_spatial, reduce_temporal
from mr_recon.fourier.common import NUFFT
from mr_recon.linops import batching_params, sense_linop


class CufiNUFFT(NUFFT):
    """Type-2 forward / type-1 adjoint with cached Plans keyed on n_trans."""

    def __init__(self, im_size: tuple, eps: float = 1e-4, dtype: str = 'complex64'):
        super().__init__(im_size)
        self.eps = float(eps)
        self.dtype = dtype
        self._fwd = {}
        self._adj = {}
        self._pts_id = None
        self.plan_kwargs = dict(
            eps=self.eps, dtype=dtype, gpu_method=1, gpu_sort=1,
            gpu_kerevalmeth=1, upsampfac=2.0, modeord=0,
        )

    def rescale_trajectory(self, trj: torch.Tensor) -> torch.Tensor:
        im = torch.tensor(self.im_size, device=trj.device, dtype=trj.dtype)
        tup = (None,) * (trj.ndim - 1) + (slice(None),)
        out = torch.pi * trj / (im[tup] / 2)
        lim = torch.pi - 1e-4
        return out.clamp(-lim, lim).contiguous()

    def _setpts_args(self, trj_pi: torch.Tensor):
        d = trj_pi.shape[-1]
        pts = trj_pi.reshape(-1, d).T.contiguous()   # (d, M), each row C-contiguous
        return [pts[i] for i in range(d)], pts.shape[1]

    def _fwd_plan(self, n_trans: int, args, M: int):
        key = (n_trans, M)
        if key not in self._fwd:
            try:
                plan = cufinufft.Plan(2, self.im_size, n_trans=n_trans,
                                      isign=-1, **self.plan_kwargs)
            except RuntimeError:
                kw = dict(self.plan_kwargs)
                kw['upsampfac'] = 1.0
                kw['gpu_kerevalmeth'] = 0
                plan = cufinufft.Plan(2, self.im_size, n_trans=n_trans,
                                      isign=-1, **kw)
            plan.setpts(*args)
            self._fwd[key] = plan
        return self._fwd[key]

    def _adj_plan(self, n_trans: int, args, M: int):
        key = (n_trans, M)
        if key not in self._adj:
            try:
                plan = cufinufft.Plan(1, self.im_size, n_trans=n_trans,
                                      isign=+1, **self.plan_kwargs)
            except RuntimeError:
                kw = dict(self.plan_kwargs)
                kw['upsampfac'] = 1.0
                kw['gpu_kerevalmeth'] = 0
                plan = cufinufft.Plan(1, self.im_size, n_trans=n_trans,
                                      isign=+1, **kw)
            plan.setpts(*args)
            self._adj[key] = plan
        return self._adj[key]

    def forward(self, img: torch.Tensor, trj: torch.Tensor) -> torch.Tensor:
        d = len(self.im_size)
        args, M = self._setpts_args(trj)
        batch = img.shape[:-d]
        n_trans = int(np.prod(batch))
        x = img.reshape(n_trans, *self.im_size).contiguous()
        y = self._fwd_plan(n_trans, args, M).execute(x)
        y = torch.as_tensor(y, device=img.device)
        if y.ndim == 1:
            y = y[None]
        scale = float(np.prod(self.im_size)) ** 0.5
        return (y.reshape(*batch, *trj.shape[1:-1]) / scale).type(img.dtype)

    def adjoint(self, ksp: torch.Tensor, trj: torch.Tensor) -> torch.Tensor:
        d = len(self.im_size)
        args, M = self._setpts_args(trj)
        trj_size = trj.shape[1:-1]
        n_trj = int(np.prod(trj_size))
        batch = ksp.shape[:-len(trj_size)]
        n_trans = int(np.prod(batch))
        c = ksp.reshape(n_trans, n_trj).contiguous()
        x = self._adj_plan(n_trans, args, M).execute(c)
        x = torch.as_tensor(x, device=ksp.device)
        if x.ndim == d:
            x = x[None]
        scale = float(np.prod(self.im_size)) ** 0.5
        return (x.reshape(*batch, *self.im_size) / scale).type(ksp.dtype)

    def clear_plans(self):
        self._fwd.clear()
        self._adj.clear()


def factor_phase_cur(ds, L: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rank-L CUR+SVD factors of the residual phase, without forming P.

    Same path as ``svd_decomp_linop(..., svd_method='cur')``: process / reduce /
    CUR / thin SVD / expand back to the native grid and full readout.
    """
    phis, alphas = ds.phis, ds.alphas
    ncomp = ds.num_compressed_bases
    if ncomp is not None:
        ncomp = min(ncomp, phis.shape[0])
    phis_n, alphas_n, spat, temp = _process_phase_coefficients(
        phis, alphas, normalize_coeffs=True, num_compressed_bases=ncomp)
    mask = ds.mask
    if ds.reduced_im_size is not None:
        phis_n = reduce_spatial(phis_n, im_size_low=ds.reduced_im_size, order=3)
        mask_r = reduce_spatial(mask, im_size_low=ds.reduced_im_size, order=3) > 0.5
        phis_n = phis_n * mask_r
    else:
        mask_r = mask > 0
    if ds.time_reduction_factor is not None:
        n_low = max(round(alphas_n.shape[1] / ds.time_reduction_factor), 2)
        alphas_n = reduce_temporal(alphas_n, num_time_low=n_low, dim=1, order=3)

    B = phis_n.shape[0]
    Rcur, Ccur = build_cur_factors(
        phis_n.reshape((B, -1)), alphas_n.reshape((B, -1)),
        rank=ds.cur_rank, normalize_method='svd', cluster_method='maxmin',
        seed=seed)
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
        temporal = expand_temporal(temporal, num_time_high=alphas.shape[1], dim=1, order=3)
    if ds.reduced_im_size is not None:
        spatial = expand_spatial(spatial, ds.im_size, order=3)
    spatial = spatial * spat
    temporal = temporal * temp
    return spatial.contiguous(), temporal.contiguous()


def make_dense_linop(ds, spatial, temporal, eps: float = 1e-4,
                     coil_batch: Optional[int] = None,
                     field_batch: int = 4):
    """Dense rank-L cuFINUFFT sense operator, plans reused across calls."""
    nufft = CufiNUFFT(ds.im_size, eps=eps)
    C = ds.C
    cb = coil_batch if coil_batch is not None else max(C // 2, 1)
    bparams = batching_params(coil_batch_size=cb, field_batch_size=field_batch)
    return sense_linop(
        ds.trj, ds.mps, dcf=ds.dcf, nufft=nufft,
        spatial_funcs=spatial, temporal_funcs=temporal, bparams=bparams,
    ), nufft
