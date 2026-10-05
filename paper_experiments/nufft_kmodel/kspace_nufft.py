"""
NUFFT-style wrapper around Chan & Haldar's non-Cartesian k-space model.

The published operators map B-spline *coefficients* to k-space (H) and to the
image (T).  This wrapper inverts T on the nominal FOV (zero-pad + deapodize)
and applies H, which is the standard NUFFT factorization:

    image  --(1/Y)-->  pad  --FFT-->  B-spline interpolate  -->  k-space

``width`` is the B-spline support in oversampled grid units, matching SigPy's
Kaiser-Bessel width: a degree-p spline has support p+1 (default p=3 => width=4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from mr_recon.fourier.common import NUFFT
from mr_recon.dtypes import complex_dtype

_KMODEL_PY = Path(__file__).resolve().parents[2] / 'noncart-kmodel' / 'python'
if str(_KMODEL_PY) not in sys.path:
    sys.path.insert(0, str(_KMODEL_PY))

from kspace_model import KSpaceModel  # noqa: E402
from kspace_model.bspline import bspline  # noqa: E402
from kspace_model.backends.pytorch_backend import ft2_pt, ift2_pt  # noqa: E402
from kspace_model.util import to_even  # noqa: E402


def snap_rho(rho: float, n: int) -> float:
    """Match KSpaceModel's even grid size so FFT bins sit on the B-spline lattice."""
    return to_even(int(round(n * rho))) / n


class kspace_nufft(NUFFT):
    """B-spline k-space model as an ``mr_recon`` NUFFT."""

    def __init__(
        self,
        im_size: tuple,
        trj: torch.Tensor,
        oversamp: float = 1.3,
        width: int = 4,
        device=None,
    ):
        if len(im_size) != 2:
            raise ValueError('KSpaceModel only supports 2D images.')
        if any(n % 2 for n in im_size):
            raise ValueError('KSpaceModel requires even image sizes.')
        if width < 1:
            raise ValueError('width must be >= 1 (B-spline support p+1).')

        super().__init__(im_size)
        self.oversamp = snap_rho(float(oversamp), im_size[0])
        if im_size[0] != im_size[1]:
            rho = [snap_rho(float(oversamp), n) for n in im_size]
        else:
            rho = self.oversamp
        self.width = int(width)
        self.degree = self.width - 1
        self.trj_size = tuple(trj.shape[:-1])
        self.M = int(np.prod(self.trj_size))

        if device is None:
            device = trj.device
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        Psi, psi_img = bspline(self.degree)
        trj_np = trj.detach().to('cpu').numpy()
        k = (trj_np[..., 0].reshape(-1), trj_np[..., 1].reshape(-1))
        self.kmodel = KSpaceModel(
            k, im_size,
            Psi=Psi, psi_img=psi_img,
            rho=rho,
            backend='pytorch',
            device=self.device,
        )
        backend = self.kmodel.backend
        self.L = tuple(int(v) for v in self.kmodel.coeff_grid_size)
        self.Hmat = backend.Hmat
        self.Hmath = backend.Hmath
        self.start = list(backend.start)
        self.end = list(backend.end)

        Y = backend.Y_vec.reshape(self.L).real
        Y_nom = Y[self.start[0]:self.end[0], self.start[1]:self.end[1]]
        self.apod = (1.0 / Y_nom.clamp(min=1e-6)).to(complex_dtype)
        self._unit_scale = float(np.prod(im_size)) ** 0.5

    def _pad(self, img: torch.Tensor) -> torch.Tensor:
        batch = img.shape[:-2]
        g = img.new_zeros(*batch, *self.L)
        s0, s1 = self.start
        e0, e1 = self.end
        g[..., s0:e0, s1:e1] = img * self.apod
        return g

    def _crop(self, g: torch.Tensor) -> torch.Tensor:
        s0, s1 = self.start
        e0, e1 = self.end
        return g[..., s0:e0, s1:e1] * self.apod.conj()

    def forward(self, img: torch.Tensor, trj: torch.Tensor) -> torch.Tensor:
        batch = img.shape[:-2]
        x = img.reshape(-1, *self.im_size) / self._unit_scale
        g = self._pad(x)
        c = ft2_pt(g).reshape(x.shape[0], -1)
        ksp = torch.sparse.mm(self.Hmat, c.T).T
        return ksp.reshape(*batch, *self.trj_size)

    def adjoint(self, ksp: torch.Tensor, trj: torch.Tensor) -> torch.Tensor:
        batch = ksp.shape[:-len(self.trj_size)]
        y = ksp.reshape(-1, self.M)
        c = torch.sparse.mm(self.Hmath, y.T).T.reshape(-1, *self.L)
        g = ift2_pt(c) * float(np.prod(self.L))
        img = self._crop(g) / self._unit_scale
        return img.reshape(*batch, *self.im_size)
