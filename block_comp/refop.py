"""
Baseline A: global coil compression + cuFINUFFT (feas.md Sec. 4.1, Sec. 9, Sec. 17).

Everything downstream is measured against this. Three pieces:

* :class:`SenseOp`   -- the complex64 reference operator, DCF kept *outside* so the
  forward/adjoint pair is a true adjoint with nothing hidden in it.
* :class:`Cufi64`    -- a minimal double-precision cuFINUFFT wrapper. ``mr_recon``'s
  ``cufi_nufft`` advertises ``dtype='complex128'`` but its data path casts to the
  module-global ``mr_recon.dtypes.complex_dtype`` (``torch.complex64``), so feas.md
  Sec. 9's ``eps_adj < 1e-8`` target is unreachable through it, and editing
  ``dtypes.py`` would perturb every other consumer of the library.
* :func:`profile_split` -- the measured FFT / interpolation split, which fixes the
  Amdahl ceiling ``(1 + rho) / rho`` on *any* method that only shrinks FFT work.

The ceiling matters more than it looks: a hierarchical method still pays the full
``C' W M`` root interpolation, exactly what Baseline A pays, so ``rho`` bounds its best
possible speedup before a line of it is written. Same bound as
``qblock_feas/rho_profile.py``.
"""
import numpy as np
import torch

from typing import Optional

import cufinufft

from mr_recon.fourier import cufi_nufft, eps_from_width, nspread_from_eps

from common import sync, repeat_time

OS_DEFAULT = 2.0        # cuFINUFFT upsampfac; 1.25 or 2.0 keep the Horner kernel
W_DEFAULT = 5           # nspread on the fine grid -> eps ~ 4.5e-5 at os=2


# ---------------------------------------------------------------------------
class SenseOp:
    """
    ``y_c(m) = sum_r x(r) s_c(r) exp(-2j pi k_m . r) / sqrt(N)``.

    ``forward`` takes ``(*im_size)`` and returns ``(C, M)``; ``adjoint`` is the exact
    Hermitian transpose. No DCF, no normalization beyond the ``1/sqrt(N)`` that
    ``cufi_nufft`` applies to both directions.
    """

    def __init__(self,
                 mps: torch.Tensor,
                 trj: torch.Tensor,
                 oversamp: float = OS_DEFAULT,
                 width: int = W_DEFAULT,
                 n_trans: Optional[int] = None):
        self.mps = mps
        self.im_size = tuple(mps.shape[1:])
        self.C = mps.shape[0]
        self.M = trj.shape[0]
        self.oversamp, self.width = float(oversamp), int(width)
        self.eps = eps_from_width(self.width, self.oversamp)
        self.nufft = cufi_nufft(self.im_size, oversamp=oversamp, width=width,
                                n_trans=n_trans or self.C)
        # Hoist the rescaled trajectory: cufi_nufft re-validates points by full
        # allclose whenever the tensor identity changes.
        self.trj = self.nufft.rescale_trajectory(trj)[None].contiguous()
        self.nufft.plan(self.trj, n_trans=n_trans or self.C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nufft.forward((self.mps * x)[None], self.trj)[0]

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        img = self.nufft.adjoint(y[None], self.trj)[0]
        return (self.mps.conj() * img).sum(0)

    def normal(self, x: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(x))

    def with_maps(self, mps: torch.Tensor) -> 'SenseOp':
        """A sibling operator sharing this one's NUFFT plan but different maps."""
        out = object.__new__(SenseOp)
        out.__dict__.update(self.__dict__)
        out.mps = mps
        return out


# ---------------------------------------------------------------------------
class Cufi64:
    """
    Double-precision cuFINUFFT, for the Sec. 9 correctness tests only.

    Same conventions as ``cufi_nufft``: type-2 forward with ``isign=-1``, type-1 adjoint
    with ``isign=+1``, both scaled by ``1/sqrt(prod(im_size))``, ``modeord=0`` so modes
    run ``-N/2 .. N/2-1``.
    """

    def __init__(self,
                 im_size: tuple,
                 trj: torch.Tensor,
                 n_trans: int = 1,
                 oversamp: float = OS_DEFAULT,
                 width: int = W_DEFAULT):
        self.im_size = tuple(im_size)
        self.d = len(im_size)
        self.eps = eps_from_width(int(width), float(oversamp))
        self.scale = float(np.prod(im_size)) ** 0.5
        self.n_trans = int(n_trans)
        kw = dict(n_trans=self.n_trans, eps=self.eps, dtype='complex128',
                  gpu_method=1, gpu_sort=1, gpu_kerevalmeth=1,
                  upsampfac=float(oversamp), modeord=0)
        im = torch.tensor(im_size, device=trj.device, dtype=torch.float64)
        pts = (torch.pi * trj.double() / (im * 0.5)).clamp(-np.pi + 1e-4, np.pi - 1e-4)
        args = [pts[:, i].contiguous() for i in range(self.d)]
        self.M = pts.shape[0]
        self.fwd_plan = cufinufft.Plan(2, self.im_size, isign=-1, **kw)
        self.adj_plan = cufinufft.Plan(1, self.im_size, isign=+1, **kw)
        self.fwd_plan.setpts(*args)
        self.adj_plan.setpts(*args)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """``(n_trans, *im_size) -> (n_trans, M)``"""
        x = img.reshape(self.n_trans, *self.im_size).to(torch.complex128).contiguous()
        return self.fwd_plan.execute(x).reshape(self.n_trans, self.M) / self.scale

    def adjoint(self, ksp: torch.Tensor) -> torch.Tensor:
        """``(n_trans, M) -> (n_trans, *im_size)``"""
        y = ksp.reshape(self.n_trans, self.M).to(torch.complex128).contiguous()
        return self.adj_plan.execute(y).reshape(self.n_trans, *self.im_size) / self.scale


class SenseOp64:
    """:class:`SenseOp` in complex128, built on :class:`Cufi64`."""

    def __init__(self, mps, trj, **kw):
        self.mps = mps.to(torch.complex128)
        self.im_size = tuple(mps.shape[1:])
        self.C = mps.shape[0]
        self.nufft = Cufi64(self.im_size, trj, n_trans=self.C, **kw)

    def forward(self, x):
        return self.nufft.forward(self.mps * x)

    def adjoint(self, y):
        return (self.mps.conj() * self.nufft.adjoint(y)).sum(0)


# ---------------------------------------------------------------------------
def _interp_only_plan(n_os, n_trans, eps, pts):
    """Type-2 interpolate-only plan (``qblock_feas/rho_cufinufft.py:122``)."""
    kw = dict(n_trans=n_trans, eps=eps, isign=-1, dtype='complex64',
              gpu_spreadinterponly=1, gpu_method=1, gpu_sort=1,
              gpu_kerevalmeth=1, upsampfac=2.0, modeord=1)
    plan = cufinufft.Plan(2, tuple(n_os), **kw)
    plan.setpts(*[pts[:, i].contiguous() for i in range(pts.shape[-1])])
    return plan


def profile_split(op: SenseOp, trj: torch.Tensor, n_reps: int = 20) -> dict:
    """
    Measure the FFT and interpolation halves of Baseline A separately.

    ``rho = t_interp / t_fft`` bounds the speedup of any method that removes only FFT
    work at ``(1 + rho) / rho``. Reported alongside the flop-counted prediction so the
    two can be compared honestly.
    """
    d = len(op.im_size)
    C = op.C
    n_os = [int(round(op.oversamp * n)) for n in op.im_size]
    im = torch.tensor(op.im_size, device=trj.device, dtype=trj.dtype)
    pts = (torch.pi * trj / (im * 0.5)).clamp(-np.pi + 1e-4, np.pi - 1e-4).contiguous()
    M = pts.shape[0]

    def fft_only():
        return torch.fft.fftn(op.mps, s=n_os, dim=tuple(range(-d, 0)))

    grid = fft_only()
    plan = _interp_only_plan(n_os, C, op.eps, pts)

    def interp_only():
        return plan.execute(grid)

    t_fft = repeat_time(fft_only, n_reps)
    t_int = repeat_time(interp_only, n_reps)
    del grid
    torch.cuda.empty_cache()

    rho = t_int['median'] / t_fft['median']
    W = nspread_from_eps(op.eps, op.oversamp) ** d
    n_grid = float(np.prod(n_os))
    return dict(
        t_fft=t_fft, t_interp=t_int, rho=float(rho),
        ceiling=float((1.0 + rho) / rho),
        # Flop-counted prediction, for comparison with the measurement
        rho_pred=float(W * M / (0.5 * n_grid * np.log2(n_grid))),
        W=int(W), M=int(M), n_os=tuple(n_os), C=int(C),
        oversamp=op.oversamp, width=op.width, eps=op.eps)
