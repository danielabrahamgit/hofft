"""
Shared problem setup for the hierarchical local coil-compression study (``feas.md``).

Three real datasets flow through one code path (feas.md Sec. 8):

* ``sos3d`` -- UCB stack-of-spirals, 3D, 294x294x150, 30 coils.
* ``sos2d`` -- the same scan decoupled by an FFT along kz. The trajectory is Cartesian
  along kz (150 integer values, in-plane spiral bit-identical across partitions), so a
  kz-FFT turns the volume into independent 2D spiral problems with the same physical
  array. Verified in ``test_tree.test_sos2d_decoupling``.
* ``hr2d``  -- ``data/highres_spiral``, 7T 300um, 2D, 734x734, 20 coils. An independent
  coil array, so a rank finding that holds on both is not array-specific.

Two grids are tracked and they are not the same thing (see ``Dataset``):

* ``im_size``     -- the native acquisition grid. **Every operator runs here.** No
  padding, so no trajectory rescaling and no aliasing subtleties.
* ``im_size_pad`` -- ``im_size`` rounded up to a multiple of ``2**L_MAX`` (and 5-smooth).
  **Only the tree/rank analysis runs here**, because a dyadic tree needs exact
  divisibility. The pad carries zero signal and is excluded from the mask, so cropping a
  padded field back to ``im_size`` is exact.

Preprocessing follows feas.md Sec. 8: whiten first (noise from the readout tail, as
``igrog/paper_figures/stack_of_spirals/main.py`` does), *then* globally coil-compress, so
compression is defined in the whitened coil space.
"""
import json
import numpy as np
import torch

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional, Sequence

from mr_recon.fourier import fft, ifft
from mr_recon.spatial import spatial_resize_poly

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MASK_THRESH = 0.9          # ESPIRiT eigenvalue threshold, the house convention
L_MAX = {2: 6, 3: 4}       # deepest dyadic level the padded grid must support
ENERGY_FRAC = 0.95         # global coil compression, matching main.py
CACHE = Path(__file__).resolve().parent / 'cache'
CACHE_VERSION = 1          # bump whenever the Dataset schema or preprocessing changes

SOS_DIR = '/local_mount/space/tiger/1/users/abrahamd/mr_data/ucb_stack_spiral/data'
SOS_IM_SIZE = (294, 294, 150)
SOS_NOISE_PTS = 100        # trailing readout samples treated as pure noise

DATASETS = ('sos3d', 'sos2d', 'hr2d')


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def next_dyadic_smooth(n: int, levels: int) -> int:
    """Smallest ``m >= n`` divisible by ``2**levels`` whose prime factors are 5-smooth."""
    step = 1 << int(levels)
    m = int(np.ceil(n / step)) * step
    while True:
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += step


def pad_grid(im_size: Sequence[int], levels: Optional[int] = None) -> tuple:
    """The dyadic-friendly grid the tree analysis runs on."""
    d = len(im_size)
    levels = L_MAX[d] if levels is None else levels
    return tuple(next_dyadic_smooth(n, levels) for n in im_size)


def pad_center(x: torch.Tensor, im_size_pad: Sequence[int]) -> torch.Tensor:
    """Center-pad the trailing ``len(im_size_pad)`` axes with zeros."""
    d = len(im_size_pad)
    src = tuple(x.shape[-d:])
    out = torch.zeros(x.shape[:-d] + tuple(im_size_pad), dtype=x.dtype, device=x.device)
    sl = tuple(slice((p - s) // 2, (p - s) // 2 + s) for s, p in zip(src, im_size_pad))
    out[(Ellipsis,) + sl] = x
    return out


def crop_center(x: torch.Tensor, im_size: Sequence[int]) -> torch.Tensor:
    """Inverse of :func:`pad_center`."""
    d = len(im_size)
    src = tuple(x.shape[-d:])
    sl = tuple(slice((s - n) // 2, (s - n) // 2 + n) for s, n in zip(src, im_size))
    return x[(Ellipsis,) + sl]


# ---------------------------------------------------------------------------
# Dataset record
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """One prepared problem: whitened, globally coil-compressed, on the native grid."""
    name: str
    im_size: tuple                  # native acquisition grid; operators run here
    im_size_pad: tuple              # dyadic grid; tree/rank analysis runs here
    trj: torch.Tensor               # (M, d) float32, cycles/FOV on im_size
    ksp: torch.Tensor               # (C, M) complex64
    mps: torch.Tensor               # (C, *im_size) complex64
    dcf: torch.Tensor               # (M,) float32
    mask: torch.Tensor              # (*im_size) bool
    img_ref: torch.Tensor           # (*im_size) complex64
    C_full: int                     # physical coils before compression
    coil_subspace: torch.Tensor     # (C_full, C) complex64, whitened -> compressed
    coil_svals: torch.Tensor        # (C_full,) float32, calibration singular values
    whiten: torch.Tensor            # (C_full, C_full) complex64, psi^{-1/2}
    meta: dict = field(default_factory=dict)

    @property
    def d(self) -> int:
        return len(self.im_size)

    @property
    def C(self) -> int:
        return self.mps.shape[0]

    @property
    def N(self) -> int:
        return int(np.prod(self.im_size))

    @property
    def M(self) -> int:
        return self.trj.shape[0]

    @property
    def n_mask(self) -> int:
        return int(self.mask.sum())

    def to(self, dev) -> 'Dataset':
        for k in ('trj', 'ksp', 'mps', 'dcf', 'mask', 'img_ref',
                  'coil_subspace', 'coil_svals', 'whiten'):
            setattr(self, k, getattr(self, k).to(dev))
        return self

    def energy_rank(self, energy: float = ENERGY_FRAC) -> int:
        """``C'`` the ``calc_coil_subspace(ksp, energy, ...)`` criterion would pick."""
        cm = self.coil_svals.cumsum(0)
        return int(torch.argwhere(cm > energy * cm[-1])[0, 0]) + 1

    def compress(self, C: int) -> 'Dataset':
        """
        A view of this dataset globally compressed to the leading ``C`` virtual coils.

        Exact because the cached coil axis is already the calibration SVD basis, ordered
        by energy -- truncation *is* the rank-``C`` compression.
        """
        C = int(min(C, self.C))
        out = Dataset(**{**self.__dict__})
        out.mps = self.mps[:C].contiguous()
        out.ksp = self.ksp[:C].contiguous()
        out.coil_subspace = self.coil_subspace[:, :C].contiguous()
        out.meta = dict(self.meta, Cprime=C)
        return out

    def summary(self) -> dict:
        return dict(name=self.name, d=self.d, im_size=self.im_size,
                    im_size_pad=self.im_size_pad, N=self.N, M=self.M,
                    C=self.C, C_full=self.C_full, n_mask=self.n_mask,
                    mask_frac=self.n_mask / self.N,
                    C_at_95=self.energy_rank(), **self.meta)


# ---------------------------------------------------------------------------
# Preprocessing primitives
# ---------------------------------------------------------------------------
def whitening_matrix(noise: torch.Tensor) -> torch.Tensor:
    """``psi^{-1/2}`` from a ``(C, ...)`` block of pure noise (Bessel-corrected)."""
    C = noise.shape[0]
    n = noise.reshape(C, -1).to(torch.complex128)
    n = n - n.mean(dim=-1, keepdim=True)
    psi = (n @ n.conj().T) / (n.shape[-1] - 1)
    vals, vecs = torch.linalg.eigh(psi)
    vals = vals.real.clamp(min=vals.real.max() * 1e-12)
    return ((vecs * vals.rsqrt()) @ vecs.conj().T).to(torch.complex64)


def coil_subspace(ksp_cal: torch.Tensor) -> tuple:
    """
    Full ``(C_full, C_full)`` rotation into the calibration coil-SVD basis, plus its
    singular values.

    Columns are ordered by energy, so truncating to the leading ``C'`` *is* the rank-``C'``
    global compression. Datasets are therefore cached once at full rank and ``C'`` is
    swept for free by :meth:`Dataset.compress`, rather than baking one threshold in.
    """
    C = ksp_cal.shape[0]
    u, s, _ = torch.linalg.svd(ksp_cal.reshape(C, -1), full_matrices=False)
    return u[:, :C].conj(), s[:C]


def apply_coils(x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """Apply a ``(C_in, C_out)`` coil matrix to the leading axis of ``x``."""
    return torch.tensordot(mat.transpose(0, 1), x, dims=([1], [0]))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_sos_raw(torch_dev: torch.device):
    """Whitening + full coil rotation and the rotated low-res maps for the SOS scan."""
    d = Path(SOS_DIR)

    # Noise from the readout tail. mmap so we touch 43 MB, not 1.2 GB.
    ksp_mm = np.load(d / 'ksp_fs.npy', mmap_mode='r')
    noise = torch.from_numpy(np.array(ksp_mm[:, -SOS_NOISE_PTS:])).to(torch_dev)
    W = whitening_matrix(noise)
    del noise

    ksp_cal = torch.from_numpy(np.load(d / 'ksp_cal.npy')).to(torch_dev)
    U, sv = coil_subspace(apply_coils(ksp_cal, W))
    del ksp_cal

    mps_low = torch.from_numpy(np.load(d / 'mps_low.npy')).to(torch_dev)
    mps_low = apply_coils(apply_coils(mps_low, W), U)     # whiten, then rotate
    evals_low = torch.from_numpy(np.load(d / 'evals_low.npy')).to(torch_dev)
    return W, U, sv, mps_low, evals_low, ksp_mm


def load_sos3d(torch_dev: torch.device, R: int = 1) -> Dataset:
    """UCB stack-of-spirals, full 3D."""
    d = Path(SOS_DIR)
    W, U, sv, mps_low, evals_low, ksp_mm = _load_sos_raw(torch_dev)
    im_size = SOS_IM_SIZE

    mps = spatial_resize_poly(mps_low, im_size, order=3)
    mps = mps / mps.abs().max()
    mask = spatial_resize_poly(evals_low, im_size, order=3) > MASK_THRESH
    del mps_low, evals_low

    trj = torch.from_numpy(np.load(d / 'trj_fs.npy')).to(torch_dev)      # (RO, I, Kz, 3)
    ksp = torch.from_numpy(np.array(ksp_mm)).to(torch_dev)               # (C, RO, I, Kz)
    ksp = apply_coils(apply_coils(ksp, W), U)
    if R > 1:
        trj, ksp = trj[:, ::R].contiguous(), ksp[:, :, ::R].contiguous()
    trj = trj.reshape(-1, 3)
    ksp = ksp.reshape(ksp.shape[0], -1)
    ksp = ksp / ksp.abs().max()

    img_ref = torch.from_numpy(np.load(d / 'img_fs.npy')).to(torch_dev)
    img_ref = img_ref / img_ref.abs().max()

    return Dataset(name='sos3d', im_size=im_size, im_size_pad=pad_grid(im_size),
                   trj=trj, ksp=ksp, mps=mps, dcf=_dcf(trj, im_size), mask=mask,
                   img_ref=img_ref, C_full=W.shape[0], coil_subspace=U,
                   coil_svals=sv, whiten=W, meta=dict(R=R))


def load_sos2d(torch_dev: torch.device, z: Optional[int] = None) -> Dataset:
    """
    One 2D slice of the stack-of-spirals scan, decoupled by an FFT along kz.

    ``ifft`` over the kz axis is exact here because kz is sampled on the integer grid
    ``-75..74`` matching the 150 image slices, so slice ``z`` of the transformed data is
    the 2D spiral k-space of slice ``z`` of the volume, up to the shared ``sqrt(Nz)``
    of the ortho convention -- which cancels in every relative error we report.
    """
    d = Path(SOS_DIR)
    W, U, sv, mps_low, evals_low, ksp_mm = _load_sos_raw(torch_dev)
    nz = SOS_IM_SIZE[2]
    z = nz // 2 if z is None else int(z)
    im_size = SOS_IM_SIZE[:2]

    mps3 = spatial_resize_poly(mps_low, SOS_IM_SIZE, order=3)
    mps = mps3[..., z] / mps3.abs().max()
    mask = (spatial_resize_poly(evals_low, SOS_IM_SIZE, order=3) > MASK_THRESH)[..., z]
    del mps_low, evals_low, mps3

    trj3 = torch.from_numpy(np.load(d / 'trj_fs.npy')).to(torch_dev)
    trj = trj3[:, :, 0, :2].reshape(-1, 2).contiguous()      # in-plane, kz-independent
    del trj3

    ksp = torch.from_numpy(np.array(ksp_mm)).to(torch_dev)
    ksp = apply_coils(apply_coils(ksp, W), U)                # (C, RO, I, Kz)
    ksp = ifft(ksp, dim=[-1])[..., z].reshape(ksp.shape[0], -1)
    ksp = ksp / ksp.abs().max()

    img_ref = torch.from_numpy(np.load(d / 'img_fs.npy')).to(torch_dev)[..., z]
    img_ref = img_ref / img_ref.abs().max()

    return Dataset(name='sos2d', im_size=im_size, im_size_pad=pad_grid(im_size),
                   trj=trj, ksp=ksp, mps=mps, dcf=_dcf(trj, im_size), mask=mask,
                   img_ref=img_ref, C_full=W.shape[0], coil_subspace=U,
                   coil_svals=sv, whiten=W, meta=dict(z=z))


def load_hr2d(torch_dev: torch.device) -> Dataset:
    """
    ``data/highres_spiral`` -- 7T 300um 2D spiral, an independent coil array.

    No noise-only region ships with this dataset, so it is used unwhitened and the maps
    are taken as supplied (already ESPIRiT-estimated). Coil compression is still applied
    so ``C'`` is defined the same way as for the SOS datasets.
    """
    p = Path('./data/highres_spiral')
    kw = dict(weights_only=True, map_location=torch_dev)
    trj = torch.load(p / 'trj.pt', **kw).reshape(-1, 2).contiguous()
    ksp = torch.load(p / 'ksp.pt', **kw)
    mps = torch.load(p / 'mps.pt', **kw)
    evals = torch.load(p / 'evals.pt', **kw)
    img_ref = torch.load(p / 'img_gt.pt', **kw)
    im_size = tuple(mps.shape[1:])

    C_full = mps.shape[0]
    Wm = torch.eye(C_full, dtype=torch.complex64, device=torch_dev)
    ksp_cal = fft(mps, dim=[-2, -1])
    U, sv = coil_subspace(ksp_cal)
    del ksp_cal

    mps = apply_coils(mps, U)
    mps = mps / mps.abs().max()
    ksp = apply_coils(ksp, U).reshape(U.shape[1], -1)
    ksp = ksp / ksp.abs().max()
    img_ref = img_ref.to(torch.complex64) / img_ref.abs().max()

    return Dataset(name='hr2d', im_size=im_size, im_size_pad=pad_grid(im_size),
                   trj=trj, ksp=ksp, mps=mps, dcf=_dcf(trj, im_size),
                   mask=evals > MASK_THRESH, img_ref=img_ref, C_full=C_full,
                   coil_subspace=U, coil_svals=sv, whiten=Wm, meta={})


def _dcf(trj: torch.Tensor, im_size: tuple) -> torch.Tensor:
    """Pipe-Menon DCF, falling back to ones if cupy is unavailable."""
    import contextlib
    import io
    from mr_recon.algs import density_compensation
    try:
        with contextlib.redirect_stderr(io.StringIO()):      # mute the tqdm bar
            return density_compensation(trj, im_size).float()
    except Exception as e:                              # pragma: no cover
        print(f'  [warn] density_compensation failed ({type(e).__name__}), using ones')
        return torch.ones(trj.shape[0], device=trj.device)


_LOADERS = dict(sos3d=load_sos3d, sos2d=load_sos2d, hr2d=load_hr2d)


def load_dataset(name: str,
                 torch_dev: torch.device,
                 use_cache: bool = True,
                 **kw) -> Dataset:
    """Load a prepared dataset, caching the result under ``block_comp/cache``."""
    tag = '_'.join([name, f'v{CACHE_VERSION}'] + [f'{k}{v}' for k, v in sorted(kw.items())])
    fpath = CACHE / f'{tag}.pt'
    if use_cache and fpath.exists():
        blob = torch.load(fpath, map_location=torch_dev, weights_only=False)
        return Dataset(**blob).to(torch_dev)
    ds = _LOADERS[name](torch_dev, **kw)
    if use_cache:
        CACHE.mkdir(exist_ok=True)
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v)
                    for k, v in ds.__dict__.items()}, fpath)
    return ds


# ---------------------------------------------------------------------------
# Timing / memory  (lifted from paper_experiments/hybrid_feas/stage2_profile.py)
# ---------------------------------------------------------------------------
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    try:
        import cupy as cp
        cp.cuda.Device().synchronize()
    except Exception:
        pass


def time_once(fn: Callable):
    sync()
    t0 = perf_counter()
    out = fn()
    sync()
    return perf_counter() - t0, out


def stats(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)),
                p90=float(np.percentile(a, 90)), n=int(a.size))


def repeat_time(fn: Callable, n: int, warmup: int = 2) -> dict:
    for _ in range(warmup):
        fn()
        sync()
    return stats([time_once(fn)[0] for _ in range(n)])


def peak_mib(fn: Callable):
    """Peak GPU allocation of one call, in MiB."""
    if not torch.cuda.is_available():
        return 0.0, fn()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    sync()
    return torch.cuda.max_memory_allocated() / 2 ** 20, out


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
def rel_err(approx: torch.Tensor, ref: torch.Tensor) -> float:
    """``||approx - ref||_2 / ||ref||_2``."""
    return float((approx - ref).norm() / ref.norm().clamp(min=1e-30))


def adjoint_error(fwd: Callable,
                  adj: Callable,
                  img_shape: tuple,
                  ksp_shape: tuple,
                  device,
                  dtype=torch.complex64,
                  n: int = 3,
                  seed: int = 0) -> float:
    """
    feas.md Sec. 9 inner-product test, ``|<Ax,y> - <x,A^H y>| / (|.| + |.|)``.

    ``fwd`` and ``adj`` must be a matched pair with no weighting hidden inside either --
    if a DCF lives in the adjoint, fold it into ``y`` before calling.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(img_shape, generator=g, dtype=torch.complex64).to(device).to(dtype)
        y = torch.randn(ksp_shape, generator=g, dtype=torch.complex64).to(device).to(dtype)
        lhs = torch.vdot(fwd(x).reshape(-1), y.reshape(-1))
        rhs = torch.vdot(x.reshape(-1), adj(y).reshape(-1))
        out.append(float(abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-30)))
    return float(np.max(out))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def gate(name: str, passed: bool, detail: str = '') -> bool:
    """Print a feas.md gate verdict in the house format."""
    bar = '=' * 24
    print(f'\n{bar} {name} {bar}')
    if detail:
        print(detail)
    print(f'[{"PASS" if passed else "FAIL"}] {name}')
    return passed


def save_results(out_dir: Path, tag: str, blob: dict, slim: Optional[dict] = None):
    """Save the full ``.pt`` plus a slim ``.json`` (house convention)."""
    out_dir.mkdir(exist_ok=True)
    torch.save(blob, out_dir / f'{tag}.pt')
    if slim is not None:
        with open(out_dir / f'{tag}.json', 'w') as f:
            json.dump(slim, f, indent=1, default=float)
    print(f'wrote {out_dir}/{tag}.{{pt,json}}')
