import copy
import math
import torch
import numpy as np

from typing import Optional, Sequence, Union
from einops import einsum

from mr_recon.linops import linop, sense_linop, batching_params
from mr_recon.algs import eigen_decomp_operator
from mr_recon.fourier import sigpy_nufft, fft, ifft, cufi_nufft
from mr_recon.utils import batch_iterator
from mr_recon.dtypes import complex_dtype, real_dtype

from .phase_coeffs import (trj_dev_to_phis_alphas, 
                           rescale_phis_alphas,
                           apply_phase_midpoints,
                           whiten_phis_alphas,
                           remove_empty_bases
)
from .decomp import (
    hofft_params, 
    als_iterations, 
    build_kern_bases, 
    lstsq_temporal,
    svd_decomp
)
from .utils import (
    gen_grd, 
    resize, 
    spatial_interp, 
    reduce_spatial, 
    expand_spatial, 
    reduce_temporal, 
    expand_temporal
)
from .matvec import matvec_naive, matvec_cur
from .spatial_init import choose_init, k_alpha_selection
from .kb import kb_apod_1d, sample_kb_kernel
from .forward_model import hofft_linop, hofft_compressed_linop
from .sparse_fit import (sparse_params, 
                         lstsq_compressed_fixed_support, 
                         sweep_smooth_interp_hyperparams, 
                         lstsq_compressed_kernels,
                         smooth_sparse_coeffs)
from .cur_ops import build_cur_factors, build_cur_factors_adaptive

def kb_nufft(trj: torch.Tensor,
             im_size: tuple,
             kern_size: tuple,
             os: float = 1.0,
             beta: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes kernel weights and the spatial factor for the KB-NUFFT model.
    
    Args
    ----
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    im_size : tuple
        Size of the image to be reconstructed.
    kern_size : tuple
        Size of the kernel.
    os : float
        Oversampling factor.
    beta : Optional[float]
        Beta parameter for the KB-NUFFT kernel.
    
    Returns
    -------
    spatial_factor : torch.Tensor
        KB-NUFFT spatial factor with shape (1, *im_size)
    kern_weights : torch.Tensor
        KB-NUFFT kernel weights with shape (1, *kern_size, *trj_size)
    """
    # Consts
    d = len(im_size)
    width = kern_size[0]
    for i in range(1, len(kern_size)):
        assert kern_size[i] == width, "Kernel size must be the same in all dimensions"
    if beta is None:
        beta = torch.pi * (((width / os) * (os - 0.5))**2 - 0.8)**0.5
        if (((width / os) * (os - 0.5))**2 - 0.8) < 0:
            beta = 1.0
    
    # Spatial factor
    rs = gen_grd(im_size).to(trj.device)
    spatial_factor = kb_apod_1d(rs / os, beta, width).prod(dim=-1)
    
    # Kernel weights
    kdevs = trj - (os * trj).round()/os
    kern_weights = sample_kb_kernel(kdevs, kern_size, os, beta)
    
    # Scaling factor correction
    spatial_factor /= width ** d
    
    # Reshape
    spatial_factor = spatial_factor[None,]
    kern_weights = kern_weights[None,].type(torch.complex64)
    
    # Apply correction linear phase
    k_corr = (width%2==0)*torch.ones(d, device=trj.device) / os / 2
    phz = torch.exp(-2j * np.pi * einsum(rs, k_corr, '... d, d -> ...'))
    spatial_factor = spatial_factor * phz

    return spatial_factor, kern_weights

def als_nufft(trj: torch.Tensor,
              im_size: tuple,
              hparams: hofft_params,
              spatial_mask: Optional[torch.Tensor] = None,
              im_size_low: Optional[tuple] = None,
              num_als_iter: Optional[int] = 100,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes least squares optimal NUFFT kernel weights and the spatial factor using ALS.
    
    Args
    ----
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    im_size : tuple
        Size of the image to be reconstructed.
    hparams : hofft_params
        HOFFT parameters.
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    im_size_low : Optional[tuple]
        Optional low resolution size for performing the decomposition
    num_als_iter : int
        Number of ALS iterations.
    
    Returns
    -------
    spatial_factor : torch.Tensor
        Apodization functions with shape (L, *im_size)
    kern_weights : torch.Tensor
        NUFFT kernel weights with shape (L, *kern_size, *trj_size)
    """
    # Consts
    d = trj.shape[-1]
    torch_dev = trj.device
    trj_size = trj.shape[:-1]
    im_size_low = (50,)*d if im_size_low is None else im_size_low
    kern_size = hparams.kern_size
    os = hparams.os
    L = hparams.L
    verbose = hparams.verbose
    spatial_init_method = hparams.spatial_init
    
    # Default spatial mask
    if spatial_mask is None:
        spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
    spatial_mask = reduce_spatial(spatial_mask, im_size_low, order=3)
    
    # Spatial and trajectory bases
    rs = gen_grd(im_size_low).to(torch_dev)
    kdevs = gen_grd(im_size_low).to(torch_dev) / os
    kern_bases = build_kern_bases(kern_size, im_size_low, os)
    kern_bases = kern_bases.to(torch_dev)
    phis = rs.moveaxis(-1, 0)
    alphas = kdevs.moveaxis(-1, 0)

    # Initialize apodization functions with eigen-vectors
    high_acc_nufft = sigpy_nufft(im_size_low, oversamp=2.0, width=6)
    kdevs_rs = high_acc_nufft.rescale_trajectory(kdevs)
    if spatial_init_method == 'eigen':
        toep_kerns = high_acc_nufft.calc_teoplitz_kernels(kdevs_rs[None])[0] # *solve_size_os
        solve_size_os = toep_kerns.shape
        def normal_op(x):
            N = x.shape[0]
            x = resize(x * spatial_mask, (N, *solve_size_os))
            x = fft(x, dim=tuple(range(-d, 0)))
            x = x * toep_kerns
            x = ifft(x, dim=tuple(range(-d, 0)))
            x = resize(x, (N, *im_size_low))
            return x * spatial_mask.conj()
        x0 = torch.randn(im_size_low, dtype=torch.complex64, device=torch_dev)
        spatial_factor, _ = eigen_decomp_operator(normal_op, x0, num_eigen=L, verbose=verbose)
    else:
        spatial_factor = choose_init(phis, alphas, hparams=hparams, spatial_init=spatial_init_method)

    # Make matrix-vector operation that applies spatially linear phase only
    class matvec_linphase(linop):
        def __init__(self):
            super().__init__(im_size_low, im_size_low)
        def forward(self, x):
            return high_acc_nufft.forward(x[None,], kdevs_rs[None,])[0] * np.prod(im_size_low) ** 0.5
        def adjoint(self, y):
            return high_acc_nufft.adjoint(y[None,], kdevs_rs[None,])[0] * np.prod(im_size_low) ** 0.5
    phase_model = matvec_linphase()
    # phis = rs.moveaxis(-1, 0)
    # alphas = kdevs.moveaxis(-1, 0)
    # phase_model = hparams.matvec_type(phis, alphas, **hparams.matvec_kwargs)
    
    # Perform ALS iterations
    spatial_factor, kern_weights = als_iterations(phase_model, kern_bases, spatial_factor, 
                                                    mask=spatial_mask,
                                                    max_iter=num_als_iter, verbose=verbose)
    
    # Interpolate spatial funcs
    kwargs = {'order': 5, 'mode': 'nearest'}
    solve_size_tensor = torch.tensor(im_size_low).to(torch_dev)
    spatial_crds = (gen_grd(im_size).to(torch_dev) + 0.5) * solve_size_tensor
    spatial_factor = spatial_interp(spatial_factor, spatial_crds, **kwargs)

    # Interpolate temporal functions
    trj_dev = trj - (os * trj).round()/os
    temporal_crds = (0.5 + trj_dev * os) * solve_size_tensor
    kern_weights = spatial_interp(kern_weights.reshape((-1, *im_size_low)), temporal_crds, **kwargs)
    kern_weights = kern_weights.reshape((L, *kern_size, *trj_size))
    
    return spatial_factor, kern_weights

def _build_reduced_terms(phis: torch.Tensor, 
                         alphas: torch.Tensor,
                         hparams: hofft_params,
                         spatial_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds the reduced spatial terms before doing a HOFFT decomposition.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    hparams : hofft_params
        HOFFT parameters.
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
        
    Returns
    -------
    phis_reduced : torch.Tensor
        Reduced spatial phase maps with shape (B, *reduced_im_size)
    alphas_reduced : torch.Tensor
        Reduced temporal phase coefficients with shape (B, *reduced_trj_size)
    kern_bases : torch.Tensor
        Reduced kernel bases with shape (L, *kern_size, *reduced_im_size)
    spatial_mask_reduced : torch.Tensor
        Reduced spatial mask with shape (*reduced_im_size)
    """
    # Consts
    im_size = phis.shape[1:]
    os = hparams.os
    kern_size = hparams.kern_size
    torch_dev = phis.device
    reduced_im_size = hparams.reduced_im_size
    time_reduction_factor = hparams.time_reduction_factor
    verbose = hparams.verbose
    
    # Default spatial mask
    if spatial_mask is None:
        spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
    
    # Reduce phi size
    if reduced_im_size is not None:
        phis_reduced = reduce_spatial(phis, im_size_low=reduced_im_size, order=3)
        # kern_bases = build_kern_bases(kern_size, 
        #                               im_size=reduced_im_size, 
        #                               os=os).to(torch_dev)
        kern_bases = build_kern_bases(kern_size, im_size=im_size, os=os).to(torch_dev)
        kern_bases = reduce_spatial(kern_bases, im_size_low=reduced_im_size, order=3)
        spatial_mask_reduced = reduce_spatial(spatial_mask, im_size_low=reduced_im_size, order=3)
    else:
        phis_reduced = phis
        kern_bases = build_kern_bases(kern_size, im_size, os).to(torch_dev)
        spatial_mask_reduced = spatial_mask
        
    # Reduce alpha size
    if time_reduction_factor is not None:
        if verbose:
            print(f'Warning, assuming that the first trajectory dimension is time')    
        num_time_low = round(alphas.shape[1] / time_reduction_factor)
        alphas_reduced = reduce_temporal(alphas, num_time_low=num_time_low, dim=1, order=3)
    else:
        alphas_reduced = alphas
        
    return phis_reduced, alphas_reduced, kern_bases, spatial_mask_reduced

def _process_phase_coefficients(phis: torch.Tensor,
                                alphas: torch.Tensor,
                                normalize_coeffs: bool = False,
                                num_compressed_bases: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process the phase coefficients.

    When ``normalize_coeffs`` is True, returns the thin SVD factors of the
    total phase A^T Φ = U Σ V^T: Φ_w = V^T, A_w = U^T, and S = diag(Σ).
    Downstream code applies S to U^T for a phase-preserving pair, or passes
    (V^T, U^T, S) to CUR so clustering can use the two whitened metrics.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    normalize_coeffs : bool
        Whether to rescale midpoints and whiten the phase coefficients.
    num_compressed_bases : Optional[int]
        Rank of the whitened SVD. Defaults to B.
        
    Returns
    -------
    phis_w : torch.Tensor
        V^T with shape (B, *im_size) (or the unwhitened phis)
    alphas_w : torch.Tensor
        U^T with shape (B, *trj_size) (or the unwhitened alphas)
    spatial_correction : torch.Tensor
        Applies constant temporal phase correction with shape (*im_size)
    temporal_correction : torch.Tensor
        Applies constant spatial phase correction with shape (*trj_size)
    """
    # Normalize phase coefficients
    if normalize_coeffs:
        # Remove empty bases
        phis, alphas = remove_empty_bases(phis, alphas)
        
        # Special case if empty
        if phis.shape[0] == 0:
            phis_one = torch.ones(*phis.shape[1:], device=phis.device, dtype=phis.dtype)[None,]
            alphas_zero = torch.zeros(*alphas.shape[1:], device=alphas.device, dtype=alphas.dtype)[None,]
            return phis_one, alphas_zero, phis_one[0] + 0j, alphas_zero * 0j + 1
        
        # Rescale and remove offsets
        phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)
        spatial_correction, temporal_correction = apply_phase_midpoints(
            phis_nrm, alphas_nrm, phis_mp, alphas_mp)
        
        # Compress/Whiten
        if num_compressed_bases is None:
            num_compressed_bases = phis.shape[0]
        else:
            assert num_compressed_bases <= phis.shape[0], \
            f"num_compressed_bases must be less or equal to {phis.shape[0]}"
        phis_w, alphas_w = whiten_phis_alphas(phis_nrm, alphas_nrm, 
                                              B_compressed=num_compressed_bases)
    else:
        phis_w = phis
        alphas_w = alphas
        spatial_correction = torch.ones_like(phis_w[0]).type(torch.complex64)
        temporal_correction = torch.ones_like(alphas_w[0]).type(torch.complex64)
    
    return phis_w, alphas_w, spatial_correction, temporal_correction

def _make_phase_model(phis: torch.Tensor,
                      alphas: torch.Tensor,
                      hparams: hofft_params):
    """
    Build the pipeline matvec.

    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    hparams : hofft_params
        HOFFT parameters.
        
    Returns
    -------
    matvec : matvec
        Phase model matvec.
    """
    kwargs = hparams.matvec_kwargs or {}
    cur_rank = getattr(hparams, 'cur_rank', None)
    if cur_rank is None:
        naive_keys = ('spatial_batch_size', 'temporal_batch_size', 'verbose')
        naive_kw = {k: kwargs[k] for k in naive_keys if k in kwargs}
        return matvec_naive(phis, alphas, **naive_kw)
    cur_keys = ('rank_phi', 'rank_alpha', 'cluster_method',
                'spatial_batch_size', 'temporal_batch_size', 'verbose')
    cur_kw = {k: kwargs[k] for k in cur_keys if k in kwargs}
    return matvec_cur(phis, alphas, cur_rank=cur_rank, normalize_method='svd', **cur_kw)


def _factor_q_axes(Q: int, bbox_len: Sequence[int]) -> tuple:
    """Split ``Q`` into per-axis counts, longest bbox axis first."""
    d = len(bbox_len)
    Q_i = [1] * d
    remaining = int(Q)
    if remaining < 1:
        raise ValueError(f'Q must be >= 1, got {Q}')
    lengths = [float(b) for b in bbox_len]
    while remaining > 1:
        ax = int(np.argmax(lengths))
        f = 2
        while remaining % f != 0:
            f += 1
            if f > remaining:
                raise ValueError(f'cannot factor Q={Q}')
        Q_i[ax] *= f
        lengths[ax] /= f
        remaining //= f
    return tuple(Q_i)


def _mask_bbox(mask: torch.Tensor) -> tuple:
    """Tight inclusive bounding box of a support mask. Returns ``(lo, hi)``."""
    d = mask.ndim
    m = mask.abs() > 0 if mask.is_complex() else mask > 0
    lo = np.zeros(d, dtype=int)
    hi = np.zeros(d, dtype=int)
    for i in range(d):
        other = tuple(j for j in range(d) if j != i)
        occ = torch.argwhere(m.any(dim=other) if other else m)[:, 0]
        if occ.numel() == 0:
            raise ValueError('spatial mask is empty')
        lo[i], hi[i] = int(occ.min()), int(occ.max())
    return lo, hi


def _cycles_to_rad(trj: torch.Tensor, im_size: Sequence[int]) -> torch.Tensor:
    """Map cycles/FOV to wrapped cuFINUFFT coordinates using the *full* ``N``."""
    N = torch.as_tensor(tuple(im_size), device=trj.device, dtype=trj.dtype)
    tup = (None,) * (trj.ndim - 1) + (slice(None),)
    x = (2.0 * math.pi) * trj / N[tup]
    twopi = 2.0 * math.pi
    x = torch.remainder(x + math.pi, twopi) - math.pi
    lim = math.pi - 1e-4
    return x.clamp(-lim, lim).contiguous().type(real_dtype)


def _window_lo(start: np.ndarray, V: np.ndarray, im_size: tuple) -> np.ndarray:
    N = np.asarray(im_size, dtype=int)
    V = np.minimum(np.asarray(V, dtype=int), N)
    return np.clip(np.asarray(start, dtype=int), 0, N - V)


def _cell_labels(mask: torch.Tensor,
                 lo: np.ndarray,
                 V: np.ndarray,
                 Q_i: tuple) -> torch.Tensor:
    """Integer cell id on ``mask``'s grid, or -1 outside the support."""
    im_size = tuple(mask.shape)
    d = len(im_size)
    sub = torch.stack(torch.meshgrid(
        *[torch.arange(n, device=mask.device) for n in im_size],
        indexing='ij'), dim=-1)
    lo_t = torch.as_tensor(lo, device=mask.device)
    V_t = torch.as_tensor(V, device=mask.device)
    Q_t = torch.as_tensor(np.asarray(Q_i), device=mask.device)
    cell = ((sub - lo_t) // V_t).clamp(min=torch.zeros_like(Q_t), max=Q_t - 1)
    cell_id = torch.zeros(im_size, dtype=torch.long, device=mask.device)
    for i in range(d):
        cell_id = cell_id * int(Q_i[i]) + cell[..., i]
    occupied = mask.abs() > 0 if mask.is_complex() else mask > 0
    return torch.where(occupied, cell_id, torch.full_like(cell_id, -1))


def _raised_cosine_1d(n: int, ov: int, device, dtype) -> torch.Tensor:
    """Length-``n`` 1D taper: 0→1 over ``2*ov`` samples, then 1, then 1→0."""
    w = torch.ones(int(n), device=device, dtype=dtype)
    n_tap = min(max(int(ov) * 2, 0), int(n))
    if n_tap <= 0:
        return w
    t = torch.arange(n_tap, device=device, dtype=dtype)
    ramp = 0.5 * (1.0 - torch.cos(math.pi * t / max(n_tap - 1, 1)))
    w[:n_tap] = ramp
    w[-n_tap:] = ramp.flip(0)
    return w


def _logical_taper(wlo: np.ndarray,
                   V_fft: np.ndarray,
                   logical_start: np.ndarray,
                   V_logical: np.ndarray,
                   ov: np.ndarray,
                   device,
                   dtype) -> torch.Tensor:
    """Separable raised-cosine on an FFT window, evaluated in logical coords."""
    d = len(V_fft)
    w = torch.ones(tuple(int(v) for v in V_fft), device=device, dtype=dtype)
    for ax in range(d):
        n_log = int(V_logical[ax])
        w1d = _raised_cosine_1d(n_log, int(ov[ax]), device, dtype)
        j = torch.arange(int(V_fft[ax]), device=device)
        log_j = int(wlo[ax]) + j - int(logical_start[ax])
        valid = (log_j >= 0) & (log_j < n_log)
        vals = torch.zeros(int(V_fft[ax]), device=device, dtype=dtype)
        if bool(valid.any()):
            vals[valid] = w1d[log_j[valid].long()]
        shape = [1] * d
        shape[ax] = int(V_fft[ax])
        w = w * vals.reshape(shape)
    return w


def _normalize_tapers(tapers: list, slices: list, shape: tuple) -> list:
    """Pointwise L1-normalize so overlapping windows form a partition of unity."""
    if not tapers:
        return tapers
    acc = torch.zeros(shape, dtype=tapers[0].dtype, device=tapers[0].device)
    for w, slc in zip(tapers, slices):
        acc[slc] = acc[slc] + w
    out = []
    for w, slc in zip(tapers, slices):
        den = acc[slc]
        out.append(torch.where(den > 0, w / den, torch.zeros_like(w)))
    return out


def _next235even(n: int) -> int:
    n = max(2, int(n))
    if n % 2:
        n += 1
    x = n
    while True:
        t = x
        while t % 2 == 0:
            t //= 2
        while t % 3 == 0:
            t //= 3
        while t % 5 == 0:
            t //= 5
        if t == 1:
            return x
        x += 2


class _QBlockSenseLinop(linop):
    """Cropped cuFINUFFT sense operator with a (possibly overlapping) partition."""

    def __init__(self,
                 mps: torch.Tensor,
                 dcf: torch.Tensor,
                 spatial_q: list,
                 temporal_q: list,
                 masks_q: list,
                 slices_q: list,
                 ramps_q: list,
                 trj_x: list,
                 im_size: tuple,
                 V: tuple,
                 bparams: batching_params,
                 oversamp: float,
                 width: int,
                 shear: Union[bool, str],
                 n_trans_cap: Optional[int] = None,
                 pack_n_trans: bool = True):
        C = mps.shape[0]
        trj_size = temporal_q[0].shape[1:]
        super().__init__(im_size, (C, *trj_size))
        self.mps = mps.type(complex_dtype)
        self.dcf = dcf.type(real_dtype)
        self.spatial_q = [s.type(complex_dtype).contiguous() for s in spatial_q]
        self.temporal_q = [t.type(complex_dtype).contiguous() for t in temporal_q]
        self.masks_q = [m.to(dtype=mps.dtype) for m in masks_q]
        self.slices_q = slices_q
        self.ramps_q = [r.type(complex_dtype).contiguous() for r in ramps_q]
        self.im_size = tuple(im_size)
        self.V = tuple(V)
        self.trj_size = trj_size
        self.bparams = bparams
        self.Q_kept = len(spatial_q)
        self.L_q = [int(s.shape[0]) for s in spatial_q]
        self.S_tot = int(sum(self.L_q))
        # Packed F/A must follow tensor ranks, not a later A.L_q overwrite
        # from the pooled allocator (those can disagree when L is raised).
        self._pack_L = list(self.L_q)
        self._pack_S = self.S_tot
        self.shear = shear
        self.C = C
        self.oversamp = float(oversamp)
        self.width = int(width)
        self._trj_x = trj_x
        self.n_trans_cap = n_trans_cap
        Nprod = float(np.prod(im_size))
        Vprod = float(np.prod(V))
        # cufi_nufft(V) divides by √V; match the global 1/√N convention.
        self.scale = (Vprod / Nprod) ** 0.5

        coil_batch = bparams.coil_batch_size or C
        field_batch = bparams.field_batch_size or 1
        self.coil_batch = min(int(coil_batch), C)
        self.field_batch = int(field_batch)

        self._streams = []
        self.nfts = []
        self.trj_b = []
        if shear in (False, None, 'none'):
            self.h_cat = torch.cat(
                [self.temporal_q[q] * self.ramps_q[q] for q in range(self.Q_kept)],
                dim=0)
        self._setup_plans(pack_n_trans)

    def _n_trans_budget(self, n_want: int, n_plans: int) -> int:
        """Cap n_trans so 2 * n_plans * n_trans * grid_bytes fits in ~35% free."""
        n_want = max(1, int(n_want))
        d = len(self.V)
        nf = _next235even(int(math.ceil(self.oversamp * max(self.V) - 1e-12)))
        grid_bytes = (nf ** d) * 8
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
        else:
            free = 8 * 1024 ** 3
        max_batch = int(0.35 * free / max(2 * max(n_plans, 1) * grid_bytes, 1))
        return max(1, min(n_want, max(1, max_batch)))

    def _setup_plans(self, pack_n_trans: bool):
        for nft in self.nfts:
            nft.clear_plans()
        self.nfts = []
        self.trj_b = []
        self._streams = []
        self.pack_n_trans = bool(pack_n_trans)
        no_shear = self.shear in (False, None, 'none')
        n_plans = 1 if no_shear else self.Q_kept

        if pack_n_trans:
            if no_shear:
                wanted = [self._pack_S * self.coil_batch]
            else:
                wanted = [L * self.coil_batch for L in self._pack_L]
        else:
            # Legacy: plan n_trans = coil_batch and chunk the factors.
            wanted = [self.coil_batch] * n_plans
        if self.n_trans_cap is not None:
            wanted = [min(w, int(self.n_trans_cap)) for w in wanted]
        self.n_trans_plan = [self._n_trans_budget(w, n_plans) for w in wanted]
        self.n_trans_wanted = wanted

        def _plan_one(x_pts, n_trans):
            nft = cufi_nufft(self.V, oversamp=self.oversamp, width=self.width,
                             n_trans=n_trans)
            nft.plan_kwargs['gpu_method'] = 1
            nft.plan_kwargs['gpu_maxbatchsize'] = int(n_trans)
            xb = x_pts[None] if x_pts.ndim == len(self.trj_size) + 1 else x_pts
            nft.plan(xb, n_trans=n_trans)
            return nft, xb if xb.ndim == len(self.trj_size) + 2 else xb[None]

        # Streams: Part A showed no overlap, and pinning a plan to
        # opts.gpu_stream broke the adjoint (rel ~ 1). Pack n_trans only.
        if no_shear:
            nft, xb = _plan_one(self._trj_x[0], self.n_trans_plan[0])
            self.nfts = [nft]
            self.trj_b = [xb]
        else:
            for q in range(self.Q_kept):
                nft, xb = _plan_one(self._trj_x[q], self.n_trans_plan[q])
                self.nfts.append(nft)
                self.trj_b.append(xb)

    def repack(self, pack_n_trans: bool):
        """Rebuild cuFINUFFT plans with or without factor packing. Same factors."""
        self._setup_plans(pack_n_trans)
        return self

    def _block_img(self, img, q):
        slc = self.slices_q[q]
        return img[slc] * self.masks_q[q]

    def _mps_q(self, q):
        slc = self.slices_q[q]
        return self.mps[(slice(None),) + slc]

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ksp = torch.zeros(self.oshape, dtype=complex_dtype, device=img.device)
        if self.shear in (False, None, 'none'):
            nft = self.nfts[0]
            trj_b = self.trj_b[0]
            for c1, c2 in batch_iterator(self.C, self.coil_batch):
                n_c = c2 - c1
                packed = img.new_zeros((n_c, self._pack_S, *self.V))
                i0 = 0
                for q in range(self.Q_kept):
                    L = self._pack_L[q]
                    Sx = self._mps_q(q)[c1:c2] * self._block_img(img, q)
                    packed[:, i0:i0 + L] = Sx[:, None] * self.spatial_q[q]
                    i0 += L
                y = nft.forward(packed[None], trj_b)[0] * self.scale
                ksp[c1:c2] += (y * self.h_cat).sum(dim=1)
        else:
            for q in range(self.Q_kept):
                ksp = ksp + self._forward_block(img, q, ksp)
        return ksp

    def _forward_block(self, img, q, ksp_like):
        nft = self.nfts[q]
        trj_b = self.trj_b[q]
        h = self.temporal_q[q] * self.ramps_q[q]
        xq = self._block_img(img, q)
        mps_q = self._mps_q(q)
        yq = ksp_like.new_zeros(ksp_like.shape)
        for c1, c2 in batch_iterator(self.C, self.coil_batch):
            Sx = mps_q[c1:c2] * xq
            Bx = Sx[:, None] * self.spatial_q[q]
            y = nft.forward(Bx[None], trj_b)[0] * self.scale
            yq[c1:c2] = (y * h).sum(dim=1)
        return yq

    def adjoint(self, ksp: torch.Tensor) -> torch.Tensor:
        img = torch.zeros(self.im_size, dtype=complex_dtype, device=ksp.device)
        if self.shear in (False, None, 'none'):
            nft = self.nfts[0]
            trj_b = self.trj_b[0]
            h = self.h_cat.conj()
            for c1, c2 in batch_iterator(self.C, self.coil_batch):
                wy = ksp[c1:c2] * self.dcf
                Hy = wy[:, None] * h
                x_all = nft.adjoint(Hy[None], trj_b)[0] * self.scale
                i0 = 0
                for q in range(self.Q_kept):
                    L = self._pack_L[q]
                    slc = self.slices_q[q]
                    xq = x_all[:, i0:i0 + L]
                    acc = (xq * self._mps_q(q)[c1:c2, None].conj()
                           * self.spatial_q[q].conj()).sum(dim=(0, 1))
                    img[slc] = img[slc] + acc * self.masks_q[q]
                    i0 += L
        else:
            for q in range(self.Q_kept):
                slc = self.slices_q[q]
                acc = self._adjoint_block(ksp, q)
                img[slc] = img[slc] + acc * self.masks_q[q]
        return img

    def _adjoint_block(self, ksp, q):
        nft = self.nfts[q]
        trj_b = self.trj_b[q]
        h = (self.temporal_q[q] * self.ramps_q[q]).conj()
        mps_q = self._mps_q(q)
        acc = torch.zeros(self.V, dtype=complex_dtype, device=ksp.device)
        for c1, c2 in batch_iterator(self.C, self.coil_batch):
            wy = ksp[c1:c2] * self.dcf
            Hy = wy[:, None] * h
            xq = nft.adjoint(Hy[None], trj_b)[0] * self.scale
            acc = acc + (xq * mps_q[c1:c2, None].conj()
                         * self.spatial_q[q].conj()).sum(dim=(0, 1))
        return acc

    def normal(self, img: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(img))

    def clear_plans(self):
        for nft in self.nfts:
            nft.clear_plans()


def qblock_svd_decomp_linop(phis: torch.Tensor,
                            alphas: torch.Tensor,
                            mps: torch.Tensor,
                            trj: torch.Tensor,
                            hparams: hofft_params,
                            Q: int = 4,
                            svd_method: str = 'direct',
                            shear: Union[bool, str] = False,
                            overlap: float = 0.1,
                            spatial_mask: Optional[torch.Tensor] = None,
                            dcf: Optional[torch.Tensor] = None,
                            bparams: batching_params = batching_params(),
                            pack_n_trans: bool = True) -> linop:
    """
    Block-partitioned CUR+SVD decomposition and cropped-cuFINUFFT linop.

    ``hparams.L`` is the total factor budget ``S_tot = sum_q L_q``, not a
    per-block rank. ``shear=False`` shares one trajectory / plan;
    ``shear='block'`` shears each block's trajectory by its affine slope.

    ``overlap`` is the fraction of the *reduced* cell extended on each side.
    The full-grid FFT size is lifted so ``V_fft / N`` matches on both grids.
    Independent rounding on each grid used to stretch the cubic upsample
    against the NUFFT crop. ``overlap=0`` restores disjoint cells. ``Q=1``
    forces ``overlap=0``.

    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    mps : torch.Tensor
        Coil sensitivities with shape (C, *im_size)
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d), cycles/FOV
    hparams : hofft_params
        HOFFT parameters. ``L`` is ``S_tot``.
    Q : int
        Requested number of spatial blocks (factored per-axis).
    shear : bool or str
        ``False`` for a shared trajectory; ``'block'`` for per-block shear.
    overlap : float
        Per-side overlap as a fraction of the cell size.
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    dcf : Optional[torch.Tensor]
        Density compensation factor with shape (*trj_size)
    bparams : batching_params
        Coil / field batch sizes for the linop.

    Returns
    -------
    linop : linop
        Block sense operator. Also stores ``Q_kept``, ``L_q``, ``S_tot``,
        ``pooled_threshold``, ``Q_per_axis``.
    """
    im_size = tuple(phis.shape[1:])
    trj_size = tuple(alphas.shape[1:])
    d = len(im_size)
    torch_dev = phis.device
    kern_size = hparams.kern_size
    os = hparams.os
    S_tot = int(hparams.L)
    verbose = hparams.verbose
    time_reduction_factor = hparams.time_reduction_factor
    reduced_im_size = hparams.reduced_im_size
    shear_block = shear in (True, 'block')
    overlap = float(overlap)
    if overlap < 0.0 or overlap >= 0.5:
        raise ValueError(f'overlap must be in [0, 0.5), got {overlap}')
    if Q <= 1:
        overlap = 0.0

    for i in range(1, d):
        assert kern_size[0] == kern_size[i], \
            "Kernel size must be isotropic for time-segmented NUFFT decomposition"
    if S_tot < 1:
        raise ValueError(f'hparams.L (S_tot) must be >= 1, got {S_tot}')

    if spatial_mask is None:
        spatial_mask_full = torch.ones(im_size, dtype=phis.dtype, device=torch_dev)
    else:
        spatial_mask_full = spatial_mask

    phis_nrm, alphas_nrm, spat, temp = _process_phase_coefficients(
        phis, alphas,
        normalize_coeffs=hparams.normalize_coeffs,
        num_compressed_bases=hparams.num_compressed_bases)
    B = phis_nrm.shape[0]
    phis_nrm, alphas_nrm, _, spatial_mask_red = _build_reduced_terms(
        phis_nrm, alphas_nrm, hparams, spatial_mask_full)
    red_size = tuple(phis_nrm.shape[1:])

    # ----------------- Split spatial into Q blocks -----------------
    lo_f, hi_f = _mask_bbox(spatial_mask_full)
    bbox_f = hi_f - lo_f + 1
    Q_i = _factor_q_axes(Q, bbox_f)
    V_full = np.array([int(math.ceil(bbox_f[i] / Q_i[i])) for i in range(d)], dtype=int)
    V_full = np.minimum(V_full, np.asarray(im_size, dtype=int))
    lo_r, hi_r = _mask_bbox(spatial_mask_red)
    bbox_r = hi_r - lo_r + 1
    V_red = np.array([int(math.ceil(bbox_r[i] / Q_i[i])) for i in range(d)], dtype=int)
    V_red = np.minimum(V_red, np.asarray(red_size, dtype=int))

    labels_f = _cell_labels(spatial_mask_full, lo_f, V_full, Q_i)
    labels_r = _cell_labels(spatial_mask_red, lo_r, V_red, Q_i)
    N_t = np.asarray(im_size, dtype=int)
    N_r = np.asarray(red_size, dtype=int)
    # Overlap and FFT size live on the full (NUFFT) grid. The reduced crop is
    # the align_corners preimage of that window — the same mapping
    # reduce_spatial / expand_spatial use on the whole image. Scaling by N_r/N_t
    # (instead of (N-1)/(N-1)) was stretching each block crop independently
    # and is why qblock NRMSE moved 2x when reduced_im_size was toggled.
    ov_f = np.maximum(np.rint(overlap * V_full).astype(int), 0)
    ov_f = np.minimum(ov_f, np.maximum((N_t - V_full) // 2, 0))
    V_fft_f = V_full + 2 * ov_f
    if np.array_equal(N_r, N_t):
        V_fft_r = V_fft_f.copy()
        ov_r = ov_f.copy()
    else:
        V_fft_r = np.rint(
            (V_fft_f - 1) * (N_r - 1) / np.maximum(N_t - 1, 1)
        ).astype(int) + 1
        V_fft_r = np.minimum(np.maximum(V_fft_r, V_red), N_r)
        extra_r = np.maximum(V_fft_r - V_red, 0)
        extra_r = extra_r - (extra_r % 2)
        V_fft_r = V_red + extra_r
        ov_r = extra_r // 2
    use_overlap = bool(np.any(ov_f > 0) or np.any(ov_r > 0))
    mask_f_bool = spatial_mask_full.abs() > 0 if spatial_mask_full.is_complex() \
        else spatial_mask_full > 0
    mask_r_bool = spatial_mask_red.abs() > 0 if spatial_mask_red.is_complex() \
        else spatial_mask_red > 0

    kept = []
    for c in range(int(np.prod(Q_i))):
        if not bool((labels_f == c).any()):
            continue
        qv = np.unravel_index(c, Q_i)
        start_f = lo_f + np.array(qv) * V_full - ov_f
        start_r = np.rint(
            start_f * (N_r - 1) / np.maximum(N_t - 1, 1)
        ).astype(int)
        wlo_f = _window_lo(start_f, V_fft_f, im_size)
        wlo_r = _window_lo(start_r, V_fft_r, red_size)
        slc_f = tuple(slice(int(wlo_f[i]), int(wlo_f[i] + V_fft_f[i])) for i in range(d))
        slc_r = tuple(slice(int(wlo_r[i]), int(wlo_r[i] + V_fft_r[i])) for i in range(d))
        r_c = (wlo_f + V_fft_f // 2 - N_t // 2).astype(np.float64) / N_t
        if use_overlap:
            occ_f = mask_f_bool[slc_f]
            occ_r = mask_r_bool[slc_r]
        else:
            occ_f = labels_f[slc_f] == c
            occ_r = labels_r[slc_r] == c
        kept.append(dict(
            slc_f=slc_f, slc_r=slc_r, r_c=r_c,
            occ_f=occ_f, occ_r=occ_r,
            wlo_f=wlo_f, wlo_r=wlo_r,
            start_f=start_f, start_r=start_r,
        ))
    Q_kept = len(kept)
    if Q_kept == 0:
        raise ValueError('no occupied blocks; check spatial_mask')
    if S_tot < Q_kept:
        raise ValueError(
            f'S_tot=hparams.L={S_tot} must be >= Q_kept={Q_kept} (Q={Q} -> {Q_i})')

    V_log_f = V_full + 2 * ov_f
    V_log_r = V_red + 2 * ov_r
    tapers_f, tapers_r = [], []
    for blk in kept:
        if use_overlap:
            tf = _logical_taper(blk['wlo_f'], V_fft_f, blk['start_f'],
                                V_log_f, ov_f, torch_dev, torch.float64)
            tr = _logical_taper(blk['wlo_r'], V_fft_r, blk['start_r'],
                                V_log_r, ov_r, torch_dev, torch.float64)
            tf = tf * blk['occ_f'].to(dtype=tf.dtype)
            tr = tr * blk['occ_r'].to(dtype=tr.dtype)
        else:
            tf = blk['occ_f'].to(dtype=torch.float64)
            tr = blk['occ_r'].to(dtype=torch.float64)
        tapers_f.append(tf)
        tapers_r.append(tr)
    tapers_f = _normalize_tapers(tapers_f, [b['slc_f'] for b in kept], im_size)
    tapers_r = _normalize_tapers(tapers_r, [b['slc_r'] for b in kept], red_size)
    for blk, tf, tr in zip(kept, tapers_f, tapers_r):
        blk['win_f'] = tf
        blk['win_r'] = tr

    n_vox_red = int(np.prod(V_fft_r))
    L_max = min(int(2 * S_tot / Q_kept + 8), n_vox_red)
    n_red_per = float(np.prod([red_size[i] / Q_i[i] for i in range(d)]))
    if n_red_per < 16 * L_max:
        raise ValueError(
            f'per-block reduced voxels ~{n_red_per:.0f} < 16*L_max={16 * L_max}. '
            f'Increase reduced_im_size (got {red_size}) or decrease Q={Q} -> {Q_i}.')

    # ----------------- Per-block affine absorption -----------------
    rs_red = gen_grd(red_size).to(device=torch_dev, dtype=torch.float64)
    C_aff = torch.zeros((Q_kept, B, d), dtype=torch.float64, device=torch_dev)
    chat = torch.zeros((Q_kept, B), dtype=torch.float64, device=torch_dev)
    phis_res = []
    for q, blk in enumerate(kept):
        slc = blk['slc_r']
        occ = blk['occ_r']
        win = blk['win_r']
        r_c = torch.as_tensor(blk['r_c'], device=torch_dev, dtype=torch.float64)
        u = rs_red[slc] - r_c
        ph = phis_nrm[(slice(None),) + slc].double()
        sel = occ.reshape(-1)
        if bool(sel.any()):
            Amat = torch.cat([
                torch.ones((int(sel.sum()), 1), dtype=torch.float64, device=torch_dev),
                u.reshape(-1, d)[sel],
            ], dim=1)
            Y = ph.reshape(B, -1)[:, sel].T
            sw = win.reshape(-1)[sel].clamp(min=0).sqrt()[:, None]
            coef = torch.linalg.lstsq(Amat * sw, Y * sw).solution
            chat[q] = coef[0]
            C_aff[q] = coef[1:].T
        if not shear_block:
            C_aff[q].zero_()
        pred = chat[q][(slice(None),) + (None,) * d]
        if shear_block:
            pred = pred + einsum(C_aff[q], u, 'b d, ... d -> b ...')
        phis_res.append(((ph - pred) * occ.double()).to(phis_nrm.dtype))

    # ----------------- Per-block SVD via CUR, pooled rank -----------------
    # Drop α DC so the SVD does not spend rank on a static phase; that DC is
    # applied as a *smooth* spatial multiplier (φ·mean(α)), not piecewise chat.
    a_mean_red = alphas_nrm.reshape((B, -1)).mean(dim=1)
    alphas_flat = alphas_nrm.reshape((B, -1)) - a_mean_red[:, None]
    U_q, V_q, S_q = [], [], []
    for q, res in enumerate(phis_res):
        phi_flat = res.reshape((B, -1))
        n_vox = phi_flat.shape[1]
        m_vec = kept[q]['occ_r'].reshape(-1).to(dtype=phis_nrm.dtype)
        if float(phi_flat.abs().max()) < 1e-8:
            # Affine absorbed the block; one constant factor remains.
            U_q.append(torch.ones((alphas_flat.shape[1], 1), dtype=torch.complex64,
                                  device=torch_dev))
            V_q.append(m_vec.to(torch.complex64)[:, None])
            S_q.append(torch.ones(1, device=torch_dev, dtype=torch.float32))
            continue
        q_svd = max(min(L_max, n_vox, alphas_flat.shape[1]), 1)
        if svd_method == 'cur':
            if hparams.cur_rank is None or hparams.cur_rank < 0:
                Rcur, Ccur, _ = build_cur_factors_adaptive(
                    phi_flat, alphas_flat, cluster_method='maxmin',
                    normalize_method='svd', verbose=verbose)
            else:
                rank = max(min(int(hparams.cur_rank), n_vox, alphas_flat.shape[1]), q_svd)
                Rcur, Ccur = build_cur_factors(
                    phi_flat, alphas_flat, rank=rank, normalize_method='svd',
                    cluster_method='maxmin')
            Qc, Tc = torch.linalg.qr(Ccur.T, mode='reduced')
            Qr, Tr = torch.linalg.qr(Rcur.T, mode='reduced')
            mid_mat = Tc @ Tr.T
            q_use = max(min(q_svd, mid_mat.shape[0], mid_mat.shape[1]), 1)
            # Full SVD of the k×k middle factor so pooled prefixes are nested
            # and deterministic (same algebraic convention as svd_lowrank + conj).
            Um, S, Vh = torch.linalg.svd(mid_mat, full_matrices=False)
            Um, S, Vh = Um[:, :q_use], S[:q_use], Vh[:q_use]
            Vm = Vh.mH.conj()
            U_q.append(Qc @ Um)
            V_q.append(Qr @ Vm)
            S_q.append(S)
        elif svd_method == 'direct':
            phase = torch.exp(-2j * torch.pi * (
                alphas_flat.T @ phi_flat
            ))  # (M, N)
            # Zero empty voxels (do not leave exp(0)=1). Same as svd_decomp_linop.
            phase = phase * m_vec
            q_use = max(min(L_max, n_vox, alphas_flat.shape[1], hparams.L), 1)
            U, S, V = torch.svd_lowrank(phase, q=q_use)
            U_q.append(U)
            V_q.append(V.conj())
            S_q.append(S)

    entries = []
    for q, S in enumerate(S_q):
        for i in range(S.numel()):
            entries.append((float(S[i].real), q, i))
    entries.sort(key=lambda t: -t[0])
    L_q = [1] * Q_kept
    leftover = S_tot - Q_kept
    for _, q, i in entries:
        if leftover <= 0:
            break
        if i == 0:
            continue
        if L_q[q] == i:
            L_q[q] += 1
            leftover -= 1
    selected = [float(S_q[q][i].real) for q in range(Q_kept) for i in range(L_q[q])]
    pooled_threshold = min(selected) if selected else 0.0

    spatial_red, temporal_red = [], []
    V_red_t = tuple(int(v) for v in V_fft_r)
    V_full_t = tuple(int(v) for v in V_fft_f)
    for q in range(Q_kept):
        L = L_q[q]
        # L = round(S_tot / Q_kept)
        # L = max(L_q)
        # L = round(L * 1.5)
        print(f'Lorig = {L_q[q]}, Lnew = {L}')
        sp = (V_q[q][:, :L] * (S_q[q][:L] ** 0.5)).T
        tm = (U_q[q][:, :L] * (S_q[q][:L] ** 0.5)).T
        spatial_red.append(sp.reshape((L, *V_red_t)))
        temporal_red.append(tm.reshape((L, *alphas_nrm.shape[1:])))

    # ----------------- Per-block upsample -----------------
    if time_reduction_factor is not None:
        alphas_w_full = expand_temporal(
            alphas_nrm, num_time_high=alphas.shape[1], dim=1, order=3)
    else:
        alphas_w_full = alphas_nrm
    a_mean = alphas_w_full.reshape(B, -1).mean(dim=1)
    a_mean_b = a_mean[(slice(None),) + (None,) * (alphas_w_full.ndim - 1)]
    alphas_ac_full = alphas_w_full - a_mean_b
    dc_red = torch.exp(
        -2j * math.pi * einsum(phis_nrm.float(), a_mean.float(), 'b ..., b -> ...'))
    if reduced_im_size is not None:
        dc_full = expand_spatial(dc_red[None], im_size, order=3)[0]
    else:
        dc_full = dc_red

    spatial_full, temporal_full, trj_x, ramps, masks, slices = [], [], [], [], [], []
    for q, blk in enumerate(kept):
        sp = spatial_red[q]
        win_r = blk['win_r'].to(dtype=sp.real.dtype if sp.is_complex() else sp.dtype)
        win_r = win_r.to(dtype=sp.dtype)
        slc_f = blk['slc_f']
        slc_r = blk['slc_r']
        if reduced_im_size is not None:
            # Same as global SVD: expand the full reduced canvas, then crop.
            # Resizing the crop alone treats it as its own image and does not
            # invert reduce_spatial's align_corners map.
            canvas = sp.new_zeros((sp.shape[0], *red_size))
            canvas[(slice(None),) + slc_r] = sp * win_r
            sp = expand_spatial(canvas, im_size, order=3)
            sp = sp[(slice(None),) + slc_f]
        else:
            sp = sp * win_r
        occ_f = blk['occ_f'].to(dtype=sp.real.dtype if sp.is_complex() else sp.dtype)
        occ_f = occ_f.to(dtype=sp.dtype)
        sp = sp * occ_f * spat[slc_f] * dc_full[slc_f]
        leaked = sp.abs() * (~blk['occ_f']).to(sp.real.dtype)
        if float(leaked.max()) > 0:
            raise RuntimeError(
                f'block {q}: spatial funcs leaked outside slice_full / occupancy')

        tm = temporal_red[q]
        if time_reduction_factor is not None:
            tm = expand_temporal(tm, num_time_high=alphas.shape[1], dim=1, order=3)

        r_c = torch.as_tensor(blk['r_c'], device=trj.device, dtype=trj.dtype)
        # AC-only zeroth: chat · mean(α) is piecewise-constant static phase and
        # is what showed up as blocking on |(img - ref)|. Smooth DC is in dc_full.
        chat_eff = chat[q] - (C_aff[q] @ r_c.double())
        zeroth = torch.exp(
            -2j * math.pi * einsum(chat_eff.float(), alphas_ac_full, 'b, b ... -> ...'))
        tm = tm * zeroth.to(dtype=tm.dtype) * temp

        if shear_block:
            trj_q = trj + einsum(
                C_aff[q].to(dtype=trj.dtype), alphas_ac_full, 'b d, b ... -> ... d')
        else:
            trj_q = trj
        ramp = torch.exp(-2j * math.pi * einsum(trj_q, r_c, '... d, d -> ...'))

        spatial_full.append(sp.contiguous())
        temporal_full.append(tm.contiguous())
        trj_x.append(_cycles_to_rad(trj_q, im_size))
        ramps.append(ramp.contiguous())
        # Taper is already in spatial_q; linop mask is occupancy only (not w²).
        masks.append(occ_f.real if occ_f.is_complex() else occ_f)
        slices.append(slc_f)

    if verbose:
        cmax = float(C_aff.abs().amax()) if shear_block else 0.0
        ratio = V_fft_f / np.maximum(N_t, 1) - V_fft_r / np.maximum(N_r, 1)
        print(f'qblock: Q={Q} -> {Q_i}, Q_kept={Q_kept}, V={V_full_t}, '
              f'V_red={V_red_t}, stride={tuple(int(v) for v in V_full)}, '
              f'overlap={overlap:g}, d(V/N)={tuple(float(x) for x in ratio)}, '
              f'S_tot={S_tot}, L_q={L_q}, pooled_thr={pooled_threshold:.3e}, '
              f'shear={shear}, |C|_max={cmax:.3g}')

    if dcf is None:
        dcf = torch.ones(trj_size, dtype=real_dtype, device=torch_dev)

    A = _QBlockSenseLinop(
        mps=mps, dcf=dcf,
        spatial_q=spatial_full, temporal_q=temporal_full,
        masks_q=masks, slices_q=slices, ramps_q=ramps, trj_x=trj_x,
        im_size=im_size, V=V_full_t, bparams=bparams,
        oversamp=os, width=kern_size[0],
        shear='block' if shear_block else False,
        pack_n_trans=pack_n_trans,
    )
    A.Q_per_axis = Q_i
    A.L_q = [int(s.shape[0]) for s in spatial_full]
    A.S_tot = int(sum(A.L_q))
    A.pooled_threshold = pooled_threshold
    A.Q_kept = Q_kept
    return A


def svd_decomp_linop(phis: torch.Tensor,
                     alphas: torch.Tensor,
                     mps: torch.Tensor,
                     trj: torch.Tensor,
                     hparams: hofft_params,
                     svd_method: str = 'cur',
                     spatial_mask: Optional[torch.Tensor] = None,
                     dcf: Optional[torch.Tensor] = None,
                     bparams: batching_params = batching_params(),
                     use_sigpy: bool = True) -> linop:
    """
    SVD decomposition and linop.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    mps : torch.Tensor
        Coil sensitivities with shape (C, *im_size)
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    hparams : hofft_params
        HOFFT parameters.
    svd_method : str
        Method to use for SVD. Options are:
        'direct' - uses torch.linalg.svd to solve for the SVD
        'lobpcg' - use LOBPCG to solve for the SVD via the matvec operator
        'power' - use power method to solve for the SVD via the matvec operator
        'cur' - performs an effiicent SVD on the CUR decomposition
        'seg' - segments in alpha space first and does SVD on segmneted space followed by a temporal least squares solve
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    dcf : Optional[torch.Tensor]
        Density compensation factor with shape (*trj_size)
    bparams : batching_params
        Batching parameters for the HOFFT linop.
    use_sigpy : bool
        Whether to use Sigpy's KB NUFFT framework.
        
    Returns
    -------
    linop : linop
        SVD decomposition and linop.
    """
    # Consts
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    d = len(im_size)
    torch_dev = phis.device
    kern_size = hparams.kern_size
    os = hparams.os
    normalize_coeffs = hparams.normalize_coeffs
    num_compressed_bases = hparams.num_compressed_bases
    reduced_im_size = hparams.reduced_im_size
    num_representative_alphas = hparams.num_representative_alphas
    verbose = hparams.verbose
    solver = hparams.solver
    lamda = hparams.lamda
    spatial_batch_size = hparams.spatial_batch_size
    time_reduction_factor = hparams.time_reduction_factor
    
    # Make sure the kernel size is isotropic
    for i in range(1, d):
        assert kern_size[0] == kern_size[i], "Kernel size must be isotropic for time-segmented NUFFT decomposition"
    
    # ----------------- Process Phase and Reduce Problem Size -----------------
    # Process phase coefficients 
    phis_nrm, alphas_nrm, spat, temp = _process_phase_coefficients(phis, alphas,
                                                                   normalize_coeffs=normalize_coeffs,
                                                                   num_compressed_bases=num_compressed_bases)
    B = phis_nrm.shape[0]
    
    # Reduce spatial and temporal dims
    phis_nrm, alphas_nrm, _, spatial_mask = _build_reduced_terms(phis_nrm, alphas_nrm, hparams, spatial_mask)
        
    # ----------------- SVD Operation -----------------
    
    # SVD decomposition via dense
    if svd_method == 'direct':
        if verbose:
            numel = np.prod(phis_nrm.shape[1:]) * np.prod(alphas_nrm.shape[1:])
            print(f'SVD direct memory usage: {8 * numel / 2**30:.2f} GB')
        phase = torch.exp(-2j * torch.pi * (
            alphas_nrm.reshape((B, -1)).T @ phis_nrm.reshape((B, -1))
        ))  # (M, N)
        phase = phase * spatial_mask.flatten()
        U, S, V = torch.svd_lowrank(phase, q=hparams.L)
        spatial_funcs  = V[:, :hparams.L] * (S[:hparams.L] ** 0.5)
        temporal_funcs = U[:, :hparams.L] * (S[:hparams.L] ** 0.5)
        spatial_funcs = spatial_funcs.T.reshape((hparams.L, *phis_nrm.shape[1:])).conj()
        temporal_funcs = temporal_funcs.T.reshape((hparams.L, *alphas_nrm.shape[1:]))
    # SVD decomposition via matvec operator
    elif svd_method == 'lobpcg' or svd_method == 'power':
        hparams_copy = copy.copy(hparams)
        hparams_copy.cur_rank = None
        hparams_copy.matvec_kwargs = {'spatial_batch_size': 2**8}
        phase_model = _make_phase_model(phis_nrm, alphas_nrm, hparams_copy)
        spatial_funcs, temporal_funcs = svd_decomp(phase_model, hparams, 
                                                   mask=spatial_mask, 
                                                   svd_method=svd_method,
                                                   verbose=verbose)
    elif svd_method == 'cur':
        normalize_method = 'svd'
        # P = C * R, clustered in the whitened (U, Σ, V) coordinates
        if hparams.cur_rank is None or hparams.cur_rank < 0:
            R, C, _ = build_cur_factors_adaptive(phis_nrm.reshape((B, -1)), 
                                                 alphas_nrm.reshape((B, -1)), 
                                                 cluster_method='maxmin',
                                                 normalize_method=normalize_method,
                                                 verbose=verbose)
        else:
            R, C = build_cur_factors(phis_nrm.reshape((B, -1)), 
                                     alphas_nrm.reshape((B, -1)), 
                                     rank=hparams.cur_rank, 
                                     normalize_method=normalize_method,
                                     cluster_method='maxmin',)
        
        # QR decompose
        Qc, Tc = torch.linalg.qr(C.T, mode='reduced') # (M, k) (k, k)
        Qr, Tr = torch.linalg.qr(R.T, mode='reduced') # (N, k) (k, k)
        
        # SVD middle part such that (Tc @ Tr.T) = Um @ S @ Vm.T
        mid_mat = Tc @ Tr.T
        # Um, S, VmH = torch.linalg.svd(mid_mat, full_matrices=False)
        # Vm = VmH.T
        Um, S, Vm = torch.svd_lowrank(mid_mat, q=hparams.L)
        Vm = Vm.conj()
        
        # Total phase = (Qc @ Um) @ S @ (Qr @ Vm).T
        U = Qc @ Um
        V = Qr @ Vm
        
        # Reshape
        spatial_funcs  = V[:, :hparams.L] * (S[:hparams.L] ** 0.5)
        temporal_funcs = U[:, :hparams.L] * (S[:hparams.L] ** 0.5)
        spatial_funcs = spatial_funcs.T.reshape((hparams.L, *phis_nrm.shape[1:]))
        temporal_funcs = temporal_funcs.T.reshape((hparams.L, *alphas_nrm.shape[1:]))
        del R, C, Qc, Tc, Qr, Tr, mid_mat, Um, S, Vm, U, V
    elif svd_method == 'seg':
        
        # SVD on segmented alpha space
        assert num_representative_alphas is not None, "num_representative_alphas must be specified for segmented SVD"
        
        # Segment alpha space
        alphas_nrm_seg = k_alpha_selection(alphas_nrm, num_representative_alphas, 
                                           method='maxmin', 
                                           train_frac=1.0,
                                           verbose=verbose)
        
        # SVD on reduced matrix
        phase = torch.exp(-2j * torch.pi * (
            alphas_nrm_seg.reshape((B, -1)).T @ phis_nrm.reshape((B, -1))
        ))  # (M, N)
        phase = phase * spatial_mask.flatten()
        U, S, Vh = torch.linalg.svd(phase, full_matrices=False)
        V = Vh.H
        spatial_funcs  = V[:, :hparams.L] * (S[:hparams.L] ** 0.5)
        spatial_funcs = spatial_funcs.T.reshape((hparams.L, *phis_nrm.shape[1:])).conj()
        
        # Temporal least squares solve
        phase_model = _make_phase_model(phis_nrm, alphas_nrm, hparams)
        kern_bases_dummy = torch.ones_like(spatial_funcs[:1])
        temporal_funcs = lstsq_temporal(phase_model, kern_bases_dummy, spatial_funcs,
                                        mask=spatial_mask,
                                        solver=solver,
                                        lamda=lamda,
                                        spatial_batch_size=spatial_batch_size)
        temporal_funcs = temporal_funcs.squeeze(1) # squeeze dummy dimension
    
    del phis_nrm, alphas_nrm
    if torch_dev.type == 'cuda':
        torch.cuda.empty_cache()

    # Expand temporal
    if time_reduction_factor is not None:
        n_out = (temporal_funcs.numel() // temporal_funcs.shape[1]) * alphas.shape[1]
        if temporal_funcs.is_cuda and n_out * temporal_funcs.element_size() > 512 * 1024 ** 2:
            temporal_funcs = expand_temporal(
                temporal_funcs.cpu(), num_time_high=alphas.shape[1], dim=1, order=3)
            if torch_dev.type == 'cuda':
                torch.cuda.empty_cache()
            temporal_funcs = temporal_funcs.to(torch_dev)
        else:
            temporal_funcs = expand_temporal(
                temporal_funcs, num_time_high=alphas.shape[1], dim=1, order=3)
        
    # Reshape temporal functions to brodcast KB kernels
    dummy_kern_size = (1,)*d
    temporal_funcs = temporal_funcs.reshape((hparams.L, *dummy_kern_size, *alphas.shape[1:]))
    
    # Expand spatial 
    if reduced_im_size is not None:
        spatial_funcs = expand_spatial(spatial_funcs, im_size, order=3)
    
    # Use Sigpy's KB NUFFT framework for forward model
    if use_sigpy:    
        # Build NUFFT
        # nft = sigpy_nufft(im_size, oversamp=os, width=kern_size[0])
        # nft.beta = nft.optimal_beta(torch_dev=torch_dev)
        nft = cufi_nufft(im_size, oversamp=os, width=kern_size[0])
        nft.plan(trj[None] if trj.ndim == d + 1 else trj)
        
        # Build linop
        temporal_funcs = temporal_funcs.reshape((hparams.L, *trj_size))
        A = sense_linop(trj, mps, dcf, nufft=nft, 
                        spatial_funcs=spatial_funcs * spat,
                        temporal_funcs=temporal_funcs * temp,
                        bparams=bparams)
    # Use HOFFT forward model with KB NUFFT weights
    else:
        # Calculate KB NUFFT weights
        nft = sigpy_nufft(im_size, oversamp=os, width=kern_size[0])
        nft.beta = nft.optimal_beta(torch_dev=torch_dev)
        spatial_factor, kern_weights = kb_nufft(trj, im_size, kern_size, 
                                                os=os, beta=nft.beta)
        
        # Combine
        spatial_factors = spatial_factor * spatial_funcs * spat
        kern_weights = kern_weights * temporal_funcs * temp
        
        # Build linop
        trj_grd = (os * trj).round()/os
        A = hofft_linop(trj=trj_grd, mps=mps, dcf=dcf, 
                        kern_weights=kern_weights, 
                        spatial_factors=spatial_factors, 
                        os_grid=os, bparams=bparams)
    
    return A

def alpha_seg_decomp_linop(phis: torch.Tensor,
                           alphas: torch.Tensor,
                           mps: torch.Tensor,
                           trj: torch.Tensor,
                           hparams: hofft_params,
                           spatial_mask: Optional[torch.Tensor] = None,
                           dcf: Optional[torch.Tensor] = None,
                           bparams: batching_params = batching_params(),
                           use_sigpy: bool = False) -> linop:
    """
    alpha-segmented NUFFT decomposition and linop.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    mps : torch.Tensor
        Coil sensitivities with shape (C, *im_size)
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    hparams : hofft_params
        HOFFT parameters.
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    dcf : Optional[torch.Tensor]
        Density compensation factor with shape (*trj_size)
    bparams : batching_params
        Batching parameters for the HOFFT linop.
    normalize_coeffs : bool
        Whether to normalize the phase coefficients.
    
    Returns
    -------
    linop : linop
        Time-segmented NUFFT decomposition and linop.
    """
    # Consts
    im_size = phis.shape[1:]
    d = len(im_size)
    torch_dev = phis.device
    kern_size = hparams.kern_size
    os = hparams.os
    normalize_coeffs = hparams.normalize_coeffs
    num_compressed_bases = hparams.num_compressed_bases
    solver = hparams.solver
    lamda = hparams.lamda
    reduced_im_size = hparams.reduced_im_size
    time_reduction_factor = hparams.time_reduction_factor
    
    # Make sure the kernel size is isotropic
    for i in range(1, d):
        assert kern_size[0] == kern_size[i], "Kernel size must be isotropic for time-segmented NUFFT decomposition"
    
    # Process phase coefficients 
    phis_nrm, alphas_nrm, spat, temp = _process_phase_coefficients(phis, alphas,
                                                                   normalize_coeffs=normalize_coeffs,
                                                                   num_compressed_bases=num_compressed_bases)
    B = phis_nrm.shape[0]
    
    # Reduce spatial and temporal dims
    hparams_copy = copy.copy(hparams)
    hparams_copy.kern_size = (1,)*d
    hparams_copy.kalpha_method = 'kmeans'
    phis_nrm, alphas_nrm, kern_bases, spatial_mask = _build_reduced_terms(phis_nrm, alphas_nrm, 
                                                                          hparams=hparams_copy, spatial_mask=spatial_mask)
    
    # Get alpha segmentation spatial maps
    spatial_funcs = choose_init(phis_nrm, alphas_nrm, 
                                hparams=hparams_copy, 
                                spatial_init='seg')
    
    # Single temporal least squares solve
    phase_model = _make_phase_model(phis_nrm, alphas_nrm, hparams_copy)
    temporal_funcs = lstsq_temporal(phase_model,
                                    kern_bases=kern_bases,
                                    spatial_factors=spatial_funcs,
                                    mask=spatial_mask,
                                    solver=solver,
                                    lamda=lamda)
    temporal_funcs = temporal_funcs.reshape((hparams.L, *hparams_copy.kern_size, *alphas_nrm.shape[1:]))
    
    # Expand spatial 
    if reduced_im_size is not None:
        spatial_funcs = expand_spatial(spatial_funcs, im_size, order=3)
        
    # Expand temporal
    if time_reduction_factor is not None:
        temporal_funcs = expand_temporal(temporal_funcs, 
                                         num_time_high=alphas.shape[1], 
                                         dim=1 + len(kern_size), order=3)
    
    # Get optimal beta parameter
    nft = sigpy_nufft(im_size, oversamp=os, width=kern_size[0])
    nft.beta = nft.optimal_beta(torch_dev=torch_dev)
    
    # Use Sigpy's KB NUFFT framework for forward model
    if use_sigpy:
        temporal_funcs = temporal_funcs.reshape((hparams.L, *alphas.shape[1:]))
        A = sense_linop(trj, mps, dcf, nufft=nft, 
                        spatial_funcs=spatial_funcs * spat,
                        temporal_funcs=temporal_funcs * temp,
                        bparams=bparams)
    # Use HOFFT forward model with KB NUFFT weights
    else:
        # Calculate KB NUFFT weights
        spatial_factor, kern_weights = kb_nufft(trj, im_size, kern_size, 
                                                os=os, beta=nft.beta)
        
        # Combine
        spatial_factors = spatial_factor * spatial_funcs * spat
        kern_weights = kern_weights * temporal_funcs * temp
        
        # Build linop
        trj_grd = (os * trj).round()/os
        A = hofft_linop(trj=trj_grd, mps=mps, dcf=dcf, 
                        kern_weights=kern_weights, 
                        spatial_factors=spatial_factors, 
                        os_grid=os, bparams=bparams)
    
    return A

def hofft_decomp_linop(phis: torch.Tensor,
                       alphas: torch.Tensor,
                       mps: torch.Tensor,
                       trj: torch.Tensor,
                       hparams: hofft_params,
                       spatial_mask: Optional[torch.Tensor] = None,
                       dcf: Optional[torch.Tensor] = None,
                       bparams: batching_params = batching_params()) -> linop:
    """
    Performs HOFFT decomposition using ALS and builds the HOFFT linop.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    mps : torch.Tensor
        coil sensitivities with shape (C, *im_size)
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    hparams : hofft_params
        HOFFT parameters.
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    dcf : Optional[torch.Tensor]
        density compensation factor with shape (*trj_size)
    bparams : batching_params
        Batching parameters for the HOFFT linop.
        
    Returns
    -------
    linop : linop
        HOFFT linop taking in an image and returning k-space data
    """
    # Consts
    im_size = phis.shape[1:]
    os = hparams.os
    spatial_init = hparams.spatial_init
    L = hparams.L
    kern_size = hparams.kern_size
    solver = hparams.solver
    lamda = hparams.lamda
    spatial_batch_size = hparams.spatial_batch_size
    verbose = hparams.verbose
    reduced_im_size = hparams.reduced_im_size
    kalpha_method = hparams.kalpha_method
    normalize_coeffs = hparams.normalize_coeffs
    num_compressed_bases = hparams.num_compressed_bases
    num_representative_alphas = hparams.num_representative_alphas
    max_als_iter = hparams.max_als_iter
    time_reduction_factor = hparams.time_reduction_factor
    
    # ----------------- Process phase coefficients -----------------
    # Combine grid deviation phase to high order phase coefficients
    phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, os)
    trj_grd = (trj * os).round() / os
    phis_stack = torch.cat([phis_dev, phis], dim=0)
    alphas_stack = torch.cat([alphas_dev, alphas], dim=0)
    phis_nrm, alphas_nrm, spat, temp = _process_phase_coefficients(phis_stack, alphas_stack,
                                                                   normalize_coeffs=normalize_coeffs,
                                                                   num_compressed_bases=num_compressed_bases)

    # ----------------- Reduce dims and setup for ALS solve -----------------
    # Reduce spatial and temporal dims
    phis_nrm, _, kern_bases, spatial_mask = _build_reduced_terms(phis_nrm, alphas_nrm,
                                                                 hparams=hparams, spatial_mask=spatial_mask)
    
    # Initialize spatial factors
    spatial_factors = choose_init(phis_nrm, alphas_nrm, 
                                  hparams=hparams, 
                                  spatial_init=spatial_init)
    
    # Reduce alpha size 
    if num_representative_alphas is not None:
        alphas_nrm_red = k_alpha_selection(alphas_nrm, num_representative_alphas, 
                                           method=kalpha_method, 
                                           train_frac=0.1,
                                           verbose=verbose)
    else:
        alphas_nrm_red = alphas_nrm
        
    # Make matvec phase model
    phase_model_red = _make_phase_model(phis_nrm, alphas_nrm_red, hparams)
    
    # ----------------- ALS Solve -----------------        
    
    # ALS to solve for kernel weights and spatial factors
    spatial_factors, kern_weights = als_iterations(phase_model_red, kern_bases, spatial_factors,
                                                    mask=spatial_mask,
                                                    max_iter=max_als_iter,
                                                    solver=solver,
                                                    lamda=lamda,
                                                    spatial_batch_size=spatial_batch_size,
                                                    verbose=verbose)
    
    # Expand temporal via one pass of temporal least squares solve
    if num_representative_alphas is not None:
        phase_model = _make_phase_model(phis_nrm, alphas_nrm, hparams)
        kern_weights = lstsq_temporal(phase_model, kern_bases, spatial_factors,
                                        mask=spatial_mask,
                                        solver=solver,
                                        lamda=lamda,
                                        spatial_batch_size=spatial_batch_size)
    kern_weights = kern_weights.reshape((L, *kern_size, *alphas_nrm.shape[1:]))
    
    # Expand spatial 
    if reduced_im_size is not None:
        spatial_factors = expand_spatial(spatial_factors, im_size, order=3)
    
    # Apply phase midpoints
    spatial_factors *= spat
    kern_weights *= temp
    
    # ----------------- Build linop -----------------        
    A = hofft_linop(trj=trj_grd, mps=mps, dcf=dcf, 
                    kern_weights=kern_weights, 
                    spatial_factors=spatial_factors, 
                    os_grid=os, bparams=bparams)
    
    return A

def sparse_hofft_decomp_linop(phis: torch.Tensor,
                              alphas: torch.Tensor,
                              mps: torch.Tensor,
                              trj: torch.Tensor,
                              hparams: hofft_params,
                              sparams: sparse_params,
                              sparsity: int = 16,
                              spatial_mask: Optional[torch.Tensor] = None,
                              dcf: Optional[torch.Tensor] = None,
                              bparams: batching_params = batching_params()) -> linop:
    """
    Performs sparse HOFFT decomposition using ALS and builds the HOFFT linop.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    mps : torch.Tensor
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    hparams : hofft_params
    """
    # Consts
    im_size = phis.shape[1:]
    os = hparams.os
    L = hparams.L
    S = sparsity
    kern_size = hparams.kern_size
    solver = hparams.solver
    num_representative_alphas = hparams.num_representative_alphas
    spatial_batch_size = hparams.spatial_batch_size
    verbose = hparams.verbose
    normalize_coeffs = hparams.normalize_coeffs
    kalpha_method = hparams.kalpha_method
    num_compressed_bases = hparams.num_compressed_bases
    max_als_iter = hparams.max_als_iter
    reduced_im_size = hparams.reduced_im_size
    spatial_subsample = sparams.spatial_subsample
    temporal_batch_size = sparams.temporal_batch_size
    assert num_representative_alphas is not None, "num_representative_alphas must be provided"
    
    # ----------------- Process phase coefficients -----------------
    # Combine grid deviation phase to high order phase coefficients
    phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, os)
    trj_grd = (trj * os).round() / os
    phis_stack = torch.cat([phis_dev, phis], dim=0)
    alphas_stack = torch.cat([alphas_dev, alphas], dim=0)
    phis_nrm, alphas_nrm, spat, temp = _process_phase_coefficients(phis_stack, alphas_stack,
                                                                   normalize_coeffs=normalize_coeffs,
                                                                   num_compressed_bases=num_compressed_bases)

    # Reduce spatial dims
    phis_nrm, _, kern_bases, spatial_mask = _build_reduced_terms(phis_nrm, alphas_nrm, hparams, spatial_mask)
    
    # Initialize spatial factors
    spatial_factors = choose_init(phis_nrm, alphas_nrm, 
                                  hparams=hparams, 
                                  spatial_init='seg')
    
    # ----------------- ALS on Q representative alpha vectors -----------------
    
    # Select Q representative alpha vectors
    Q = num_representative_alphas
    betas = k_alpha_selection(alphas_nrm, Q, 
                              method=kalpha_method, 
                              train_frac=1.0, 
                              verbose=verbose)
    
    # Decomposition on reduced alpha vectors
    phase_model_red = _make_phase_model(phis_nrm, betas, hparams)
    spatial_factors, compressed_kernels = als_iterations(phase_model_red, kern_bases, spatial_factors,
                                                         mask=spatial_mask,
                                                         max_iter=max_als_iter, 
                                                         solver=solver,
                                                         lamda=hparams.lamda,
                                                         spatial_batch_size=spatial_batch_size,
                                                         verbose=verbose)
    # als_iterations returns (L, K, Q); linop expects (L, *kern_size, Q)
    compressed_kernels = compressed_kernels.reshape((L, *kern_size, Q))
    
    # ----------------- Solve for Sparse Coefficients -----------------
    # Least squares approach
    if sparams.interp_type == 'lstsq':
        sparse_inds, sparse_coeffs = lstsq_compressed_fixed_support(
            phis_nrm, alphas_nrm,
            spatial_factors=spatial_factors,
            compressed_kernels=compressed_kernels,
            kern_bases=kern_bases,
            betas=betas,
            sparsity=S,
            hparams=hparams,
            spatial_mask=spatial_mask,
            spatial_subsample=spatial_subsample,
            temporal_batch_size=temporal_batch_size,
            lamda=sparams.lamda,
            verbose=verbose)
        # lstsq returns (T, S); linop wants (S, *trj_size)
        sparse_inds = sparse_inds.T.reshape((S, *alphas_nrm.shape[1:]))
        sparse_coeffs = sparse_coeffs.T.reshape((S, *alphas_nrm.shape[1:]))
    # Smooth distance-based approach
    else:
        # Tune d and p against a validation subset of exact HOFFT kernels
        kernel = sparams.interp_type
        d_opt, p_opt, errors = sweep_smooth_interp_hyperparams(
            phis_nrm, alphas_nrm, spatial_factors, compressed_kernels, betas, kern_bases,
            sparsity=S,
            hparams=hparams, sparams=sparams,
            spatial_mask=spatial_mask, verbose=verbose,
        )
        if verbose:
            msg = f'Strategy 2.5 auto-tune: picked d={d_opt}'
            if kernel == 'inv_dist':
                msg += f', p={p_opt}'
            print(f'{msg} (validation error {errors.min().item():.4g})')

        # Already returns (S, *trj_size)
        sparse_inds, sparse_coeffs = smooth_sparse_coeffs(
            alphas_nrm, betas,
            sparsity=S,
            kernel=kernel,
            d=d_opt, p=p_opt,
            eps=sparams.eps,
            temporal_batch_size=temporal_batch_size,
        )
        
    # # Solve for kernels given sparse coefficients
    # compressed_kernels = lstsq_compressed_kernels(phis_nrm, alphas_nrm, 
    #                                               spatial_factors=spatial_factors,
    #                                               kern_bases=kern_bases,
    #                                               sparse_inds=sparse_inds,
    #                                               sparse_coeffs=sparse_coeffs,
    #                                               hparams=hparams,
    #                                               num_kernels=Q,
    #                                               spatial_mask=spatial_mask,
    #                                               spatial_subsample=spatial_subsample,
    #                                               temporal_batch_size=temporal_batch_size,
    #                                               lamda=sparams.lamda,
    #                                               verbose=verbose,)
    
    # Expand spatial via upsampling
    if reduced_im_size is not None:
        spatial_factors = expand_spatial(spatial_factors, im_size, order=3)
        
    # Build sparse linop
    spatial_factors *= spat # Apply phase midpoints
    A = hofft_compressed_linop(trj=trj_grd, mps=mps, dcf=dcf,
                                compressed_kernels=compressed_kernels,
                                sparse_idxs=sparse_inds,
                                sparse_coeffs=sparse_coeffs,
                                spatial_factors=spatial_factors,
                                temporal_factors=temp,
                                os_grid=os, bparams=bparams)
    
    return A
