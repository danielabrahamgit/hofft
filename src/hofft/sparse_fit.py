"""
Direct sparse fitting of compressed HOFFT kernels.

We directly fit the compressed HOFFT kernels to the high order phase matrix
(see math_docs/sparse_fit.md). The high order phase matrix is
    Phi[n, m] = exp(-j2pi phi(r_n) . alpha(t_m))     shape (N, M)
and we approximate it as
    Phi ~= E @ H_comp @ C
where
    E[n, l + L*w] = g_l(r_n) * kappa_w(r_n)          shape (N, L*W)
        (g_l are the HOFFT spatial factors, kappa_w the Fourier kernel bases)
    H_comp                                            shape (L*W, Q)
        compressed basis HOFFT kernels (Q << M), from the 'k_alphas'
        decomposition -- the learned kernels for the Q representative betas
        ARE H_comp.
    C                                                 shape (Q, M)
        coefficients with S-sparse columns (||c_m||_0 <= S).

Two ways to solve for the S-sparse columns of C given known H_comp:

``lstsq_compressed_fixed_support``: the atom indices come from alpha-space
nearest neighbors to the k_alphas betas, and coefficients are obtained via a
single batched least-squares solve.

``smooth_sparse_coeffs``: no dictionary, Gram matrix, or phase-model matvec
needed -- the sparse support is again the S nearest betas, and the
coefficients are set directly from normalized alpha-space distances to those
betas (considerably cheaper to compute than the least-squares solve).
"""

import torch
import numpy as np

from einops import einsum
from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm

from mr_recon.dtypes import complex_dtype

from .decomp import hofft_params, lstsq_temporal
from .matvec import subsample_idx

__all__ = [
    'sparse_params',
    'lstsq_compressed_fixed_support',
    'smooth_sparse_coeffs',
    'sweep_smooth_interp_hyperparams',
    'batch_lstsq_fixed_support',
]


@dataclass
class sparse_params:
    """
    Parameters for sparse decomposition.

    Attributes
    ----------
    Q : int
        Number of representative beta vectors to use.
    S : int
        Number of sparse coefficients to keep.
    spatial_subsample : Optional[int]
        Fixed voxel count for the subsampled least-squares normal equations.
        None uses all voxels.
    temporal_batch_size : Optional[int]
        Batch size over the trajectory (temporal) dimension.
    spatial_batch_size : Optional[int]
        Batch size for the spatial dimension.
    interp_type: str
        Type of interpolation to use for the smooth sparse decomposition.
        'lstsq' -> least-squares solver
        'rbf' -> smooth distance-based sparse rbf interpolation
        'inv_dist' -> smooth distance-based sparse inverse-distance interpolation
    lamda : float
        Regularization parameter for the least-squares solver.
    num_validation : Optional[int]
        If given, autotunes the smooth sparse interpolation kernel parameters with num_validation held-out trajectory points.
    """
    Q: int = 500
    S: int = 5
    spatial_subsample: Optional[int] = None
    temporal_batch_size: Optional[int] = 2**15
    spatial_batch_size: Optional[int] = None
    interp_type: str = 'lstsq'
    lamda: float = 1e-6
    num_validation: Optional[int] = None


def _flatten_spatial(phis: torch.Tensor,
                     spatial_factors: torch.Tensor,
                     kern_bases: torch.Tensor,
                     spatial_mask: Optional[torch.Tensor],
                     spatial_subsample: Optional[int],
                     seed: int = 0) -> tuple:
    """
    Flattens the spatial tensors to (-1, R), optionally subsampling voxels, and
    builds the combined encoding matrix E and its mask-weighted conjugate.

    Args
    ----
    phis : torch.Tensor
        spatial phase bases with shape (B, *im_size)
    spatial_factors : torch.Tensor
        HOFFT spatial factors g_l with shape (L, *im_size)
    kern_bases : torch.Tensor
        Fourier kernel bases kappa_w with shape (W, *im_size)
    spatial_mask : Optional[torch.Tensor]
        image weighting mask with shape (*im_size)
    spatial_subsample : Optional[int]
        number of voxels to keep (randomized sketch over N). None keeps all.
    seed : int
        seed for the (reproducible) voxel subset

    Returns
    -------
    phis_flt : torch.Tensor
        (possibly subsampled) flattened phis with shape (B, R)
    E : torch.Tensor
        encoding matrix with shape (L*W, R), row index l*W + w
    Ew : torch.Tensor
        mask-weighted conjugate conj(E) * |mask|^2 with shape (L*W, R)
    """
    B = phis.shape[0]
    L = spatial_factors.shape[0]
    W = kern_bases.shape[0]
    R = int(np.prod(phis.shape[1:]))
    torch_dev = phis.device

    phis_flt = phis.reshape((B, R))
    sf_flt = spatial_factors.reshape((L, R))
    kb_flt = kern_bases.reshape((W, R))
    if spatial_mask is None:
        mask_flt = torch.ones(R, dtype=complex_dtype, device=torch_dev)
    else:
        mask_flt = spatial_mask.reshape(R)

    # Randomized sketch over the voxel (N) dimension
    if spatial_subsample is not None and spatial_subsample < R:
        vidx = subsample_idx(R, spatial_subsample, torch_dev, mode='fixed', seed=seed)
        phis_flt = phis_flt[:, vidx]
        sf_flt = sf_flt[:, vidx]
        kb_flt = kb_flt[:, vidx]
        mask_flt = mask_flt[vidx]

    # E[lw, r] = g_l(r) * kappa_w(r), flattened l-major to match (L, *kern, Q)
    E = einsum(sf_flt, kb_flt, 'L R, W R -> L W R').reshape((L * W, -1))
    Ew = E.conj() * (mask_flt.abs() ** 2)

    return phis_flt, E, Ew

def _nearest_beta_support(alphas: torch.Tensor,
                          betas: torch.Tensor,
                          sparsity: int,
                          trj_size: tuple) -> torch.Tensor:
    """S-nearest betas in alpha space; returns sparse_inds with shape (S, *trj_size)."""
    B = alphas.shape[0]
    T = int(np.prod(trj_size))
    Q = betas.shape[0]
    S = min(sparsity, Q)
    alphas_flt = alphas.reshape((B, T)).T  # (T, B)
    dists = torch.cdist(alphas_flt, betas)  # (T, Q)
    support = torch.topk(dists, S, dim=-1, largest=False).indices  # (T, S)
    return support.T.reshape((S, *trj_size))

def smooth_sparse_coeffs(alphas: torch.Tensor,
                         betas: torch.Tensor,
                         sparsity: int,
                         kernel: str = 'rbf',
                         d: float = 1.0,
                         p: float = 2.0,
                         eps: float = 1e-3,
                         temporal_batch_size: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Strategy 2.5 (see math_docs/sparse_fit.md): smooth distance-based sparse
    interpolation coefficients in alpha space.

    Unlike ``lstsq_compressed_fixed_support``, this needs no dictionary, Gram
    matrix, or phase-model matvec -- the sparse support is the S nearest betas
    (as in ``_nearest_beta_support``) and the coefficients are set directly
    from normalized alpha-space distances to those betas:
        c_{q,m} = w(delta_{q,m}) / sum_{q' in support(m)} w(delta_{q',m})
        delta_{q,m} = ||alpha_m - beta_q||_2 / d_m
        d_m = d * max_{q in support(m)} ||alpha_m - beta_q||_2
    with w(delta) = exp(-delta^2 / 2) ('rbf') or w(delta) = 1 / (delta^p + eps)
    ('inv_dist').

    Args
    ----
    alphas : torch.Tensor
        temporal phase coefficients with shape (B, *trj_size)
    betas : torch.Tensor
        representative alpha vectors (e.g. from K_alphas_init) with shape (Q, B)
    sparsity : int
        number of nonzero atoms (S) per column of C
    kernel : str
        'rbf' -> w(delta) = exp(-delta^2 / 2)
        'inv_dist' -> w(delta) = 1 / (delta^p + eps)
    d : float
        distance-scaling hyperparameter (see math_docs/sparse_fit.md)
    p : float
        power for the inverse-distance kernel
    eps : float
        regularization for the inverse-distance kernel and the coefficient
        normalization
    temporal_batch_size : Optional[int]
        Batch size for computing top k distances.

    Returns
    -------
    sparse_inds : torch.Tensor
        sparse indices (long) with shape (S, *trj_size), values in [0, Q)
    sparse_coeffs : torch.Tensor
        sparse coefficients (complex) with shape (S, *trj_size)
    """
    B = alphas.shape[0]
    trj_size = alphas.shape[1:]
    T = int(np.prod(trj_size))
    Q = betas.shape[0]
    S = min(sparsity, Q)
    tbs = T if temporal_batch_size is None else temporal_batch_size
    
    # Compute top k distances in batches
    alphas_flt = alphas.reshape((B, T)).T  # (T, B)
    dists_topk = torch.empty((T, S), dtype=torch.float32, device=alphas.device)
    idx_topk = torch.empty((T, S), dtype=torch.long, device=alphas.device)
    for t1 in range(0, T, tbs):
        t2 = min(t1 + tbs, T)
        dists_batch = torch.cdist(alphas_flt[t1:t2], betas)  # (T, Q)
        dists_topk_batch, idx_topk_batch = torch.topk(dists_batch, S, 
                                                      dim=-1, 
                                                      largest=False)  # (T, S), ascending
        dists_topk[t1:t2] = dists_topk_batch
        idx_topk[t1:t2] = idx_topk_batch

    d_m = d * dists_topk.amax(dim=-1, keepdim=True).clamp(min=eps)  # (T, 1)
    delta = dists_topk / d_m  # (T, S)

    if kernel == 'rbf':
        w = torch.exp(-0.5 * delta ** 2)
    elif kernel == 'inv_dist':
        w = 1.0 / (delta ** p + eps)
    else:
        raise ValueError(f"kernel '{kernel}' is not supported, use 'rbf' or 'inv_dist'")
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=eps)

    sparse_inds = idx_topk.T.reshape((S, *trj_size))
    sparse_coeffs = w.T.reshape((S, *trj_size)).type(complex_dtype)
    return sparse_inds, sparse_coeffs

def sweep_smooth_interp_hyperparams(phis: torch.Tensor,
                                    alphas: torch.Tensor,
                                    spatial_factors: torch.Tensor,
                                    compressed_kernels: torch.Tensor,
                                    betas: torch.Tensor,
                                    kern_bases: torch.Tensor,
                                    sparsity: int,
                                    hparams: hofft_params,
                                    kernel: str = 'rbf',
                                    d_grid: torch.Tensor = torch.logspace(-1, 0.7, 10),
                                    p_grid: torch.Tensor = torch.linspace(0.1, 8.0, 8),
                                    eps: float = 1e-3,
                                    num_val: int = 2**10,
                                    spatial_mask: Optional[torch.Tensor] = None,
                                    verbose: bool = True) -> tuple[float, Optional[float], torch.Tensor]:
    """
    Tunes the (d, p) hyperparameters of ``smooth_sparse_coeffs`` (Strategy 2.5,
    see math_docs/sparse_fit.md) against a validation subset of "exact" HOFFT
    kernels.

    For ``num_val`` randomly held-out trajectory points, computes the exact
    least-squares-optimal HOFFT kernels h_m (via ``lstsq_temporal``, holding the
    already-fitted ``spatial_factors`` fixed) and compares them against the
    Strategy 2.5 compressed reconstruction hat{h}_m = H_comp @ c_m for every
    (d, p) in the grid (p is ignored for kernel='rbf'). Returns the pair that
    minimizes the relative Frobenius error over the validation set.

    This is only tractable on a small validation subset -- ``lstsq_temporal``
    solves an exact (L*W, L*W) normal-equations system per call, so it should
    not be run over the full trajectory.

    Args
    ----
    phis : torch.Tensor
        spatial phase bases with shape (B, *im_size) -- the same (possibly
        spatially-reduced) phis used to obtain spatial_factors / compressed_kernels
        (e.g. phis_reduced, matching kern_bases' spatial shape)
    alphas : torch.Tensor
        temporal phase coefficients with shape (B, *trj_size), from which the
        validation subset is drawn
    spatial_factors : torch.Tensor
        HOFFT spatial factors g_l with shape (L, *im_size) (already fixed, e.g.
        from K_alphas_init, matching phis/kern_bases' spatial shape)
    compressed_kernels : torch.Tensor
        compressed HOFFT kernels H_comp with shape (L, *kern_size, Q)
    betas : torch.Tensor
        representative alpha vectors with shape (Q, B)
    kern_bases : torch.Tensor
        Fourier kernel bases kappa_w(r) with shape (W, *im_size), W = prod(kern_size)
    sparsity : int
        number of nonzero atoms (S) per column of C
    hparams : hofft_params
        HOFFT parameters (matvec_type/kwargs build the validation phase model;
        solver/lamda are used for the lstsq_temporal normal-equations solve)
    kernel : str
        'rbf' or 'inv_dist' (see smooth_sparse_coeffs)
    d_grid : tuple
        candidate values of d to sweep
    p_grid : tuple
        candidate values of p to sweep (only used when kernel == 'inv_dist')
    eps : float
        regularization, passed through to smooth_sparse_coeffs
    num_val : int
        number of held-out trajectory points to validate against
    spatial_mask : Optional[torch.Tensor]
        image weighting mask with shape (*im_size), passed to lstsq_temporal
    verbose : bool
        whether to print progress

    Returns
    -------
    best_d : float
        the d value minimizing the validation error
    best_p : Optional[float]
        the p value minimizing the validation error (None if kernel == 'rbf')
    errors : torch.Tensor
        relative Frobenius validation errors, shape (len(d_grid), len(p_grid))
        for kernel == 'inv_dist', or (len(d_grid),) for kernel == 'rbf'
    """
    B = phis.shape[0]
    trj_size = alphas.shape[1:]
    T = int(np.prod(trj_size))
    L = spatial_factors.shape[0]
    W = kern_bases.shape[0]
    Q = compressed_kernels.shape[-1]
    torch_dev = phis.device
    S = min(sparsity, Q)

    # Validation subset of trajectory points, held out from the sparse fit itself
    num_val = min(num_val, T)
    val_idx = subsample_idx(T, num_val, torch_dev, mode='fixed', seed=2)
    alphas_val = alphas.reshape((B, T))[:, val_idx]  # (B, num_val)

    # Exact HOFFT kernels h_m for the validation subset, holding spatial_factors fixed
    pm = hparams.matvec_type(phis, alphas_val, **hparams.matvec_kwargs)
    h_true = lstsq_temporal(pm, kern_bases, spatial_factors,
                            mask=spatial_mask, solver=hparams.solver, lamda=hparams.lamda)
    h_true = h_true.reshape((L * W, num_val))
    H_comp = compressed_kernels.reshape((L * W, Q))

    single_p = kernel != 'inv_dist'
    p_iter = (None,) if single_p else tuple(p_grid)

    errors = torch.full((len(d_grid), len(p_iter)), float('inf'), device=torch_dev)
    best_err, best_d, best_p = float('inf'), d_grid[0], p_iter[0]
    for i, d in enumerate(tqdm(d_grid, desc='Sweep smooth-interp hyperparams', disable=not verbose)):
        for j, p in enumerate(p_iter):
            sparse_inds, sparse_coeffs = smooth_sparse_coeffs(
                alphas_val, betas, sparsity=S, kernel=kernel, d=d,
                p=(2.0 if p is None else p), eps=eps)
            h_approx = torch.zeros_like(h_true)
            for s in range(S):
                h_approx += H_comp[:, sparse_inds[s]] * sparse_coeffs[s]
            err = (torch.linalg.norm(h_true - h_approx) / torch.linalg.norm(h_true)).item()
            errors[i, j] = err
            if err < best_err:
                best_err, best_d, best_p = err, d, p

    if single_p:
        errors = errors[:, 0]
    return best_d, best_p, errors

def batch_lstsq_fixed_support(corr0: torch.Tensor,
                              gram: torch.Tensor,
                              support: torch.Tensor,
                              lamda: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched LS on a fixed support: min || corr0 - gram[:,Z] c || with Z = support.

    Parameters
    ----------
    corr0 : torch.Tensor
        Target correlations D^H x with shape (T, Q).
    gram : torch.Tensor
        Dictionary Gram D^H D with shape (Q, Q).
    support : torch.Tensor
        Fixed atom indices with shape (T, S).

    Returns
    -------
    support : torch.Tensor
        Same as input, shape (T, S).
    coeffs : torch.Tensor
        LS coefficients with shape (T, S).
    """
    T, S = support.shape
    torch_dev = corr0.device
    g_ss = gram[support[:, :, None], support[:, None, :]]  # (T, S, S)
    rhs = corr0.gather(1, support)  # (T, S)
    eye = torch.eye(S, dtype=complex_dtype, device=torch_dev)
    coeffs = torch.linalg.solve(g_ss + lamda * eye, rhs.unsqueeze(-1))[..., 0]
    return support, coeffs

def lstsq_compressed_fixed_support(phis: torch.Tensor,
                                 alphas: torch.Tensor,
                                 spatial_factors: torch.Tensor,
                                 compressed_kernels: torch.Tensor,
                                 kern_bases: torch.Tensor,
                                 betas: torch.Tensor,
                                 sparsity: int,
                                 hparams: hofft_params,
                                 spatial_mask: Optional[torch.Tensor] = None,
                                 spatial_subsample: Optional[int] = None,
                                 temporal_batch_size: Optional[int] = None,
                                 lamda: float = 1e-6,
                                 verbose: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Finds least squares optimal interpolation coefficients for the sparse HOFFT kernels.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase bases with shape (B, *im_size)
    alphas : torch.Tensor
        temporal phase coefficients with shape (B, *trj_size)
    spatial_factors : torch.Tensor
        HOFFT spatial factors g_l with shape (L, *im_size)
    compressed_kernels : torch.Tensor
        compressed HOFFT kernels H_comp with shape (L, *kern_size, Q)
    kern_bases : torch.Tensor
        Fourier kernel bases kappa_w(r) with shape (W, *im_size), W = prod(kern_size)
    sparsity : int
        number of nonzero atoms (S) per column of C
    hparams : hofft_params
        HOFFT parameters (matvec_type/kwargs build the validation phase model;
        solver/lamda are used for the lstsq_temporal normal-equations solve)
    spatial_mask : Optional[torch.Tensor]
        image weighting mask with shape (*im_size), passed to lstsq_temporal
    spatial_subsample : Optional[int]
        number of voxels to keep (randomized sketch over N). None keeps all.
    temporal_batch_size : Optional[int]
        Batch size over the trajectory (temporal) dimension.
    lamda : float
        Regularization parameter for the least-squares solver.
    verbose : bool
        whether to print progress

    Returns
    -------
    sparse_idxs : torch.Tensor
        sparse indices (long) with shape (T, S), values in [0, Q)
    sparse_coeffs : torch.Tensor
        sparse coefficients (complex) with shape (T, S)
    """
    torch_dev = phis.device
    B = phis.shape[0]
    trj_size = alphas.shape[1:]
    T = int(np.prod(trj_size))
    L = spatial_factors.shape[0]
    Q = compressed_kernels.shape[-1]
    S = min(sparsity, Q)
    tbs = T if temporal_batch_size is None else temporal_batch_size
    save_mem = False

    H = compressed_kernels.reshape((L * kern_bases.shape[0], Q))
    phis_flt, E, Ew = _flatten_spatial(phis, spatial_factors, kern_bases,
                                       spatial_mask, spatial_subsample, seed=0)
    D = einsum(H, E, 'lw Q, lw R -> Q R')
    Dw = einsum(H.conj(), Ew, 'lw Q, lw R -> Q R')
    gram = einsum(Dw, D, 'Q1 R, Q2 R -> Q1 Q2')

    alphas_flt = alphas.reshape((B, T))
    if not save_mem:
        pm = hparams.matvec_type(phis_flt, alphas_flt, **hparams.matvec_kwargs)
        corr = pm.forward(Dw)  # (Q, T)
        corr_t = corr.T.contiguous()  # (T, Q)
        support = _nearest_beta_support(alphas, betas, S, trj_size)
        support_flt = support.reshape((S, T)).T.contiguous()  # (T, S)
    else:
        corr_t = None
        support_flt = None

    sparse_idxs = torch.empty((T, S), dtype=torch.long, device=torch_dev)
    sparse_coeffs = torch.empty((T, S), dtype=complex_dtype, device=torch_dev)
    for t1 in tqdm(range(0, T, tbs), 'Fixed-support LS', disable=not verbose):
        t2 = min(t1 + tbs, T)
        
        # Grab batch of correlations and support
        if save_mem:
            pm = hparams.matvec_type(phis_flt, alphas_flt[:, t1:t2], **hparams.matvec_kwargs)
            corr_batch = pm.forward(Dw).T
            support_batch = _nearest_beta_support(alphas_flt[:, t1:t2], betas, S, (t2-t1,)).T
        else:
            corr_batch = corr_t[t1:t2]
            support_batch = support_flt[t1:t2]
        
        
        idx, coef = batch_lstsq_fixed_support(corr_batch, gram, support_batch, 
                                              lamda=lamda)
        sparse_idxs[t1:t2] = idx
        sparse_coeffs[t1:t2] = coef
    return sparse_idxs, sparse_coeffs
