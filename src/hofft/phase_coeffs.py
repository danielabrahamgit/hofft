"""
This file contains the functions to process the phase coefficients for the HOFFT model.

Most spatio-temporal phase patterns in MRI can be decomposed as:
phi(r, t) = sum_k phi_k(r) * alpha_k(t)

Where:
- phi_k(r) are the spatial phase bases
- alpha_k(t) are the temporal phase coefficients

Below we provide functions to express, resize, and process these phase coefficients.
"""
import torch
import numpy as np

from mr_recon.utils import gen_grd
from einops import einsum
from fast_pytorch_kmeans import KMeans

def coco_bases(x: torch.Tensor, 
               y: torch.Tensor, 
               z: torch.Tensor) -> torch.Tensor:
    """
    Phase bases for concomitant field encoding.
    
    Args
    ----
    x : torch.Tensor
        x coordinates wuth shape (...)
    y : torch.Tensor
        y coordinates with shape (...)
    z : torch.Tensor
        z coordinates with shape (...)

    Returns
    -------
    phis : torch.Tensor
        Phase bases with shape (4, ...)
    """
    assert x.shape == y.shape
    assert z.shape == x.shape
    tup = (None,) + (slice(None),) * x.ndim
    x = x[tup]
    y = y[tup]
    z = z[tup]
    return torch.cat([
        z * z,
        x * x + y * y,
        x * z,
        y * z
    ], dim=0)

def sph_bases(x: torch.Tensor, 
              y: torch.Tensor, 
              z: torch.Tensor) -> torch.Tensor:
    """
    Phase bases for spherical harmonic functions
    
    Args
    ----
    x : torch.Tensor
        x coordinates with shape (...)
    y : torch.Tensor
        y coordinates with shape (...)
    z : torch.Tensor
        z coordinates with shape (...)

    Returns
    -------
    phis : torch.Tensor
        Phase bases with shape (16, ...)
    """
    assert x.shape == y.shape
    assert z.shape == x.shape
    tup = (None,) + (slice(None),) * x.ndim
    x = x[tup]
    y = y[tup]
    z = z[tup]
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2
    x3 = x ** 3
    y3 = y ** 3
    z3 = z ** 3
    return torch.cat([
        torch.ones_like(x),
        x,
        y,
        z,
        x * y,
        z * y,
        3 * z2 - (x2 + y2 + z2),
        x * z,
        x2 - y2,
        3 * y * x2 - y3, 
        x * y * z,
        (5 * z2 - (x2 + y2 + z2)) * y,
        5 * z3 - 3 * z * (x2 + y2 + z2),
        (5 * z2 - (x2 + y2 + z2)) * x,
        z * x2 - z * y2,
        x3 - 3 * x * y2
    ], dim=0)

def b0_to_phis_alphas(b0_map: torch.Tensor,
                      trj_size: tuple,
                      ro_dim: int,
                      dt: float,
                      repeat_empty_dims: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert B0 field map to phi and alpha notation

    Args:
    -----
    b0_map : torch.Tensor
        B0 field map in Hz with shape (*im_size)
    trj_size : tuple
        Size of the trajectory, arbitrary dimensions
    ro_dim : int
        Readout dimension, but be in [0, len(trj_size))
    dt : float
        Time step in seconds
    repeat_empty_dims : bool
        If True, repeat the empty dimensions to match the shape of the trajectory, since only the readout dimension is non-empty.

    Returns:
    --------
    phis : torch.Tensor
        Phase basis in radians with shape (1, *im_size),
        normalized to range [-1/2, 1/2]
    alphas : torch.Tensor
        Phase coefficients with shape (1, *trj_size)
    """
    
    # Normalize b0_map to range [-1/2, 1/2]
    scale = 2 * b0_map.abs().max()
    # scale = 1.0
    phis = b0_map / scale
    
    # Make alphas
    ts = torch.arange(trj_size[ro_dim], device=b0_map.device, dtype=torch.float32) * dt * scale
    tup = (slice(None),) + (None,) * (len(trj_size) - 1)
    alphas = ts[tup].moveaxis(0, ro_dim)
    
    # Repeat empty dimensions so that alphas.shape[1:] == trj_size
    if repeat_empty_dims:
        alphas = alphas.expand(trj_size)
    
    # Return
    return phis[None,], alphas[None,]
    
def coco_to_phis_alphas(trj: torch.Tensor,
                        spatial_crds: torch.Tensor,
                        field_strength: float,
                        ro_dim: int,
                        dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert trajectory to concomitant fields phi and alpha notation

    Args
    ----
    trj : torch.Tensor
        K-space trajectory with shape (*trj_size, 3) in units of 1/meters
    spatial_crds : torch.Tensor
        Spatial coordinates with shape (*im_size, 3) in units of meters
    field_strength : float
        Field strength in units of Teslas
    ro_dim : int
        Readout dimension, but be in [0, len(trj_size))
    dt : float
        Sampling time along readout in units of seconds

    Returns
    -------
    phis : torch.Tensor
        Phase basis in radians with shape (4, *im_size)
    alphas : torch.Tensor
        Phase coefficients with shape (4, *trj_size)
    """
    # Consts
    trj = trj.swapaxes(0, ro_dim)
    d = trj.shape[-1]
    trj_size = trj.shape[:-1]
    gamma_bar = 42.5774e6 # Hz / T
    assert d == 3
    assert d == spatial_crds.shape[-1]

    # Get gradient from trj
    trj = trj.type(torch.float32)
    g = torch.diff(trj, dim=0) / (dt * gamma_bar)
    g = torch.cat((g, g[-1:]), dim=0)
    
    # Build phis and alphas
    alphas = torch.zeros((4, *trj_size), dtype=trj.dtype, device=trj.device)
    X, Y, Z = spatial_crds[..., 0], spatial_crds[..., 1], spatial_crds[..., 2]
    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
    phis = coco_bases(X, Y, Z)
    alphas[0] = gx ** 2 + gy ** 2
    alphas[1] = (gz ** 2) / 4
    alphas[2] = -gx * gz 
    alphas[3] = -gy * gz
    alphas /= 2 * field_strength

    # Integral on alphas, gamma_bar to map T to phase
    alphas = torch.cumulative_trapezoid(alphas, dx=dt, dim=1) * gamma_bar
    alphas = torch.cat([alphas[:, :1] * 0, alphas], dim=1)
    
    return phis, alphas.swapaxes(1, ro_dim+1)

def trj_dev_to_phis_alphas(trj: torch.Tensor, 
                           im_size: tuple[int, ...], 
                           os: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Express the spatio-temporal phase pattern that comes up when viewing the deviation of a non-Cartesian trajectory from the nearest Cartesian trajectory.
    
    Args
    ----
    trj : torch.Tensor
        The non-Cartesian trajectory, shape (*trj_size, d)
    im_size : tuple[int, ...]
        The image size, shape (*im_size) len(im_size) == d
    os : float
        The oversampling factor

    Returns
    -------
    phis : torch.Tensor
        The spatial phase bases, shape (B, *im_size)
    alphas : torch.Tensor
        The temporal phase coefficients, shape (B, *trj_size)
    """
    # Get the nearest Cartesian trajectory
    trj_cart = (trj * os).round() / os
    
    # Get the deviation of the non-Cartesian trajectory from the nearest Cartesian trajectory
    trj_dev = trj - trj_cart
    
    # Get the spatial phase bases
    phis = gen_grd(im_size).to(trj.device).moveaxis(-1, 0)
    
    # Get the temporal phase coefficients
    alphas = trj_dev.moveaxis(-1, 0)
    
    return phis, alphas

def rescale_phis_alphas(phis: torch.Tensor,
                        alphas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase model is 
    phase(r, t) = 2pi sum_k phi_k(r) * alpha_k(t)
    
    We will use the following rescaling:
    phase(r,t) = 2pi sum_k (phi_nrm_k(r) + phi_mp_k) * (alpha_nrm(t) + alpha_mp_k)
    where phi_nrm are normalized to be between [-1/2, 1/2]. 
    
    This is useful for two reasons:
    1. alpha_nrm tells you how many phase wraps accumulate, which is convenient for guesstimating how many segments should be used.
    2. This effectively 'equalizes' the contribution of each phase basis, making clustering easier.
    
    Args
    -----
    phis : torch.Tensor
        Phase basis with shape (B, *im_size)
    alphas : torch.Tensor
        Phase coefficients with shape (B, *trj_size)
        
    Returns
    --------
    phis_nrm : torch.Tensor
        Normalized phase basis with shape (B, *im_size)
    phis_mp : torch.Tensor
        Phase basis midpoints with shape (B,)
    alphas_nrm : torch.Tensor
        Normalized phase coefficients with shape (B, *trj_size)
    alphas_mp : torch.Tensor
        Phase coefficients midpoints with shape (B,)
    """
    # Consts
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    B = phis.shape[0]
    R = np.prod(im_size)
    T = np.prod(trj_size)
    assert B == alphas.shape[0]
    
    # Flatten everything
    phis_flt = phis.reshape((B, R))
    alphas_flt = alphas.reshape((B, T))
    
    # Center alphas and phis
    idx_flat = torch.argwhere(phis_flt.std(dim=1) < 1e-6)[:, 0]
    phis_mp = (phis_flt.min(dim=1).values + phis_flt.max(dim=1).values)/2
    phis_mp[idx_flat] = 0.0
    alphas_mp = (alphas_flt.min(dim=1).values + alphas_flt.max(dim=1).values)/2
    phis_flt_cent = phis_flt - phis_mp[:, None]
    alphas_flt_cent = alphas_flt - alphas_mp[:, None]
    
    # Rescale phis to be between [-1/2, 1/2]    
    scales = phis_flt_cent.abs().max(dim=1).values * 2
    scales[idx_flat] = phis_flt_cent[idx_flat].abs().max(dim=1).values
    phis_flt_cent /= scales[:, None]
    phis_mp /= scales
    alphas_flt_cent *= scales[:, None]
    alphas_mp *= scales
    
    # Reshape and return
    phis_nrm = phis_flt_cent.reshape((B, *im_size))
    alphas_nrm = alphas_flt_cent.reshape((B, *trj_size))
    return phis_nrm, phis_mp, alphas_nrm, alphas_mp

def compress_phis_alphas(phis: torch.Tensor,
                         alphas: torch.Tensor,
                         B_compressed: int = 5,
                         eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compress the number of bases in the phase model using SVD.
    This method was developed by ChatGPT :) 
    
    Args
    ----
    phis : torch.Tensor
        The spatial phase bases, shape (B, *im_size)
    alphas : torch.Tensor
        The temporal phase coefficients, shape (B, *trj_size)
    B_compressed : int
        The number of compressed bases to  compress to
        
    Returns
    -------
    phis_compressed : torch.Tensor
        The compressed spatial phase bases, shape (B_compressed, *im_size)
    alphas_compressed : torch.Tensor
        The compressed temporal phase coefficients, shape (B_compressed, *trj_size)
    """
    # Consts
    B = phis.shape[0]
    R = np.prod(phis.shape[1:])
    T = np.prod(alphas.shape[1:])
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    assert B == alphas.shape[0]
    assert B_compressed <= B, "B_compressed must be less than or equal to B"
    
    # Flatten everything
    P = phis.reshape((B, R)).T
    A = alphas.reshape((B, T))
    
    # Small BxB Gram matrices
    H = A @ A.H # B B
    G = P.H @ P # B B
    G += torch.eye(B, device=G.device, dtype=G.dtype) * eps
    
    # Cholesky and Solve orthogonal basis
    L = torch.linalg.cholesky(G)
    # Q = P @ torch.linalg.inv(L).H
    Q = torch.linalg.solve(L.H, P, left=False)
    
    # Decompose and form new SVD terms
    K = L.H @ H @ L # B B
    U, S, _ = torch.linalg.svd(K, full_matrices=False) # B B
    # S, U = torch.linalg.eigh(K)
    U = Q @ U # B R
    S = S ** 0.5 # B
    V = A.H @ P.H @ U @ torch.diag(S ** -1).type(U.dtype) # B T

    # Reshape and return
    phis_compressed   = (U[:, :B_compressed] * (S[:B_compressed] ** 0.5)).T.reshape((B_compressed, *im_size))
    alphas_compressed = (V[:, :B_compressed] * (S[:B_compressed] ** 0.5)).T.reshape((B_compressed, *trj_size))
    return phis_compressed, alphas_compressed

def apply_phase_midpoints(phis_nrm: torch.Tensor,
                          alphas_nrm: torch.Tensor,
                          phis_mp: torch.Tensor,
                          alphas_mp: torch.Tensor,
                          spatial_factors: torch.Tensor,
                          temporal_factors: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply phase midpoints to the phase coefficients.
    
    Args
    ----
    phis_nrm : torch.Tensor
        The spatial phase bases, shape (B, *im_size)
    alphas_nrm : torch.Tensor
        The temporal phase coefficients, shape (B, *trj_size)
    phis_mp : torch.Tensor
        The spatial phase midpoints, shape (B,)
    alphas_mp : torch.Tensor
        The temporal phase midpoints, shape (B,)
    spatial_factors : torch.Tensor
        The spatial factors, shape (..., *im_size)
    temporal_factors : torch.Tensor
        The temporal factors, shape (... *trj_size)

    Returns
    -------
    spatial_factors : torch.Tensor
        The spatial factors, shape (..., *im_size)
    temporal_factors : torch.Tensor
        The temporal factors, shape (... *trj_size)
    """
    spatial_mp = torch.exp(-2j * torch.pi * einsum(phis_nrm, alphas_mp, 'B ..., B -> ...')) # *im_size
    temporal_mp = torch.exp(-2j * torch.pi * einsum(alphas_nrm, phis_mp, 'B ..., B -> ...')) # *trj_size
    temporal_mp *= torch.exp(-2j * torch.pi * (phis_mp @ alphas_mp))
    temporal_factors *= temporal_mp
    spatial_factors *= spatial_mp
    return spatial_factors, temporal_factors
    
def uniform_quantization(coeffs: torch.Tensor, 
                         grid_spacing: float,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform quantization of phase coefficients.
    
    Args
    ----
    coeffs : torch.Tensor
        phi or alpha phase coefficients with shape (B, N)
    grid_spacing : float
        Grid spacing in units of the coefficient units
        
    Returns
    -------
    coeffs_quant : torch.Tensor
        Smaller set of quantized coefficients (B, M), M <= N
    inds_quant : torch.Tensor
        Indices mapping from original coefficients to quantized coefficients with shape (N,) in [0, M)
    """
    # Consts
    assert coeffs.ndim == 2
    
    # Quantize coefficients
    coeffs_quant = (coeffs / grid_spacing).round() * grid_spacing
    coeffs_quant, inds_quant = coeffs_quant.unique(dim=1, return_inverse=True)
    
    return coeffs_quant, inds_quant

def kmeans_quantization(coeffs: torch.Tensor, 
                        K: int,
                        max_iter: int = 1000,
                        mode: str = 'euclidean') -> tuple[torch.Tensor, torch.Tensor]:
    """
    K-means quantization of phase coefficients.
    
    Args
    ----
    coeffs : torch.Tensor
        phi or alpha phase coefficients with shape (B, N)
    K : int
        Number of coefficients to quantize to
    max_iter : int
        Maximum number of iterations for K-means
    mode : str
        Mode for K-means clustering
        
    Returns
    -------
    coeffs_quant : torch.Tensor
        Smaller set of quantized coefficients (B, M), M <= N
    inds_quant : torch.Tensor
        Indices mapping from original coefficients to quantized coefficients with shape (N,) in [0, M)
    """
    # Consts
    assert coeffs.ndim == 2
    
    # Quantize data using K-means
    torch_dev = coeffs.device
    verbose = 0
    if (torch_dev.index == -1) or (torch_dev.index is None):
        kmeans = KMeans(n_clusters=K,
                        max_iter=max_iter,
                        verbose=verbose,
                        mode=mode)
        inds_quant = kmeans.fit_predict(coeffs.T)
    else:
        with torch.cuda.device(torch_dev):
            kmeans = KMeans(n_clusters=K,
                            max_iter=max_iter,
                            verbose=verbose,
                            mode=mode)
            inds_quant = kmeans.fit_predict(coeffs.T)
    coeffs_quant = kmeans.centroids.T

    
    return coeffs_quant, inds_quant