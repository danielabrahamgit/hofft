"""
This file contains the functions to process the phase coefficients for the HOFFT model.

Most spatio-temporal phase patterns in MRI can be decomposed as:
phi(r, t) = sum_b phi_b(r) * alpha_b(t)

Where:
- phi_b(r) are the spatial phase bases
- alpha_b(t) are the temporal phase coefficients
- there are B spatial phase bases and B temporal phase coefficients

Below we provide functions to express, resize, and process these phase coefficients.
"""
import torch
import numpy as np

from typing import Optional
from einops import einsum
from .utils import gen_grd

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
    alphas = torch.zeros((4, *trj_size), dtype=g.dtype, device=g.device)
    X, Y, Z = spatial_crds[..., 0], spatial_crds[..., 1], spatial_crds[..., 2]
    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
    phis = coco_bases(X, Y, Z)
    alphas[0] = gx ** 2 + gy ** 2
    alphas[1] = (gz ** 2) / 4
    alphas[2] = -gx * gz 
    alphas[3] = -gy * gz
    alphas /= 2 * field_strength

    # Integral on alphas, gamma_bar to map T to phase
    for b in range(alphas.shape[0]): # more memory efficient
        alphas[b, 1:] = torch.cumulative_trapezoid(alphas[b], dx=dt, dim=0) * gamma_bar
    alphas = alphas[:, 1:]
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

def whiten_phis_alphas(phis: torch.Tensor,
                       alphas: torch.Tensor,
                       B_compressed: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Whitens the covariance matrix of the phi bases, and applies a similar transformation to the alpha coefficients to preserve total phase.
    
    Args
    ----
    phis : torch.Tensor
        Phase basis with shape (B, *im_size)
    alphas : torch.Tensor
        Phase coefficients with shape (B, *trj_size)
    B_compressed : Optional[int]
        Number of compressed bases to use for whitening.
        If None, all bases are used.

    Returns
    -------
    phis_whitened : torch.Tensor
        Whitened phase basis with shape (B, *im_size)
    alphas_whitened : torch.Tensor
        Whitened phase coefficients with shape (B, *trj_size)
    """
    # Consts
    im_size = phis.shape[1:]
    B = phis.shape[0]
    R = np.prod(im_size)
    assert B == alphas.shape[0]
    if B_compressed is None:
        B_compressed = B
    
    # Eigen decompose phis
    cov_mat = einsum(phis, phis, 'B1 ..., B2 ... -> B1 B2') / R
    evals, evecs = torch.linalg.eigh(cov_mat)
    evecs = evecs[:, :B_compressed] # B B'
    evals = evals[:B_compressed] # B'
    
    # Compute whitening matrix for phi, alpha
    W_phi = (evals[:, None] ** -0.5) * evecs.T # B' B
    W_alpha = (evals[:, None] ** +0.5) * evecs.T # B' B
    
    # Check that whitening is within tolerance
    I = W_phi @ cov_mat @ W_phi.T
    I_targ = torch.eye(B_compressed, device=phis.device, dtype=phis.dtype)
    err = (I - I_targ).abs().max()
    if err > 1e-4:
        print(f'Warning: Whitening matrix is not exact, max|I - I_targ| error = {err:1.2e} > 1e-4')
        
    # Whiten
    phis_whitened = einsum(W_phi, phis, 'Bc B, B ... -> Bc ...')
    alphas_whitened = einsum(W_alpha, alphas, 'Bc B, B ... -> Bc ...')
    
    return phis_whitened, alphas_whitened

def rescale_phis_alphas(phis: torch.Tensor,
                        alphas: torch.Tensor,
                        offset: str = 'midpoint') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f"""
    Phase model is 
    Phase = 2pi Alpha.T @ Phi 
    with Alpha.shape (B, M) and Phi.shape (B, N).
    
    We want to split things according to 
    Phi = Phi_nrm + Phi_ofs, Phi_ofs.shape (B,)
    Alpha = Alpha_nrm + Alpha_ofs, Alpha_ofs.shape (B,)
    
    where Phi_nrm[b] is normalized to be between [-1/2, 1/2] and Phi_ofs[b] is the ofsset of the phase basis.
    and Alpha_nrm[b] represents the number of phase wraps now and Alpha_ofs[b] is the offset of the phase coefficients.
    
    This is useful for two reasons:
    1. alpha_nrm tells you how many phase wraps accumulate
    2. This removes offset terms in alpha or phi that the HOFFT kernels would otherwise have to account for.
    
    Args
    -----
    phis : torch.Tensor
        Phase basis with shape (B, *im_size)
    alphas : torch.Tensor
        Phase coefficients with shape (B, *trj_size)
    offset : str
        The offset to use, either 'midpoint', 'mean', or 'median'
        
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
    N = np.prod(im_size)
    M = np.prod(trj_size)
    assert B == alphas.shape[0]
    
    # Flatten everything
    phis_flt = phis.reshape((B, N))
    alphas_flt = alphas.reshape((B, M))
    
    # Compute offset terms
    if offset == 'midpoint':
        phis_ofs = (phis_flt.min(dim=1).values + phis_flt.max(dim=1).values)/2
        alphas_ofs = (alphas_flt.min(dim=1).values + alphas_flt.max(dim=1).values)/2
    elif offset == 'mean':
        phis_ofs = phis_flt.mean(dim=1)
        alphas_ofs = alphas_flt.mean(dim=1)
    elif offset == 'median':
        phis_ofs = phis_flt.median(dim=1).values
        alphas_ofs = alphas_flt.median(dim=1).values
    
    # Identify any indices with no spatial variation
    idx_flat = torch.argwhere(phis_flt.std(dim=1) < 1e-6)[:, 0]
    phis_ofs[idx_flat] = 0.0
    
    # Centered phis and alphas
    phis_flt_cent = phis_flt - phis_ofs[:, None]
    alphas_flt_cent = alphas_flt - alphas_ofs[:, None]
    
    # Rescale phis to be between [-1/2, 1/2], or [0, 1] if no spatial variation
    scales = phis_flt_cent.abs().max(dim=1).values * 2
    scales[idx_flat] = phis_flt_cent[idx_flat].abs().max(dim=1).values
    phis_ofs /= scales
    phis_nrm = phis_flt_cent / scales[:, None]
    alphas_ofs *= scales
    alphas_nrm = alphas_flt_cent * scales[:, None]
        
    # Reshape and return
    return phis_nrm.reshape((B, *im_size)), phis_ofs, alphas_nrm.reshape((B, *trj_size)), alphas_ofs

def whiten_phis_alphas(phis: torch.Tensor,
                       alphas: torch.Tensor,
                       B_compressed: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A = PHI.T @ ALPHA with shape (im_size, trj_size).
    This function both whitens the PHI bases and finds the best rank-B_compressed approximation to A using SVD, efficiently.
    
    Args
    ----
    phis : torch.Tensor
        The spatial phase bases, shape (B, *im_size)
    alphas : torch.Tensor
        The temporal phase coefficients, shape (B, *trj_size)
    B_compressed : int
        The number of bases to compress to
        
    Returns
    -------
    phis_compressed : torch.Tensor
        The compressed spatial phase bases, shape (B_compressed, *im_size)
    alphas_compressed : torch.Tensor
        The compressed temporal phase coefficients, shape (B_compressed, *trj_size)
    """
    # Consts
    B = phis.shape[0]
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    N = np.prod(im_size)
    M = np.prod(trj_size)
    assert B == alphas.shape[0]
    assert B_compressed <= B, "B_compressed must be less than or equal to B"
    
    # Flatten everything
    A = alphas.reshape((B, M))
    P = phis.reshape((B, N))
    # Total phase = A.T @ P
    
    # QR decompose
    # Total phase = Qa @ Ra @ Rp.T @ Qp.T
    Qa, Ra = torch.linalg.qr(A.T, mode='reduced')
    Qp, Rp = torch.linalg.qr(P.T, mode='reduced')

    # SVD middle part such that (Ra @ Rp.T) = Um @ S @ Vm.T
    # This happens on a BxB matrix, so it's extremely fast.
    mid_mat = Ra @ Rp.T
    Um, S, Vmt = torch.linalg.svd(mid_mat, full_matrices=False)
    Vm = Vmt.T
    
    # Total phase = (Qa @ Um) @ S @ (Qp @ Vm).T
    # Total phase = (   U   ) @ S @ (   V   ).T
    U = Qa @ Um
    V = Qp @ Vm

    # Set Phi' = V.T / 2pi and  Alpha' = 2pi * S @ U.T
    phis_new   = (V[:, :B_compressed]).T / (2 * torch.pi)
    alphas_new = (U[:, :B_compressed] * S[:B_compressed]).T * 2 * torch.pi
    
    # Reshape and return
    return phis_new.reshape((B_compressed, *im_size)), alphas_new.reshape((B_compressed, *trj_size))

def apply_phase_midpoints(phis_nrm: torch.Tensor,
                          alphas_nrm: torch.Tensor,
                          phis_mp: torch.Tensor,
                          alphas_mp: torch.Tensor,
                          spatial_factors: Optional[torch.Tensor] = None,
                          temporal_factors: Optional[torch.Tensor] = None,) -> tuple[torch.Tensor, torch.Tensor]:
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
        defaults to ones with shape (*im_size)
    temporal_factors : torch.Tensor
        The temporal factors, shape (... *trj_size)
        defaults to ones with shape (*trj_size)

    Returns
    -------
    spatial_factors : torch.Tensor
        The spatial factors, shape (..., *im_size)
    temporal_factors : torch.Tensor
        The temporal factors, shape (... *trj_size)
    """
    if spatial_factors is None:
        spatial_factors = torch.ones_like(phis_nrm[0]).type(torch.complex64)
    if temporal_factors is None:
        temporal_factors = torch.ones_like(alphas_nrm[0]).type(torch.complex64)
    spatial_mp = torch.exp(-2j * torch.pi * einsum(phis_nrm, alphas_mp, 'B ..., B -> ...')) # *im_size
    temporal_mp = torch.exp(-2j * torch.pi * einsum(alphas_nrm, phis_mp, 'B ..., B -> ...')) # *trj_size
    temporal_mp *= torch.exp(-2j * torch.pi * (phis_mp @ alphas_mp))
    temporal_factors *= temporal_mp
    spatial_factors *= spatial_mp
    return spatial_factors, temporal_factors