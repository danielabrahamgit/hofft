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

import matplotlib.pyplot as plt

from typing import Optional
from einops import einsum
from .utils import gen_grd
from .linalg import svd_product

def visualize_alpha_space(phis: torch.Tensor,
                          alphas: torch.Tensor,
                          B_compressed: Optional[int] = None,
                          npts_plot: int = 5000) -> None:
    """
    Visualize alpha space.
    
    Args
    ----
    phis : torch.Tensor
        The spatial phase bases, shape (B, *im_size)
    alphas : torch.Tensor
        The temporal phase coefficients, shape (B, *trj_size)
    B_compressed : Optional[int]
        The number of bases to compress to, must be 2 or 3
    npts_plot : int
        The number of points to plot
    """
    # Consts
    B = phis.shape[0]
    
    # Compress
    if B_compressed is not None:
        assert B_compressed in [2, 3], "B_compressed must be 2 or 3"
        phis, alphas= whiten_phis_alphas(phis, alphas, B_compressed)
        B = B_compressed
    else:
        assert B in [2, 3], "B must be 2 or 3 if B_compressed is not provided"
        
    # Grab random points to plot
    alphas_flt = alphas.reshape((B, -1)).cpu()
    rnd_inds = torch.randperm(alphas_flt.shape[1])[:npts_plot]
    alphas_flt = alphas_flt[:, rnd_inds]
    
    # Normalize phis to range [-1/2, 1/2]
    phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis.cpu(), alphas_flt, quantiles=(0.0, 1.0))
    # alphas_nrm += alphas_mp[:, None]

    
    # Plot phis as images
    if phis_nrm.ndim == 3:
        slc = slice(None)
    else:
        slc = (slice(None), phis.shape[1]//2, slice(None))
    plt.figure(figsize=(10, 5))
    for b in range(B):
        plt.subplot(1, B, b+1)
        plt.imshow(phis_nrm[b][slc].rot90(), cmap='RdBu_r')
        plt.colorbar()
        plt.title(f'phi_{b}')
        plt.axis('off')
    plt.tight_layout()
        
    # Plot alpha scatter
    if B == 2:
        plt.figure(figsize=(10, 5))
        plt.scatter(alphas_nrm[0], alphas_nrm[1], marker='.', alpha=0.2)
        plt.tight_layout()
        # plt.axis('equal')
    if B == 3:
        # 3D scatter
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(alphas_nrm[0], alphas_nrm[1], alphas_nrm[2], marker='.', alpha=0.2)
        plt.tight_layout()
        # plt.axis('equal')
    # plt.xlim(-6, 6)
    # plt.ylim(-6, 6)
    # if B == 3:
    #     plt.zlim(-6, 6)

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

def remove_empty_bases(phis: torch.Tensor,
                       alphas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove empty bases from the phase coefficients.
    
    Args
    ----
    phis : torch.Tensor
        The spatial phase bases, shape (B, *im_size)
    alphas : torch.Tensor
        The temporal phase coefficients, shape (B, *trj_size)
    
    Returns
    -------
    phis : torch.Tensor
        The spatial phase bases, shape (B_new, *im_size)
    alphas : torch.Tensor
        The temporal phase coefficients, shape (B_new, *trj_size)
    """
    B = phis.shape[0]
    energy = (phis.reshape((B, -1)).abs().mean(dim=1)
            * alphas.reshape((B, -1)).abs().mean(dim=1))
    idxs = torch.argwhere(energy > 1e-6)[:, 0]
    phis, alphas = phis[idxs], alphas[idxs]
    return phis, alphas

def rescale_phis_alphas(phis: torch.Tensor,
                        alphas: torch.Tensor,
                        quantiles: tuple[float, float] = (0.0, 1.0),
                        mask: Optional[torch.Tensor] = None,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    
    # Detault mask
    if mask is None:
        mask = torch.ones_like(phis[0])
    
    # Flatten everything
    phis_flt = phis.reshape((B, N))
    alphas_flt = alphas.reshape((B, M))
    mask = mask.reshape((N,))
    
    # phi = S * (phi + phis_ofs)
    qs = torch.tensor(quantiles, device=phis_flt.device)
    try:
        plow, phigh = torch.quantile(phis_flt[:, mask > 0], q=qs, dim=1)
    except:
        phis_rnd = phis_flt[:, mask > 0]
        rnd_inds = torch.randperm(phis_rnd.shape[1])[:10_000]
        phis_rnd = phis_rnd[:, rnd_inds]
        plow, phigh = torch.quantile(phis_rnd, q=qs, dim=1)
    scales = (phigh - plow)
    phis_ofs = (plow + phigh) / 2 / scales
    
    # Identify any indices with no spatial variation
    idx_flat = torch.argwhere((plow - phigh).abs() < 1e-6)[:, 0]
    phis_ofs[idx_flat] = 0.0
    scales[idx_flat] = plow[idx_flat]
    
    # Rescale phis to be between [-1/2, 1/2], or [0, 1] if no spatial variation
    phis_nrm = phis_flt / scales[:, None] - phis_ofs[:, None]
    
    # Carry scaling term into alpha
    alphas_ofs = (alphas_flt * scales[:, None]).mean(dim=1)
    alphas_nrm = (alphas_flt * scales[:, None]) - alphas_ofs[:, None]
        
    # Reshape and return
    return phis_nrm.reshape((B, *im_size)), phis_ofs, alphas_nrm.reshape((B, *trj_size)), alphas_ofs

def whiten_phis_alphas(phis: torch.Tensor,
                       alphas: torch.Tensor,
                       B_compressed: Optional[int] = None,
                       return_singular_values: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Thin SVD of the total phase Φ A without forming the dense N×M matrix.

        Φ A = U Σ V^H
        returns Φ_w = U^T,  A_w = Σ V^H
    
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
    phis_w : torch.Tensor
        U^T with shape (B_compressed, *im_size)
    alphas_w : torch.Tensor
        V^H with shape (B_compressed, *trj_size)
    """
    # Consts
    B = phis.shape[0]
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    N = np.prod(im_size)
    M = np.prod(trj_size)
    assert B == alphas.shape[0]
    if B_compressed is None:
        B_compressed = B
    assert B_compressed <= B, "B_compressed must be less than or equal to B"

    # Φ A with Φ : (N, B) and A : (B, M)
    U, S, Vh = svd_product(phis.reshape((B, N)).mT,
                           alphas.reshape((B, M)),
                           rank=B_compressed)
    phis_new = U.mT.reshape((B_compressed, *im_size))
    alphas_new = Vh.reshape((B_compressed, *trj_size))

    # Return
    if return_singular_values:
        return phis_new, alphas_new, S
    else:
        p = 1
        alphas_new = einsum(alphas_new, S ** p, 'B ..., B -> B ...')
        phis_new = einsum(phis_new, S ** (1-p), 'B ..., B -> B ...')
        return phis_new, alphas_new

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

def remove_linear_terms(phis: torch.Tensor,
                        alphas: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove spatially linear terms from the phase coefficients.
    
    Args
    ----
    phis : torch.Tensor
        The spatial phase bases, shape (B, *im_size)
    alphas : torch.Tensor
        The temporal phase coefficients, shape (B, *trj_size)
    mask : Optional[torch.Tensor]
        spatial mask with shape (*im_size)
        
    Returns
    -------
    phis_new : torch.Tensor
        The spatial phase bases with linear terms removed, shape (B, *im_size)
    trj_term : torch.Tensor
        The temporal phase coefficients for removed linear terms, shape (*trj_size, d)
    zeroth_order : torch.Tensor
        The zeroth order phase offset, shape (*trj_size)
    """
    # Consts
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    d = len(im_size)
    B = phis.shape[0]
    N = np.prod(im_size)
    
    # Build linear bases
    crds = gen_grd(im_size).to(phis.device)
    
    # Default mask
    if mask is None:
        mask = torch.ones_like(phis[0])
    
    # Least squares fit out linear terms
    mask_flt = mask.reshape((N,)).float()[:, None]
    Amat = crds.reshape((N, d))
    Amat = torch.cat([(Amat[:, :1] * 0 + 1),
                       Amat], dim=1) # offset term
    Bmat = phis.reshape((B, N)).T
    coeffs = torch.linalg.lstsq(Amat * mask_flt, 
                                Bmat * mask_flt).solution # Shape (d+1, B)
    
    # Remove linear terms from phis
    phis_hat = (Amat @ coeffs).T.reshape((B, *im_size))
    phis_new = phis - phis_hat
    
    # Build alphas_lin
    alphas_lin = torch.zeros((d+1, *trj_size), device=alphas.device, dtype=alphas.dtype)
    for b in range(phis.shape[0]):
        alphas_lin += einsum(coeffs[:, b], alphas[b], 'd, ... -> d ...') 
        
    # Split into zeroth order and trj term
    zeroth_order = alphas_lin[0]
    trj_term = alphas_lin[1:].moveaxis(0, -1)
    
    # Return
    return phis_new, trj_term, zeroth_order
        
    