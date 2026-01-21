"""
Contains tools for reducing the spatial or temporal dimension sizes.
"""

import torch

from mr_recon.spatial import spatial_resize_poly, spatial_interp
from mr_recon.utils import gen_grd
from typing import Optional, Callable, Union
from dataclasses import dataclass
from math import floor, ceil
from einops import einsum
from tqdm import tqdm

@dataclass
class reduce_params: 
    spatial_reduce_size: Optional[tuple] = None
    spatial_reduce_order: int = 3
    alpha_reduce_width: Optional[int] = None
    alpha_reduce_grid_spacing: Union[float, tuple] = 0.5
    alpha_reduce_use_apod: bool = False
    alpha_interp_batch_size: Optional[int] = None
    """
    Parameters for reducing the spatial or temporal dimension sizes.
    
    Attributes
    ----------
    spatial_reduce_size : Optional[tuple]
        Optional low resolution size for performing the decomposition
    spatial_reduce_order : int
        Order of the polynomial interpolation
    alpha_reduce_width : Optional[int]
        Width of the kernel
    alpha_reduce_grid_spacing : Union[float, tuple]
        Step sizes for the alpha deviations for each alpha direction, len(alpha_reduce_grid_spacing) = B
    alpha_reduce_use_apod : bool
        Whether to solve for the apodization function
    alpha_interp_batch_size : Optional[int]
        Batch size for the temporal dimension
    """
    
def reduce_spatial(spatial_data: torch.Tensor, 
                   im_size_low: tuple, 
                   order: int = 3) -> torch.Tensor:
    """
    Reduce the spatial dimensions of the data using 
    grid interpolation. Data must lie on a regular grid.
    
    Args
    ----
    data : torch.Tensor
        Data to reduce with shape (..., *im_size)
    im_size_low : tuple
        Low resolution spatial size
    order : int, optional
        Order of the polynomial interpolation
        
    Returns
    -------
    data_low : torch.Tensor
        Reduced data with shape (..., *im_size_low)
    """
    return spatial_resize_poly(spatial_data, 
                               im_size=im_size_low, 
                               order=order, 
                               mode='nearest')
    
def expand_spatial(spatial_data: torch.Tensor, 
                   im_size_high: tuple, 
                   order: int = 3) -> torch.Tensor:
    """
    Expand the spatial dimensions of the data using 
    grid interpolation. Data must lie on a regular grid.
    
    Args
    ----
    spatial_data : torch.Tensor
        Spatial data to expand with shape (..., *im_size)
    im_size_high : tuple
        High resolution spatial size
    order : int, optional
        Order of the polynomial interpolation
        
    Returns
    -------
    spatial_data_high : torch.Tensor
        Expanded spatial data with shape (..., *im_size_high)
    """
    # return spatial_resize_poly(spatial_data, 
    #                            im_size=im_size_high, 
    #                            order=order, 
    #                            mode='nearest')
    
    kwargs = {'order': order, 'mode': 'nearest'}
    torch_dev = spatial_data.device
    im_size_low = spatial_data.shape[-len(im_size_high):]
    im_size_low_tensor = torch.tensor(im_size_low).to(torch_dev)
    spatial_crds = (gen_grd(im_size_high).to(torch_dev) + 0.5) * im_size_low_tensor
    spatial_data_high = spatial_interp(spatial_data, spatial_crds, **kwargs)
    return spatial_data_high

def _solve_1d_kern(phi: torch.Tensor,
                   W: int = 2,
                   dalpha: float = 0.5,
                   Nkerns: int = 100,
                   ptol: float = 1e-4,
                   solve_apod: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    f"""
    Solves for 1d interpolation weights for interpolating in an alpha direction.
    
    Equation to solve is
    e^(-j 2pi delta_alpha * phi(r)) = sum_l w_l(delta_alpha) e^(-j 2pi alpha_l * phi) * A(phi)
    where A(phi) is an optional apodization function.
    
    Args
    ----
    phi : torch.Tensor
        Single spatial phase basis function with shape (R,)
    W : int
        Width of the kernel
    dalpha : float
        Step size for the alpha deviations
    Nkerns : int
        Number of kernel weights to solve for
    ptol : float
        Percent tolerance for ALS convergence
    solve_apod : bool
        Whether to solve for the apodization function
        
    Returns
    -------
    weights : torch.Tensor
        Kernel weights with shape (Nkerns,)
    delta_alphas : torch.Tensor
        Delta alphas with shape (Nkerns,)
    apod : torch.Tensor
        Apodization function with shape (R,)
    """
    # Consts
    phi_flt = phi.flatten()
    torch_dev = phi.device
    
    # Gen alpha grid points
    alphas_grd = torch.arange(-floor(W/2), ceil(W/2), device=torch_dev) * dalpha 
    alphas_grd -= alphas_grd.mean()
    
    # Alpha deviations
    delta_alphas = torch.linspace(-dalpha/2, dalpha/2, Nkerns, device=torch_dev)
    
    # Apodization function
    apod = torch.ones(len(phi_flt), device=torch_dev, dtype=torch.complex64)
    
    # Setup least squares to get kernel weights
    def _solve_kern_weights(apod: torch.Tensor) -> torch.Tensor:
        # Build matrices
        A = torch.exp(-2j * torch.pi * alphas_grd[None, :] * phi_flt[:, None]) * apod[:, None] # R W
        B = torch.exp(-2j * torch.pi * delta_alphas[None, :] * phi_flt[:, None]) # R Nkerns
        
        # Solve least squares
        weights = torch.linalg.solve(A.H @ A, A.H @ B).T # Nkerns W
        return weights
    
    # Setup least squares to get apodization function
    def _solve_apod(weights: torch.Tensor) -> torch.Tensor:
        # Build matrices
        A = torch.exp(-2j * torch.pi * alphas_grd[None, :] * phi_flt[:, None]) # R W
        Aw = einsum(A, weights, 'R W, N W -> R N')
        B = torch.exp(-2j * torch.pi * delta_alphas[None, :] * phi_flt[:, None]) # R Nkerns
        
        # Solve least squares
        AHA = (Aw.conj() * Aw).sum(dim=-1)
        AHB = (Aw.conj() * B).sum(dim=-1)
        apod = AHB / AHA
        
        return apod
    
    if solve_apod:
        for i in tqdm(range(1000), 'Mini ALS for apodization'):
            # ALS
            weights = _solve_kern_weights(apod)
            apod = _solve_apod(weights)
            
            # Check convergence
            if i > 0:
                potl_apod = (apod - apod_prev).norm() / apod_prev.norm()
                potl_weights = (weights - weights_prev).norm() / weights_prev.norm()
                if potl_apod < ptol and potl_weights < ptol:
                    break
                
            # Update previous values
            apod_prev = apod.clone()
            weights_prev = weights.clone()
    else:
        # Solve for kernel weights
        weights = _solve_kern_weights(apod)
    
    return weights, delta_alphas, apod

def alpha_interp_kerns(phis: torch.Tensor,
                       W: int = 2,
                       dalphas: Union[float, tuple] = 0.5,
                       Nkerns: int = 100,
                       solve_apod: bool = False) -> torch.Tensor:
    """
    Solves for 1d interpolation weights for interpolating in an alpha direction.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase basis functions with shape (B, *im_size)
    W : int
        Width of the kernel
    dalphas : Optional[tuple]
        Step sizes for the alpha deviations for each alpha direction, len(dalphas) = B
    Nkerns : int
        Number of kernel weights to solve for
    solve_apod : bool
        Whether to solve for the apodization function
        
    Returns
    -------
    weights : torch.Tensor
        Kernel weights with shape (B, Nkerns, W)
    delta_alphas : torch.Tensor
        Delta alphas with shape (B, Nkerns)
    """
    # Consts
    B = phis.shape[0]
    phis_flt = phis.reshape((B, -1))
    if isinstance(dalphas, float):
        dalphas = (dalphas,) * B
    elif isinstance(dalphas, tuple):
        assert len(dalphas) == B, "dalphas must be a tuple of length B"
    
    # Solve for kernel weights
    weights = []
    delta_alphas = []
    apods = []
    for b in range(B):
        weight, delta_alpha, apod = _solve_1d_kern(phis_flt[b], W, dalphas[b], Nkerns, solve_apod=solve_apod)
        weights.append(weight)
        delta_alphas.append(delta_alpha)
        apods.append(apod)
        
    # Stack
    weights = torch.stack(weights, dim=0)
    delta_alphas = torch.stack(delta_alphas, dim=0)
    apods = torch.stack(apods, dim=0)
    apods = apods.reshape((B, *phis.shape[1:]))
    
    return weights, delta_alphas, apods

def reduce_temporal(alphas: torch.Tensor,
                    W: int = 2,
                    dalphas: Optional[tuple] = None) -> torch.Tensor:
    """
    Reduce the temporal dimensions of the data using 
    kernel interpolation
    
    Args
    ----
    alphas : torch.Tensor
        Temporal alpha coefficients with shape (B, *trj_size)
    W : int
        Width of the kernel
    dalphas : Optional[tuple]
        Step sizes for the alpha deviations for each alpha direction, len(dalphas) = B
        
    Returns
    -------
    alphas_unq : torch.Tensor
        Unique alpha coefficients with shape (N, B)
    alpha_kern : torch.Tensor
        Alpha kernel with shape (W^B, B)
    alpha_to_unq_idx : Callable
        Function to go from alpha to index of alphas_unq
    """
    # Consts
    B = alphas.shape[0]
    if isinstance(dalphas, float):
        dalphas = (dalphas,) * B
    elif isinstance(dalphas, tuple):
        assert len(dalphas) == B, "dalphas must be a tuple of length B"
        
    # Build kernel
    alpha_kern = torch.arange(-floor(W/2), ceil(W/2), device=alphas.device, dtype=torch.float32)
    # alpha_kern -= alpha_kern.mean()
    alpha_kern = [alpha_kern * dalphas[b] for b in range(B)]
    alpha_kern = torch.stack(torch.meshgrid(*alpha_kern, indexing='ij'), dim=-1).reshape((-1, B))
    
    # Setup mappings from alpha to grid ints to ints
    alphas_flt = alphas.reshape((B, -1)).T # T B
    dalphas_tensor = torch.tensor(dalphas, device=alphas.device)
    base = int(((alphas_flt / dalphas_tensor).max() - (alphas_flt / dalphas_tensor).min() + W).ceil().item())
    max_int_long = 2 ** 63 - 1
    assert base ** B <= max_int_long
    gint_ofs = base // 2
    mults = base ** torch.arange(B, device=alphas.device, dtype=torch.float32)
    def alpha_to_gint(alpha: torch.Tensor) -> torch.Tensor:
        return (alpha / dalphas_tensor).round() + gint_ofs
    def gint_to_int(gint: torch.Tensor) -> torch.Tensor:
        return gint @ mults
    def int_to_gint(int: torch.Tensor) -> torch.Tensor:
        return (int.unsqueeze(-1) / mults).floor() % base
    def gint_to_alpha(gint: torch.Tensor) -> torch.Tensor:
        return (gint - gint_ofs) * dalphas_tensor
    
    # Get unique grid points only
    alphas_grd_all = alpha_kern[:, None, :] + alphas_flt[None, :, :] # W^B T B # TODO FIXME MEMORY INTENSIVE
    gints_all = alpha_to_gint(alphas_grd_all.reshape((-1, B)))
    gints_unq = torch.unique(gints_all, dim=0)
    int_unq = gint_to_int(gints_unq)
    ret = torch.sort(int_unq)
    int_unq_sorted = ret.values
    int_unq_inds = ret.indices
    alphas_unq = gint_to_alpha(gints_unq)[int_unq_inds]

    # Functionality to go from alpha to index of alphas_unq
    def alpha_to_unq_idx(alpha: torch.Tensor) -> torch.Tensor:
        gint = alpha_to_gint(alpha)
        ints = gint_to_int(gint)
        ints = ints.long()
        return torch.searchsorted(int_unq_sorted, ints)

    return alphas_unq, alpha_kern, alpha_to_unq_idx

def expand_temporal(lowres_data: torch.Tensor,
                    alphas: torch.Tensor,
                    dalphas: tuple,
                    weights: torch.Tensor,
                    delta_alphas: torch.Tensor,
                    alpha_kern: torch.Tensor,
                    alpha_to_unq_idx: Callable,
                    temporal_batch_size: Optional[int] = None) -> torch.Tensor:
    """
    Expand the temporal dimensions of the data using 
    kernel interpolation.
    
    Args
    ----
    lowres_data : torch.Tensor
        Low resolution temporal data to expand with shape (..., G)
    alphas : torch.Tensor
        Temporal alpha coefficients with shape (B, *trj_size)
    weights : torch.Tensor
        Kernel weights with shape (B, Nkerns, W)
    delta_alphas : torch.Tensor
        Delta alphas with shape (B, Nkerns)
    alpha_kern : torch.Tensor
        Alpha kernel with shape (W^B, B)
    alpha_to_unq_idx : Callable
        Function to go from alpha to index of alphas_unq
    temporal_batch_size : Optional[int]
        Batch size for the temporal dimension
        
    Returns
    -------
    data_out : torch.Tensor
        Expanded temporal data with shape (..., *trj_size)
    """
    # Consts
    torch_dev = alphas.device
    W = weights.shape[-1]
    B = alphas.shape[0]
    G = lowres_data.shape[-1]
    arb_size = lowres_data.shape[:-1]
    trj_size = alphas.shape[1:]
    alphas_flt = alphas.reshape((B, -1)).T # T B
    lowres_data_flt = lowres_data.reshape((-1, G))
    dalphas_tensor = torch.tensor(dalphas, device=torch_dev)
    if temporal_batch_size is None:
        temporal_batch_size = len(alphas_flt)
    
    # Kernel weights
    alpha_devs = alphas_flt - (alphas_flt / dalphas_tensor).round() * dalphas_tensor # T B
    delta_delta_alphas = delta_alphas[:, 1] - delta_alphas[:, 0]
    weight_idxs = (((alpha_devs - delta_alphas[:, 0])) / delta_delta_alphas).round().long() # T B
    
    # Interpolate
    data_out = torch.zeros((len(lowres_data_flt),) + (len(alphas_flt),) , device=torch_dev, dtype=lowres_data.dtype)
    for t1 in tqdm(range(0, len(alphas_flt), temporal_batch_size), 'Alpha interpolation'):
        
        # Batch of temporal stuff
        t2 = min(t1 + temporal_batch_size, len(alphas_flt))
        weights_exp = torch.ones((t2-t1,) + (W,)*B, device=torch_dev, dtype=lowres_data.dtype)
        idxs_kerns = alpha_to_unq_idx(alphas_flt[t1:t2, None, :] + alpha_kern) # T W^B
        data_kerns = lowres_data_flt[:, idxs_kerns] # N T W^B
        
        # Build weights and sum
        for b in range(B):
            weights_temp = weights[b, weight_idxs[t1:t2, b], :] # T W
            tup = (None,)*b + (slice(None),) + (None,)*(B-b-1)
            weights_exp *= weights_temp[(slice(None),) + tup] # T W ... W
        weights_exp = weights_exp.reshape((t2-t1, -1))
        data_out[:, t1:t2] = (data_kerns * weights_exp).sum(dim=-1) # N T 
    
    # Reshape output
    data_out = data_out.reshape(arb_size + trj_size)
    return data_out
                
