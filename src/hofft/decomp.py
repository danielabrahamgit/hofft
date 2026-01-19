import re
import torch
import numpy as np

from typing import Optional, Union
from einops import einsum

from mr_recon.utils import gen_grd, resize
from mr_recon.spatial import spatial_interp
from mr_recon.algs import eigen_decomp_operator, lin_solve
from mr_recon.imperfections.field import alpha_segementation
from mr_recon.fourier import fft, ifft, sigpy_nufft
from mr_recon.linops import linop
from mr_recon.dtypes import complex_dtype

from .matvec import matvec, matvec_naive

from tqdm import tqdm
from dataclasses import dataclass, field
from einops import rearrange, einsum


__all__ = [
    'build_kern_bases',
    'funcs_to_phase',
    'kb_nufft',
    'als_iterations',
    'lstsq_spatial',
    'lstsq_temporal',
    'hofft_params',
]

@dataclass
class hofft_params: 
    kern_size: tuple
    os: float
    L: int
    matvec_type: matvec = matvec_naive
    matvec_kwargs: dict = field(default_factory=dict)
    spatial_init: Union[torch.Tensor, str] = 'seg'
    verbose: bool = True
    """
    Parameters for HOFFT models.
    
    Attributes
    ----------
    kern_size : tuple
        Size of the kernel, must have the same number of dimensions as the image.
    os : Optional[float]
        Oversampling factor.
    L : Optional[int]
        Number of apodization functions.
    matvec : matvec
        Matrix-vector operation for the HOFFT model, must be a subclass of matvec. Options are:
        'matvec_naive' - naive implementation of the matrix-vector product (default)
        'matvec_type3' - type-3 nufft implementation of the matrix-vector product
        'matvec_svd' - SVD implementation of the matrix-vector product
        'matvec_cur' - CUR implementation of the matrix-vector product
        'matvec_histogram' - Histogram-based matrix-vector product
    matvec_kwargs : dict
        Keyword arguments for initalizing the matrix vector product (batch sizes, interpolation order, etc.)
    spatial_init : Union[torch.Tensor, str]
        If string:
        'seg' - uses segmentation method to initialize spatial factors
        'eigen' - uses eigen-decomposition method to initialize spatial factors
        'k_alphas' - uses K representative alphas to initialize spatial factors
        If torch.Tensor:
        Initial spatial factors with shape (L, *solve_size)
    verbose : Optional[bool]
        If True, prints progress
    """

def build_kern_bases(kern_size: tuple,
                      im_size: tuple,
                      os: Optional[float] = 1.0) -> torch.Tensor:
    """
    Builds the kernel bases for the HOFFT model.    
    
    Args
    ----
    kern_size : tuple
        Size of the kernel.
    im_size : tuple
        Size of the image.
    os : Optional[float]
        Oversampling factor.
        
    Returns
    -------
    kern_bases : torch.Tensor
        Kernel bases with shape (prod(kern_size), *im_size).
    """
    d = len(im_size)
    rs = gen_grd(im_size)
    kern = gen_grd(kern_size, kern_size).reshape((-1, d)) / os
    phz = einsum(kern, rs, 'K D, ... D -> K ...')
    kern_bases = torch.exp(-2j * np.pi * phz)
    return kern_bases

def funcs_to_phase(kern_weights: torch.Tensor,
                   spatial_factors: torch.Tensor,
                   os: Optional[float] = 1.0) -> torch.Tensor:
    """
    Convert kernel weights and spatial factors to phase maps
    
    Args
    ----
    kern_weights : torch.Tensor
        Kernel weights with shape (L, *kern_size, *trj_size)
    spatial_factors : torch.Tensor
        Spatial factors with shape (L, *im_size)
    os : Optional[float]
        Oversampling factor.
        
    Returns
    -------
    phz : torch.Tensor
        Phase maps with shape (*trj_size, *im_size)
    """
    # Consts
    kern_size = kern_weights.shape[1:-1]
    im_size = spatial_factors.shape[1:]
    torch_dev = kern_weights.device
    d = len(im_size)
    trj_size = kern_weights.shape[(d+1):]
    L = kern_weights.shape[0]
    K = np.prod(kern_size)
    T = np.prod(trj_size)
    
    # Flatten
    kern_weights_flt = kern_weights.reshape((L, K, T))
    
    # Make kernel bases
    rs = gen_grd(im_size).to(torch_dev)
    kern = gen_grd(kern_size, kern_size).to(torch_dev).reshape((-1, d)) / os
    phz = einsum(kern, rs, 'K D, ... D -> K ...')
    kern_bases = torch.exp(-2j * np.pi * phz)
    
    # Apply
    phz = einsum(kern_weights_flt, kern_bases, 'L K T, K ... -> L T ...')
    phz = einsum(phz, spatial_factors, 'L T ..., L ... -> T ...')
    
    return phz.reshape((*trj_size, *im_size))

def als_iterations(phase_model: matvec, 
                   kern_bases: torch.Tensor,
                   spatial_factors_init: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   max_iter: Optional[int] = 100,
                   verbose: Optional[bool] = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform ALS iterations to solve for spatial factors and kernel weights.
    
    Args
    ----
    phase_model : mr_recon.linop
        high order phase matrix-vector operation
    kern_bases : torch.Tensor
        kernel bases with shape (K, *im_size)
    spatial_factors_init : torch.Tensor
        initial spatial factors with shape (L, *im_size)
    mask : torch.Tensor, optional
        image weighting mask with shape im_size
    max_iter : int, optional
        maximum number of iterations
    verbose : bool, optional
        whether to print progress
    
    Returns
    -------
    kernel_weights : torch.Tensor
        kernel weights with shape (L, K, *trj_size)
    spatial_factors : torch.Tensor
        spatial factors with shape (L, *im_size)
    """
        
    # Default mask
    if mask is None:
        mask = torch.ones_like(spatial_factors_init[0])
        
    # Weights only 
    if max_iter == 0:
        kernel_weights = lstsq_temporal(phase_model, kern_bases, spatial_factors_init, mask=mask)
        return kernel_weights, spatial_factors_init
    
    # Stopping criteria
    kwargs_allclose = {'atol': 0.0, 'rtol': 1e-2}
    
    # Momentum term
    # momentum = lambda k : k / (k + 3)
    momentum = lambda k : .8
    k0 = 0
    
    # ALS till max_iter
    spatial_factors_prev = spatial_factors_init
    kernel_weights_prev = None
    for k in tqdm(range(max_iter), 'ALS iterations', disable=not verbose):
        
        # ALS weight updates
        kernel_weights = lstsq_temporal(phase_model, kern_bases, spatial_factors_prev, mask=mask)
        if k > k0:
            beta = momentum(k)
            kernel_weights = kernel_weights + beta * (kernel_weights_prev - kernel_weights)
        
        # ALS apodization updates
        spatial_factors = lstsq_spatial(phase_model, kern_bases, kernel_weights, mask=mask)
        if k > k0:
            spatial_factors = spatial_factors + beta * (spatial_factors_prev - spatial_factors)
        
        # TODO REMOVEME
        # Check convergence
        if k > 0 and \
            torch.allclose(kernel_weights, kernel_weights_prev, **kwargs_allclose) and \
            torch.allclose(spatial_factors, spatial_factors_prev, **kwargs_allclose):
            break
        
        # Update previous values
        kernel_weights_prev = kernel_weights
        spatial_factors_prev = spatial_factors
        
    return kernel_weights, spatial_factors
    
def lstsq_spatial(phase_model: matvec, 
                  kern_bases: torch.Tensor, 
                  kernel_weights: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  k_batch_size: Optional[int] = None,
                  t_batch_size: Optional[int] = None,
                  solver: Optional[str] = 'pinv',
                  lamda: Optional[float] = 0.0,) -> torch.Tensor:
    """    
    This function optimizes for the spatial factors 
    given fixed kernel weights via least squares.
    
    Args
    ----
    phase_model : mr_recon.linops.linop
        linear operator for the phase operator
    kern_bases : torch.Tensor
        kernel bases with shape (K, *im_size)
    kernel_weights : torch.Tensor
        kernel weights with shape (L, K, *trj_size)
    mask : torch.Tensor, optional
        image weighting mask with shape im_size
    k_batch_size : int, optional
        batch size for kernel bases
    t_batch_size : int, optional
        batch size for temporal weights
    solver : str, optional
        solver for least squares
    lamda : float, optional
        regularization parameter for least squares
    
    Returns
    -------
    spatial_factors : torch.Tensor
        solution with shape (L, *im_size)
    """
    # Consts
    torch_dev = kernel_weights.device
    im_size = kern_bases.shape[1:]
    trj_size = kernel_weights.shape[2:]
    kernel_weights_flt = rearrange(kernel_weights, 'L K ... -> L K (...)')
    L = kernel_weights.shape[0]
    K = kern_bases.shape[0]
    T = np.prod(trj_size)
    assert phase_model.ishape == im_size
    assert phase_model.oshape == trj_size
    
    # Default
    if k_batch_size is None:
        k_batch_size = K
    if t_batch_size is None:
        t_batch_size = T
    if mask is None:
        mask = torch.ones(im_size, dtype=complex_dtype, device=torch_dev)
    kern_bases *= mask
    
    AHA = torch.zeros((*im_size, L, L), dtype=complex_dtype, device=torch_dev)
    AHB = torch.zeros((*im_size, L), dtype=complex_dtype, device=torch_dev)
    
    # Build cross terms
    cross_terms = torch.zeros((L, K, L, K), dtype=complex_dtype, device=torch_dev)
    for t1 in range(0, T, t_batch_size):
        t2 = min(t1 + t_batch_size, T)
        kernel_weights_batch = kernel_weights_flt[:, :, t1:t2] # L K T
        cross_terms += einsum(kernel_weights_batch.conj(), kernel_weights_batch, 'L1 K1 T, L2 K2 T -> L1 K1 L2 K2')
    
    # Build AHB and AHA matrices
    # TODO add masking behaviour here
    for k1 in range(0, K, k_batch_size):
        k2 = min(k1 + k_batch_size, K)
        
        # AHB
        kernel_weights_batch = kernel_weights[:, k1:k2, ...]
        kernel_weights_batch = rearrange(kernel_weights_batch, 'L K ... -> (L K) ...')
        imgs_batch = phase_model.adjoint(kernel_weights_batch).reshape((L, (k2-k1), *im_size)).conj() * mask
        AHB += einsum(imgs_batch, kern_bases[k1:k2].conj(), 'L K ..., K ... -> ... L')
        
        # AHA   
        kern_cross = kern_bases[k1:k2, None].conj() * kern_bases[None, :]
        AHA += einsum(kern_cross, cross_terms[:, k1:k2], 'K1 K2 ..., L1 K1 L2 K2 -> ... L1 L2')
        
    # Solve least squares
    spatial_factors = lin_solve(AHA, AHB[..., None], solver=solver, lamda=lamda)[..., 0] # *im_size L
    spatial_factors = rearrange(spatial_factors, '... L -> L ...') * mask
    
    return spatial_factors   
   
def lstsq_temporal(phase_model: matvec, 
                   kern_bases: torch.Tensor, 
                   spatial_factors: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None,
                   lk_batch_size: Optional[int] = None,
                   solver: Optional[str] = 'pinv',
                   lamda: Optional[float] = 0.0,) -> torch.Tensor:
    """
    This function optimizes for the kernel weights given fixed spatial factors.
    
    Args
    ----
    phase_model : mr_recon.linops.linop
        linear operator for the phase operator
    kern_bases : torch.Tensor
        kernel bases with shape (K, *im_size)
    spatial_factors : torch.Tensor
        spatial factors with shape (L, *im_size)
    mask : torch.Tensor, optional
        image weighting mask with shape im_size
    k_batch_size : int, optional
        batch size over combined L and K dimension
    solver : str, optional
        solver for least squares
    lamda : float, optional
        regularization parameter for least squares    
    
    Returns
    -------
    kernel_weights : torch.Tensor
        weights with shape (L, K, *trj_size)
    """
    # Consts
    L = spatial_factors.shape[0]
    K = kern_bases.shape[0]
    torch_dev = spatial_factors.device
    im_size = kern_bases.shape[1:]
    trj_size = phase_model.oshape
    assert phase_model.ishape == im_size
    
    # Default
    if lk_batch_size is None:
        lk_batch_size = L * K
    if mask is None:
        mask = torch.ones(im_size, dtype=complex_dtype, device=torch_dev)
    
    bases = einsum(kern_bases, spatial_factors, 'K ..., L ... -> L K ...') * mask
    AHA = torch.zeros((L, K, L, K), dtype=kern_bases.dtype, device=kern_bases.device)
    AHB = torch.zeros((L, K, *trj_size), dtype=kern_bases.dtype, device=kern_bases.device)
    
    # Inds for both l and k
    l_inds = torch.arange(L, device=torch_dev)
    k_inds = torch.arange(K, device=torch_dev)
    linds, kinds = torch.meshgrid(l_inds, k_inds, indexing='ij')
    linds = linds.flatten()
    kinds = kinds.flatten()
    
    # Build AHA and AHB
    for lk1 in range(0, L*K, lk_batch_size):
        lk2 = min(lk1 + lk_batch_size, L*K)
        ls = linds[lk1:lk2]
        ks = kinds[lk1:lk2]

        AHA[ls, ks] += einsum(bases[ls, ks].conj(), bases, 'lk ..., L K ... -> lk L K')
        temp_batch = phase_model.forward(bases[ls, ks].conj()) # (L K) *trj_size
        AHB[ls, ks, ...] += temp_batch
    
    # Solve least squares
    AHA_flt = rearrange(AHA, 'L1 K1 L2 K2 -> (L1 K1) (L2 K2)')
    AHB_flt = rearrange(AHB, 'L K ... -> (L K) (...)')
    kernel_weights_flt = lin_solve(AHA_flt, AHB_flt, solver=solver, lamda=lamda) # (L K) (...)
    kernel_weights = kernel_weights_flt.reshape(AHB.shape)
        
    return kernel_weights