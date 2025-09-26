"""
This file performs the following decomposition:
    e^{-j2pi phase(r, t)} = sum_l sum_k b_l(r) e_k(r) h_{l,k}(t)
    
Where:
- b_l are the apodization functions,
- e_k(r) are the fixed (likely fourier) kernel bases,
- h_{l,k}(t) are the kernel weights.

The main idea is to alternating least squares (als). We alternate 
between solving for the spatial apodization functions b_l(r), 
and then for temporal kernel weights h_{l,k}(t).

The functions below assume some operator access to a phase_model object. 
This object performs the following two important operations:
- phase_model.forward(x) = sum_r x(r) e^{-j 2pi phi(r, t)} 
- phase_model.adjoint(y) = sum_t y(t) e^{j 2pi phi(r, t)}
"""

import torch
import numpy as np

from typing import Optional
from einops import rearrange, einsum
from tqdm import tqdm

from mr_recon.algs import eigen_decomp_operator, lin_solve
from mr_recon.utils import quantize_data
from mr_recon.linops import linop
from mr_recon.dtypes import complex_dtype

__all__ = [
    'als_iterations', 
    'lstsq_spatial', 
    'lstsq_temporal'
]

# TODO:
# - masking doesn't work well

def init_apods(phase_model: linop,
               L: int,
               method: Optional[str] = 'rnd',
               torch_dev: Optional[torch.device] = torch.device('cpu')) -> torch.Tensor:
    """
    Initialize apodization functions.
    
    Args:
    -----
    phase_model : mr_recon.linop
        linear operator for the phase operator
    L : int
        number of apodization functions
    method : str, optional
        method for initialization. Options are:
        - 'rnd': random initialization
        - 'svd': singular value decomposition
        - 'seg': clusters phase coefficients
        - 'ones': all ones
    torch_dev : torch.device, optional
        torch device to run on
    
    Returns:
    --------
    apods : torch.Tensor
        apodization functions with shape (L, *im_size)
    """
    
    if method == 'rnd':
        apods = torch.randn(L, *phase_model.ishape, dtype=complex_dtype, device=torch_dev)
    elif method == 'svd':
        x0 = torch.randn(phase_model.ishape, dtype=complex_dtype, device=torch_dev)
        L_batch_size = L
        def normal(x):
            # x has shape (L, *im_size)
            ys = []
            for l1 in range(0, x.shape[0], L_batch_size):
                l2 = min(l1 + L_batch_size, x.shape[0])
                ys.append(phase_model.normal(x[l1:l2]))
            return torch.cat(ys, dim=0)
        apods, _ = eigen_decomp_operator(normal, x0, num_eigen=L)
        apods = apods.conj()
    elif method == 'ones':
        apods = torch.ones(L, *phase_model.ishape, dtype=complex_dtype, device=torch_dev)
    elif method == 'seg':
        alphas = phase_model.alphas
        phis = phase_model.phis
        B = alphas.shape[0]
        alphas = alphas.reshape((B, -1))
        print(f'Warning: Segmentation with fast type3nufft might not work too well ...')
        if B == 1:
            clusts, _ = quantize_data(alphas.T, L, method='uniform')
        else:
            clusts, _ = quantize_data(alphas.T, L, method='cluster')
        apods = torch.exp(-2j * torch.pi * einsum(clusts, phis, 'L B, B ... -> L ...'))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return apods

def als_iterations(phase_model: linop, 
                   kern_bases: torch.Tensor,
                   apods_init: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   max_iter: Optional[int] = 100,
                   verbose: Optional[bool] = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform ALS iterations to solve for apodization functions and weights.
    
    Args
    ----
    phase_model : mr_recon.linop
        linear operator for the phase operator
    kern_bases : torch.Tensor
        kernel bases with shape (K, *im_size)
    apods_init : torch.Tensor
        initial apodization functions with shape (L, *im_size)
    mask : torch.Tensor, optional
        image weighting mask with shape im_size
    max_iter : int, optional
        maximum number of iterations
    verbose : bool, optional
        whether to print progress
    
    Returns
    -------
    weights : torch.Tensor
        weights with shape (L, K, *trj_size)
    apods : torch.Tensor
        apodization functions with shape (L, *im_size)
    """
    # Default mask
    if mask is None:
        mask = torch.ones_like(apods_init[0])
        
    # Weights only 
    if max_iter == 0:
        weights = lstsq_temporal(phase_model, kern_bases, apods_init, mask=mask)
        return weights, apods_init
    
    # Stopping criteria
    kwargs_allclose = {'atol': 0.0, 'rtol': 1e-2}
    
    # Momentum term
    # momentum = lambda k : k / (k + 3)
    momentum = lambda k : .8
    k0 = 0
    
    # ALS till max_iter
    apods_prev = apods_init
    weights_prev = None
    for k in tqdm(range(max_iter), 'ALS iterations', disable=not verbose):
        
        # ALS weight updates
        weights = lstsq_temporal(phase_model, kern_bases, apods_prev, mask=mask)
        if k > k0:
            beta = momentum(k)
            weights = weights + beta * (weights_prev - weights)
        
        # ALS apodization updates
        apods = lstsq_spatial(phase_model, kern_bases, weights, mask=mask)
        if k > k0:
            apods = apods + beta * (apods_prev - apods)
        
        # Check convergence
        if k > 0 and \
            torch.allclose(weights, weights_prev, **kwargs_allclose) and \
            torch.allclose(apods, apods_prev, **kwargs_allclose):
            break
        
        # Update previous values
        weights_prev = weights
        apods_prev = apods
        
        
    return weights, apods
    
def lstsq_spatial(phase_model: linop, 
                  kern_bases: torch.Tensor, 
                  weights: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  k_batch_size: Optional[int] = None,
                  t_batch_size: Optional[int] = None,
                  solver: Optional[str] = 'pinv',
                  lamda: Optional[float] = 0.0,) -> torch.Tensor:
    """    
    This function optimizes for the apodization functions 
    given fixed weights via least squares.
    
    Args
    ----
    phase_model : mr_recon.linops.linop
        linear operator for the phase operator
    kern_bases : torch.Tensor
        kernel bases with shape (K, *im_size)
    weights : torch.Tensor
        weights with shape (L, K, *trj_size)
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
    apods : torch.Tensor
        solution with shape (L, *im_size)
    """
    # Consts
    torch_dev = weights.device
    im_size = kern_bases.shape[1:]
    trj_size = weights.shape[2:]
    weights_flt = rearrange(weights, 'L K ... -> L K (...)')
    L = weights.shape[0]
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
        weights_batch = weights_flt[:, :, t1:t2] # L K T
        cross_terms += einsum(weights_batch.conj(), weights_batch, 'L1 K1 T, L2 K2 T -> L1 K1 L2 K2')
    
    # Build AHB and AHA matrices
    for k1 in range(0, K, k_batch_size):
        k2 = min(k1 + k_batch_size, K)
        
        # AHB
        weights_batch = weights[:, k1:k2, ...]
        weights_batch = rearrange(weights_batch, 'L K ... -> (L K) ...')
        imgs_batch = phase_model.adjoint(weights_batch).reshape((L, (k2-k1), *im_size)).conj() * mask
        AHB += einsum(imgs_batch, kern_bases[k1:k2].conj(), 'L K ..., K ... -> ... L')
        
        # AHA
        kern_cross = kern_bases[k1:k2, None].conj() * kern_bases[None, :]
        AHA += einsum(kern_cross, cross_terms[:, k1:k2], 'K1 K2 ..., L1 K1 L2 K2 -> ... L1 L2')
        
    # Solve least squares
    apods = lin_solve(AHA, AHB[..., None], solver=solver, lamda=lamda)[..., 0] # *im_size L
    apods = rearrange(apods, '... L -> L ...') * mask
    
    return apods   
    
def lstsq_temporal(phase_model: linop, 
                   kern_bases: torch.Tensor, 
                   apods: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None,
                   lk_batch_size: Optional[int] = None,
                   solver: Optional[str] = 'pinv',
                   lamda: Optional[float] = 0.0,) -> torch.Tensor:
    """
    This function optimizes for the weights given fixed apodization functions.
    
    Args
    ----
    phase_model : mr_recon.linops.linop
        linear operator for the phase operator
    kern_bases : torch.Tensor
        kernel bases with shape (K, *im_size)
    apods : torch.Tensor
        apodization functions with shape (L, *im_size)
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
    weights : torch.Tensor
        weights with shape (L, K, *trj_size)
    """
    # Consts
    L = apods.shape[0]
    K = kern_bases.shape[0]
    torch_dev = apods.device
    im_size = kern_bases.shape[1:]
    trj_size = phase_model.oshape
    assert phase_model.ishape == im_size
    
    # Default
    if lk_batch_size is None:
        lk_batch_size = L * K
    if mask is None:
        mask = torch.ones(im_size, dtype=complex_dtype, device=torch_dev)
    
    bases = einsum(kern_bases, apods, 'K ..., L ... -> L K ...') * mask
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
    soln_flt = lin_solve(AHA_flt, AHB_flt, solver=solver, lamda=lamda) # (L K) (...)
    soln = soln_flt.reshape(AHB.shape)
        
    return soln