import re
import torch
import numpy as np

from einops import einsum
from typing import Union

from mr_recon.utils import gen_grd, pick_K_vectors
from mr_recon.algs import eigen_decomp_operator
from mr_recon.imperfections.field import alpha_segementation
from mr_recon.dtypes import complex_dtype

from .als import als_iterations
from .decomp import hofft_params

def choose_init(phis: torch.Tensor,
                alphas: torch.Tensor,
                hparams: hofft_params,
                spatial_init: Union[str, torch.Tensor] = '100_alphas',
                k_alphas_method: str = 'minmax') -> torch.Tensor:
    
    # Initialize spatial factors
    if isinstance(spatial_init, torch.Tensor):
        spatial_factors = spatial_init
    elif isinstance(spatial_init, str):
        if spatial_init == 'eigen':
            spatial_factors = eigen_init(phis, alphas, hparams)
        elif spatial_init == 'seg':
            spatial_factors = alpha_seg_init(phis, alphas, hparams)
        elif 'alphas' in spatial_init:
            splt = spatial_init.split('_')
            if len(splt) == 2:
                K = int(splt[0])
                num_als_iter = 100
            elif len(splt) == 3:
                K = int(splt[0])
                num_als_iter = int(splt[2])
            spatial_factors = K_alphas_init(phis, alphas, hparams,
                                           method=k_alphas_method,
                                           spatial_init_method='seg',
                                           num_als_iter=num_als_iter, K=K)
        else:
            raise ValueError(f'Invalid spatial_init {spatial_init}. Supported methods are seg and eigen.')
    else:
        raise ValueError("spatial_init must be a torch.Tensor or a string")
    
    return spatial_factors

def K_alphas_init(phis: torch.Tensor,
                  alphas: torch.Tensor,
                  hparams: hofft_params,
                  method: str = 'minmax',
                  spatial_init_method: str = 'eigen',
                  num_als_iter: int = 100,
                  K: int = 500,) -> torch.Tensor:
    """
    Initialize spatial factors by running ALS on K representative alphas.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis functions, shape (B, *im_size).
    alphas : torch.Tensor
        temporal phase basis functions, shape (B, *trj_size).
    hparams : hofft_params
        hofft parameters.
    method : str
        method to pick K representative alphas. Options are 'minmax' and 'kmeans', and 'random'
    apod_init_method : str
        method to initialize spatial factors before ALS. Options are 'seg' and 'eigen
    num_als_iter : int
        number of ALS iterations to run.
    K : int
        number of representative alphas to pick.
        
    Returns
    -------
    spatial_factors : torch.Tensor
        initialized spatial factors, shape (L, *im_size).
    """
    # Consts
    im_size = phis.shape[1:]
    torch_dev = phis.device
    B = phis.shape[0]
    d = len(im_size)
    kern_size = hparams.kern_size
    os = hparams.os
    
    # Prep ALS algorithm
    rs = gen_grd(im_size).to(torch_dev)
    kern = gen_grd(kern_size, kern_size).to(torch_dev)
    kern = kern.reshape((-1, d)) / os
    kern_bases = torch.exp(-2j * np.pi * einsum(kern, rs,
                                                'K d, ... d -> K ...'))
    
    # Use other apod_init functions to get initial apods
    if spatial_init_method == 'seg':
        spatial_factors_init = alpha_seg_init(phis, alphas, hparams)
    elif spatial_init_method == 'eigen':
        spatial_factors_init = eigen_init(phis, alphas, hparams)
    else:
        raise ValueError(f'Invalid spatial_init_method {spatial_init_method}. Supported methods are seg and eigen.')
    
    # Pick K representative alphas
    k_alphas, _ = pick_K_vectors(vectors=alphas.reshape((B,-1)).T, K=K, 
                                 sigma=0, method=method)
    k_alphas = k_alphas.T # shape (B, K)
    
    # Make matvec phase model
    phase_model = hparams.matvec_type(phis, k_alphas, **hparams.matvec_kwargs)
    
    # ALS agorithm
    _, spatial_factors = als_iterations(phase_model, kern_bases, spatial_factors_init, 
                              max_iter=num_als_iter,
                              verbose=True)
    
    return spatial_factors
    
def alpha_seg_init(phis: torch.Tensor,
                   alphas: torch.Tensor,
                   hparams: hofft_params) -> torch.Tensor:
    """
    Initialize apodizations using alpha segmentation.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis functions, shape (B, *im_size).
    alphas : torch.Tensor
        temporal phase basis functions, shape (B, *trj_size).
    hparams : hofft_params
        hofft parameters.
        
    Returns
    -------
    spatial_factors : torch.Tensor
        initialized spatial factors, shape (L, *im_size).
    """
    L = hparams.L
    spatial_factors, _ = alpha_segementation(phis, alphas, L=L, L_batch_size=L, interp_type='zero', use_type3=False)
    return spatial_factors

def eigen_init(phis: torch.Tensor,
               alphas: torch.Tensor,
               hparams: hofft_params) -> torch.Tensor:
    """
    Initialize spatial factors using eigen-decomposition of the system matrix.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis functions, shape (B, *im_size).
    alphas : torch.Tensor
        temporal phase basis functions, shape (B, *trj_size).
        
    Returns
    -------
    spatial_factors : torch.Tensor
        initialized spatial factors, shape (L, *im_size).
    """
    # Consts
    im_size = phis.shape[1:]
    torch_dev = phis.device
    L = hparams.L
    
    # Make matvec phase model
    phase_model = hparams.matvec_type(phis, alphas, **hparams.matvec_kwargs)
    
    # Eigen-decomp
    x0 = torch.randn(im_size, dtype=complex_dtype, device=torch_dev)
    spatial_factors, _ = eigen_decomp_operator(phase_model.normal, x0, num_eigen=L, 
                                               num_iter=15,
                                               lobpcg=True,
                                               largest=True)
    
    return spatial_factors

