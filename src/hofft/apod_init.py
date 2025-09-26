import torch
import numpy as np

from einops import einsum

from mr_recon.utils import gen_grd, pick_K_vectors
from mr_recon.algs import eigen_decomp_operator
from mr_recon.imperfections.field import alpha_segementation
from mr_recon.linops import type3_nufft_naive, type3_nufft
from mr_recon.dtypes import complex_dtype

from hofft.als import als_iterations
from hofft.model import hofft_params

def K_alphas_apod_init(phis: torch.Tensor,
                       alphas: torch.Tensor,
                       hparams: hofft_params,
                       method: str = 'minmax',
                       apod_init_method: str = 'eigen',
                       num_als_iter: int = 100,
                       K: int = 500,) -> torch.Tensor:
    """
    Initialize apodizations by running ALS on K representative alphas.
    
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
        method to initialize apodizations before ALS. Options are 'seg' and 'eigen
    num_als_iter : int
        number of ALS iterations to run.
    K : int
        number of representative alphas to pick.
        
    Returns
    -------
    apods : torch.Tensor
        initialized apodizations, shape (L, *im_size).
    """
    # Consts
    im_size = phis.shape[1:]
    torch_dev = phis.device
    B = phis.shape[0]
    d = len(im_size)
    use_type3 = hparams.use_type3
    kern_size = hparams.kern_size
    os = hparams.os
    
    # Prep ALS algorithm
    rs = gen_grd(im_size).to(torch_dev)
    kern = gen_grd(kern_size, kern_size).to(torch_dev)
    kern = kern.reshape((-1, d)) / os
    kern_bases = torch.exp(-2j * np.pi * einsum(kern, rs,
                                                'K d, ... d -> K ...'))
    
    # Use other apod_init functions to get initial apods
    if apod_init_method == 'seg':
        apods_init_init = alpha_seg_apod_init(phis, alphas, hparams)
    elif apod_init_method == 'eigen':
        apods_init_init = eigen_apod_init(phis, alphas, hparams)
    else:
        raise ValueError(f'Invalid apod_init_method {apod_init_method}. Supported methods are seg and eigen.')
    
    # Pick K representative alphas
    k_alphas, _ = pick_K_vectors(vectors=alphas.reshape((B,-1)).T, K=K, 
                                 sigma=0, method=method)
    k_alphas = k_alphas.T # shape (B, K)
    
    # Make type3 object using k_alphas
    if use_type3:
        t3n = type3_nufft(phis, k_alphas, use_toep=True)
    else:
        t3n = type3_nufft_naive(phis, k_alphas)
    
    # ALS agorithm
    _, apods = als_iterations(t3n, kern_bases, apods_init_init, 
                              max_iter=num_als_iter,
                              verbose=True)
    
    return apods
    
def alpha_seg_apod_init(phis: torch.Tensor,
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
    apods : torch.Tensor
        initialized apodizations, shape (L, *im_size).
    """
    L = hparams.L
    apods, _ = alpha_segementation(phis, alphas, L=L, L_batch_size=L, interp_type='zero', use_type3=False)
    return apods

def eigen_apod_init(phis: torch.Tensor,
                     alphas: torch.Tensor,
                     hparams: hofft_params) -> torch.Tensor:
    """
    Initialize apodizations using eigen-decomposition of the system matrix.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis functions, shape (B, *im_size).
    alphas : torch.Tensor
        temporal phase basis functions, shape (B, *trj_size).
        
    Returns
    -------
    apods : torch.Tensor
        initialized apodizations, shape (L, *im_size).
    """
    # Consts
    im_size = phis.shape[1:]
    torch_dev = phis.device
    L = hparams.L
    use_type3 = hparams.use_type3
    
    # Make type3 object
    if use_type3:
        t3n = type3_nufft(phis, alphas, use_toep=True)
    else:
        t3n = type3_nufft_naive(phis, alphas)
    
    # Eigen-decomp
    x0 = torch.randn(im_size, dtype=complex_dtype, device=torch_dev)
    apods, _ = eigen_decomp_operator(t3n.normal, x0, num_eigen=L, 
                                     num_iter=15,
                                     lobpcg=True,
                                     largest=True)
    
    return apods

