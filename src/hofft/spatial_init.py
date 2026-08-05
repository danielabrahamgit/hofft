import torch
import numpy as np

from einops import einsum
from typing import Union, Optional

from mr_recon.utils import gen_grd, pick_K_vectors
from mr_recon.algs import eigen_decomp_operator

from .decomp import hofft_params, als_iterations

def k_alpha_selection(alphas: torch.Tensor,
                      k: int,
                      method: str,
                      train_frac: float = 0.1,
                      verbose: bool = False) -> torch.Tensor:
    """
    Select K representative alphas.
    
    Args
    ----
    alphas : torch.Tensor
        temporal phase basis functions, shape (B, ...).
    k : int
        number of representative alphas to pick.
    method : str
        method to pick K representative alphas. Options are 'maxmin' and 'kmeans', and 'random'
    train_frac : float
        fraction of alphas to use for training.
    verbose : bool
        if True, print output.
        
    Returns
    -------
    betas : torch.Tensor
        representative alphas, shape (L, B).
    """
    B = alphas.shape[0]
    alphas_flt = alphas.reshape((B,-1))
    M = alphas_flt.shape[1]
    ntrain = int(M * train_frac)
    if verbose:
        print(f'Using {ntrain} out of {M} alphas for selecting {k} representative alphas')
    train_idxs = torch.randperm(alphas_flt.shape[1])[:ntrain]
    alphas_train = alphas_flt[:, train_idxs]
    betas, _ = pick_K_vectors(vectors=alphas_train.T, K=k, 
                              sigma=0, method=method)
    return betas

def choose_init(phis: torch.Tensor,
                alphas: torch.Tensor,
                hparams: hofft_params,
                spatial_mask: Optional[torch.Tensor] = None,
                spatial_init: Union[str, torch.Tensor] = '100_alphas_100') -> torch.Tensor:
    """
    Choose the spatial initialization method.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis functions, shape (B, *im_size).
    alphas : torch.Tensor
        temporal phase basis functions, shape (B, *trj_size).
    hparams : hofft_params
        hofft parameters.
    spatial_mask : Optional[torch.Tensor]
        spatial mask with shape (*im_size)
    spatial_init : Union[str, torch.Tensor]
        If string:
        'eigen' - uses eigen-decomposition method to initialize spatial factors
        'seg' - uses segmentation method to initialize spatial factors
        'ones' - uses ones to initialize spatial factors
        'k_alphas' - uses K representative alphas to initialize spatial factors
        If torch.Tensor:
        Initial spatial factors with shape (L, *solve_size)
        
    Returns
    -------
    spatial_factors : torch.Tensor
        initialized spatial factors, shape (L, *im_size).
    """
    # Initialize spatial factors
    if isinstance(spatial_init, torch.Tensor):
        spatial_factors = spatial_init
    elif isinstance(spatial_init, str):
        if spatial_init == 'eigen':
            spatial_factors = eigen_init(phis, alphas, hparams)
        elif spatial_init == 'seg':
            spatial_factors = alpha_seg_init(phis, alphas, hparams)
        elif spatial_init == 'ones':
            spatial_factors = torch.ones(hparams.L, *phis.shape[1:], dtype=torch.complex64, device=phis.device)
        elif 'alphas' in spatial_init:
            splt = spatial_init.split('_')
            if len(splt) == 2:
                K = int(splt[0])
                num_als_iter = 100
            elif len(splt) == 3:
                K = int(splt[0])
                num_als_iter = int(splt[2])
            spatial_factors = K_alphas_init(phis, alphas, hparams,
                                           spatial_init_method='seg',
                                           spatial_mask=spatial_mask,
                                           num_als_iter=num_als_iter, K=K)
        else:
            raise ValueError(f'Invalid spatial_init {spatial_init}. Supported methods are seg and eigen.')
    else:
        raise ValueError("spatial_init must be a torch.Tensor or a string")
    
    # # Kb apod
    # im_size = phis.shape[1:]
    # width = hparams.kern_size[0]
    # d = len(im_size)
    # os = hparams.os
    # from mr_recon.fourier import sigpy_nufft
    # nft = sigpy_nufft(im_size, oversamp=os, width=width)
    # beta = nft.optimal_beta(torch_dev=phis.device)
    # rs = gen_grd(im_size).to(phis.device)
    # apod = kb_apod_1d(rs / os, beta, width).prod(dim=-1)
    # apod *= apod.abs().max()
    # apod = apod[None,]
    # k_corr = (width%2==0)*torch.ones(d, device=phis.device) / os / 2
    # phz = torch.exp(-2j * np.pi * einsum(rs, k_corr, '... d, d -> ...'))
    # apod = apod * phz
    # spatial_factors = spatial_factors * apod
    
    return spatial_factors

def K_alphas_init(phis: torch.Tensor,
                  alphas: torch.Tensor,
                  hparams: hofft_params,
                  spatial_mask: Optional[torch.Tensor] = None,
                  spatial_init_method: str = 'seg',
                  num_als_iter: int = 100,
                  K: int = 500,
                  return_kernels: bool = False,) -> torch.Tensor:
    """
    Initialize spatial factors by running ALS on K representative alphas.

    The decomposition is performed by ``als_compressed``, which fits the explicit
    spatial bases b_q(r) = exp(-j2pi sum_b phi_b(r) k_alpha_b,q) of the K
    representatives, batching over the voxel (N) dimension. This avoids the
    (kern x kern x *im_size) cross-term materialized by the matvec-based ALS,
    making the initialization tractable in 3D.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis functions, shape (B, *im_size).
    alphas : torch.Tensor
        temporal phase basis functions, shape (B, *trj_size).
    hparams : hofft_params
        hofft parameters.
    spatial_mask : Optional[torch.Tensor]
        spatial mask with shape (*im_size)
    spatial_init_method : str
        method to initialize spatial factors before ALS. Options are 'seg' and 'eigen', and 'ones'
    num_als_iter : int
        number of ALS iterations to run.
    K : int
        number of representative alphas to pick.
    return_kernels : bool
        if True, also return the learned HOFFT kernels for the K representative
        alphas (these serve as the compressed kernels H_comp) and the
        representative alphas (betas) themselves.
    spatial_batch_size : Optional[int]
        batch size over the voxel (N) dimension for the ALS solve. None uses all
        voxels at once; set this to bound memory in 3D.
        
    Returns
    -------
    spatial_factors : torch.Tensor
        initialized spatial factors, shape (L, *im_size).
    compressed_kernels : torch.Tensor
        (only if return_kernels) learned kernels with shape (L, *kern_size, K).
    betas : torch.Tensor
        (only if return_kernels) representative alphas with shape (K, B).
    """
    # Consts
    verbose = hparams.verbose
    im_size = phis.shape[1:]
    torch_dev = phis.device
    B = phis.shape[0]
    d = len(im_size)
    L = hparams.L
    kern_size = hparams.kern_size
    os = hparams.os
    spatial_batch_size = hparams.spatial_batch_size
    method = hparams.kalpha_method
    
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
    elif spatial_init_method == 'ones':
        spatial_factors_init = torch.ones(L, *phis.shape[1:], dtype=torch.complex64, device=phis.device)
    else:
        raise ValueError(f'Invalid spatial_init_method {spatial_init_method}. Supported methods are seg, eigen, and ones.')
    
    # Pick K representative alphas
    k_alphas = k_alpha_selection(alphas, K, method, verbose=verbose)
    k_alphas = k_alphas.T # shape (B, K)
    
    # Explicit spatial bases of the K representatives:
    spatial_bases = torch.exp(-2j * np.pi * einsum(phis, k_alphas, 
                                                   'B ..., B Q -> Q ...'))
    
    # HOFFT decomp via voxel-batched ALS (memory efficient in 3D). This solves the
    # same least squares as the matvec-based ALS but batches over N, avoiding the
    # (kern x kern x *im_size) cross term. Note: anderson_order is not used here.
    # spatial_factors, kernel_weights = als_compressed(spatial_bases=spatial_bases,
    #                                                  kern_bases=kern_bases,
    #                                                  spatial_factors_init=spatial_factors_init,
    #                                                  mask=spatial_mask,
    #                                                  max_iter=num_als_iter,
    #                                                  spatial_batch_size=spatial_batch_size,
    #                                                  solver=hparams.solver,
    #                                                  lamda=hparams.lamda,
    #                                                  verbose=verbose)
    phase_model = hparams.matvec_type(phis, k_alphas, **hparams.matvec_kwargs)
    spatial_factors, kernel_weights = als_iterations(phase_model=phase_model,
                                                     kern_bases=kern_bases,
                                                     spatial_factors_init=spatial_factors_init,
                                                     mask=spatial_mask,
                                                     max_iter=num_als_iter,
                                                     solver=hparams.solver,
                                                     lamda=hparams.lamda,
                                                     spatial_batch_size=spatial_batch_size,
                                                     verbose=verbose)
    
    if return_kernels:
        # kernel_weights has shape (L, prod(kern_size), K); reshape to kernel grid.
        # The K representative alphas (k_alphas.T) are the betas.
        compressed_kernels = kernel_weights.reshape((L, *kern_size, K))
        betas = k_alphas.T  # (K, B)
        return spatial_factors, compressed_kernels, betas
    
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
    kalpha_method = hparams.kalpha_method
    verbose = hparams.verbose
    L = hparams.L
    
    # Pick K representative alphas
    betas = k_alpha_selection(alphas, L, kalpha_method, verbose=verbose)
    spatial_factors = einsum(phis, betas, 'B ..., L B -> L ...')
    spatial_factors = torch.exp(-2j * torch.pi * spatial_factors)
    
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
    x0 = torch.randn(im_size, dtype=torch.complex64, device=torch_dev)
    spatial_factors, _ = eigen_decomp_operator(phase_model.normal, x0, num_eigen=L, 
                                               num_iter=15,
                                               lobpcg=True,
                                               largest=True)
    
    return spatial_factors

