import torch

from einops import einsum
from typing import Union, Optional

from mr_recon.utils import pick_K_vectors
from .linalg import eigen_decomp_operator
from .decomp import hofft_params

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
        representative alphas, shape (B, k).
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
    return betas.T

def choose_init(phis: torch.Tensor,
                alphas: torch.Tensor,
                hparams: hofft_params,
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
    spatial_factors = einsum(phis, betas, 'B ..., B L -> L ...')
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

