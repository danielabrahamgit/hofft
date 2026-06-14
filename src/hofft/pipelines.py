import torch
import numpy as np

from typing import Optional, Union
from einops import einsum

from mr_recon.imperfections.field import b0_to_phis_alphas, alpha_segementation
from mr_recon.utils import gen_grd, resize
from mr_recon.linops import linop, batching_params
from mr_recon.spatial import spatial_resize_poly, spatial_interp
from mr_recon.algs import eigen_decomp_operator
from mr_recon.fourier import sigpy_nufft, fft, ifft

from .decomp import (
    hofft_params, 
    als_iterations, 
    build_kern_bases, 
    als_anderson_iterations,
    als_compressed,
    als_iterations_tempinit,
)
from .spatial_init import choose_init
from .sgd import train_net_apod
from .kb import kb_apod_1d, sample_kb_kernel
from .forward_model import hofft_linop
from .reduce import expand_temporal, reduce_temporal, alpha_interp_kerns, reduce_params, expand_spatial
from .sparse_decomp import sparse_params, sparse_alpha_segmentation

def kb_nufft(trj: torch.Tensor,
             im_size: tuple,
             kern_size: tuple,
             os: float = 1.0,
             beta: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes kernel weights and the spatial factor for the KB-NUFFT model.
    
    Args
    ----
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    im_size : tuple
        Size of the image to be reconstructed.
    kern_size : tuple
        Size of the kernel.
    os : float
        Oversampling factor.
    beta : Optional[float]
        Beta parameter for the KB-NUFFT kernel.
    
    Returns
    -------
    spatial_factor : torch.Tensor
        KB-NUFFT spatial factor with shape (1, *im_size)
    kern_weights : torch.Tensor
        KB-NUFFT kernel weights with shape (1, *kern_size, *trj_size)
    """
    # Consts
    d = len(im_size)
    width = kern_size[0]
    for i in range(1, len(kern_size)):
        assert kern_size[i] == width, "Kernel size must be the same in all dimensions"
    if beta is None:
        beta = torch.pi * (((width / os) * (os - 0.5))**2 - 0.8)**0.5
        if (((width / os) * (os - 0.5))**2 - 0.8) < 0:
            beta = 1.0
    
    # Spatial factor
    rs = gen_grd(im_size).to(trj.device)
    spatial_factor = kb_apod_1d(rs / os, beta, width).prod(dim=-1)
    
    # Kernel weights
    kdevs = trj - (os * trj).round()/os
    kern_weights = sample_kb_kernel(kdevs, kern_size, os, beta)
    
    # Scaling factor correction
    spatial_factor /= width ** d
    
    # Reshape
    spatial_factor = spatial_factor[None,]
    kern_weights = kern_weights[None,].type(torch.complex64)
    
    # Apply correction linear phase
    k_corr = (width%2==0)*torch.ones(d, device=trj.device) / os / 2
    phz = torch.exp(-2j * np.pi * einsum(rs, k_corr, '... d, d -> ...'))
    spatial_factor = spatial_factor * phz

    return spatial_factor, kern_weights

# TODO work in progress
def mlp_hofft(phis: torch.Tensor,
              alphas: torch.Tensor,
              im_size: tuple,
              hparams: hofft_params,
              opt_apods: bool = True,
              epochs: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
    """
    Multi-apodization HOFFT model, allowing for arbitrary non-linear phase
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *solve_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    im_size : tuple
        Size of the image to be reconstructed.
    hparams : hofft_params
        HOFFT parameters.
    opt_apods : bool
        If True, optimizes the apodization functions.
        If False, uses the initial apodization functions.
    epochs : int
        Number of epochs to train the MLP.
        
    Returns
    -------
    weights : torch.Tensor
        NUFFT kernel weights with shape (L, *kern_size, *trj_size)
    apods : torch.Tensor
        Apodization functions with shape (L, *im_size)
    kern_model : nn.Module
        The learned kernel model.
    """
    # Consts
    solve_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    torch_dev = phis.device
    d = len(im_size)
    B = phis.shape[0]
    kern_size = hparams.kern_size
    os = hparams.os
    L = hparams.L
    apods_init = hparams.apods_init
    use_type3 = hparams.use_type3
    verbose = hparams.verbose
    
    # Initialize apodization functions
    if isinstance(apods_init, torch.Tensor):
        apods = apods_init
    elif isinstance(apods_init, str):
        if apods_init == 'eigen':
            apods = eigen_apod_init(phis, alphas, hparams)
        elif apods_init == 'seg':
            apods = alpha_seg_apod_init(phis, alphas, hparams)
        elif re.fullmatch(r"\d+_alphas", apods_init):
            K = int(apods_init.split('_')[0])
            apods = K_alphas_apod_init(phis, alphas, hparams,
                                       method='minmax',
                                       apod_init_method='seg',
                                       num_als_iter=100, K=K)
        else:
            raise ValueError(f'Invalid apods_init {apods_init}. Supported methods are seg, eigen, and k_alphas.')
    else:
        raise ValueError("apods_init must be a torch.Tensor or a string")
    
    # Train MLP
    weights, apods, kern_model = train_net_apod(phis, alphas, apods, kern_size, os, opt_apods=opt_apods, epochs=epochs)
    
    # Interpolate spatial funcs
    kwargs = {'order': 3, 'mode': 'nearest'}
    solve_size_tensor = torch.tensor(solve_size).to(torch_dev)
    spatial_crds = (gen_grd(im_size).to(torch_dev) + 0.5) * solve_size_tensor
    apods = spatial_interp(apods, spatial_crds, **kwargs)
    
    return weights, apods, kern_model

def als_nufft(trj: torch.Tensor,
              im_size: tuple,
              hparams: hofft_params,
              spatial_mask: Optional[torch.Tensor] = None,
              im_size_low: Optional[tuple] = None,
              num_als_iter: Optional[int] = 100,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes least squares optimal NUFFT kernel weights and the spatial factor using ALS.
    
    Args
    ----
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    im_size : tuple
        Size of the image to be reconstructed.
    hparams : hofft_params
        HOFFT parameters.
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    im_size_low : Optional[tuple]
        Optional low resolution size for performing the decomposition
    num_als_iter : int
        Number of ALS iterations.
    
    Returns
    -------
    spatial_factor : torch.Tensor
        Apodization functions with shape (L, *im_size)
    kern_weights : torch.Tensor
        NUFFT kernel weights with shape (L, *kern_size, *trj_size)
    """
    # Consts
    d = trj.shape[-1]
    torch_dev = trj.device
    trj_size = trj.shape[:-1]
    im_size_low = (50,)*d if im_size_low is None else im_size_low
    kern_size = hparams.kern_size
    os = hparams.os
    L = hparams.L
    verbose = hparams.verbose
    spatial_init_method = hparams.spatial_init
    
    # Default spatial mask
    if spatial_mask is None:
        spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
    spatial_mask = spatial_resize_poly(spatial_mask, im_size_low, order=3)
    
    # Spatial and trajectory bases
    rs = gen_grd(im_size_low).to(torch_dev)
    kdevs = gen_grd(im_size_low).to(torch_dev) / os
    kern_bases = build_kern_bases(kern_size, im_size_low, os)
    kern_bases = kern_bases.to(torch_dev)
    phis = rs.moveaxis(-1, 0)
    alphas = kdevs.moveaxis(-1, 0)

    # Initialize apodization functions with eigen-vectors
    high_acc_nufft = sigpy_nufft(im_size_low, oversamp=2.0, width=6)
    kdevs_rs = high_acc_nufft.rescale_trajectory(kdevs)
    if spatial_init_method == 'eigen':
        toep_kerns = high_acc_nufft.calc_teoplitz_kernels(kdevs_rs[None])[0] # *solve_size_os
        solve_size_os = toep_kerns.shape
        def normal_op(x):
            N = x.shape[0]
            x = resize(x * spatial_mask, (N, *solve_size_os))
            x = fft(x, dim=tuple(range(-d, 0)))
            x = x * toep_kerns
            x = ifft(x, dim=tuple(range(-d, 0)))
            x = resize(x, (N, *im_size_low))
            return x * spatial_mask.conj()
        x0 = torch.randn(im_size_low, dtype=torch.complex64, device=torch_dev)
        spatial_factor, _ = eigen_decomp_operator(normal_op, x0, num_eigen=L, verbose=verbose)
    else:
        spatial_factor = choose_init(phis, alphas, hparams=hparams, spatial_init=spatial_init_method)

    # Make matrix-vector operation that applies spatially linear phase only
    class matvec_linphase(linop):
        def __init__(self):
            super().__init__(im_size_low, im_size_low)
        def forward(self, x):
            return high_acc_nufft.forward(x[None,], kdevs_rs[None,])[0] * np.prod(im_size_low) ** 0.5
        def adjoint(self, y):
            return high_acc_nufft.adjoint(y[None,], kdevs_rs[None,])[0] * np.prod(im_size_low) ** 0.5
    phase_model = matvec_linphase()
    # phis = rs.moveaxis(-1, 0)
    # alphas = kdevs.moveaxis(-1, 0)
    # phase_model = hparams.matvec_type(phis, alphas, **hparams.matvec_kwargs)
    
    # Perform ALS iterations
    if hparams.anderson_order is None: 
        kern_weights, spatial_factor = als_iterations(phase_model, kern_bases, spatial_factor, 
                                                      mask=spatial_mask,
                                                      max_iter=num_als_iter, verbose=verbose)
    else:
        kern_weights, spatial_factor = als_anderson_iterations(phase_model, kern_bases, spatial_factor, 
                                                               anderson_order=hparams.anderson_order,
                                                               mask=spatial_mask,
                                                               max_iter=num_als_iter, verbose=verbose)
    
    # Interpolate spatial funcs
    kwargs = {'order': 5, 'mode': 'nearest'}
    solve_size_tensor = torch.tensor(im_size_low).to(torch_dev)
    spatial_crds = (gen_grd(im_size).to(torch_dev) + 0.5) * solve_size_tensor
    spatial_factor = spatial_interp(spatial_factor, spatial_crds, **kwargs)

    # Interpolate temporal functions
    trj_dev = trj - (os * trj).round()/os
    temporal_crds = (0.5 + trj_dev * os) * solve_size_tensor
    kern_weights = spatial_interp(kern_weights.reshape((-1, *im_size_low)), temporal_crds, **kwargs)
    kern_weights = kern_weights.reshape((L, *kern_size, *trj_size))
    
    return spatial_factor, kern_weights

def als_hofft(phis: torch.Tensor,
              alphas: torch.Tensor,
              hparams: hofft_params,
              rparams: Optional[reduce_params] = reduce_params(),
              spatial_mask: Optional[torch.Tensor] = None,
              num_als_iter: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    General pipeline for getting a HOFFT linop using ALS.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    hparams : hofft_params
        HOFFT parameters.
    rparams : Optional[reduce_params]
        Parameters specifying how to reduce the spatial (phi) and temporal (alpha) dimensions
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    num_als_iter : Optional[int]
        Number of ALS iterations.
        
    Returns
    -------
    spatial_factor : torch.Tensor
        Spatial factors with shape (L, *im_size)
    kern_weights : torch.Tensor
        NUFFT kernel weights with shape (L, *kern_size, *trj_size)
    """
    # Consts
    im_size = phis.shape[1:]
    torch_dev = phis.device
    kern_size = hparams.kern_size
    os = hparams.os
    L = hparams.L
    spatial_init = hparams.spatial_init
    verbose = hparams.verbose
    
    # Default spatial mask
    if spatial_mask is None:
        spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
    
    # Reduce phi size
    if rparams.spatial_reduce_size is not None:
        phis_reduced = spatial_resize_poly(phis, 
                                           im_size=rparams.spatial_reduce_size, 
                                           order=rparams.spatial_reduce_order)
        kern_bases = build_kern_bases(kern_size, 
                                      im_size=rparams.spatial_reduce_size, 
                                      os=os).to(torch_dev)
        spatial_mask = spatial_resize_poly(spatial_mask, 
                                           im_size=rparams.spatial_reduce_size, 
                                           order=rparams.spatial_reduce_order)
    else:
        kern_bases = build_kern_bases(kern_size, im_size, os).to(torch_dev)
        phis_reduced = phis
    
    # Reduce alpha size
    if rparams.alpha_reduce_width is not None:
        ret = alpha_interp_kerns(phis_reduced, 
                                 W=rparams.alpha_reduce_width, 
                                 dalphas=rparams.alpha_reduce_grid_spacing, 
                                 solve_apod=rparams.alpha_reduce_use_apod)
        weights, delta_alphas, apods = ret
        ret = reduce_temporal(alphas, 
                              W=rparams.alpha_reduce_width, 
                              dalphas=rparams.alpha_reduce_grid_spacing)
        alphas_reduced, alpha_kern, alpha_to_unq_idx = ret
        alphas_reduced = alphas_reduced.T
        print(alphas.numel() / alphas_reduced.numel())
    else:
        alphas_reduced = alphas
    
    # Make matvec phase model
    phase_model = hparams.matvec_type(phis_reduced, alphas_reduced, 
                                      **hparams.matvec_kwargs)
    
    # Initialize spatial factors
    spatial_factors = choose_init(phis_reduced, alphas_reduced, 
                                  hparams=hparams, 
                                  spatial_mask=spatial_mask,
                                  spatial_init=spatial_init)

    # ALS to solve for kernel weights and spatial factors
    if hparams.anderson_order is None:
        spatial_factors, kern_weights = als_iterations(phase_model, kern_bases, spatial_factors,
                                                       mask=spatial_mask,
                                                       max_iter=num_als_iter, 
                                                       verbose=verbose)
    else:
        spatial_factors, kern_weights = als_anderson_iterations(phase_model, kern_bases, spatial_factors, 
                                                                anderson_order=hparams.anderson_order,
                                                                mask=spatial_mask,
                                                                max_iter=num_als_iter, verbose=verbose)
    kern_weights = kern_weights.reshape((L, *kern_size, *alphas_reduced.shape[1:]))

    # Expand phis 
    if rparams.spatial_reduce_size is not None:
        spatial_factors = expand_spatial(spatial_factors, 
                                         im_size_high=im_size,
                                         order=rparams.spatial_reduce_order)
    
    # Expand alphas
    if rparams.alpha_reduce_width is not None:
        kern_weights = expand_temporal(kern_weights, alphas, 
                                       dalphas=rparams.alpha_reduce_grid_spacing,
                                       weights=weights,
                                       delta_alphas=delta_alphas,
                                       alpha_kern=alpha_kern,
                                       alpha_to_unq_idx=alpha_to_unq_idx,
                                       temporal_batch_size=rparams.alpha_interp_batch_size)
        apods = expand_spatial(apods,
                               im_size_high=im_size,
                               order=rparams.spatial_reduce_order)
        spatial_factors *= apods.prod(dim=0)
    
    return spatial_factors, kern_weights

def als_hofft_kbinit(phis: torch.Tensor,
                     alphas: torch.Tensor,
                     trj: torch.Tensor,
                     hparams: hofft_params,
                     rparams: Optional[reduce_params] = reduce_params(),
                     spatial_mask: Optional[torch.Tensor] = None,
                     num_als_iter: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    General pipeline for getting a HOFFT linop using ALS with a splitting model 
    for the spatio-temporal phase.
    
    Args
    ----
    phis : torch.Tensor
        High order spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        High order temporal phase coefficients with shape (B, *trj_size)
    trj : torch.Tensor
        Non-Cartesian trajectory with shape (*trj_size, d)
    hparams : hofft_params
        HOFFT parameters.
    Q : int
        Number of spatial basis functions.
    rparams : Optional[reduce_params]
        Parameters specifying how to reduce the spatial (phi) and temporal (alpha) dimensions
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    num_als_iter : Optional[int]
        Number of ALS iterations.
        
    Returns
    -------
    spatial_factor : torch.Tensor
        Spatial factors with shape (L, *im_size)
    kern_weights : torch.Tensor
        NUFFT kernel weights with shape (L, *kern_size, *trj_size)
    """
    # Consts
    im_size = phis.shape[1:]
    torch_dev = phis.device
    kern_size = hparams.kern_size
    os = hparams.os
    L = hparams.L
    K = np.prod(kern_size)
    trj_size = trj.shape[:-1]
    spatial_init = hparams.spatial_init
    verbose = hparams.verbose
    
    # Default spatial mask
    if spatial_mask is None:
        spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
    
    # Reduce phi size
    if rparams.spatial_reduce_size is not None:
        phis_reduced = spatial_resize_poly(phis, 
                                           im_size=rparams.spatial_reduce_size, 
                                           order=rparams.spatial_reduce_order)
        kern_bases = build_kern_bases(kern_size, 
                                      im_size=rparams.spatial_reduce_size, 
                                      os=os).to(torch_dev)
        spatial_mask = spatial_resize_poly(spatial_mask, 
                                           im_size=rparams.spatial_reduce_size, 
                                           order=rparams.spatial_reduce_order)
    else:
        kern_bases = build_kern_bases(kern_size, im_size, os).to(torch_dev)
        phis_reduced = phis
    
    # Reduce alpha size
    if rparams.alpha_reduce_width is not None:
        ret = alpha_interp_kerns(phis_reduced, 
                                 W=rparams.alpha_reduce_width, 
                                 dalphas=rparams.alpha_reduce_grid_spacing, 
                                 solve_apod=rparams.alpha_reduce_use_apod)
        weights, delta_alphas, apods = ret
        ret = reduce_temporal(alphas, 
                              W=rparams.alpha_reduce_width, 
                              dalphas=rparams.alpha_reduce_grid_spacing)
        alphas_reduced, alpha_kern, alpha_to_unq_idx = ret
        alphas_reduced = alphas_reduced.T
        print(alphas.numel() / alphas_reduced.numel())
    else:
        alphas_reduced = alphas
        
    # Make matvec phase model
    phase_model = hparams.matvec_type(phis_reduced, alphas_reduced, 
                                      **hparams.matvec_kwargs)
    # Initialize spatial factors
    spatial_factors = choose_init(phis_reduced, alphas_reduced, 
                                  hparams=hparams, 
                                  spatial_mask=spatial_mask,
                                  spatial_init=spatial_init)

    # Solve high order splitting coeffs
    from mr_recon.imperfections.field import phi_alpha_svd, alpha_segementation
    bs, cs, _ = alpha_segementation(phis_reduced, alphas_reduced, 
                                    L=L, interp_type='lstsq', use_type3=False,
                                    manual_spatial_funcs=spatial_factors,
                                    verbose=verbose)
    
    # Initialize kernel weights via KB
    from mr_recon.fourier import sigpy_nufft
    nft = sigpy_nufft(im_size, oversamp=os, width=kern_size[0])
    beta = nft.optimal_beta(torch_dev=torch_dev)
    apod, kb_weights = kb_nufft(trj, im_size, kern_size,
                                os=os, beta=beta)
    kb_weights = kb_weights.reshape((1, K, *trj_size))
    kern_weights_init = kb_weights * cs[:, None, ...]
    
    # HOFFT decomp with kernel weights initialized via KB
    spatial_factors, kern_weights = als_iterations_tempinit(phase_model, kern_bases, kern_weights_init,
                                                            mask=spatial_mask,
                                                            max_iter=num_als_iter, verbose=verbose)
    kern_weights = kern_weights.reshape((L, *kern_size, *alphas_reduced.shape[1:]))

    # Expand phis 
    if rparams.spatial_reduce_size is not None:
        spatial_factors = expand_spatial(spatial_factors, 
                                         im_size_high=im_size,
                                         order=rparams.spatial_reduce_order)
    
    # Expand alphas
    if rparams.alpha_reduce_width is not None:
        kern_weights = expand_temporal(kern_weights, alphas, 
                                       dalphas=rparams.alpha_reduce_grid_spacing,
                                       weights=weights,
                                       delta_alphas=delta_alphas,
                                       alpha_kern=alpha_kern,
                                       alpha_to_unq_idx=alpha_to_unq_idx,
                                       temporal_batch_size=rparams.alpha_interp_batch_size)
        apods = expand_spatial(apods,
                               im_size_high=im_size,
                               order=rparams.spatial_reduce_order)
        spatial_factors *= apods.prod(dim=0)
    
    return spatial_factors, kern_weights

def als_hofft_compressed(phis: torch.Tensor,
                         alphas: torch.Tensor,
                         hparams: hofft_params,
                         sparams: sparse_params,
                         spatial_mask: Optional[torch.Tensor] = None,
                         num_als_iter: int = 100,
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    General pipeline for getting a HOFFT linop using ALS with a compression model 
    for the spatio-temporal phase.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    hparams : hofft_params
        HOFFT parameters.
    sparams : sparse_params
        Sparse decomposition parameters.
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    num_als_iter : Optional[int]
        Number of ALS iterations.
    return_compressed : bool
        Whether to return the compressed kernels.
        
    Returns
    -------
    spatial_factor : torch.Tensor
        Spatial factors with shape (L, *im_size)
    kern_weights : torch.Tensor
        NUFFT kernel weights with shape (L, *kern_size, *trj_size)
    sparse_inds : torch.Tensor
        Sparse indices with shape (K, *trj_size) in [0, Q)
    sparse_coeffs : torch.Tensor
        Sparse coefficients with shape (K, *trj_size)
    """
    # Consts
    im_size = phis.shape[1:]
    torch_dev = phis.device
    kern_size = hparams.kern_size
    os = hparams.os
    L = hparams.L
    spatial_init = hparams.spatial_init
    verbose = hparams.verbose
    reduced_im_size = hparams.reduced_im_size
    Q = sparams.Q
    K = sparams.K
    
    # Default spatial mask
    if spatial_mask is None:
        spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
    
    # Reduce phi size
    if reduced_im_size is not None:
        phis_reduced = spatial_resize_poly(phis, 
                                           im_size=reduced_im_size, 
                                           order=3)
        kern_bases = build_kern_bases(kern_size, 
                                      reduced_im_size, 
                                      os=os).to(torch_dev)
        spatial_mask = spatial_resize_poly(spatial_mask, 
                                           im_size=reduced_im_size, 
                                           order=3)
    else:
        kern_bases = build_kern_bases(kern_size, im_size, os).to(torch_dev)
        phis_reduced = phis
    
    # Initialize spatial factors
    spatial_factors = choose_init(phis_reduced, alphas, 
                                  hparams=hparams, 
                                  spatial_mask=spatial_mask,
                                  spatial_init=spatial_init)

    # Apply splitting decomp
    sparse_rets = sparse_alpha_segmentation(phis_reduced, alphas, sparams,
                                            verbose=verbose)
    spatial_bases, sparse_inds, sparse_coeffs = sparse_rets
    Q = spatial_bases.shape[0] # may differ from sparams.Q for grid methods
    
    # HOFFT decomp on split bases
    spatial_factors, compressed_kernels = als_compressed(spatial_bases=spatial_bases,
                                                         kern_bases=kern_bases, 
                                                         spatial_factors_init=spatial_factors,
                                                         spatial_batch_size=sparams.spatial_batch_size,
                                                         mask=spatial_mask,
                                                         max_iter=num_als_iter,
                                                         lamda=hparams.lamda,
                                                         solver=hparams.solver,
                                                         verbose=verbose)
    compressed_kernels = compressed_kernels.reshape((L, *kern_size, Q))

    # Expand spatial
    if reduced_im_size is not None:
        spatial_factors = expand_spatial(spatial_factors, 
                                         im_size_high=im_size,
                                         order=3)
        
    return spatial_factors, compressed_kernels, sparse_inds, sparse_coeffs

def b0_correction(b0: torch.Tensor,
                  trj: torch.Tensor,
                  mps: torch.Tensor,
                  dcf: Optional[torch.Tensor],
                  ro_dim: int,
                  dt: float,
                  hparams: hofft_params,
                  bparams: Optional[batching_params] = batching_params(),
                  num_als_iter=100) -> linop:
    """
    Creates a HOFFT linop for B0 correction.
    
    Args
    ----
    b0 : torch.Tensor
        B0 map of shape (*solve_size) in Hz
    trj : torch.Tensor
        Trajectory of shape (*trj_size, d), where d = len(im_size)
    mps : torch.Tensor
        Coil sensitivity maps of shape (C, *im_size), where C is the number of coils
    ro_dim : int
        Readout dimension in trj
    dt : float
        Sampling dwell time in seconds
    hparams : hofft_params
        hofft parameters.
    bparams : batching_params
        batching parameters linop
    num_als_iter : int
        number of ALS iterations to run.
    
    Returns
    -------
    linop
        HOFFT linop for B0 correction
    """
    # Consts
    torch_dev = b0.device
    solve_size = b0.shape
    im_size = mps.shape[1:]
    trj_size = trj.shape[:-1]
    
    # Make phis and alphas
    phis_b0, alphas_b0 = b0_to_phis_alphas(b0, trj_size, ro_dim, dt)
    phis_kdev = gen_grd(solve_size).to(torch_dev).moveaxis(-1, 0) # (d, *im_size)
    trj_grd = (trj * hparams.os).round() / hparams.os
    alphas_kdev = (trj - trj_grd).moveaxis(-1, 0) # (d, *trj_size)
    alphas = torch.cat([alphas_b0.expand_as(alphas_kdev)[:1], alphas_kdev], dim=0) # (B+d, *trj_size)
    phis = torch.cat([phis_b0, phis_kdev], dim=0) # (B+d, *im_size)
    phis = phis[1:]
    alphas = alphas[1:]
    
    # Call ALS hofft
    weights, apods = als_hofft(phis, alphas, im_size, hparams, 
                               num_als_iter=num_als_iter)
    # weights = torch.zeros((hparams.L, *hparams.kern_size, *trj_size), device=torch_dev, dtype=torch.complex64)
    # apods = torch.zeros((hparams.L, *im_size), device=torch_dev, dtype=torch.complex64)
    
    # Build linop
    linop = hofft_linop(trj_grd, mps, 
                        weights=weights, 
                        apods=apods,
                        dcf=dcf,
                        os_grid=hparams.os,
                        bparams=bparams)
    
    return linop