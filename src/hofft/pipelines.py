import copy
import torch
import numpy as np

from typing import Optional, Union
from einops import einsum

from mr_recon.imperfections.field import b0_to_phis_alphas
from mr_recon.linops import linop, sense_linop, batching_params
from mr_recon.algs import eigen_decomp_operator
from mr_recon.fourier import sigpy_nufft, fft, ifft

from .matvec import matvec_naive

from .phase_coeffs import (trj_dev_to_phis_alphas, 
                           rescale_phis_alphas,
                           apply_phase_midpoints,
                           whiten_phis_alphas
)
from .decomp import (
    hofft_params, 
    als_iterations, 
    build_kern_bases, 
    als_anderson_iterations
)
from .utils import gen_grd, resize, spatial_interp, reduce_spatial, expand_spatial
from .spatial_init import choose_init, K_alphas_init
from .sgd import train_sparse_net, training_params
from .kb import kb_apod_1d, sample_kb_kernel
from .forward_model import hofft_linop, hofft_compressed_linop
from .sparse_fit import (
    sparse_params,
    lstsq_compressed_fixed_support,
    smooth_sparse_coeffs,
    sweep_smooth_interp_hyperparams,
)

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

def mlp_hofft_compressed(phis: torch.Tensor,
                         alphas: torch.Tensor,
                         hparams: hofft_params,
                         sparams: sparse_params,
                         tparams: training_params = training_params(),
                         spatial_mask: Optional[torch.Tensor] = None,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Learns compressed HOFFT kernels using a sparse MLP
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *solve_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    hparams : hofft_params
        HOFFT parameters.
    sparams : sparse_params
        Sparse decomposition parameters.
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    num_als_iter : int
        Number of ALS iterations.
    
    Returns
    -------
    spatial_factors : torch.Tensor
        Spatial factors with shape (L, *im_size)
    compressed_kernels : torch.Tensor
        Compressed kernel dictionary with shape (L, *kern_size, Q)
    bias_kern : torch.Tensor
        Bias kernel with shape (L, *kern_size)
    sparse_inds : torch.Tensor
        Sparse indices (long) with shape (S, *trj_size), values in [0, Q)
    sparse_coeffs : torch.Tensor
        Sparse coefficients (real softmax weights, cast to complex64) with shape (S, *trj_size)
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
    S = sparams.S

    # Default spatial mask
    if spatial_mask is None:
        spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
    
    # Reduce phi size
    if reduced_im_size is not None:
        phis_reduced = spatial_resize_poly(phis, 
                                           im_size=reduced_im_size, 
                                           order=3)
        spatial_mask = spatial_resize_poly(spatial_mask, 
                                           im_size=reduced_im_size, 
                                           order=3)
    else:
        phis_reduced = phis
    
    # Initialize spatial factors
    spatial_factors = choose_init(phis_reduced, alphas, 
                                  hparams=hparams, 
                                  spatial_mask=spatial_mask,
                                  spatial_init=spatial_init)
    
    # Train sparse MLP
    ret = train_sparse_net(phis_reduced, alphas, 
                           spatial_factors_init=spatial_factors, 
                           hparams=hparams, sparams=sparams, tparams=tparams, 
                           opt_spatial_factors=False,
                           spatial_mask=spatial_mask)
    spatial_factors, compressed_kernels, bias_kern, sparse_inds, sparse_coeffs = ret
    
    # Expand spatial factors
    spatial_factors = expand_spatial(spatial_factors, 
                                     im_size_high=im_size, 
                                     order=3)
    return spatial_factors, compressed_kernels, bias_kern, sparse_inds, sparse_coeffs
    
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
        spatial_factor, kern_weights = als_iterations(phase_model, kern_bases, spatial_factor, 
                                                      mask=spatial_mask,
                                                      max_iter=num_als_iter, verbose=verbose)
    else:
        spatial_factor, kern_weights = als_anderson_iterations(phase_model, kern_bases, spatial_factor, 
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

def time_seg_decomp_linop(phis: torch.Tensor,
                          alphas: torch.Tensor,
                          mps: torch.Tensor,
                          trj: torch.Tensor,
                          hparams: hofft_params,
                          spatial_mask: Optional[torch.Tensor] = None,
                          dcf: Optional[torch.Tensor] = None,
                          bparams: batching_params = batching_params(),
                          normalize_coeffs: bool = False,
                          use_sigpy: bool = False) -> linop:
    """
    Time-segmented NUFFT decomposition and linop.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    mps : torch.Tensor
        Coil sensitivities with shape (C, *im_size)
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    hparams : hofft_params
        HOFFT parameters.
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    dcf : Optional[torch.Tensor]
        Density compensation factor with shape (*trj_size)
    bparams : batching_params
        Batching parameters for the HOFFT linop.
    normalize_coeffs : bool
        Whether to normalize the phase coefficients.
    
    Returns
    -------
    linop : linop
        Time-segmented NUFFT decomposition and linop.
    """
    # Consts
    B = phis.shape[0]
    im_size = phis.shape[1:]
    trj_size = trj.shape[:-1]
    d = len(im_size)
    torch_dev = phis.device
    kern_size = hparams.kern_size
    os = hparams.os
    
    # Make sure the kernel size is isotropic
    for i in range(1, d):
        assert kern_size[0] == kern_size[i], "Kernel size must be isotropic for time-segmented NUFFT decomposition"
    
    # Normalize phase coefficients
    if normalize_coeffs:
        phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)
        spat, temp = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp)
        phis_nrm, alphas_nrm = whiten_phis_alphas(phis_nrm, alphas_nrm, 
                                                  B_compressed=B)
    else:
        phis_nrm = phis
        alphas_nrm = alphas
        phis_mp = torch.zeros(B, device=torch_dev, dtype=torch.float32)
        alphas_mp = torch.zeros(B, device=torch_dev, dtype=torch.float32)
        spat = torch.ones_like(phis_nrm[0]).type(torch.complex64)
        temp = torch.ones_like(alphas_nrm[0]).type(torch.complex64)
    
    
    # Single temporal least squares solve, num_als_iter=0 does exactly that
    hparams_copy = copy.copy(hparams)
    hparams_copy.spatial_init = 'seg' # time seg
    hparams_copy.kern_size = (1,)*d
    num_als_iter = 0
    spatial_funcs, temporal_funcs = als_hofft(phis_nrm, alphas_nrm, hparams_copy, 
                                              spatial_mask=spatial_mask, 
                                              num_als_iter=num_als_iter)
    
    # Get optimal beta parameter
    nft = sigpy_nufft(im_size, oversamp=os, width=kern_size[0])
    nft.beta = nft.optimal_beta(torch_dev=torch_dev)
    
    # Use Sigpy's KB NUFFT framework for forward model
    if use_sigpy:
        temporal_funcs = temporal_funcs.reshape((hparams.L, *alphas_nrm.shape[1:]))
        A = sense_linop(trj, mps, dcf, nufft=nft, 
                        spatial_funcs=spatial_funcs * spat,
                        temporal_funcs=temporal_funcs * temp,
                        bparams=bparams)
    # Use HOFFT forward model with KB NUFFT weights
    else:
        # Calculate KB NUFFT weights
        spatial_factor, kern_weights = kb_nufft(trj, im_size, kern_size, 
                                                os=os, beta=nft.beta)
        
        # Combine
        spatial_factors = spatial_factor * spatial_funcs * spat
        kern_weights = kern_weights * temporal_funcs * temp
        
        # Build linop
        trj_grd = (os * trj).round()/os
        A = hofft_linop(trj=trj_grd, mps=mps, dcf=dcf, 
                        kern_weights=kern_weights, 
                        spatial_factors=spatial_factors, 
                        os_grid=os, bparams=bparams)
    
    return A

def hofft_decomp_linop(phis: torch.Tensor,
                       alphas: torch.Tensor,
                       mps: torch.Tensor,
                       trj: torch.Tensor,
                       hparams: hofft_params,
                       B_compressed: Optional[int] = None,
                       normalize_coeffs: bool = True,
                       sparams: Optional[sparse_params] = None,
                       num_als_iter: int = 100,
                       spatial_mask: Optional[torch.Tensor] = None,
                       dcf: Optional[torch.Tensor] = None,
                       bparams: batching_params = batching_params()) -> linop:
    """
    Performs HOFFT decomposition using ALS and builds the HOFFT linop.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    mps : torch.Tensor
        coil sensitivities with shape (C, *im_size)
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    hparams : hofft_params
        HOFFT parameters.
    B_compressed : Optional[int]
        Number of compressed field bases to reduce computation
    normalize_coeffs : bool
        Whether to normalize the phase coefficients.
    sparams : Optional[sparse_params]
        Sparse decomposition parameters. If None, uses the full HOFFT forward model (memory intensive).
    num_als_iter : int
        Number of ALS iterations.
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    dcf : Optional[torch.Tensor]
        density compensation factor with shape (*trj_size)
    bparams : batching_params
        Batching parameters for the HOFFT linop.
        
    Returns
    -------
    linop : linop
        HOFFT linop taking in an image and returning k-space data
    """
    # Consts
    B = phis.shape[0]
    im_size = phis.shape[1:]
    d = len(im_size)
    os = hparams.os
    torch_dev = phis.device
    
    # ----------------- Process phase coefficients -----------------
    # Combine grid deviation phase to high order phase coefficients
    phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, os)
    trj_grd = (trj * os).round() / os
    phis_stack = torch.cat([phis_dev, phis], dim=0)
    alphas_stack = torch.cat([alphas_dev, alphas], dim=0)
    
    # Normalize phase coefficients
    if normalize_coeffs:
        phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack)
        spat, temp = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp)
        if B_compressed is None:
            B_compressed = B + d
        else:
            assert B_compressed <= B + d, "B_compressed must be less or equal to B + d"
        phis_nrm, alphas_nrm = whiten_phis_alphas(phis_nrm, alphas_nrm, 
                                                  B_compressed=B_compressed)
    else:
        phis_nrm = phis_stack
        alphas_nrm = alphas_stack
        phis_mp = torch.zeros(B, device=torch_dev, dtype=torch.float32)
        alphas_mp = torch.zeros(B, device=torch_dev, dtype=torch.float32)
        spat = torch.ones_like(phis_nrm[0]).type(torch.complex64)
        temp = torch.ones_like(alphas_nrm[0]).type(torch.complex64)
    
    # ----------------- HOFFT Decomposition -----------------
    
    # Full HOFFT Decomposition
    if sparams is None:
        spatial_factors, kern_weights = als_hofft(phis_nrm, alphas_nrm,
                                                  spatial_mask=spatial_mask,
                                                  hparams=hparams,
                                                  num_als_iter=num_als_iter)
        spatial_factors *= spat
        kern_weights *= temp
        A = hofft_linop(trj=trj_grd, mps=mps, dcf=dcf, 
                        kern_weights=kern_weights, 
                        spatial_factors=spatial_factors, 
                        os_grid=os, bparams=bparams)
    # Sparse HOFFT Decomposition
    else:
        # Least squares optimal sparse decomposition
        if sparams.interp_type == 'lstsq':
            ret = als_hofft_sparse_lstsq(phis_nrm, alphas_nrm, 
                                         spatial_mask=spatial_mask,
                                         hparams=hparams, 
                                         sparams=sparams, 
                                         num_als_iter=num_als_iter)
        # Smooth sparse decomposition
        else:
            ret = als_hofft_sparse_smooth(phis_nrm, alphas_nrm, 
                                          spatial_mask=spatial_mask,
                                          hparams=hparams, 
                                          sparams=sparams, 
                                          num_als_iter=num_als_iter)
            
        # Build sparse linop
        spatial_factors, compressed_kernels, sparse_idxs, sparse_coeffs = ret
        spatial_factors *= spat
        A = hofft_compressed_linop(trj=trj_grd, mps=mps, dcf=dcf,
                                   compressed_kernels=compressed_kernels,
                                   sparse_idxs=sparse_idxs,
                                   sparse_coeffs=sparse_coeffs,
                                   spatial_factors=spatial_factors,
                                   temporal_factors=temp,
                                   os_grid=os, bparams=bparams)
    
    return A

def als_hofft(phis: torch.Tensor,
              alphas: torch.Tensor,
              hparams: hofft_params,
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
    spatial_reduce_size = hparams.reduced_im_size
    solver = hparams.solver
    lamda = hparams.lamda
    spatial_batch_size = hparams.spatial_batch_size
    
    # Default spatial mask
    if spatial_mask is None:
        spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
    
    # Reduce phi size
    if spatial_reduce_size is not None:
        phis_reduced = reduce_spatial(phis, 
                                      im_size_low=spatial_reduce_size, 
                                      order=3)
        # kern_bases = build_kern_bases(kern_size, 
        #                               im_size=spatial_reduce_size, 
        #                               os=os).to(torch_dev)
        kern_bases = build_kern_bases(kern_size, 
                                      im_size=im_size, 
                                      os=os).to(torch_dev)
        kern_bases = reduce_spatial(kern_bases, 
                                    im_size_low=spatial_reduce_size, 
                                    order=3)
        spatial_mask = reduce_spatial(spatial_mask, 
                                      im_size_low=spatial_reduce_size, 
                                      order=3)
    else:
        kern_bases = build_kern_bases(kern_size, im_size, os).to(torch_dev)
        phis_reduced = phis
    
    # Make matvec phase model
    phase_model = hparams.matvec_type(phis_reduced, alphas, 
                                      **hparams.matvec_kwargs)
    
    # Initialize spatial factors
    spatial_factors = choose_init(phis_reduced, alphas, 
                                  hparams=hparams, 
                                  spatial_mask=spatial_mask,
                                  spatial_init=spatial_init)

    # ALS to solve for kernel weights and spatial factors
    if hparams.anderson_order is None:
        spatial_factors, kern_weights = als_iterations(phase_model, kern_bases, spatial_factors,
                                                       mask=spatial_mask,
                                                       max_iter=num_als_iter, 
                                                       solver=solver,
                                                       lamda=lamda,
                                                       spatial_batch_size=spatial_batch_size,
                                                       verbose=verbose)
    else:
        spatial_factors, kern_weights = als_anderson_iterations(phase_model, kern_bases, spatial_factors, 
                                                                anderson_order=hparams.anderson_order,
                                                                mask=spatial_mask,
                                                                solver=solver,
                                                                lamda=lamda,
                                                                max_iter=num_als_iter, verbose=verbose)
    kern_weights = kern_weights.reshape((L, *kern_size, *alphas.shape[1:]))

    # Expand phis 
    if spatial_reduce_size is not None:
        spatial_factors = expand_spatial(spatial_factors, 
                                         im_size_high=im_size,
                                         order=3)
    
    return spatial_factors, kern_weights

def als_hofft_sparse_lstsq(phis: torch.Tensor,
                           alphas: torch.Tensor,
                           hparams: hofft_params,
                           sparams: sparse_params,
                           spatial_mask: Optional[torch.Tensor] = None,
                           num_als_iter: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finds least squares optimal interpolation coefficients for the sparse HOFFT kernels.
    
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
    num_als_iter : int
        Number of ALS iterations.
        
    Returns
    -------
    spatial_factors : torch.Tensor
        Spatial factors with shape (L, *im_size)
    compressed_kernels : torch.Tensor
        Compressed HOFFT kernels with shape (L, *kern_size, Q)
    sparse_idxs : torch.Tensor
        Sparse indices with shape (S, *trj_size) in [0, Q)
    sparse_coeffs : torch.Tensor
        Sparse coefficients with shape (S, *trj_size)
    """
    im_size = phis.shape[1:]
    torch_dev = phis.device
    kern_size = hparams.kern_size
    os = hparams.os
    verbose = hparams.verbose
    reduced_im_size = hparams.reduced_im_size
    Q = sparams.Q
    S = sparams.S
    lamda = sparams.lamda
    spatial_subsample = sparams.spatial_subsample
    temporal_batch_size = sparams.temporal_batch_size
    
    # Default spatial mask
    if spatial_mask is None:
        spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)

    if reduced_im_size is not None:
        phis_reduced = reduce_spatial(phis, im_size_low=reduced_im_size, order=3)
        kern_bases = build_kern_bases(kern_size, reduced_im_size, os=os).to(torch_dev)
        # kern_bases = build_kern_bases(kern_size, im_size, os=os).to(torch_dev)
        # kern_bases = reduce_spatial(kern_bases, im_size_low=reduced_im_size, order=3)
        spatial_mask_red = reduce_spatial(spatial_mask, im_size_low=reduced_im_size, order=3)
    else:
        phis_reduced = phis
        kern_bases = build_kern_bases(kern_size, im_size, os=os).to(torch_dev)
        spatial_mask_red = spatial_mask

    spatial_factors, compressed_kernels, betas = K_alphas_init(
        phis_reduced, alphas,
        hparams=hparams,
        spatial_mask=spatial_mask_red,
        spatial_init_method='seg',
        num_als_iter=num_als_iter,
        K=Q,
        return_kernels=True,
    )

    sparse_inds, sparse_coeffs = lstsq_compressed_fixed_support(
        phis_reduced, alphas,
        spatial_factors=spatial_factors,
        compressed_kernels=compressed_kernels,
        kern_bases=kern_bases,
        betas=betas,
        sparsity=S,
        hparams=hparams,
        spatial_mask=spatial_mask_red,
        spatial_subsample=spatial_subsample,
        temporal_batch_size=temporal_batch_size,
        lamda=lamda,
        verbose=verbose,
    )
    
    # Reshape
    sparse_coeffs = sparse_coeffs.moveaxis(-1, 0)
    sparse_inds = sparse_inds.moveaxis(-1, 0)

    if reduced_im_size is not None:
        spatial_factors = expand_spatial(spatial_factors, im_size_high=im_size, order=3)

    return spatial_factors, compressed_kernels, sparse_inds, sparse_coeffs

def als_hofft_sparse_smooth(phis: torch.Tensor,
                            alphas: torch.Tensor,
                            hparams: hofft_params,
                            sparams: sparse_params,
                            spatial_mask: Optional[torch.Tensor] = None,
                            num_als_iter: int = 100,
                            d: float = 1.0,
                            p: float = 2.0,
                            eps: float = 1e-3,
                            d_grid: torch.Tensor = torch.logspace(-1, 0.7, 10),
                            p_grid: torch.Tensor = torch.linspace(1, 8.0, 8)
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Strategy 2.5 (see math_docs/sparse_fit.md): k_alphas ``H_comp`` + smooth
    distance-based sparse interpolation for ``C``.

    Same compressed kernels as ``sparse_fit_hofft_omp`` / ``sparse_fit_hofft_lstsq_support``,
    but replaces the per-signal OMP / least-squares solve with normalized
    RBF or inverse-distance weights to the S-nearest-beta support in alpha
    space -- no dictionary, Gram matrix, or phase-model matvec is needed, so
    this is considerably cheaper to compute.

    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    hparams : hofft_params
        HOFFT parameters.
    sparams : sparse_params
        Sparse decomposition parameters. sparams.Q sets the number of compressed
        kernels (representative betas); sparams.S sets the sparsity level.
    spatial_mask : Optional[torch.Tensor]
        Spatial mask with shape (*im_size)
    num_als_iter : int
        Number of ALS iterations for the 'k_alphas' decomposition.
    kernel : str
        'rbf' -> w(delta) = exp(-delta^2 / 2), or 'inv_dist' -> w(delta) = 1 / (delta^p + eps).
    d : float
        Distance-scaling hyperparameter (see math_docs/sparse_fit.md). Ignored
        (overwritten) if auto_tune is True.
    p : float
        Power for the inverse-distance kernel. Ignored (overwritten) if
        auto_tune is True and kernel == 'inv_dist'.
    eps : float
        Regularization for the inverse-distance kernel and coefficient normalization.
    auto_tune : bool
        If True, picks d (and p, for 'inv_dist') via ``sweep_smooth_interp_hyperparams``:
        the k_alphas spatial_factors are held fixed and compared against the
        exact ("full") HOFFT kernels on a validation subset of num_val
        trajectory points (see math_docs/sparse_fit.md).
    d_grid : tuple
        Candidate d values to sweep when auto_tune is True.
    p_grid : tuple
        Candidate p values to sweep when auto_tune is True and kernel == 'inv_dist'.
    num_val : int
        Number of held-out trajectory points used for auto_tune.

    Returns
    -------
    spatial_factors : torch.Tensor
        Spatial factors with shape (L, *im_size)
    compressed_kernels : torch.Tensor
        Compressed HOFFT kernels with shape (L, *kern_size, Q)
    sparse_inds : torch.Tensor
        Sparse indices with shape (S, *trj_size) in [0, Q)
    sparse_coeffs : torch.Tensor
        Sparse coefficients with shape (S, *trj_size)
    """
    im_size = phis.shape[1:]
    torch_dev = phis.device
    kern_size = hparams.kern_size
    os = hparams.os
    verbose = hparams.verbose
    reduced_im_size = hparams.reduced_im_size
    spatial_init = hparams.spatial_init
    Q = sparams.Q
    S = sparams.S
    num_valid = sparams.num_validation
    kernel = sparams.interp_type
    temporal_batch_size = sparams.temporal_batch_size
    assert kernel in ['rbf', 'inv_dist'], "Invalid interpolation type"

    if spatial_mask is None:
        spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)

    if reduced_im_size is not None:
        phis_reduced = reduce_spatial(phis, im_size_low=reduced_im_size, order=3)
        # kern_bases = build_kern_bases(kern_size, reduced_im_size, os=os).to(torch_dev)
        kern_bases = build_kern_bases(kern_size, im_size, os=os).to(torch_dev)
        kern_bases = reduce_spatial(kern_bases, im_size_low=reduced_im_size, order=3)
        spatial_mask_red = reduce_spatial(spatial_mask, im_size_low=reduced_im_size, order=3)
    else:
        phis_reduced = phis
        kern_bases = build_kern_bases(kern_size, im_size, os=os).to(torch_dev)
        spatial_mask_red = spatial_mask

    # k_alphas decomposition: spatial factors, compressed kernels H_comp, and
    # the Q representative betas (same as sparse_fit_hofft_omp / lstsq_support).
    spatial_factors, compressed_kernels, betas = K_alphas_init(
        phis_reduced, alphas,
        hparams=hparams,
        spatial_mask=spatial_mask_red,
        spatial_init_method=spatial_init,
        num_als_iter=num_als_iter,
        K=Q,
        return_kernels=True,
    )    

    # Tune d (and p) against a validation subset of exact HOFFT kernels, using
    # the already-fixed spatial_factors from the k_alphas decomposition above.
    if num_valid is not None:
        d, p, errors = sweep_smooth_interp_hyperparams(
            phis_reduced, alphas, spatial_factors, compressed_kernels, betas, kern_bases,
            sparsity=S, hparams=hparams, kernel=kernel,
            d_grid=d_grid, p_grid=p_grid, eps=eps, num_val=num_valid,
            spatial_mask=spatial_mask_red, verbose=verbose,
        )
        if verbose:
            msg = f'Strategy 2.5 auto-tune: picked d={d}'
            if kernel == 'inv_dist':
                msg += f', p={p}'
            print(f'{msg} (validation error {errors.min().item():.4g})')

    # Solve for the sparse coefficients via smooth distance-based weights (Strategy 2.5)
    sparse_inds, sparse_coeffs = smooth_sparse_coeffs(
        alphas, betas, sparsity=S,
        kernel=kernel, d=d, p=p, eps=eps,
        temporal_batch_size=temporal_batch_size,
    )
    
    # Expand spatial
    if reduced_im_size is not None:
        spatial_factors = expand_spatial(spatial_factors, im_size_high=im_size, order=3)

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