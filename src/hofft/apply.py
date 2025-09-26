import re
import torch
import numpy as np

from typing import Optional, Union
from einops import einsum, rearrange
from dataclasses import dataclass

from mr_recon.utils import gen_grd, resize, quantize_data
from mr_recon.spatial import spatial_interp
from mr_recon.algs import eigen_decomp_operator, lin_solve
from mr_recon.imperfections.field import alpha_segementation
from mr_recon.fourier import fft, ifft, sigpy_nufft, matrix_nufft, torchkb_nufft
from mr_recon.linops import linop, type3_nufft_naive, type3_nufft
from mr_recon.dtypes import complex_dtype

from hofft.apod_init import K_alphas_apod_init, alpha_seg_apod_init, eigen_apod_init
from hofft.als import als_iterations, lstsq_temporal
from hofft.sgd import train_net_apod
from hofft.kb import kb_weights_1d, kb_apod_1d, sample_kb_kernel, _gen_kern_bases
from hofft.model import hofft_params

__all__ = [
    'funcs_to_phase',
    'kb_nufft',
    'als_nufft',
    'als_hofft',
    'mlp_hofft',
]

def funcs_to_phase(weights: torch.Tensor,
                   apods: torch.Tensor,
                   os: Optional[float] = 1.0) -> torch.Tensor:
    """
    Convert weights and apodization functions to phase maps
    
    Args
    ----
    weights : torch.Tensor
        NUFFT kernel weights with shape (L, *kern_size, *trj_size)
    apods : torch.Tensor
        Apodization functions with shape (L, *im_size)
    os : Optional[float]
        Oversampling factor.
        
    Returns
    -------
    phz : torch.Tensor
        Phase maps with shape (*trj_size, *im_size)
    """
    # Consts
    kern_size = weights.shape[1:-1]
    im_size = apods.shape[1:]
    torch_dev = weights.device
    d = len(im_size)
    trj_size = weights.shape[(d+1):]
    L = weights.shape[0]
    K = np.prod(kern_size)
    T = np.prod(trj_size)
    
    # Flatten
    weights_flt = weights.reshape((L, K, T))
    
    # Make kernel bases
    rs = gen_grd(im_size).to(torch_dev)
    kern = gen_grd(kern_size, kern_size).to(torch_dev).reshape((-1, d)) / os
    phz = einsum(kern, rs, 'K D, ... D -> K ...')
    kern_bases = torch.exp(-2j * np.pi * phz)
    
    # Apply
    phz = einsum(weights_flt, kern_bases, 'L K T, K ... -> L T ...')
    phz = einsum(phz, apods, 'L T ..., L ... -> T ...')
    
    return phz.reshape((*trj_size, *im_size))

def kb_nufft(trj: torch.Tensor,
             im_size: tuple,
             kern_size: tuple,
             os: Optional[float] = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-apodization non-uniform FFT (NUFFT)
    
    Args
    ----
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    im_size : tuple
        Size of the image to be reconstructed.
    kern_size : tuple
        Size of the kernel.
    os : Optional[float]
        Oversampling factor.
    
    Returns
    -------
    weights : torch.Tensor
        KB-NUFFT kernel weights with shape (1, *kern_size, *trj_size)
    apods : torch.Tensor
        KB-NUFFT apodization functions with shape (1, *im_size)
    """
    # Consts
    d = len(im_size)
    trj_size = trj.shape[:-1]
    width = kern_size[0]
    for i in range(1, len(kern_size)):
        assert kern_size[i] == width, "Kernel size must be the same in all dimensions"
    beta = torch.pi * (((width / os) * (os - 0.5))**2 - 0.8)**0.5
    if (((width / os) * (os - 0.5))**2 - 0.8) < 0:
        beta = 1.0
    
    # Apodization function
    rs = gen_grd(im_size).to(trj.device)
    apod = kb_apod_1d(rs / os, beta, width).prod(dim=-1)
    
    # Kernel weights
    kdevs = trj - (os * trj).round()/os
    weights = sample_kb_kernel(kdevs, kern_size, os, beta)
    
    # Scaling factor correction
    # kdev = torch.zeros((1, d), dtype=torch.float32, device=trj.device)
    # weights_scale = sample_kb_kernel(kdev, kern_size, os, beta).flatten()
    # kern_bases = _gen_kern_bases(im_size, kern_size, os).to(trj.device)
    # ones_est = einsum(weights_scale.type(complex_dtype), apod * kern_bases, 
    #                   'L, L ... -> ...')
    # # apod = apod / ones_est
    # apod = torch.exp(-ones_est.angle()) * apod / ones_est.abs().mean()
    apod /= width ** d
    
    # Reshape
    apods = apod[None,]
    weights = weights[None,].type(complex_dtype)
    
    # # Apply correction linear phase
    # k_corr = (width%2==0)*torch.ones(d, device=trj.device) / os / 2
    # phz = torch.exp(-2j * np.pi * einsum(rs, k_corr, '... d, d -> ...'))
    # apods = apods * phz

    return weights, apods

def als_nufft(trj: torch.Tensor,
              im_size: tuple,
              kern_size: tuple,
              os: Optional[float] = 1.0,
              L: Optional[int] = 1,
              num_als_iter: Optional[int] = 100,
              init_method: Optional[str] = 'eigen',
              solve_size: Optional[tuple] = None,
              verbose: Optional[bool] = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-apodization non-uniform FFT (NUFFT)
    
    Args
    ----
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    im_size : tuple
        Size of the image to be reconstructed.
    kern_size : tuple
        Size of the kernel.
    os : Optional[float]
        Oversampling factor.
    L : Optional[int]
        Number of apodization functions.
    num_als_iter : Optional[int]
        Number of ALS iterations.
    init_method : Optional[str]
        Method to initialize apodization functions. Options are 'eigen' or 'seg'.
    solve_size : Optional[tuple]
        solves problem on a smaller grid size. Defaults to 50 in each dimension.
    verbose : Optional[bool]
        If True, prints progress of ALS iterations.
    
    Returns
    -------
    weights : torch.Tensor
        NUFFT kernel weights with shape (L, *kern_size, *trj_size)
    apods : torch.Tensor
        Apodization functions with shape (L, *im_size)
    """
    # Consts
    d = trj.shape[-1]
    torch_dev = trj.device
    trj_size = trj.shape[:-1]
    solve_size = (50,)*d if solve_size is None else solve_size
    
    # Spatial and kspace bases
    rs = gen_grd(solve_size).to(torch_dev)
    kdevs = gen_grd(solve_size).to(torch_dev) / os
    
    # Make kernel bases
    kern = gen_grd(kern_size, kern_size).to(torch_dev).reshape((-1, d)) / os
    phz = einsum(kern, rs, 'K D, ... D -> K ...')
    kern_bases = torch.exp(-2j * np.pi * phz)

    # Initialize apodization functions with eigen-vectors
    # high_acc_nufft = matrix_nufft(solve_size)
    high_acc_nufft = sigpy_nufft(solve_size, oversamp=2.0, width=6)
    # high_acc_nufft = torchkb_nufft(solve_size, torch_dev)
    kdevs_rs = high_acc_nufft.rescale_trajectory(kdevs)
    if init_method == 'eigen':
        toep_kerns = high_acc_nufft.calc_teoplitz_kernels(kdevs_rs[None])[0] # *solve_size_os
        solve_size_os = toep_kerns.shape
        def normal_op(x):
            N = x.shape[0]
            x = resize(x, (N, *solve_size_os))
            x = fft(x, dim=tuple(range(-d, 0)))
            x = x * toep_kerns
            x = ifft(x, dim=tuple(range(-d, 0)))
            x = resize(x, (N, *solve_size))
            return x
        x0 = torch.randn(solve_size, dtype=complex_dtype, device=torch_dev)
        apods, _ = eigen_decomp_operator(normal_op, x0, num_eigen=L, verbose=verbose)
    else:
        apods, _ = alpha_segementation(rs.moveaxis(-1, 0), kdevs.moveaxis(-1,0), L=L, L_batch_size=L, use_type3=False)

    # ALS to solve for weights and apods
    class phase_model(linop):
        def __init__(self):
            super().__init__(solve_size, solve_size)
        def forward(self, x):
            return high_acc_nufft.forward(x[None,], kdevs_rs[None,])[0]
        def adjoint(self, y):
            return high_acc_nufft.adjoint(y[None,], kdevs_rs[None,])[0]
    t3n = phase_model()
    # t3n = type3_nufft_naive(rs.moveaxis(-1, 0), kdevs.moveaxis(-1, 0))
    # t3n = type3_nufft(rs.moveaxis(-1, 0), kdevs.moveaxis(-1, 0))
    weights, apods = als_iterations(t3n, kern_bases, apods, max_iter=num_als_iter, verbose=verbose)

    # Interpolate spatial funcs
    kwargs = {'order': 3, 'mode': 'nearest'}
    solve_size_tensor = torch.tensor(solve_size).to(torch_dev)
    spatial_crds = (gen_grd(im_size).to(torch_dev) + 0.5) * solve_size_tensor
    apods = spatial_interp(apods, spatial_crds, **kwargs)

    # Interpolate temporal functions
    trj_dev = trj - (os * trj).round()/os
    temporal_crds = (0.5 + trj_dev * os) * solve_size_tensor
    weights = spatial_interp(weights.reshape((-1, *solve_size)), temporal_crds, **kwargs)
    weights = weights.reshape((L, *kern_size, *trj_size))
    
    return weights, apods

def als_hofft(phis: torch.Tensor,
              alphas: torch.Tensor,
              im_size: tuple,
              hparams: hofft_params,
              num_als_iter: Optional[int] = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-apodization HOFFT model, allowing for arbitrary non-linear phase
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *solve_size)
        where solve_size is likely smaller than the image size, but has the same number of dimensions.
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    im_size : tuple
        Size of the image to be reconstructed.
    hparams : hofft_params
        HOFFT parameters.
    num_als_iter : Optional[int]
        Number of ALS iterations.
        
    Returns
    -------
    weights : torch.Tensor
        NUFFT kernel weights with shape (L, *kern_size, *trj_size)
    apods : torch.Tensor
        Apodization functions with shape (L, *im_size)
    """
    # Consts
    trj_size = alphas.shape[1:]
    solve_size = phis.shape[1:]
    torch_dev = phis.device
    d = len(im_size)
    kern_size = hparams.kern_size
    os = hparams.os
    L = hparams.L
    apods_init = hparams.apods_init
    use_type3 = hparams.use_type3
    verbose = hparams.verbose
    
    # Make kernel bases
    rs = gen_grd(solve_size).to(torch_dev)
    kern = gen_grd(kern_size, kern_size).to(torch_dev).reshape((-1, d)) / os
    phz = einsum(kern, rs, 'K D, ... D -> K ...')
    kern_bases = torch.exp(-2j * np.pi * phz)
    
    # Make type3 object
    if use_type3:
        t3n = type3_nufft(phis, alphas, use_toep=True)
    else:
        t3n = type3_nufft_naive(phis, alphas)

    # Initialize apodization functions
    if isinstance(apods_init, torch.Tensor):
        apods = apods_init
    elif isinstance(apods_init, str):
        if 'eig' in apods_init:
            apods = eigen_apod_init(phis, alphas, hparams)
        elif 'seg' in apods_init:
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

    # ALS to solve for weights and apods
    weights, apods = als_iterations(t3n, kern_bases, apods, max_iter=num_als_iter, verbose=verbose)

    # Interpolate spatial funcs
    kwargs = {'order': 3, 'mode': 'nearest'}
    solve_size_tensor = torch.tensor(solve_size).to(torch_dev)
    spatial_crds = (gen_grd(im_size).to(torch_dev) + 0.5) * solve_size_tensor
    apods = spatial_interp(apods, spatial_crds, **kwargs)
    
    # Reshape weights
    weights = weights.reshape((L, *kern_size, *trj_size))
    
    return weights, apods

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

def coil_hofft(phis: torch.Tensor,
               alphas: torch.Tensor,
               mps: torch.Tensor,
               hparams: hofft_params,
               num_als_iter: Optional[int] = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes HOFFT coefficients when coil sensitivity maps are used.
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *solve_size)
        where solve_size is likely smaller than the image size, but has the same number of dimensions.
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    mps : torch.Tensor
        Coil sensitivity maps with shape (C, *im_size)
    hparams : hofft_params
        HOFFT parameters.
    num_als_iter : Optional[int]
        Number of ALS iterations.
        
    Returns
    -------
    weights : torch.Tensor
        NUFFT kernel weights with shape (C, C, L, *kern_size, *trj_size)
    apods : torch.Tensor
        Apodization functions with shape (L, *im_size)
    """
    # Consts
    im_size = mps.shape[1:]
    trj_size = alphas.shape[1:]
    solve_size = phis.shape[1:]
    torch_dev = phis.device
    d = len(im_size)
    kern_size = hparams.kern_size
    os = hparams.os
    L = hparams.L
    apods_init = hparams.apods_init
    use_type3 = hparams.use_type3
    verbose = hparams.verbose
    
    # Make kernel bases
    rs = gen_grd(solve_size).to(torch_dev)
    kern = gen_grd(kern_size, kern_size).to(torch_dev).reshape((-1, d)) / os
    phz = einsum(kern, rs, 'K D, ... D -> K ...')
    kern_bases = torch.exp(-2j * np.pi * phz)
    
    # Make type3 object
    if use_type3:
        t3n = type3_nufft(phis, alphas, use_toep=True)
    else:
        t3n = type3_nufft_naive(phis, alphas)

    

def idonttrustthis(phis: torch.Tensor,
                   alphas: torch.Tensor,
                   im_size: tuple,
                   kern_size: tuple,
                   L: int,
                   rank : int,
                   os: Optional[float] = 1.0,
                   use_type3: Optional[bool] = True,
                   verbose: Optional[bool] = True,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Will prob delete this function in exactly 20 minutes.
    
    Args:
    -----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *solve_size)
        where solve_size is likely smaller than the image size, but has the same number of dimensions.
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    im_size : tuple
        Size of the image to be reconstructed.
    kern_size : tuple
        Size of the kernel, must have the same number of dimensions as the image.
    L : int
        Number of apodization functions.
    rank : int
        Rank of SVD step, needs ot be greater than L * prod(kern_size)
    os : Optional[float]
        Oversampling factor.
    use_type3 : Optional[bool]
        If True, uses type3 nufft for the forward and adjoint operations.
    verbose : Optional[bool]
        If True, prints progress of ALS iterations.
    
    Returns:
    --------
    weights : torch.Tensor
        NUFFT kernel weights with shape (L, *kern_size, *trj_size)
    apods : torch.Tensor
        Apodization functions with shape (L, *im_size)
    """
    # Consts
    trj_size = alphas.shape[1:]
    solve_size = phis.shape[1:]
    torch_dev = phis.device
    d = len(im_size)
    B = phis.shape[0]
    K = np.prod(kern_size)
    assert rank >= L * K, "Rank must be greater than L * prod(kern_size)"
    
    # Make kernel bases
    rs = gen_grd(solve_size).to(torch_dev)
    kern = gen_grd(kern_size, kern_size).to(torch_dev).reshape((-1, d)) / os
    phz = einsum(kern, rs, 'K D, ... D -> K ...')
    kern_bases = torch.exp(-2j * np.pi * phz)
    
    # Spatial eigen-vectors
    if use_type3:
        t3n = type3_nufft(phis, alphas, use_toep=True)
    else:
        t3n = type3_nufft_naive(phis, alphas)
    x0 = torch.randn(solve_size, dtype=complex_dtype, device=torch_dev)
    spatial_vecs, _ = eigen_decomp_operator(t3n.normal, x0, num_eigen=rank, verbose=verbose,
                                            num_iter=100)
    
    # Solve for kernel coefficiets for each spatial vector
    E = kern_bases.reshape((K, -1)).T # R K
    B = spatial_vecs.reshape((rank, -1)).T # R r
    EHE = E.H @ E # K K
    EHB = E.H @ B # K r
    coeffs = lin_solve(EHE, EHB, lamda=0.0, solver='solve') # K r
    
    # Cluster coefficients into L groups
    coeffs_reim = torch.cat([coeffs.real, coeffs.imag], dim=0)
    cents, idxs = quantize_data(coeffs_reim.T, L, method='cluster')
    cents = cents[:, :K] + 1j * cents[:, K:]
    
    # For each cluster, solve for apodization functions
    apods = []
    for i in range(L):
        
        print((idxs == i).sum(), f'l = {i+1}')
        
        # Grab data for this group
        inds_i = torch.argwhere(idxs == i)[:, 0] # g
        vi = B[:, inds_i] # R g
        ci = coeffs[:, inds_i] # K g
        
        # Solve least squares diagonal form
        Eci = E @ ci # R g
        Eci_vi = (Eci.conj() * vi).sum(dim=-1) # R
        Eci_Eci = (Eci.conj() * Eci).sum(dim=-1) # R
        apods.append(Eci_vi / (Eci_Eci + 1e-3))
    apods = torch.stack(apods, dim=0).reshape((L, *solve_size))
    
    # Run a temporal least squares, and done.
    weights = lstsq_temporal(t3n, kern_bases, apods)
    weights = weights.reshape((L, *kern_size, *trj_size))
    
    # Interpolate spatial funcs
    kwargs = {'order': 3, 'mode': 'nearest'}
    solve_size_tensor = torch.tensor(solve_size).to(torch_dev)
    spatial_crds = (gen_grd(im_size).to(torch_dev) + 0.5) * solve_size_tensor
    apods = spatial_interp(apods, spatial_crds, **kwargs)
    
    return weights, apods
    
def also_this_one(phis: torch.Tensor,
                  alphas: torch.Tensor,
                  im_size: tuple,
                  kern_size: tuple,
                  L: int,
                  rank : int,
                  os: Optional[float] = 1.0,
                  use_type3: Optional[bool] = True,
                  verbose: Optional[bool] = True,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    I also don't trust this, but at least this one is not a chatGPT idea.
    
    Args:
    -----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *solve_size)
        where solve_size is likely smaller than the image size, but has the same number of dimensions.
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    im_size : tuple
        Size of the image to be reconstructed.
    kern_size : tuple
        Size of the kernel, must have the same number of dimensions as the image.
    L : int
        Number of apodization functions.
    rank : int
        Rank of SVD step, needs ot be greater than L * prod(kern_size)
    os : Optional[float]
        Oversampling factor.
    use_type3 : Optional[bool]
        If True, uses type3 nufft for the forward and adjoint operations.
    verbose : Optional[bool]
        If True, prints progress of ALS iterations.
    
    Returns:
    --------
    weights : torch.Tensor
        NUFFT kernel weights with shape (L, *kern_size, *trj_size)
    apods : torch.Tensor
        Apodization functions with shape (L, *im_size)
    """
    # Consts
    trj_size = alphas.shape[1:]
    solve_size = phis.shape[1:]
    torch_dev = phis.device
    d = len(im_size)
    B = phis.shape[0]
    K = np.prod(kern_size)
    assert rank >= L * K, "Rank must be greater than L * prod(kern_size)"
    
    # Make kernel bases
    rs = gen_grd(solve_size).to(torch_dev)
    kern = gen_grd(kern_size, kern_size).to(torch_dev).reshape((-1, d)) / os
    phz = einsum(kern, rs, 'K D, ... D -> K ...')
    kern_bases = torch.exp(-2j * np.pi * phz)
    
    # Spatial eigen-vectors
    if use_type3:
        t3n = type3_nufft(phis, alphas, use_toep=True)
    else:
        t3n = type3_nufft_naive(phis, alphas)
    x0 = torch.randn(solve_size, dtype=complex_dtype, device=torch_dev)
    # apods, _ = eigen_decomp_operator(t3n.normal, x0, num_eigen=L, verbose=verbose,
    #                                  num_iter=100)
    spatial_vecs, _ = eigen_decomp_operator(t3n.normal, x0, num_eigen=rank, verbose=verbose,
                                            num_iter=100)
    
    # Define linop 
    def sym_op(x):
        # x has shape L *solve_size
        xk = einsum(x, kern_bases, 'L ..., K ... -> L K ...')
        xk_proj = einsum(xk, spatial_vecs.conj(), 'L K ..., r ... -> L K r')
        xk_back = einsum(xk_proj, spatial_vecs, 'L K r, r ... -> L K ...')
        x_back = einsum(xk_back, kern_bases.conj(), 'L K ..., K ... -> L ...')
        return K * x - x_back
    
    apods, evals = eigen_decomp_operator(sym_op, x0, num_eigen=L, verbose=verbose,
                                       num_iter=100, largest=False)
    
    # Run a temporal least squares, and done.
    weights = lstsq_temporal(t3n, kern_bases, apods)
    weights = weights.reshape((L, *kern_size, *trj_size))
    
    # Interpolate spatial funcs
    kwargs = {'order': 3, 'mode': 'nearest'}
    solve_size_tensor = torch.tensor(solve_size).to(torch_dev)
    spatial_crds = (gen_grd(im_size).to(torch_dev) + 0.5) * solve_size_tensor
    apods = spatial_interp(apods, spatial_crds, **kwargs)
    
    return weights, apods
    
    