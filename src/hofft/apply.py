import torch
import numpy as np

from typing import Optional
from einops import einsum, rearrange

from mr_recon.utils import gen_grd, resize, quantize_data
from mr_recon.spatial import spatial_interp
from mr_recon.algs import eigen_decomp_operator, lin_solve
from mr_recon.imperfections.field import alpha_segementation
from mr_recon.fourier import fft, ifft, sigpy_nufft, matrix_nufft, torchkb_nufft, _torch_apodize
from mr_recon.linops import linop, type3_nufft_naive, type3_nufft
from mr_recon.dtypes import complex_dtype

from hofft.als import als_iterations, lstsq_temporal
from hofft.sgd import train_net_apod

__all__ = [
    'kb_nufft',
    'als_nufft',
    'als_hofft',
    'mlp_hofft',
]

def kb_weights_1d(x: torch.Tensor,
                  beta: float) -> torch.Tensor:
    """
    Kaiser Bessel kernel function in 1D
    
    Parameters:
    -----------
    x : torch.tensor
        input scaled between [-1, 1] arb shape
    beta : float
        beta parameter for kaiser bessel kernel
    
    Returns:
    ----------
    k : torch.tensor
        kaiser bessel evaluation kb(x) with same shape as x
    """
    x = beta * (1 - x**2) ** 0.5
    t = x / 3.75
    k1 = (  1
            + 3.5156229 * t**2
            + 3.0899424 * t**4
            + 1.2067492 * t**6
            + 0.2659732 * t**8
            + 0.0360768 * t**10
            + 0.0045813 * t**12)
    k2 = (  x**-0.5
            * torch.exp(x)
            * (
                0.39894228
                + 0.01328592 * t**-1
                + 0.00225319 * t**-2
                - 0.00157565 * t**-3
                + 0.00916281 * t**-4
                - 0.02057706 * t**-5
                + 0.02635537 * t**-6
                - 0.01647633 * t**-7
                + 0.00392377 * t**-8))
    return k1 * (x < 3.75) + k2 * (x >= 3.75)
 
def kb_apod_1d(x: torch.Tensor,
               beta: float,
               width: float) -> torch.Tensor:
    """
    1D apodization for the Kasier Bessel Kernel
    
    Parameters:
    -----------
    x : torch.Tensor <float>
        Arb shape signal between [-1/2, 1/2]
    beta : float
        beta parameter for kaiser bessel kernel
    width : float
        width of the kernel in pixels
        
    Returns:
    --------
    apod : torch.Tensor <float>
        apodization evaluated at input x
    """

    apod = (
        beta**2 - (np.pi * width * x) ** 2
    ) ** 0.5
    apod /= torch.sinh(apod)
    return apod
 
def kb_nufft(trj: torch.Tensor,
             im_size: tuple,
             kern_size: tuple,
             os: Optional[float] = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-apodization non-uniform FFT (NUFFT)
    
    Args:
    -----
    trj : torch.Tensor
        k-space trajectory with shape (*trj_size, d)
    im_size : tuple
        Size of the image to be reconstructed.
    kern_size : tuple
        Size of the kernel.
    os : Optional[float]
        Oversampling factor.
    
    Returns:
    --------
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
    kern = gen_grd(kern_size, kern_size).reshape((-1, d)).to(trj.device)
    kern -= kern.mean(dim=0)
    kdevs = (trj - (os * trj).round()/os)
    pts = (kdevs[..., None, :] * os - kern) / (width / 2)
    weights = torch.prod(kb_weights_1d(pts, beta), dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    
    # Reshape
    apods = apod[None,]
    weights = weights.moveaxis(-1, 0).reshape((1, *kern_size, *trj_size))
    weights = weights.type(complex_dtype)
    
    # Apply correction linear phase
    k_corr = torch.ones(d, device=trj.device) / os / 2
    phz = torch.exp(-2j * np.pi * einsum(rs, k_corr, '... d, d -> ...'))

    return weights, apods * phz

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
    
    Args:
    -----
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
    
    Returns:
    --------
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
    high_acc_nufft = matrix_nufft(solve_size)
    # high_acc_nufft = sigpy_nufft(solve_size, oversamp=2.0, width=6)
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
              kern_size: tuple,
              os: Optional[float] = 1.0,
              L: Optional[int] = 1,
              num_als_iter: Optional[int] = 100,
              init_method: Optional[str] = 'eigen',
              use_type3: Optional[bool] = True,
              verbose: Optional[bool] = True,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-apodization HOFFT model, allowing for arbitrary non-linear phase
    
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
    os : Optional[float]
        Oversampling factor.
    L : Optional[int]
        Number of apodization functions.
    num_als_iter : Optional[int]
        Number of ALS iterations.
    init_method : Optional[str]
        Method to initialize apodization functions. Options are 'eigen' or 'seg'.
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
    
    # Make kernel bases
    rs = gen_grd(solve_size).to(torch_dev)
    kern = gen_grd(kern_size, kern_size).to(torch_dev).reshape((-1, d)) / os
    phz = einsum(kern, rs, 'K D, ... D -> K ...')
    kern_bases = torch.exp(-2j * np.pi * phz)

    # Initialize apodization functions with eigen-vectors
    if use_type3:
        t3n = type3_nufft(phis, alphas, use_toep=True)
    else:
        t3n = type3_nufft_naive(phis, alphas)
    if init_method == 'eigen':
        x0 = torch.randn(solve_size, dtype=complex_dtype, device=torch_dev)
        apods, _ = eigen_decomp_operator(t3n.normal, x0, num_eigen=L, verbose=verbose)
    else:
        apods, _ = alpha_segementation(phis, alphas, L=L, L_batch_size=L, use_type3=use_type3)

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
              kern_size: tuple,
              os: Optional[float] = 1.0,
              L: Optional[int] = 1,
              opt_apods: Optional[bool] = True,
              epochs: Optional[int] = 100,
              use_type3: Optional[bool] = True,
              verbose: Optional[bool] = True,) -> tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
    """
    Multi-apodization HOFFT model, allowing for arbitrary non-linear phase
    
    Args:
    -----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *solve_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    im_size : tuple
        Size of the image to be reconstructed.
    kern_size : tuple
        Size of the kernel, must have the same number of dimensions as the image.
    os : Optional[float]
        Oversampling factor.
    L : Optional[int]
        Number of apodization functions.
    opt_apods : Optional[bool]
        If True, optimizes the apodization functions.
        If False, uses the initial apodization functions.
    num_als_iter : Optional[int]
        Number of ALS iterations.
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
    kern_model : nn.Module
        The learned kernel model.
    """
    # Consts
    solve_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    torch_dev = phis.device
    d = len(im_size)
    B = phis.shape[0]
    
    # Initialize apodization functions with eigen-vectors
    if use_type3:
        t3n = type3_nufft(phis, alphas, use_toep=True)
    else:
        t3n = type3_nufft_naive(phis, alphas)
    x0 = torch.randn(solve_size, dtype=complex_dtype, device=torch_dev)
    apods_init, _ = eigen_decomp_operator(t3n.normal, x0, num_eigen=L, verbose=verbose)
    
    weights, apods, kern_model = train_net_apod(phis, alphas, apods_init, kern_size, os, opt_apods=opt_apods, epochs=epochs)
    
    # Interpolate spatial funcs
    kwargs = {'order': 3, 'mode': 'nearest'}
    solve_size_tensor = torch.tensor(solve_size).to(torch_dev)
    spatial_crds = (gen_grd(im_size).to(torch_dev) + 0.5) * solve_size_tensor
    apods = spatial_interp(apods, spatial_crds, **kwargs)
    
    return weights, apods, kern_model

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
    
    