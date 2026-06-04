import torch
import numpy as np

from tqdm import tqdm
from dataclasses import dataclass, field
from einops import rearrange, einsum
from typing import Optional, Union
from einops import einsum

from mr_recon.utils import gen_grd
from mr_recon.algs import lin_solve
from mr_recon.dtypes import complex_dtype

from .matvec import matvec, matvec_naive
from .kernel_regressors import kernel_regressor


__all__ = [
    'hofft_params',
    'build_kern_bases',
    'funcs_to_phase',
    'als_iterations',
    'als_anderson_iterations',
    'lstsq_spatial',
    'lstsq_temporal',
]

@dataclass
class hofft_params: 
    kern_size: tuple
    os: float
    L: int
    reduced_im_size: Optional[tuple] = None
    matvec_type: matvec = matvec_naive
    matvec_kwargs: dict = field(default_factory=dict)
    spatial_init: Union[torch.Tensor, str] = 'seg'
    anderson_order: Optional[int] = None
    kalpha_method: str = 'maxmin'
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
    reduced_im_size: Optional[tuple]
        Optional low resolution size for performing the decomposition
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
        'ones' - uses ones to initialize spatial factors
        'seg' - uses segmentation method to initialize spatial factors
        'eigen' - uses eigen-decomposition method to initialize spatial factors
        'k_alphas' - uses K representative alphas to initialize spatial factors
        If torch.Tensor:
        Initial spatial factors with shape (L, *solve_size)
    anderson_order : Optional[int]
        Order of the Anderson acceleration. Only used if anderson_order is not None.
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
    spatial_factors : torch.Tensor
        spatial factors with shape (L, *im_size)
    kernel_weights : torch.Tensor
        kernel weights with shape (L, K, *trj_size)
    """
        
    # Default mask
    if mask is None:
        mask = torch.ones_like(spatial_factors_init[0])
        
    # Weights only 
    if max_iter == 0:
        kernel_weights = lstsq_temporal(phase_model, kern_bases, spatial_factors_init, mask=mask)
        return spatial_factors_init, kernel_weights
    
    # Stopping criteria
    kwargs_allclose = {'atol': 0.0, 'rtol': 1e-3}
    
    # Momentum term
    # momentum = lambda k : k / (k + 3)
    # momentum = lambda k : 0.8
    momentum = lambda k : 0.0
    k0 = 0
    
    # Least squares parameters
    lamda = 1e0 * 0
    # solver = 'solve'
    solver = 'pinv'
    
    # ALS till max_iter
    spatial_factors_prev = spatial_factors_init
    kernel_weights_prev = None
    tbar = tqdm(range(max_iter), 'ALS iterations', disable=not verbose)
    for k in tbar:
        
        # ALS weight updates
        kernel_weights = lstsq_temporal(phase_model, kern_bases, spatial_factors_prev, 
                                        mask=mask,
                                        lamda=lamda,
                                        kernel_weights_prev=kernel_weights_prev,
                                        solver=solver)
        if k > k0:
            beta = momentum(k)
            kernel_weights = kernel_weights + beta * (kernel_weights_prev - kernel_weights)
        kernel_weights = kernel_weights.nan_to_num(0.0)
        
        # ALS apodization updates
        spatial_factors = lstsq_spatial(phase_model, kern_bases, kernel_weights, 
                                        mask=mask,
                                        lamda=lamda,
                                        spatial_factors_prev=spatial_factors_prev,
                                        solver=solver)
        spatial_factors = spatial_factors.nan_to_num(0.0)
        if k > k0:
            spatial_factors = spatial_factors + beta * (spatial_factors_prev - spatial_factors)
        
        # TODO REMOVEME
        # Check convergence
        if k > 0 and k % 10 == 0:
            if torch.allclose(kernel_weights, kernel_weights_prev, **kwargs_allclose) and \
               torch.allclose(spatial_factors, spatial_factors_prev, **kwargs_allclose):
                break
        
        # Update previous values
        kernel_weights_prev = kernel_weights
        spatial_factors_prev = spatial_factors
        
    return spatial_factors, kernel_weights

def als_iterations_tempinit(phase_model: matvec, 
                            kern_bases: torch.Tensor,
                            kernel_weights_init: torch.Tensor,
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
    kernel_weights_init : torch.Tensor
        initial kernel weights with shape (L, K, *trj_size)
    mask : torch.Tensor, optional
        image weighting mask with shape im_size
    max_iter : int, optional
        maximum number of iterations
    verbose : bool, optional
        whether to print progress
    
    Returns
    -------
    spatial_factors : torch.Tensor
        spatial factors with shape (L, *im_size)
    kernel_weights : torch.Tensor
        kernel weights with shape (L, K, *trj_size)
    """
        
    # Default mask
    if mask is None:
        mask = torch.ones_like(kern_bases[0])
        
    # Weights only 
    if max_iter == 0:
        spatial_factors = lstsq_spatial(phase_model, kern_bases, kernel_weights_init, mask=mask)
        return spatial_factors, kernel_weights_init
    
    # Stopping criteria
    kwargs_allclose = {'atol': 0.0, 'rtol': 1e-3}
    
    # Momentum term
    momentum = lambda k : 0.0
    k0 = 0
    
    # Least squares parameters
    lamda = 1e0 * 0
    # solver = 'solve'
    solver = 'pinv'
    
    # ALS till max_iter
    spatial_factors_prev = None
    kernel_weights_prev = kernel_weights_init
    tbar = tqdm(range(max_iter), 'ALS iterations', disable=not verbose)
    for k in tbar:
        
        # ALS spatial updates
        spatial_factors = lstsq_spatial(phase_model, kern_bases, kernel_weights_prev, 
                                        mask=mask,
                                        lamda=lamda,
                                        spatial_factors_prev=spatial_factors_prev,
                                        solver=solver)
        spatial_factors = spatial_factors.nan_to_num(0.0)
        if k > k0:
            spatial_factors = spatial_factors + beta * (spatial_factors_prev - spatial_factors)
        
        # ALS temporal updates
        kernel_weights = lstsq_temporal(phase_model, kern_bases, spatial_factors, 
                                        mask=mask,
                                        lamda=lamda,
                                        kernel_weights_prev=kernel_weights_prev,
                                        solver=solver)
        if k > k0:
            beta = momentum(k)
            kernel_weights = kernel_weights + beta * (kernel_weights_prev - kernel_weights)
        kernel_weights = kernel_weights.nan_to_num(0.0)
        
        # Update previous values
        kernel_weights_prev = kernel_weights
        spatial_factors_prev = spatial_factors
        
    return spatial_factors, kernel_weights
  
def als_anderson_iterations(phase_model: matvec, 
                            kern_bases: torch.Tensor,
                            spatial_factors_init: torch.Tensor,
                            anderson_order: int = 3,
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
    anderson_order : int
        order of the Anderson acceleration
    mask : torch.Tensor, optional
        image weighting mask with shape im_size
    max_iter : int, optional
        maximum number of iterations
    verbose : bool, optional
        whether to print progress
    
    Returns
    -------
    spatial_factors : torch.Tensor
        spatial factors with shape (L, *im_size)
    kernel_weights : torch.Tensor
        kernel weights with shape (L, K, *trj_size)
    """
        
    # Default mask
    if mask is None:
        mask = torch.ones_like(spatial_factors_init[0])
        
    # First iteration
    kernel_weights = lstsq_temporal(phase_model, kern_bases, spatial_factors_init, mask=mask)
    kernel_weights = kernel_weights.nan_to_num(0.0)
    if max_iter == 0:
        return kernel_weights, spatial_factors_init
    spatial_factors = lstsq_spatial(phase_model, kern_bases, kernel_weights, mask=mask)
    spatial_factors = spatial_factors.nan_to_num(0.0)
    
    # Initialize Anderson acceleration
    xs = torch.zeros((anderson_order, kernel_weights.numel() + spatial_factors.numel()), 
                     dtype=kernel_weights.dtype, device=kernel_weights.device)
    xs[-1] = torch.cat((kernel_weights.flatten(), spatial_factors.flatten()))
    fs = torch.zeros_like(xs)
    
    # Stopping criteria
    kwargs_allclose = {'atol': 0.0, 'rtol': 1e-3}
    
    # Iteration in convenient form
    def f(x):
        kws = x[:kernel_weights.numel()].reshape(kernel_weights.shape)
        sfs = x[kernel_weights.numel():].reshape(spatial_factors.shape)
        
        kws_next = lstsq_temporal(phase_model, kern_bases, sfs, mask=mask)
        kws_next = kws_next.nan_to_num(0.0)
        sfs_next = lstsq_spatial(phase_model, kern_bases, kws_next, mask=mask)
        sfs_next = sfs_next.nan_to_num(0.0)
        
        fx = torch.cat((kws_next.flatten(), sfs_next.flatten()))
        return fx
    
    # ALS till max_iter
    tbar = tqdm(range(max_iter), 'Anderson ALS Iterations', disable=not verbose)
    for k in tbar:
        
        order = min(k+1, anderson_order)
        
        # Compute f(x_k)
        fk = f(xs[-1])
        fs[:-1] = fs[1:].clone()
        fs[-1] = fk
        
        # Solve for x_{k+1} via anderson minimization
        G = (fs - xs)[-order:].T # (M+N) x (anderson_order)
        # G = torch.vstack((G.real, G.imag))
        A = torch.zeros((order+1, order+1), dtype=torch.float32, device=G.device)
        A[:-1, :-1] = (G.H @ G).real
        A[:-1, -1] = 1.0
        A[-1, :-1] = 1.0
        b = torch.zeros((order+1), dtype=torch.float32, device=G.device)
        b[-1] = 1.0
        coeffs = torch.linalg.solve(A, b)[:-1]
        xk1 = fs[-order:].T @ coeffs.type(torch.complex64)
        
        # Update xs
        xs[:-1] = xs[1:].clone()
        xs[-1] = xk1
        
        # TODO REMOVEME
        # Check convergence
        if k > 0 and k % 10 == 0:
            if torch.allclose(xs[-2], xs[-1], **kwargs_allclose):
                break
            
    kernel_weights = xs[-1][:kernel_weights.numel()].reshape(kernel_weights.shape)
    spatial_factors = xs[-1][kernel_weights.numel():].reshape(spatial_factors.shape)
        
    return spatial_factors, kernel_weights

def als_compressed(spatial_bases: torch.Tensor, 
                   kern_bases: torch.Tensor, 
                   spatial_factors_init: torch.Tensor,
                   mask: Optional[torch.Tensor] = None, 
                   max_iter: int = 100,
                   spatial_batch_size: Optional[int] = None,
                   temporal_batch_size: Optional[int] = None,
                   solver: Optional[str] = 'pinv', 
                   lamda: Optional[float] = 0.0,
                   verbose: bool = True,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function optimizes for the kernel weights and spatial factors given 
    a compression model for the spatio-temporal phase.
    
    Args
    ----
    spatial_bases : torch.Tensor
        spatial bases 'b' with shape (Q, *im_size)
    kern_bases : torch.Tensor
        kernel bases with shape (K, *im_size)
    spatial_factors_init : torch.Tensor
        initial spatial factors with shape (L, *im_size)
    mask : torch.Tensor, optional
        image weighting mask with shape (*im_size)
    max_iter : int, optional
        maximum number of iterations
    spatial_batch_size : int, optional
        batch size over spatial dimension
    temporal_batch_size : int, optional
        batch size over temporal dimension
    solver : str, optional
        solver for least squares
    lamda : float, optional, optional
        regularization parameter for least squares
    verbose : bool, optional
        whether to print progress
        
    Returns
    -------
    spatial_factors : torch.Tensor
        spatial factors with shape (L, *im_size)
    compressed_kernels: torch.Tensor
        kernel basis weights with shape (L, K, Q)
    """
    # Consts
    Q = spatial_bases.shape[0]
    K = kern_bases.shape[0]
    L = spatial_factors_init.shape[0]
    torch_dev = spatial_bases.device
    im_size = kern_bases.shape[1:]
    dtype = spatial_bases.dtype
    
    # Flatten spatial and temporal axes
    spatial_bases_flt = rearrange(spatial_bases, 'Q ... -> Q (...)')
    kern_bases_flt = rearrange(kern_bases, 'K ... -> K (...)')
    spatial_factors_flt = rearrange(spatial_factors_init, 'L ... -> L (...)')
    N = spatial_bases_flt.shape[1]
    assert N == np.prod(im_size)
    
    # Default
    if spatial_batch_size is None:
        spatial_batch_size = N
    if temporal_batch_size is None:
        temporal_batch_size = N
    if mask is None:
        mask = torch.ones((N,), dtype=complex_dtype, device=torch_dev)
    mask_flt = mask.flatten()
    assert mask_flt.shape[0] == N
    
    # Helpers
    def _lstsq_temporal(spatial_factors: torch.Tensor, 
                        spatial_bases: torch.Tensor,
                        kern_bases: torch.Tensor,
                        mask: torch.Tensor,
                        N_batch_size: int,
                        solver: str = 'pinv',
                        lamda: float = 0.0,) -> torch.Tensor:
        """
        This function optimizes for the kernel basis weights given fixed spatial factors.
        
        Args
        ----
        spatial_factors : torch.Tensor
            spatial factors with shape (L, N)
        spatial_bases : torch.Tensor
            spatial bases with shape (Q, N)
        kern_bases : torch.Tensor
            kernel bases with shape (K, N)
        mask : torch.Tensor
            image weighting mask with shape (N,)
        lk_batch_size : int
            batch size over L x K dimension
        solver : str
            solver for least squares
        lamda : float
            regularization parameter for least squares
            
        Returns
        -------
        compressed_kernels : torch.Tensor
            compressed kernel weights with shape (L, K, Q)
        """
        # Combine kernel and spatial bases
        bases = einsum(kern_bases, spatial_factors, 'K N, L N -> L K N') * mask
        bases = rearrange(bases, 'L K N -> (L K) N')
        
        # Matrices to compute
        M = torch.zeros((L * K, L * K), dtype=dtype, device=torch_dev)
        f = torch.zeros((L * K, Q), dtype=dtype, device=torch_dev)
        
        # Build matrices in batches
        for n1 in range(0, N, N_batch_size):
            n2 = min(n1 + N_batch_size, N)

            M += einsum(bases[:, n1:n2].conj(), bases[:, n1:n2], 
                                'lk1 N, lk2 N -> lk1 lk2')
            f += einsum(bases[:, n1:n2].conj() * mask[n1:n2], spatial_bases[:, n1:n2], 
                                'lk N, Q N -> lk Q')
        
        # Solve least squares
        compressed_kernels = lin_solve(M, f, solver=solver, lamda=lamda) # (L K) Q
        compressed_kernels = rearrange(compressed_kernels, 
                                       '(L K) Q -> L K Q', 
                                       L=L, K=K, Q=Q)
        
        return compressed_kernels
        
    def _lstsq_spatial(compressed_kernels: torch.Tensor,
                       spatial_bases: torch.Tensor,
                       kern_bases: torch.Tensor,
                       mask: torch.Tensor,
                       N_batch_size: int,
                       solver: str = 'pinv',
                       lamda: float = 0.0,) -> torch.Tensor:
        """
        This function optimizes for the spatial factors given fixed compressed kernel weights.
        
        Args:
        ----
        compressed_kernels : torch.Tensor
            compressed kernel weights with shape (L, K, Q)
        spatial_bases : torch.Tensor
            spatial bases with shape (Q, N)
        kern_bases : torch.Tensor
            kernel bases with shape (K, N)
        mask : torch.Tensor
            image weighting mask with shape (N,)
        l_batch_size : int
            batch size over L dimension
        solver : str
            solver for least squares
        lamda : float
            regularization parameter for least squares
            
        Returns
        -------
        spatial_factors : torch.Tensor
            spatial factors with shape (L, N)
        """
        
        # Matrices to compute
        M = torch.zeros((N, L, L), dtype=dtype, device=torch_dev)
        f = torch.zeros((N, L), dtype=dtype, device=torch_dev)
        
        # TODO make sure mask is being used properly
        for n1 in range(0, N, N_batch_size):
            n2 = min(n1 + N_batch_size, N)
            
            # M matrix
            left = einsum(kern_bases[:, n1:n2].conj() * mask[n1:n2], 
                          compressed_kernels.conj(),
                          'K N, L K Q -> N L Q')
            right = einsum(kern_bases[:, n1:n2] * mask[n1:n2], 
                          compressed_kernels,
                          'K N, L K Q -> N L Q')
            M[n1:n2] = einsum(left, right, 
                              'N L1 Q, N L2 Q -> N L1 L2')
            
            # f vector
            inner = einsum(compressed_kernels.conj(),
                           kern_bases[:, n1:n2].conj() * mask[n1:n2],
                           'L K Q, K N -> N Q L')
            f[n1:n2] = einsum(inner, spatial_bases[:, n1:n2], 
                              'N Q L, Q N -> N L')
    
    
        # Solve least squares
        spatial_factors = lin_solve(M, f[..., None], solver=solver, lamda=lamda)[..., 0]
        
        return spatial_factors.T

    # ALS Iterations
    spatial_factors_prev = spatial_factors_flt
    compressed_kernels_prev = None
    tbar = tqdm(range(max_iter), 'ALS iterations', disable=not verbose)
    for k in tbar:
        
        # ALS weight updates
        compressed_kernels = _lstsq_temporal(spatial_factors=spatial_factors_prev,
                                             spatial_bases=spatial_bases_flt,
                                             kern_bases=kern_bases_flt,
                                             mask=mask_flt,
                                             N_batch_size=spatial_batch_size,
                                             solver=solver,
                                             lamda=lamda)
        compressed_kernels = compressed_kernels.nan_to_num(0.0)
        
        # ALS apodization updates
        spatial_factors = _lstsq_spatial(compressed_kernels=compressed_kernels,
                                         spatial_bases=spatial_bases_flt,
                                         kern_bases=kern_bases_flt,
                                         mask=mask_flt,
                                         N_batch_size=spatial_batch_size,
                                         solver=solver,
                                         lamda=lamda)
        spatial_factors = spatial_factors.nan_to_num(0.0)
        
        # Update previous values
        compressed_kernels_prev = compressed_kernels
        spatial_factors_prev = spatial_factors
        
    # Reshape spatial dimensions
    spatial_factors = spatial_factors.reshape((L, *im_size))
        
    return spatial_factors, compressed_kernels

def lstsq_spatial(phase_model: matvec, 
                  kern_bases: torch.Tensor, 
                  kernel_weights: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  spatial_factors_prev: Optional[torch.Tensor] = None,
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
    spatial_factors_prev : torch.Tensor, optional
        previous spatial factors with shape (L, *im_size)
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
    kern_bases_masked = kern_bases * mask
    
    AHA = torch.zeros((*im_size, L, L), dtype=complex_dtype, device=torch_dev)
    AHB = torch.zeros((*im_size, L), dtype=complex_dtype, device=torch_dev)
    
    # Build cross terms
    cross_terms = torch.zeros((L, K, L, K), dtype=complex_dtype, device=torch_dev)
    for t1 in range(0, T, t_batch_size):
        t2 = min(t1 + t_batch_size, T)
        kernel_weights_batch = kernel_weights_flt[:, :, t1:t2] # L K T
        cross_terms += einsum(kernel_weights_batch.conj(), kernel_weights_batch, 'L1 K1 T, L2 K2 T -> L1 K1 L2 K2')
    
    # Build AHB and AHA matrices
    for k1 in range(0, K, k_batch_size):
        k2 = min(k1 + k_batch_size, K)
        
        # AHB
        kernel_weights_batch = kernel_weights[:, k1:k2, ...]
        kernel_weights_batch = rearrange(kernel_weights_batch, 'L K ... -> (L K) ...')
        imgs_batch = phase_model.adjoint(kernel_weights_batch).reshape((L, (k2-k1), *im_size)).conj()
        AHB += einsum(imgs_batch * mask, kern_bases_masked[k1:k2].conj(), 'L K ..., K ... -> ... L')
        
        # AHA   
        kern_cross = kern_bases_masked[k1:k2, None].conj() * kern_bases_masked[None, :]
        AHA += einsum(kern_cross, cross_terms[:, k1:k2], 'K1 K2 ..., L1 K1 L2 K2 -> ... L1 L2')
        
    # Solve least squares
    if spatial_factors_prev is None:
        spatial_factors = lin_solve(AHA, AHB[..., None], solver=solver, lamda=lamda)[..., 0] # *im_size L
    else:
        spatial_factors = lin_solve(AHA, (AHB + lamda * spatial_factors_prev.moveaxis(0, -1))[..., None], 
                                    solver=solver, lamda=lamda)[..., 0] # *im_size L
    spatial_factors = rearrange(spatial_factors, '... L -> L ...')
    
    return spatial_factors   
   
def lstsq_temporal(phase_model: matvec, 
                   kern_bases: torch.Tensor, 
                   spatial_factors: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None,
                   kernel_weights_prev: Optional[torch.Tensor] = None,
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
    kernel_weights_prev : torch.Tensor, optional
        previous kernel weights with shape (L, K, *trj_size)
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
        temp_batch = phase_model.forward(bases[ls, ks].conj() * mask) # (L K) *trj_size
        AHB[ls, ks, ...] += temp_batch
    
    # Solve least squares
    AHA_flt = rearrange(AHA, 'L1 K1 L2 K2 -> (L1 K1) (L2 K2)')
    AHB_flt = rearrange(AHB, 'L K ... -> (L K) (...)')
    if kernel_weights_prev is None:
        kernel_weights_flt = lin_solve(AHA_flt, AHB_flt, solver=solver, lamda=lamda) # (L K) (...)
    else:
        kernel_weights_prev_flt = rearrange(kernel_weights_prev, 'L K ... -> (L K) (...)')
        kernel_weights_flt = lin_solve(AHA_flt, AHB_flt + lamda * kernel_weights_prev_flt, 
                                       solver=solver, lamda=lamda) # (L K) (...)
    kernel_weights = kernel_weights_flt.reshape(AHB.shape)
        
    return kernel_weights

def lstsq_temporal_regressor(phase_model: matvec, 
                             kern_bases: torch.Tensor, 
                             spatial_factors: torch.Tensor, 
                             kern_regressor: kernel_regressor,
                             mask: Optional[torch.Tensor] = None,
                             lk_batch_size: Optional[int] = None,
                             solver: Optional[str] = 'pinv',
                             lamda: Optional[float] = 0.0,) -> torch.Tensor:
    """
    This function optimizes for the kernel weights given fixed spatial factors.
    The kernel weights are constrained to be a sparse linear combination of splatting kernels.
    
    Args
    ----
    phase_model : mr_recon.linops.linop
        linear operator for the phase operator
    kern_bases : torch.Tensor
        kernel bases with shape (K, *im_size)
    spatial_factors : torch.Tensor
        spatial factors with shape (L, *im_size)
    kern_regressor : kernel_regressor
        kernel regression model
    mask : torch.Tensor, optional
        image weighting mask with shape im_size
    lk_batch_size : int, optional
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
  