import torch
import numpy as np

from tqdm import tqdm
from dataclasses import dataclass, field
from einops import rearrange, einsum
from typing import Optional, Union
from einops import einsum

from .utils import gen_grd
from .matvec import matvec, matvec_naive
from .linalg import svd_operator, lin_solve

__all__ = [
    'hofft_params',
    'build_kern_bases',
    'als_iterations',
    'lstsq_spatial',
    'lstsq_temporal',
]

@dataclass
class hofft_params: 
    # Model params
    kern_size: tuple
    os: float
    L: int
    # Other
    matvec_type: matvec = matvec_naive
    matvec_kwargs: dict = field(default_factory=dict)
    spatial_init: Union[torch.Tensor, str] = 'seg'
    spatial_batch_size: Optional[int] = None
    normalize_coeffs: bool = True
    cur_rank: Optional[int] = None
    # Reduction
    kalpha_method: str = 'maxmin'
    reduced_im_size: Optional[tuple] = None
    time_reduction_factor: Optional[float] = None
    num_compressed_bases: Optional[int] = None
    num_representative_alphas: Optional[int] = None
    # Solver
    max_als_iter: int = 100
    lamda: float = 0.0
    solver: str = 'pinv'
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
    cur_rank : Optional[int]
        Pipeline matvec switch. None uses matvec_naive on (V^T, S U^T).
        Negative uses adaptive-rank CUR; positive uses fixed-rank CUR
        with that rank. Both CUR paths take the whitened (V^T, U^T, S).
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
    kalpha_method : str
        Method to use for selecting the K alpha vectors.
    solver : str
        Solver to use for least squares.
    lamda : float
        Regularization parameter for least squares.
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

def svd_decomp(phase_model: matvec,
               hparams: hofft_params,
               mask: Optional[torch.Tensor] = None,
               svd_method: str = 'lobpcg',
               max_iter: int = 100,
               tol: float = 1e-2,
               verbose: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs an SVD to decompose the phase model into spatial and temporal factors.
    
    Args
    ----
    phase_model : matvec
        phase model to decompose
    hparams : hofft_params
        HOFFT parameters
    mask : torch.Tensor, optional
        image weighting mask with shape im_size
    svd_method : str, optional
        method to use for SVD. Options are:
        'lobpcg' - use LOBPCG to solve for the SVD
        'power' - use power method to solve for the SVD
    max_iter : int, optional
        maximum number of iterations
    tol : float, optional
        tolerance for SVD convergence
    verbose : bool, optional
        whether to print progress
        
    Returns
    -------
    spatial_factors : torch.Tensor
        spatial factors with shape (L, *im_size)
    temporal_factors : torch.Tensor
        temporal factors with shape (L, *trj_size)
    """
    # Consts
    im_size = phase_model.im_size
    trj_size = phase_model.trj_size
    M = int(np.prod(trj_size))
    N = int(np.prod(im_size))
    
    if svd_method == 'lobpcg':
        use_lobpcg = True
    elif svd_method == 'power':
        use_lobpcg = False
    else:
        raise ValueError(f'Invalid SVD method: {svd_method}')
    
    # Perform SVD on a tall matrix
    if N < M:
        A = lambda x: phase_model.forward(x * mask)
        AHA = lambda x: phase_model.adjoint(phase_model.forward(x * mask)) * mask.conj()
        inp_example = torch.zeros(phase_model.im_size, 
                                dtype=torch.complex64, 
                                device=phase_model.phis.device)
        U, S, Vh = svd_operator(A, AHA, inp_example, 
                                rank=hparams.L,
                                num_iter=max_iter,
                                lobpcg=use_lobpcg,
                                tol=tol,
                                verbose=verbose)
        U = U.moveaxis(-1, 0)
    # Perform SVD on a wide matrix, use A.H instead of A
    else:    
        AH = lambda y: phase_model.adjoint(y) * mask.conj()
        AAH = lambda y: phase_model.forward(phase_model.adjoint(y) * mask.abs().square())
        inp_example = torch.zeros(phase_model.trj_size, 
                                dtype=torch.complex64, 
                                device=phase_model.alphas.device)
        Vh, S, U = svd_operator(AH, AAH, inp_example, 
                                rank=hparams.L,
                                num_iter=max_iter,
                                lobpcg=use_lobpcg,
                                tol=tol,
                                verbose=verbose)
        Vh = Vh.moveaxis(-1, 0).conj()
        U = U.conj()
    
    temporal_factors = einsum(U, S ** 0.5, 'L ..., L -> L ...')
    spatial_factors = einsum(Vh, S ** 0.5, 'L ..., L -> L ...')
    
    return spatial_factors, temporal_factors

def als_iterations(phase_model: matvec, 
                   kern_bases: torch.Tensor,
                   spatial_factors_init: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   max_iter: Optional[int] = 100,
                   solver: str = 'pinv',
                   lamda: float = 0.0,
                   spatial_batch_size: Optional[int] = None,
                   n_err_pts: Optional[int] = 1000,
                   rel_err_tol: Optional[float] = 0.5e-2,
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
    solver : str, optional
        solver for least squares
    lamda : float, optional
        regularization parameter for least squares
    spatial_batch_size : int, optional
        batch size over spatial dimension
    n_err_pts : int, optional
        number of error points for convergence check
    rel_err_tol : float, optional
        relative error tolerance for convergence check
    verbose : bool, optional
        whether to print progress
    
    Returns
    -------
    spatial_factors : torch.Tensor
        spatial factors with shape (L, *im_size)
    kernel_weights : torch.Tensor
        kernel weights with shape (L, K, *trj_size)
    """
    # Consts
    L = spatial_factors_init.shape[0]
    K = kern_bases.shape[0]
    N = phase_model.R
    M = phase_model.T
        
    # Default mask
    if mask is None:
        mask = torch.ones_like(spatial_factors_init[0])
        
    # Weights only 
    if max_iter == 0:
        kernel_weights = lstsq_temporal(phase_model, kern_bases, spatial_factors_init, 
                                        mask=mask,
                                        lamda=lamda,
                                        solver=solver,
                                        spatial_batch_size=spatial_batch_size)
        return spatial_factors_init, kernel_weights
    
    # Pick random space and time indices for error computation
    mask_flt = mask.flatten()
    idxs_nonz = torch.argwhere(mask_flt.abs() > 0)[:, 0]
    idxs_space = idxs_nonz[torch.randperm(len(idxs_nonz))[:n_err_pts].to(kern_bases.device)]
    idxs_time = torch.randperm(M)[:n_err_pts].to(kern_bases.device)
    kern_bases_flt = kern_bases.reshape((K, -1))[:,idxs_space]

    
    # Momentum term (off for now, didn't seem to help)
    momentum = lambda k : 0.0
    k0 = 0
    
    # ALS till max_iter
    spatial_factors_prev = spatial_factors_init
    kernel_weights_prev = None
    phz_est_prev = None
    tbar = tqdm(range(max_iter), 'ALS iterations', disable=not verbose)
    for k in tbar:
        
        # ALS weight updates
        kernel_weights = lstsq_temporal(phase_model, kern_bases, spatial_factors_prev, 
                                        mask=mask,
                                        lamda=lamda,
                                        solver=solver,
                                        spatial_batch_size=spatial_batch_size,
                                        kernel_weights_prev=kernel_weights_prev)
        if k > k0:
            beta = momentum(k)
            kernel_weights = kernel_weights + beta * (kernel_weights_prev - kernel_weights)
        kernel_weights = kernel_weights.nan_to_num(0.0)
        
        # ALS apodization updates
        spatial_factors = lstsq_spatial(phase_model, kern_bases, kernel_weights, 
                                        mask=mask,
                                        solver=solver,
                                        lamda=lamda,
                                        spatial_batch_size=spatial_batch_size,
                                        spatial_factors_prev=spatial_factors_prev,)
        spatial_factors = spatial_factors.nan_to_num(0.0)
        if k > k0:
            spatial_factors = spatial_factors + beta * (spatial_factors_prev - spatial_factors)
                    
        
        # Compute phase error over subset of space and time points
        spatial_flt = spatial_factors.reshape((L, -1))[:, idxs_space]
        kernel_flt = kernel_weights.reshape((L, K, -1))[:, :, idxs_time]
        phz_est = einsum(kern_bases_flt, spatial_flt, 'K N, L N -> L K N')
        phz_est = einsum(phz_est, kernel_flt, 'L K N, L K M -> M N')
        if k > 0:
            rel_err = (phz_est - phz_est_prev).norm() / phz_est_prev.norm()
            if rel_err < rel_err_tol:
                if verbose:
                    print(f'Converged at iteration {k} with relative phase error {rel_err*100:.2f}%')
                break
        phz_est_prev = phz_est
        
        # Update previous values
        kernel_weights_prev = kernel_weights
        spatial_factors_prev = spatial_factors
        
    return spatial_factors, kernel_weights

def als_compressed(spatial_bases: torch.Tensor, 
                   kern_bases: torch.Tensor, 
                   spatial_factors_init: torch.Tensor,
                   mask: Optional[torch.Tensor] = None, 
                   max_iter: int = 100,
                   spatial_batch_size: Optional[int] = None,
                   solver: Optional[str] = 'solve', 
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
    if mask is None:
        mask = torch.ones((N,), dtype=torch.complex64, device=torch_dev)
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
        # Matrices to compute
        M = torch.zeros((L * K, L * K), dtype=dtype, device=torch_dev)
        f = torch.zeros((L * K, Q), dtype=dtype, device=torch_dev)
        # Build matrices in batches (bases built per-batch to limit L*K*N memory)
        for n1 in range(0, N, N_batch_size):
            n2 = min(n1 + N_batch_size, N)

            bases = einsum(kern_bases[:, n1:n2], spatial_factors[:, n1:n2], 
                           'K N, L N -> L K N') * mask[n1:n2]
            bases = rearrange(bases, 'L K N -> (L K) N')
            
            M += einsum(bases.conj(), bases, 
                                'lk1 N, lk2 N -> lk1 lk2')
            f += einsum(bases.conj() * mask[n1:n2], spatial_bases[:, n1:n2], 
                                'lk N, Q N -> lk Q')
        
        # Tiny diagonal jitter so LU stays stable if M is rank deficient
        if solver == 'solve' and lamda == 0:
            diag_scale = M.diagonal(dim1=-2, dim2=-1).real.mean()
            M.diagonal(dim1=-2, dim2=-1).add_(1e-6 * diag_scale)
        
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
        
        mask2 = mask * mask
        
        # Matrices to compute
        M = torch.empty((N, L, L), dtype=dtype, device=torch_dev)
        f = torch.empty((N, L), dtype=dtype, device=torch_dev)
        for n1 in range(0, N, N_batch_size):
            n2 = min(n1 + N_batch_size, N)
            
            # A[n,l,q] = sum_k ck[l,k,q] kern[k,n] (one GEMM, shared by M and f)
            A = einsum(compressed_kernels, kern_bases[:, n1:n2], 
                       'L K Q, K N -> N L Q')
            Ah = A.conj() * mask2[n1:n2, None, None]
            
            # M matrix
            M[n1:n2] = einsum(Ah, A, 'N L1 Q, N L2 Q -> N L1 L2')
            
            # f vector (mask^2 to match the weighting in M)
            f[n1:n2] = einsum(Ah, spatial_bases[:, n1:n2], 
                              'N L Q, Q N -> N L')
        
        # Tiny diagonal jitter so batched LU stays stable where mask zeros out voxels
        if solver == 'solve' and lamda == 0:
            diag_scale = M.diagonal(dim1=-2, dim2=-1).real.mean()
            M.diagonal(dim1=-2, dim2=-1).add_(1e-6 * diag_scale)
    
        # Solve least squares
        spatial_factors = lin_solve(M, f[..., None], solver=solver, lamda=lamda)[..., 0]
        
        return spatial_factors.T

    # If max_iter is 0, just optimize the kernel weights
    if max_iter == 0:
        compressed_kernels = _lstsq_temporal(spatial_factors=spatial_factors_flt,
                                             spatial_bases=spatial_bases_flt,
                                             kern_bases=kern_bases_flt,
                                             mask=mask_flt,
                                             N_batch_size=spatial_batch_size,
                                             solver=solver,
                                             lamda=lamda)
        compressed_kernels = compressed_kernels.nan_to_num(0.0)
        spatial_factors = spatial_factors_flt.reshape((L, *im_size))
        return spatial_factors, compressed_kernels

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
                  spatial_batch_size: Optional[int] = None,
                  solver: Optional[str] = 'pinv',
                  lamda: Optional[float] = 0.0,) -> torch.Tensor:
    """    
    This function optimizes for the spatial factors 
    given fixed kernel weights via least squares.
    
    Spatial dimensions are flattened and the normal equations are assembled
    and solved in batches of voxels to limit peak memory (important for large
    FOVs where a full (N, L, L) solve does not fit on GPU).
    
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
    spatial_batch_size : int, optional
        number of voxels per assemble/solve batch. None uses all voxels.
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
    L = kernel_weights.shape[0]
    K = kern_bases.shape[0]
    N = int(np.prod(im_size))
    assert phase_model.ishape == im_size
    assert phase_model.oshape == kernel_weights.shape[2:]

    if mask is None:
        mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
    if spatial_batch_size is None:
        spatial_batch_size = N

    # Flatten spatial dims
    mask_flt = mask.reshape(N)
    kern_bases_flt = (kern_bases * mask).reshape((K, N))  # K N
    kernel_weights_flt = rearrange(kernel_weights, 'L K ... -> L K (...)')  # L K T
    prev_flt = None
    if spatial_factors_prev is not None:
        prev_flt = spatial_factors_prev.reshape((L, N))

    # Gram over trajectory (small: L K L K)
    cross_terms = einsum(
        kernel_weights_flt.conj(), kernel_weights_flt,
        'L1 K1 T, L2 K2 T -> L1 K1 L2 K2',
    )

    # Adjoint of all (L*K) temporal weights -> (L, K, N)
    kw_all = rearrange(kernel_weights, 'L K ... -> (L K) ...')
    imgs_flt = phase_model.adjoint(kw_all).reshape((L, K, N)).conj()

    # Assemble + solve in spatial batches
    spatial_factors_flt = torch.empty((L, N), dtype=torch.complex64, device=torch_dev)
    for n1 in range(0, N, spatial_batch_size):
        n2 = min(n1 + spatial_batch_size, N)
        kb = kern_bases_flt[:, n1:n2]            # K Nb
        m = mask_flt[n1:n2]                      # Nb
        imgs_b = imgs_flt[:, :, n1:n2] * m       # L K Nb

        AHB = einsum(imgs_b, kb.conj(), 'L K N, K N -> N L')
        kern_cross = kb[:, None].conj() * kb[None, :]  # K1 K2 Nb
        AHA = einsum(kern_cross, cross_terms, 'K1 K2 N, L1 K1 L2 K2 -> N L1 L2')

        if prev_flt is None:
            rhs = AHB
        else:
            rhs = AHB + lamda * prev_flt[:, n1:n2].moveaxis(0, -1)
        sol = lin_solve(AHA, rhs[..., None], solver=solver, lamda=lamda)[..., 0]  # Nb L
        spatial_factors_flt[:, n1:n2] = sol.moveaxis(-1, 0)

    return spatial_factors_flt.reshape((L, *im_size))
   
def lstsq_temporal(phase_model: matvec, 
                   kern_bases: torch.Tensor, 
                   spatial_factors: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None,
                   kernel_weights_prev: Optional[torch.Tensor] = None,
                   spatial_batch_size: Optional[int] = None,
                   solver: Optional[str] = 'pinv',
                   lamda: Optional[float] = 0.0,) -> torch.Tensor:
    """
    This function optimizes for the kernel weights given fixed spatial factors.

    Spatial dimensions are flattened and the normal equations are accumulated
    in batches of voxels (AHA via spatial Gram chunks; AHB via linearity of
    ``phase_model.forward`` on spatially supported inputs).
    
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
    spatial_batch_size : int, optional
        number of voxels per assemble batch. None uses all voxels.
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
    N = int(np.prod(im_size))
    T = int(np.prod(trj_size))
    assert phase_model.ishape == im_size

    if mask is None:
        mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
    if spatial_batch_size is None:
        spatial_batch_size = N

    # Flatten spatial dims
    kern_flt = kern_bases.reshape((K, N))
    spat_flt = spatial_factors.reshape((L, N))
    mask_flt = mask.reshape(N)

    AHA = torch.zeros((L, K, L, K), dtype=torch.complex64, device=torch_dev)
    AHB = torch.zeros((L * K, T), dtype=torch.complex64, device=torch_dev)
    # Reused forward input: only one spatial slab is nonzero per batch
    x = torch.zeros((L * K, *im_size), dtype=torch.complex64, device=torch_dev)
    x_flt = x.reshape((L * K, N))

    for n1 in range(0, N, spatial_batch_size):
        n2 = min(n1 + spatial_batch_size, N)
        m = mask_flt[n1:n2]
        # bases = spatial_factors * kern_bases * mask  (same as before)
        b = spat_flt[:, None, n1:n2] * kern_flt[None, :, n1:n2] * m  # L K Nb

        AHA += einsum(b.conj(), b, 'L1 K1 N, L2 K2 N -> L1 K1 L2 K2')

        # AHB += Phi @ (conj(bases) * mask), via spatial support of this slab
        x_flt.zero_()
        x_flt[:, n1:n2] = rearrange(b.conj() * m, 'L K N -> (L K) N')
        AHB += phase_model.forward(x).reshape((L * K, T))

    # Solve least squares on the small (L*K, L*K) system
    AHA_flt = rearrange(AHA, 'L1 K1 L2 K2 -> (L1 K1) (L2 K2)')
    if kernel_weights_prev is None:
        kernel_weights_flt = lin_solve(AHA_flt, AHB, solver=solver, lamda=lamda)
    else:
        kernel_weights_prev_flt = rearrange(kernel_weights_prev, 'L K ... -> (L K) (...)')
        kernel_weights_flt = lin_solve(
            AHA_flt, AHB + lamda * kernel_weights_prev_flt,
            solver=solver, lamda=lamda,
        )
    return kernel_weights_flt.reshape((L, K, *trj_size))
