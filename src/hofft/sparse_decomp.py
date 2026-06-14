import torch
import numpy as np

from einops import einsum
from typing import Optional
from dataclasses import dataclass

from mr_recon.utils import pick_K_vectors
from mr_recon.dtypes import complex_dtype
from tqdm import tqdm

from .phase_coeffs import rescale_phis_alphas
from .matvec import subsample_idx

__all__ = [
    'select_betas',
    'sparse_alpha_segmentation',
]

@dataclass
class sparse_params:
    Q: int = 500
    K: int = 5
    beta_method: str = 'maxmin'
    interp_method: str = 'lstsq'
    lamda: float = 0.0
    spatial_subsample: Optional[int] = None
    normalize_coeffs: bool = True
    temporal_batch_size: Optional[int] = None
    spatial_batch_size: Optional[int] = None
    grid_spacing: float = 0.5
    grid_width: int = 2
    """
    Parameters for sparse decomposition.
    
    Attributes
    ----------
    Q : int
        Number of representative beta vectors to use.
    K : int
        Number of sparse coefficients to keep. Equals Q if None.
    beta_method : str
        Method to use for alpha segmentation: 
        - ```grid``` - generates a grid in alphas space with spacing <dalpha>
        - ```maxmin``` - maximum minimum method
        - ```kmeans``` - k-means clustering
    interp_method : str
        Method to use for interpolating the sparse coefficients.
        - ```lstsq``` - least squares interpolation
        - ```barycentric``` / ```barycentric_nn``` - barycentric interpolation
        - ```rbf``` - unstructured RBF interpolation with global weights
        - ```grid``` - structured grid interpolation. Q and K are ignored and 
          instead determined by grid_spacing and grid_width (K = grid_width^B).
    lamda : float
        Regularization parameter for the sparse coefficients. Used by 'lstsq' 
        and 'rbf'.
    spatial_subsample : Optional[int]
        Fixed voxel count for subsampled lstsq/RBF normal equations. None uses 
        all voxels.
    normalize_coeffs : bool
        Whether to normalize the phi terms to be ubniased between [-1/2, 1/2].
    temporal_batch_size : Optional[int]
        Batch size for the temporal dimension.
    spatial_batch_size : Optional[int]
        Batch size for the spatial dimension.
    grid_spacing : float
        Spacing (delta beta) of the beta grid. Used when interp_method or 
        beta_method is 'grid'.
    grid_width : int
        Width (W_beta) of the sparse interpolation stencil per alpha dimension. 
        Used when interp_method or beta_method is 'grid'.
    """
    
def select_betas(alphas: torch.Tensor,
                 num_betas: int,
                 beta_method: str = 'maxmin',
                 grid_width: int = 2,
                 grid_spacing: float = 0.5) -> torch.Tensor:
    """
    Selects the beta vectors for the alpha segmentation. The alpha vectors are B-dimensional, and we seek 'num_betas' 
    representative beta vectors that can be used to interpolate in alpha space. 
    
    Args
    ----
    alphas : torch.Tensor
        Alpha vectors with shape (B, ...)
    num_betas : int
        Number of representative beta vectors to select
    beta_method : str
        Method to use for selecting the beta vectors:
        - ```grid``` - generates a sparse grid in alpha space around the alpha trajectory
        - ```maxmin``` - picks num_betas alpha vectors that are maximally separated in alpha space
        - ```kmeans``` - k-means clustering
        
    Returns
    -------
    betas : torch.Tensor
        Representative beta vectors with shape (num_betas, B)
    """
    # Consts
    B = alphas.shape[0]
    alphas_flt = alphas.reshape((B, -1)).T # T B
    
    # Grid method: union of each time point's grid_width^B stencil grid points,
    # using the same stencil + mixed-radix unique-coordinate logic as the
    # structured grid interpolation (see _grid_stencil_crds / _grid_unique_betas).
    if beta_method == 'grid':
        grid_crds = _grid_stencil_crds(alphas_flt.T, grid_spacing, grid_width) # B T W
        betas, _ = _grid_unique_betas(grid_crds, grid_spacing, alphas.dtype)
        print(f'Warning, number of betas is now {betas.shape[0]} instead of {num_betas}!')
    # other methods
    else:
        betas, _ = pick_K_vectors(alphas.T, K=num_betas, method=beta_method, return_idxs=False)
    
    return betas

def _grid_stencil_crds(alphas: torch.Tensor,
                       grid_spacing: float,
                       grid_width: int) -> torch.Tensor:
    """
    Integer grid coordinates of each time point's interpolation stencil, per 
    dimension. ceil(alpha/db - W/2) + {0, ..., W-1} gives the W nearest 
    consecutive grid coordinates (rounding ties broken consistently to the left).
    
    Args
    ----
    alphas : torch.Tensor
        temporal coefficients with shape (B, T)
    grid_spacing : float
        spacing (delta beta) of the beta grid
    grid_width : int
        width (W_beta) of the interpolation stencil per dimension
        
    Returns
    -------
    grid_crds : torch.Tensor
        integer grid coordinates with shape (B, T, W)
    """
    W = grid_width
    db = grid_spacing
    base_crds = torch.ceil(alphas / db - W / 2) # B T
    grid_crds = base_crds[..., None] + torch.arange(W, device=alphas.device) # B T W
    return grid_crds

def _grid_unique_betas(grid_crds: torch.Tensor,
                       grid_spacing: float,
                       dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collapses the per-time-point stencil grid coordinates into the unique set of 
    grid points (the betas) and the indices that map each time point's stencil 
    back into that set.
    
    Each time point's cartesian product of per-dimension stencil coordinates is 
    encoded as a mixed-radix integer code (first dimension varies slowest). The 
    unique codes define the Q grid points, which are then inverted back to 
    coordinates and scaled by the grid spacing.
    
    Args
    ----
    grid_crds : torch.Tensor
        integer grid coordinates with shape (B, T, W)
    grid_spacing : float
        spacing (delta beta) of the beta grid
    dtype : torch.dtype
        floating dtype for the returned betas
        
    Returns
    -------
    betas : torch.Tensor
        unique grid points with shape (Q, B)
    sparse_inds : torch.Tensor
        per-time-point stencil indices with shape (T, K) in [0, Q), K = W^B
    """
    B, T, W = grid_crds.shape
    torch_dev = grid_crds.device
    crds_int = grid_crds.to(torch.int64) # B T W
    mins = crds_int.amin(dim=(1, 2)) # B
    extents = crds_int.amax(dim=(1, 2)) - mins + 1 # B
    
    # Per-time-point cartesian product of stencil coords encoded as mixed-radix
    # integer codes (first dimension varies slowest).
    codes = torch.zeros((T, 1), dtype=torch.int64, device=torch_dev)
    for b in range(B):
        codes = (codes[:, :, None] * extents[b] 
                 + (crds_int[b] - mins[b])[:, None, :]).reshape((T, -1))
    K = codes.shape[1] # W^B
    
    # Unique grid points define the Q betas; invert codes back to coordinates
    unique_codes, sparse_inds = torch.unique(codes.reshape(-1), return_inverse=True)
    sparse_inds = sparse_inds.reshape((T, K))
    Q = unique_codes.shape[0]
    beta_crds = torch.zeros((Q, B), dtype=torch.int64, device=torch_dev)
    for b in range(B - 1, -1, -1):
        beta_crds[:, b] = unique_codes % extents[b] + mins[b]
        unique_codes = unique_codes // extents[b]
    betas = beta_crds.to(dtype) * grid_spacing
    return betas, sparse_inds

def _lstsq_voxel_inds(R: int,
                      spatial_subsample: Optional[int],
                      device: torch.device) -> Optional[torch.Tensor]:
    """Fixed voxel subset for exact A^H b einsum or RBF normal equations."""
    if spatial_subsample is None or spatial_subsample >= R:
        return None
    return subsample_idx(R, spatial_subsample, device, mode='fixed', seed=0)

def sparse_alpha_segmentation(phis: torch.Tensor, 
                              alphas: torch.Tensor, 
                              sparams: sparse_params,
                              verbose: bool = True,
                              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs alpha segmentation with sparsity. Spatio-temporal phase becomes:
    phase(r,t) = sum_k spatial_bases[sparse_inds[k,t]](r) * sparse_coeffs[k,t]
    
    Args
    ----
    phis : torch.Tensor
        Spatial phase basis with shape (B, *im_size)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    sparams : sparse_params
        Sparse decomposition / interpolation configuration.
    verbose : bool
        whether to print progress
        
    Returns
    -------
    spatial_bases : torch.Tensor
        Spatial bases with shape (Q, *im_size)
    sparse_inds : torch.Tensor
        Sparse indices with shape (K, *trj_size)
    sparse_coeffs : torch.Tensor
        Sparse coefficients with shape (K, *trj_size)
        
    """
    Q = sparams.Q
    K_sparsity = sparams.K
    beta_method = sparams.beta_method
    interp_method = sparams.interp_method
    lamda = sparams.lamda
    spatial_subsample = sparams.spatial_subsample
    normalize_coeffs = sparams.normalize_coeffs
    temporal_batch_size = sparams.temporal_batch_size
    grid_spacing = sparams.grid_spacing
    grid_width = sparams.grid_width
    
    # Consts
    B = phis.shape[0]
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    R = np.prod(im_size)
    T = np.prod(trj_size)
    
    if temporal_batch_size is None:
        temporal_batch_size = T
    assert K_sparsity <= Q, "K must be less than or equal to Q"
    
    # Flatten
    phis_flt = phis.reshape((B, R))
    alphas_flt = alphas.reshape((B, T))
    
    # Normalize coefficients
    if normalize_coeffs:
        phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_flt, alphas_flt)
        # Full phis for spatial bases / midpoint factor; normalized phis only for 
        # the grid 1D stencil fits (phi_mp is spatially constant per dim and must 
        # not enter the dalpha-local solves).
        phis = phis_nrm + phis_mp[:, None]
        phis_grid = phis_nrm
        alphas = alphas_nrm
    else:
        phis = phis_flt
        phis_grid = phis_flt
        alphas = alphas_flt
        alphas_mp = torch.zeros_like(alphas_flt[:, 0])
    
    # The alphas_mp factor re-introduces the temporal midpoint that was removed
    # during normalization, so that betas only need to span the centered alphas.
    midpoint_factor = torch.exp(-2j * np.pi * einsum(phis, alphas_mp, 
                                                     'B R, B -> R'))
    voxel_inds = _lstsq_voxel_inds(R, spatial_subsample, phis.device)
    
    # Structured grid interpolation determines betas, indices, and coefficients
    # jointly from the grid structure.
    if interp_method == 'grid':
        betas, apod, sparse_inds, sparse_coeffs = _structured_grid_interp(
            phis_grid, alphas, 
            grid_spacing=grid_spacing, 
            grid_width=grid_width,
            num_apod_iter=20,
            num_dalpha=2 ** 7,
            lamda=lamda, 
            verbose=verbose)
        Q = betas.shape[0]
        K_sparsity = sparse_inds.shape[1]
        if verbose:
            print(f'Structured grid interpolation: Q = {Q}, K = {K_sparsity}')
        
        # Spatial bases b_q(r) = a(r) exp(-j2pi sum_b phi_b(r) * beta_q,b)
        spatial_bases = torch.exp(-2j * np.pi * einsum(phis, betas, 
                                                       'B R, Q B -> Q R'))
        spatial_bases = spatial_bases * (apod * midpoint_factor)
        
        # Reshape to expected output shapes
        spatial_bases = spatial_bases.reshape((Q, *im_size))
        sparse_inds = sparse_inds.T.reshape((K_sparsity, *trj_size))
        sparse_coeffs = sparse_coeffs.T.reshape((K_sparsity, *trj_size))
        return spatial_bases, sparse_inds, sparse_coeffs
    
    # Pick Q representatives of alphas (betas has shape (Q, B))
    betas = select_betas(alphas, 
                         num_betas=Q, 
                         beta_method=beta_method,
                         grid_width=grid_width,
                         grid_spacing=grid_spacing)
    Q = betas.shape[0]

    # Compute spatial bases b_q(r) = exp(-j2pi sum_b phi_b(r) * beta_q,b)
    spatial_bases = torch.exp(-2j * np.pi * einsum(phis, betas, 
                                                   'B R, Q B -> Q R'))
    spatial_bases = spatial_bases * midpoint_factor
    
    # Find indices of the K_sparsity closest representatives for each time point.
    # topk_q -||a - beta_q||^2 = topk_q (2 a . beta_q - ||beta_q||^2), which only
    # needs a (batch, Q) matrix instead of the (T, Q, B) broadcasted difference.
    alphas_T = alphas.T # T B
    betas_sq = (betas ** 2).sum(dim=-1) # Q
    sparse_inds = torch.empty((T, K_sparsity), dtype=torch.long, device=alphas.device)
    for t1 in tqdm(range(0, T, temporal_batch_size), 
                   desc='Selecting Sparse Indices', 
                   disable=not verbose):
        t2 = min(t1 + temporal_batch_size, T)
        neg_dists = 2 * (alphas_T[t1:t2] @ betas.T) - betas_sq # batch Q
        sparse_inds[t1:t2] = torch.topk(neg_dists, K_sparsity, dim=-1).indices
    
    # Solve for the sparse coefficients
    if interp_method == 'barycentric':
        sparse_coeffs = _barycentric_coeffs(alphas, betas, sparse_inds, 
                                            temporal_batch_size, nonneg=False, 
                                            lamda=lamda, verbose=verbose)
    elif interp_method == 'barycentric_nn':
        sparse_coeffs = _barycentric_coeffs(alphas, betas, sparse_inds, 
                                            temporal_batch_size, nonneg=True, 
                                            lamda=lamda, verbose=verbose)
    elif interp_method == 'lstsq':
        sparse_coeffs = _lstsq_coeffs(phis, alphas + alphas_mp[:, None], 
                                      spatial_bases, 
                                      sparse_inds, temporal_batch_size, 
                                      lamda=lamda, 
                                      voxel_inds=voxel_inds,
                                      verbose=verbose)
    elif interp_method == 'rbf':
        sparse_coeffs = _rbf_coeffs(phis, alphas + alphas_mp[:, None], 
                                    alphas, betas, spatial_bases, 
                                    sparse_inds, temporal_batch_size, 
                                    num_rnd_voxels=spatial_subsample,
                                    lamda=lamda, 
                                    voxel_inds=voxel_inds,
                                    verbose=verbose)
    else:
        raise NotImplementedError(f"interp_method '{interp_method}' is not implemented. "
                                  f"Supported methods are 'barycentric', 'lstsq', "
                                  f"'rbf', and 'grid'.")
    
    # Reshape to expected output shapes
    spatial_bases = spatial_bases.reshape((Q, *im_size))
    sparse_inds = sparse_inds.T.reshape((K_sparsity, *trj_size))
    sparse_coeffs = sparse_coeffs.T.reshape((K_sparsity, *trj_size))
    
    return spatial_bases, sparse_inds, sparse_coeffs

def _structured_grid_interp(phis: torch.Tensor,
                            alphas: torch.Tensor,
                            grid_spacing: float,
                            grid_width: int,
                            num_apod_iter: int = 20,
                            num_dalpha: int = 2 ** 7,
                            lamda: float = 0.0,
                            verbose: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Structured grid interpolation of the temporal phase coefficients:
    exp(-j2pi phi(r) . alpha(t)) ~= a(r) sum_k exp(-j2pi phi(r) . betas[sparse_inds[k,t]]) * sparse_coeffs[k,t]
    where the betas live on a regular grid with spacing <grid_spacing>, and each 
    time point interpolates from a compact stencil of grid_width^B grid points.
    
    The problem decouples into B 1D problems. For each alpha dimension b, we 
    decompose alpha_b(t) = (stencil center) + dalpha, dalpha in [-db/2, db/2], 
    and jointly fit a spatial apodization a_b(r) and stencil weight functions 
    w_b(dalpha) (W_beta of them) via alternating least squares:
    exp(-j2pi phi_b(r) dalpha) ~= a_b(r) sum_w exp(-j2pi phi_b(r) s_w dbeta) w_b[w](dalpha)
    where s_w are the fixed stencil offsets around the stencil center. The 
    weight functions are tabulated on <num_dalpha> linearly spaced dalpha 
    samples and linearly interpolated at each time point's actual dalpha. 
    Since the stencil phase columns relative to the stencil center are the same 
    for every time point (up to a unit-modulus factor that cancels in the least 
    squares), the table entries are exact per-sample solutions; the only 
    approximation is the linear interpolation between table nodes.
    
    The final apodization is a(r) = prod_b a_b(r) and the sparse coefficients 
    are the tensor products c[t] = kron(w_1(dalpha_1(t)), ..., w_B(dalpha_B(t))).
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis with shape (B, R), ideally normalized to a unit range
    alphas : torch.Tensor
        (centered) temporal coefficients with shape (B, T)
    grid_spacing : float
        spacing (delta beta) of the beta grid
    grid_width : int
        width (W_beta) of the interpolation stencil per alpha dimension
    num_apod_iter : int
        number of alternating iterations to fit a_b(r). 0 disables apodization.
    num_dalpha : int
        number of linearly spaced dalpha samples for the weight tables
    lamda : float
        regularization parameter for the stencil weight solves
    verbose : bool
        whether to print progress
        
    Returns
    -------
    betas : torch.Tensor
        unique grid points with shape (Q, B)
    apod : torch.Tensor
        spatial apodization a(r) with shape (R,)
    sparse_inds : torch.Tensor
        sparse grid indices with shape (T, K) in [0, Q), K = grid_width^B
    sparse_coeffs : torch.Tensor
        sparse coefficients with shape (T, K)
    """
    B, R = phis.shape
    T = alphas.shape[1]
    W = grid_width
    db = grid_spacing
    D = num_dalpha
    torch_dev = phis.device
    
    # Integer grid coordinates of each time point's stencil, per dimension.
    grid_crds = _grid_stencil_crds(alphas, db, W) # B T W
    base_crds = grid_crds[..., 0] # B T
    
    # Stencil offsets relative to the stencil center (integers for odd W, 
    # half-integers for even W) and per-time-point deviations from the center
    offsets = torch.arange(W, device=torch_dev, dtype=torch.float32) - (W - 1) / 2 # W
    dalphas = alphas - (base_crds + (W - 1) / 2) * db # B T, in [-db/2, db/2]
    
    # Linearly spaced dalpha samples for the weight tables
    dalpha_table = torch.linspace(-db / 2, db / 2, D, device=torch_dev) # D
    
    apods = torch.ones((B, R), dtype=complex_dtype, device=torch_dev)
    weights = torch.zeros((B, T, W), dtype=complex_dtype, device=torch_dev)
    for b in tqdm(range(B), desc='Structured Grid Interp', disable=not verbose):
        phi_b = phis[b] # R
        apod_b = apods[b] # R
        
        # Apodization-free stencil columns and targets on the dalpha table
        E = torch.exp(-2j * np.pi * phi_b * (offsets[:, None] * db)) # W R
        target = torch.exp(-2j * np.pi * phi_b * dalpha_table[:, None]) # D R
        
        # Alternate between table weights and apodization
        for _ in range(num_apod_iter):
            gram = _grid_gram(phi_b, apod_b, W, db, lamda)
            w_table = _grid_weights_solve(phi_b, apod_b, gram, 
                                          dalpha_table, offsets, db) # D W
            pred = w_table @ E # D R
            numer = (pred.conj() * target).sum(dim=0)
            denom = (pred.abs() ** 2).sum(dim=0)
            apod_b = numer / denom.clamp(min=1e-12 * denom.max())
        apods[b] = apod_b
        
        # Final weight table with the fitted apodization
        gram = _grid_gram(phi_b, apod_b, W, db, lamda)
        w_table = _grid_weights_solve(phi_b, apod_b, gram, 
                                      dalpha_table, offsets, db) # D W
        
        # Linearly interpolate the table at each time point's actual dalpha
        pos = (dalphas[b] / db + 0.5) * (D - 1) # T, in [0, D-1]
        left = pos.floor().long().clamp(0, D - 2) # T
        frac = (pos - left).clamp(0, 1)[:, None].type(complex_dtype) # T 1
        weights[b] = (1 - frac) * w_table[left] + frac * w_table[left + 1]
    
    # Per-time-point kron of stencil weights (first dimension varies slowest, 
    # matching the mixed-radix grid index flattening in _grid_unique_betas).
    sparse_coeffs = torch.ones((T, 1), dtype=complex_dtype, device=torch_dev)
    for b in range(B):
        sparse_coeffs = (sparse_coeffs[:, :, None] 
                         * weights[b][:, None, :]).reshape((T, -1))
    
    # Unique grid points define the Q betas and per-time-point stencil indices
    betas, sparse_inds = _grid_unique_betas(grid_crds, db, alphas.dtype)
    
    # # 3D figure
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # inds = torch.randperm(T)[:10_000]
    # ax.scatter(betas[:, 0].cpu(), betas[:, 1].cpu(), betas[:, 2].cpu(), marker='x')
    # ax.scatter(alphas[0, inds].cpu(), alphas[1, inds].cpu(), alphas[2, inds].cpu(), marker='.', alpha=0.4)
    # plt.show()
    # quit()
    
    apod = apods.prod(dim=0) # R
    return betas, apod, sparse_inds, sparse_coeffs

def _grid_gram(phi: torch.Tensor,
               apod: torch.Tensor,
               grid_width: int,
               grid_spacing: float,
               lamda: float = 0.0) -> torch.Tensor:
    """
    Gram matrix of the apodized 1D stencil columns a(r) exp(-j2pi phi(r) g_w db).
    Since the stencil coordinates are always W consecutive integers, the Gram 
    entries only depend on the coordinate differences (w1 - w2), making the 
    matrix identical for every time point.
    
    Args
    ----
    phi : torch.Tensor
        single spatial phase basis with shape (R,)
    apod : torch.Tensor
        spatial apodization with shape (R,)
    grid_width : int
        width of the interpolation stencil
    grid_spacing : float
        spacing of the beta grid
    lamda : float
        Tikhonov regularization added to the diagonal
        
    Returns
    -------
    gram : torch.Tensor
        (regularized) Gram matrix with shape (W, W)
    """
    W = grid_width
    torch_dev = phi.device
    diffs = torch.arange(-(W - 1), W, device=torch_dev) * grid_spacing # 2W-1
    h = einsum(apod.abs().square().type(complex_dtype), 
               torch.exp(-2j * np.pi * phi[:, None] * diffs), 
               'R, R D -> D') # 2W-1
    # gram[k, l] = (A^H A)[k, l] = h((l - k) * db) for consecutive stencil coords
    idxs = torch.arange(W, device=torch_dev)
    gram = h[idxs[None, :] - idxs[:, None] + W - 1] # W W
    gram = gram + lamda * torch.eye(W, dtype=complex_dtype, device=torch_dev)
    return gram

def _grid_weights_solve(phi: torch.Tensor,
                        apod: torch.Tensor,
                        gram: torch.Tensor,
                        dalpha: torch.Tensor,
                        offsets: torch.Tensor,
                        grid_spacing: float) -> torch.Tensor:
    """
    Solves the 1D stencil weight problem for each dalpha sample
    min_w sum_r |exp(-j2pi phi(r) dalpha[d]) - a(r) sum_w exp(-j2pi phi(r) s[w] db) w[d,w]|^2
    via the normal equations gram @ w = rhs.
    
    Args
    ----
    phi : torch.Tensor
        single spatial phase basis with shape (R,)
    apod : torch.Tensor
        spatial apodization with shape (R,)
    gram : torch.Tensor
        precomputed Gram matrix from _grid_gram with shape (W, W)
    dalpha : torch.Tensor
        dalpha samples with shape (D,)
    offsets : torch.Tensor
        stencil offsets (in units of grid_spacing) with shape (W,)
    grid_spacing : float
        spacing of the beta grid
        
    Returns
    -------
    weights : torch.Tensor
        stencil weights with shape (D, W)
    """
    # rhs_w = sum_r conj(a(r)) exp(-j2pi phi(r) (dalpha[d] - s[w] db))
    deltas = dalpha[:, None] - offsets * grid_spacing # D W
    rhs = einsum(apod.conj(), 
                 torch.exp(-2j * np.pi * phi * deltas[..., None]), 
                 'R, D W R -> D W')
    return torch.linalg.solve(gram, rhs[..., None])[..., 0]

def _barycentric_coeffs(alphas: torch.Tensor,
                        betas: torch.Tensor,
                        sparse_inds: torch.Tensor,
                        temporal_batch_size: int,
                        nonneg: bool = True,
                        lamda: float = 0.0,
                        verbose: bool = True) -> torch.Tensor:
    """
    Solves for barycentric (sum-to-one) interpolation coefficients such that
    alpha(t) ~= sum_k betas[sparse_inds[k,t]] * sparse_coeffs[k,t]
    in a least squares sense, subject to sum_k sparse_coeffs[k,t] = 1.
    
    The sum-to-one constraint ensures the temporal midpoint factor in the
    spatial bases factors out cleanly during reconstruction.
    
    If nonneg is True, the coefficients are additionally constrained to be 
    non-negative, making alpha(t) a convex combination of its representatives.
    Since the resulting problem is a small convex QP (with the global optimum 
    living on one of the faces spanned by a subset of the K representatives), 
    and K is small, we solve it exactly via active-set enumeration: for each 
    non-empty subset of representatives we solve the equality-constrained 
    problem on that face, discard infeasible (negative) solutions, and keep the 
    lowest-residual feasible solution per time point.
    
    Args
    ----
    alphas : torch.Tensor
        (centered) temporal coefficients with shape (B, T)
    betas : torch.Tensor
        representative alpha vectors with shape (Q, B)
    sparse_inds : torch.Tensor
        sparse representative indices with shape (T, K) in [0, Q)
    temporal_batch_size : int
        batch size over the temporal dimension to limit memory
    nonneg : bool
        whether to additionally enforce non-negative coefficients
    lamda : float
        Regularization parameter for the sparse coefficients
    verbose : bool
        whether to print progress
        
    Returns
    -------
    sparse_coeffs : torch.Tensor
        sparse coefficients with shape (T, K)
    """
    T, K = sparse_inds.shape
    torch_dev = alphas.device
    dtype = betas.dtype
    alphas_T = alphas.T # T B
    
    # Precompute the active sets (non-empty subsets of the K representatives)
    if nonneg:
        assert K <= 20, f"Active-set enumeration is exponential in K (got K={K})."
        mask_ints = torch.arange(1, 1 << K, device=torch_dev)[:, None] # S 1
        active_sets = ((mask_ints >> torch.arange(K, device=torch_dev)) & 1).to(dtype) # S K
        eye = torch.eye(K + 1, dtype=dtype, device=torch_dev)
    
    sparse_coeffs = torch.zeros((T, K), dtype=complex_dtype, device=torch_dev)
    for t1 in tqdm(range(0, T, temporal_batch_size), 
                   desc='Barycentric Sparse Solve', 
                   disable=not verbose):
        t2 = min(t1 + temporal_batch_size, T)
        Tb = t2 - t1
        
        # Selected representatives and (real) normal equations for this batch
        selected_betas = betas[sparse_inds[t1:t2]] # Tb K B
        BtB = einsum(selected_betas, selected_betas, 'Tb K1 B, Tb K2 B -> Tb K1 K2') # Tb K K
        Bta = einsum(selected_betas, alphas_T[t1:t2], 'Tb K B, Tb B -> Tb K') # Tb K
        
        # KKT system enforcing sum_k c_k = 1:
        # [[BtB, 1], [1^T, 0]] @ [c; lam] = [Bta; 1]
        base_M = torch.zeros((Tb, K + 1, K + 1), dtype=dtype, device=torch_dev)
        base_M[:, :K, :K] = BtB + lamda * torch.eye(K, dtype=dtype, device=torch_dev)
        base_M[:, :K, K] = 1.0
        base_M[:, K, :K] = 1.0
        base_rhs = torch.zeros((Tb, K + 1), dtype=dtype, device=torch_dev)
        base_rhs[:, :K] = Bta
        base_rhs[:, K] = 1.0
        
        # Sum-to-one only (no active-set enumeration needed)
        if not nonneg:
            sol = (torch.linalg.pinv(base_M) @ base_rhs[..., None])[..., 0]
            sparse_coeffs[t1:t2] = sol[:, :K].to(complex_dtype)
            continue
        
        # Active-set enumeration over all non-empty subsets of representatives
        best_res = torch.full((Tb,), float('inf'), dtype=dtype, device=torch_dev)
        best_coeffs = torch.zeros((Tb, K), dtype=dtype, device=torch_dev)
        for active in active_sets:
            # Force inactive coefficients to zero: replace their KKT rows with 
            # identity rows (lam row, index K, is always kept active).
            row_keep = torch.cat([active, torch.ones(1, dtype=dtype, device=torch_dev)]) # K+1
            M = base_M * row_keep[None, :, None] + (eye * (1 - row_keep))[None]
            rhs = base_rhs * row_keep[None, :]
            
            c = torch.linalg.solve(M, rhs[..., None])[..., 0][:, :K] # Tb K
            
            # Feasibility (non-negativity) and least-squares residual (up to a 
            # constant ||alpha||^2 that is shared across all subsets)
            feasible = (c >= -1e-9).all(dim=-1) # Tb
            res = einsum(c, BtB, c, 'Tb K1, Tb K1 K2, Tb K2 -> Tb') \
                  - 2 * einsum(c, Bta, 'Tb K, Tb K -> Tb') # Tb
            
            improve = feasible & (res < best_res)
            best_res = torch.where(improve, res, best_res)
            best_coeffs = torch.where(improve[:, None], c, best_coeffs)
        
        sparse_coeffs[t1:t2] = best_coeffs.to(complex_dtype)
    
    return sparse_coeffs

def _rbf_beta_neighbor_dists(betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise beta distances (Q, Q) with diagonal masked, and neighbor indices (Q,)."""
    diff = betas[:, None, :] - betas[None, :, :] # Q Q B
    dists = diff.norm(dim=-1).clone() # Q Q
    dists.fill_diagonal_(float('inf'))
    nn = dists.argmin(dim=-1) # Q
    return dists, nn

def _rbf_sigma_beta_spacing(betas: torch.Tensor,
                            sigma_reg: float = 1e-4) -> torch.Tensor:
    """Isotropic RBF covariances from nearest-neighbor beta spacing."""
    dists, _ = _rbf_beta_neighbor_dists(betas)
    B = betas.shape[1]
    scale = dists.min(dim=-1).values.square().clamp(min=sigma_reg ** 2) # Q
    eye_B = torch.eye(B, dtype=betas.dtype, device=betas.device)
    return scale[:, None, None] * eye_B + sigma_reg * eye_B

def _rbf_empirical_cov(assigned: torch.Tensor,
                       sigma_reg: float = 1e-4) -> torch.Tensor:
    """Sample covariance of assigned alpha vectors (n, B) with regularization."""
    n, B = assigned.shape
    eye = torch.eye(B, dtype=assigned.dtype, device=assigned.device)
    centered = assigned - assigned.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(n - 1, 1)
    return cov + sigma_reg * eye

def _rbf_sigma_nearest_beta_cov(alphas: torch.Tensor,
                                betas: torch.Tensor,
                                sigma_reg: float = 1e-4) -> torch.Tensor:
    """
    Per-beta RBF covariances from the empirical covariance of all alphas whose 
    nearest representative is beta_q. Uses beta-spacing isotropic covariances as 
    a fallback when fewer than B+1 samples are assigned to a cell.
    """
    B, T = alphas.shape
    Q = betas.shape[0]
    alphas_T = alphas.T # T B
    
    neg_dists = 2 * (alphas_T @ betas.T) - (betas ** 2).sum(dim=-1) # T Q
    nearest = neg_dists.argmax(dim=-1) # T
    
    sigmas = _rbf_sigma_beta_spacing(betas, sigma_reg)
    for q in range(Q):
        assigned = alphas_T[nearest == q]
        if assigned.shape[0] <= B:
            continue
        sigmas[q] = _rbf_empirical_cov(assigned, sigma_reg)
    return sigmas

def _rbf_kernel_values(alphas: torch.Tensor,
                       betas: torch.Tensor,
                       sigmas: torch.Tensor) -> torch.Tensor:
    """
    Evaluates Gaussian RBF kernels h_q(alpha) = exp(-1/2 (alpha-beta_q)^T 
    Sigma_q^{-1} (alpha-beta_q)) at all time points.
    
    Args
    ----
    alphas : torch.Tensor
        (centered) temporal coefficients with shape (B, T)
    betas : torch.Tensor
        representative alpha vectors with shape (Q, B)
    sigmas : torch.Tensor
        covariance matrices with shape (Q, B, B)
        
    Returns
    -------
    h : torch.Tensor
        RBF values with shape (T, Q)
    """
    T = alphas.shape[1]
    Q = betas.shape[0]
    alphas_T = alphas.T # T B
    
    h = torch.zeros((T, Q), dtype=betas.dtype, device=alphas.device)
    for q in range(Q):
        diff = alphas_T - betas[q] # T B
        inv_sigma = torch.linalg.inv(sigmas[q])
        mahal = einsum(diff, inv_sigma, diff, 'T b1, b1 b2, T b2 -> T')
        h[:, q] = torch.exp(-0.5 * mahal)
    return h

def _rbf_coeffs(phis: torch.Tensor,
                alphas_target: torch.Tensor,
                alphas_centered: torch.Tensor,
                betas: torch.Tensor,
                spatial_bases: torch.Tensor,
                sparse_inds: torch.Tensor,
                temporal_batch_size: int,
                num_rnd_voxels: Optional[int] = None,
                lamda: float = 0.0,
                voxel_inds: Optional[torch.Tensor] = None,
                verbose: bool = True) -> torch.Tensor:
    """
    Unstructured RBF interpolation: estimates global complex weights w_q and 
    forms sparse coefficients c_{m,k} = w_{Z_{m,k}} h_{Z_{m,k}}(alpha(t_m)).
    
    The weights solve
    min_w sum_m || sum_k w_{Z_{m,k}} h_{Z_{m,k}}(alpha_m) b_{Z_{m,k}}(r) 
              - exp(-j2pi phi(r) . alpha(t_m)) ||_2^2
    over all spatial locations r, where h_q are Gaussian RBFs centered at beta_q.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis with shape (B, R)
    alphas_target : torch.Tensor
        full temporal coefficients (with midpoint restored) with shape (B, T)
    alphas_centered : torch.Tensor
        centered temporal coefficients used for RBF evaluation with shape (B, T)
    betas : torch.Tensor
        representative alpha vectors with shape (Q, B)
    spatial_bases : torch.Tensor
        spatial bases with shape (Q, R)
    sparse_inds : torch.Tensor
        sparse representative indices with shape (T, K) in [0, Q)
    temporal_batch_size : int
        batch size over the temporal dimension to limit memory
    num_rnd_voxels : int
        number of random voxels to sample for the normal equations solve
    lamda : float
        Tikhonov regularization on the global weights
    voxel_inds : Optional[torch.Tensor]
        Flat voxel indices for subsampled normal equations. None uses all voxels.
    verbose : bool
        whether to print progress
        
    Returns
    -------
    sparse_coeffs : torch.Tensor
        sparse coefficients with shape (T, K)
    """
    T, K = sparse_inds.shape
    Q = betas.shape[0]
    R = phis.shape[1]
    torch_dev = spatial_bases.device
    
    if voxel_inds is not None:
        phis = phis[:, voxel_inds]
        spatial_bases = spatial_bases[:, voxel_inds]
    
    sigmas = _rbf_sigma_beta_spacing(betas)
    # sigmas = _rbf_sigma_nearest_beta_cov(alphas_centered, betas)
    sigmas *= 2
    h = _rbf_kernel_values(alphas_centered, betas, sigmas) # T Q
    
    h_active = h.gather(1, sparse_inds) # T K
    h_active = h_active / h_active.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    
    # Masked RBF values for the global weight solve (only K bases active per m).
    h_masked = torch.zeros((T, Q), dtype=h.dtype, device=torch_dev)
    h_masked.scatter_(1, sparse_inds, h_active)
    
    BHB = einsum(spatial_bases.conj(), spatial_bases, 
                 'Q1 R, Q2 R -> Q1 Q2')
    lamda_I = lamda * torch.eye(Q, dtype=complex_dtype, device=torch_dev)
    
    # Random voxel inds
    voxel_count = R
    
    G = torch.zeros((Q, Q), dtype=complex_dtype, device=torch_dev)
    rhs = torch.zeros(Q, dtype=complex_dtype, device=torch_dev)
    for t1 in tqdm(range(0, T, temporal_batch_size), 
                   desc='RBF Sparse Solve', 
                   disable=not verbose):
        t2 = min(t1 + temporal_batch_size, T)
        
        # Grab random voxel indices
        if num_rnd_voxels is None:
            voxel_inds = slice(None)
        else:
            if voxel_count + num_rnd_voxels > R:
                voxel_inds_all = torch.randperm(R, device=torch_dev)
                voxel_count = 0
            slc = slice(
                voxel_count,
                voxel_count + num_rnd_voxels)
            voxel_inds = voxel_inds_all[slc]
            voxel_count += num_rnd_voxels
        
        target = torch.exp(-2j * np.pi * einsum(phis[:, voxel_inds], 
                                                alphas_target[:, t1:t2], 
                                                'B R, B Tb -> Tb R'))
        h_batch = h_masked[t1:t2].type(complex_dtype) # Tb Q
        
        G = G + einsum(h_batch.conj(), BHB, h_batch, 
                       'Tb q1, q1 q2, Tb q2 -> q1 q2')
        AHb = einsum(spatial_bases[:, voxel_inds].conj(), target, 'Q R, Tb R -> Tb Q')
        rhs = rhs + einsum(AHb, h_batch, 'Tb Q, Tb Q -> Q')
    
    weights = torch.linalg.solve(G + lamda_I, rhs[..., None])[..., 0] # Q
    
    sparse_coeffs = weights[sparse_inds] * h_active
    return sparse_coeffs.to(complex_dtype)

def _lstsq_coeffs(phis: torch.Tensor,
                  alphas: torch.Tensor,
                  spatial_bases: torch.Tensor,
                  sparse_inds: torch.Tensor,
                  temporal_batch_size: int,
                  lamda: float = 0.0,
                  voxel_inds: Optional[torch.Tensor] = None,
                  verbose: bool = True) -> torch.Tensor:
    """
    Solves for the sparse coefficients that minimize, per time point t,
    || exp(-j2pi phi(r) . alpha(t)) - sum_k b[sparse_inds[k,t]](r) * sparse_coeffs[k,t] ||^2
    over spatial locations r, in a least squares sense.
    
    Args
    ----
    phis : torch.Tensor
        spatial phase basis with shape (B, R)
    alphas : torch.Tensor
        temporal coefficients with shape (B, T)
    spatial_bases : torch.Tensor
        spatial bases with shape (Q, R)
    sparse_inds : torch.Tensor
        sparse representative indices with shape (T, K) in [0, Q)
    temporal_batch_size : int
        batch size over the temporal dimension to limit memory
    lamda : float
        Regularization parameter for the sparse coefficients
    voxel_inds : Optional[torch.Tensor]
        Flat voxel indices for subsampled A^H b. None uses all voxels.
        
    Returns
    -------
    sparse_coeffs : torch.Tensor
        sparse coefficients with shape (T, K)
    """
    T, K = sparse_inds.shape
    torch_dev = spatial_bases.device
    
    if voxel_inds is not None:
        phis = phis[:, voxel_inds]
        spatial_bases = spatial_bases[:, voxel_inds]
    
    lamda_I = lamda * torch.eye(K, dtype=complex_dtype, device=torch_dev)
    BHB = einsum(spatial_bases.conj(), spatial_bases, 
                 'Q1 R, Q2 R -> Q1 Q2')
    
    sparse_coeffs = torch.zeros((T, K), dtype=complex_dtype, device=torch_dev)
    for t1 in tqdm(range(0, T, temporal_batch_size), 
                   desc='Full Least Squares Sparse Solve', 
                   disable=not verbose):
        t2 = min(t1 + temporal_batch_size, T)
        target = torch.exp(-2j * np.pi * einsum(phis, alphas[:, t1:t2], 
                                                'B R, B Tb -> Tb R'))
        A = spatial_bases[sparse_inds[t1:t2]]
        AHA = BHB[sparse_inds[t1:t2, :, None], sparse_inds[t1:t2, None, :]] 
        AHb = einsum(A.conj(), target, 'Tb K R, Tb R -> Tb K')
        sparse_coeffs[t1:t2] = torch.linalg.solve(AHA + lamda_I, 
                                                  AHb[..., None])[..., 0]
    
    return sparse_coeffs
    