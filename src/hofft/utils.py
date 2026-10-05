"""
Contains tools for reducing the spatial dimension size.
"""

import itertools

import torch
import torch.nn.functional as F
from typing import Optional
from flash_kmeans import batch_kmeans_Euclid
from fast_pytorch_kmeans import KMeans

__all__ = [
    'resize',
    'gen_grd',
    'normalize',
    'spatial_interp',
    'spatial_resize_poly',
    'reduce_spatial',
    'expand_spatial',
    'reduce_temporal',
    'expand_temporal',
]

# ---------------- General ----------------
def _expand_shapes(*shapes):
    """
    Given iterable of shapes, returns shapes with appropriate empty 
    dimensions such that all returned shapes have the same number of dimensions.
    
    Direct copy from Frank Ong's Sigpy library.
    """
    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape for shape in shapes]

    return tuple(shapes_exp)

def resize(input, oshape, ishift=None, oshift=None):
    """
    Resize with zero-padding or cropping.
    
    Direct copy from Frank Ong's Sigpy library.

    Parameters:
    -----------
    input : torch.Tensor
        Input array with arb shape
    oshape : tuple
        Output shape with same number of dimensions as input
    ishift : tuple
        Shift of input array.
    oshift : tuple
        Shift of output array.

    Returns:
        array: Zero-padded or cropped result.
    """
    
    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [
        min(i - si, o - so)
        for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)
    ]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape1, dtype=input.dtype, device=input.device)
    input = input.reshape(ishape1)
    output[oslice] = input[islice]

    return output.reshape(oshape)

def gen_grd(im_size: tuple, 
            fovs: Optional[tuple] = None,
            balanced: Optional[bool] = False) -> torch.Tensor:
    """
    Generates a grid of points given image size and FOVs

    Parameters:
    -----------
    im_size : tuple
        image dimensions
    fovs : tuple
        field of views, same size as im_size
    
    Returns:
    --------
    grd : torch.Tensor
        grid of points with shape (*im_size, len(im_size))
    """
    if fovs is None:
        fovs = (1,) * len(im_size)
    if balanced:
        lins = [
            fovs[i] * torch.linspace(-1/2, 1/2, im_size[i]) 
            for i in range(len(im_size))
            ]
    else:
        lins = [
            fovs[i] * torch.arange(-(im_size[i]//2), im_size[i]//2 + (im_size[i] % 2)) / (im_size[i]) 
            for i in range(len(im_size))
            ]
    grds = torch.meshgrid(*lins, indexing='ij')
    grd = torch.cat(
        [g[..., None] for g in grds], dim=-1)
        
    return grd.type(torch.float32)

def normalize(shifted, target, ofs=False, mag=True):
    """
    Assumes the following scaling/shifting offset:

    shifted = a * target + b

    solves for a, b and returns the corrected data

    Parameters:
    -----------
    shifted : np.ndarray
        data to be corrected
    target : np.ndarray
        reference data
    ofs : bool
        include b offset in the correction
    mag : bool
        use magnitude of data for correction
    
    Returns:
    --------
    np.ndarray
        corrected data
    """
    if mag:
        col1 = shifted.abs().flatten()
        y = target.abs().flatten()
    else:
        col1 = shifted.flatten()
        y = target.flatten()

    if ofs:
        col2 = col1 * 0 + 1
        A = torch.stack([col1, col2], dim=-1)
        a, b = torch.linalg.lstsq(A, y, rcond=None).solution
    else:
        b = 0
        a = torch.linalg.lstsq(col1[:, None], y, rcond=None).solution

    out = a * shifted + b
    return out

# ---------------- Interpolation ----------------
def _cubic_kernel(t: torch.Tensor, a: float = -0.75) -> torch.Tensor:
    """
    Keys cubic convolution kernel, evaluated at (signed) distances `t`.
    a=-0.75 matches the coefficient used by torch's own grid_sample bicubic
    mode, so the two interpolation paths below agree with each other.
    """
    t = t.abs()
    t2 = t * t
    t3 = t2 * t
    near = (a + 2) * t3 - (a + 3) * t2 + 1
    far = a * t3 - 5 * a * t2 + 8 * a * t - 4 * a
    return torch.where(t <= 1, near, torch.where(t < 2, far, torch.zeros_like(t)))

def _interp_taps_and_weights(coord: torch.Tensor,
                             size: int,
                             order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1D interpolation stencil for a batch of continuous coordinates.

    Args
    ----
    coord : torch.Tensor
        continuous coordinates with shape (M,), in index units [0, size - 1]
    size : int
        extent of this dimension
    order : int
        0 (nearest), 1 (linear), or 3 (cubic convolution)

    Returns
    -------
    idxs : torch.Tensor
        tap indices (long), clamped to [0, size - 1] ('nearest' boundary mode),
        with shape (M, T)
    weights : torch.Tensor
        per-tap weights (same dtype as coord) summing to 1 per row, shape (M, T)
    """
    if order == 0:
        idx = coord.round().long().clamp(0, size - 1)
        weights = torch.ones_like(idx, dtype=coord.dtype)
        return idx[:, None], weights[:, None]
    elif order == 1:
        floor = coord.floor()
        frac = coord - floor
        idx0 = floor.long()
        idxs = torch.stack([idx0, idx0 + 1], dim=-1).clamp(0, size - 1)
        weights = torch.stack([1 - frac, frac], dim=-1)
        return idxs, weights
    elif order == 3:
        floor = coord.floor()
        frac = coord - floor
        idx0 = floor.long()
        offsets = torch.arange(-1, 3, device=coord.device, dtype=coord.dtype)  # (4,)
        idxs = (idx0[:, None] + offsets[None, :].long()).clamp(0, size - 1)
        dists = offsets[None, :] - frac[:, None]
        weights = _cubic_kernel(dists)
        return idxs, weights
    else:
        raise NotImplementedError(f'order={order} is not supported (only 0, 1, 3 are implemented)')

def _interp_general_nd(spatial_input: torch.Tensor,
                       coords: torch.Tensor,
                       order: int) -> torch.Tensor:
    """
    Fully general N-D separable polynomial interpolation (a map_coordinates
    equivalent), implemented from scratch in pure PyTorch. Supports any
    number of spatial dimensions K, with 'nearest' (clamp-to-edge) boundary
    handling. This is the fallback path used whenever the faster
    ``_interp_grid_sample`` can't handle the requested (K, order) combo
    (e.g. cubic interpolation in 3D, or K > 3).

    Args
    ----
    spatial_input : torch.Tensor
        input tensor with shape (N, d0, ..., d_{K-1})
    coords : torch.Tensor
        target coordinates with shape (M, K)
    order : int
        0, 1, or 3 (see ``_interp_taps_and_weights``)

    Returns
    -------
    out : torch.Tensor
        interpolated values with shape (N, M)
    """
    N, *im_size = spatial_input.shape
    K = len(im_size)
    torch_dev = spatial_input.device
    x_flt = spatial_input.reshape(N, -1)

    per_dim_idx, per_dim_w = [], []
    for d in range(K):
        idxs, weights = _interp_taps_and_weights(coords[:, d], im_size[d], order)
        per_dim_idx.append(idxs)   # (M, T)
        per_dim_w.append(weights)  # (M, T)
    T = per_dim_idx[0].shape[1]

    strides = [1] * K
    for d in range(K - 2, -1, -1):
        strides[d] = strides[d + 1] * im_size[d + 1]

    M = coords.shape[0]
    out = torch.zeros((N, M), dtype=spatial_input.dtype, device=torch_dev)
    for combo in itertools.product(range(T), repeat=K):
        flat_idx = sum(per_dim_idx[d][:, combo[d]] * strides[d] for d in range(K))
        weight = per_dim_w[0][:, combo[0]]
        for d in range(1, K):
            weight = weight * per_dim_w[d][:, combo[d]]
        out = out + x_flt[:, flat_idx] * weight[None, :].to(x_flt.dtype)

    return out

def _interp_grid_sample(spatial_input: torch.Tensor,
                        coords: torch.Tensor,
                        order: int) -> torch.Tensor:
    """
    1D/2D/3D map_coordinates equivalent backed by ``torch.nn.functional.grid_sample``,
    with 'nearest' (clamp-to-edge, via padding_mode='border') boundary handling.
    Supports order 0/1 in 1D, 2D, and 3D, and order 3 (native bicubic) in 1D
    and 2D only -- torch has no tricubic grid_sample mode, so 3D cubic
    interpolation must go through ``_interp_general_nd`` instead. 1D is done
    via the standard grid_sample trick of adding a dummy size-1 spatial axis
    and treating it as 2D with H=1 (its weights sum to 1, so the dummy axis
    cancels out exactly).

    Args
    ----
    spatial_input : torch.Tensor
        input tensor with shape (N, *im_size), K = len(im_size) in {1, 2, 3}
    coords : torch.Tensor
        target coordinates with shape (M, K)
    order : int
        0, 1, or (1D/2D only) 3

    Returns
    -------
    out : torch.Tensor
        interpolated values with shape (N, M)
    """
    N, *im_size = spatial_input.shape
    K = len(im_size)
    M = coords.shape[0]
    gs_mode = {0: 'nearest', 1: 'bilinear', 3: 'bicubic'}[order]
    if K == 3 and gs_mode == 'bicubic':
        raise NotImplementedError('torch grid_sample has no tricubic mode for 3D input')

    # grid_sample expects normalized coords in (x, y[, z]) order, i.e. reversed
    # relative to the (d0, ..., d_{K-1}) axis order used everywhere else here.
    size = torch.tensor(im_size, dtype=coords.dtype, device=coords.device)
    norm = (2 * coords / (size - 1).clamp(min=1) - 1).flip(-1)

    def _sample(x: torch.Tensor) -> torch.Tensor:
        inp = x.unsqueeze(1)  # (N, 1, *im_size)
        if K == 1:
            inp = inp.unsqueeze(2)  # (N, 1, 1, size) -- dummy H=1 axis
            grid = torch.cat([norm, -torch.ones_like(norm)], dim=-1)  # (M, 2), dummy y
            grid = grid.view(1, M, 1, 2).expand(N, M, 1, 2)
            out = F.grid_sample(inp, grid, mode=gs_mode, padding_mode='border', align_corners=True)
            return out[:, 0, :, 0]  # (N, M)
        elif K == 2:
            grid = norm.view(1, M, 1, 2).expand(N, M, 1, 2)
            out = F.grid_sample(inp, grid, mode=gs_mode, padding_mode='border', align_corners=True)
            return out[:, 0, :, 0]  # (N, M)
        else:
            grid = norm.view(1, M, 1, 1, 3).expand(N, M, 1, 1, 3)
            out = F.grid_sample(inp, grid, mode=gs_mode, padding_mode='border', align_corners=True)
            return out[:, 0, :, 0, 0]  # (N, M)

    if spatial_input.is_complex():
        return torch.complex(_sample(spatial_input.real), _sample(spatial_input.imag))
    return _sample(spatial_input)

def spatial_interp(spatial_input: torch.Tensor,
                   coords: torch.Tensor,
                   order: Optional[int] = 3,
                   mode: Optional[str] = 'nearest') -> torch.Tensor:
    """
    Perform polynomial interpolation on the spatial dimensions of a tensor
    at specified coordinates. Pure PyTorch (no scipy/cupy/sigpy): 1D/2D/3D go
    through the fast ``torch.nn.functional.grid_sample``-backed path, and
    everything else (including 3D cubic) falls back to a fully general N-D
    separable interpolator. order=3 uses cubic convolution (Keys, a=-0.75,
    matching torch's own bicubic) rather than scipy's prefiltered
    interpolating B-spline -- visually/numerically very close, but not
    bit-identical to the old scipy.ndimage.map_coordinates results.

    Args
    ----
    spatial_input : torch.Tensor
        Input tensor of shape (N, d0, d1, ..., d_{K-1}).
    coords : torch.Tensor
        target coordinates with shape (M, K)
    order : int, optional
        The order of the spline interpolation (default is cubic, order=3).
        Only 0 (nearest), 1 (linear), and 3 (cubic) are implemented.
    mode : str, optional
        How to handle points outside the boundaries. Only 'nearest'
        (clamp-to-edge) is implemented.

    Returns
    -------
    out : torch.Tensor
        An array of interpolated values with shape (N, M). That is, for each
        batch element, the tensor is evaluated at the provided coordinates.
    """
    assert mode == 'nearest', f"only mode='nearest' is supported, got {mode!r}"
    assert order in (0, 1, 3), f"order={order} is not supported (only 0, 1, 3 are implemented)"
    K = coords.shape[-1]
    assert spatial_input.ndim == K + 1, "Input tensor must have K spatial dimensions."
    N = spatial_input.shape[0]
    crds_flt = coords.reshape((-1, K))

    can_grid_sample = K in (1, 2, 3) and not (K == 3 and order == 3)
    interp = _interp_grid_sample if can_grid_sample else _interp_general_nd
    # 3D cubic especially: _interp_general_nd stores (M, 4) idx/weight per
    # axis. Chunk so a 320^3 upsample does not allocate multi-GB temporaries.
    max_m = 1 << 20
    Mtot = crds_flt.shape[0]
    if Mtot <= max_m:
        out = interp(spatial_input, crds_flt, order)
    else:
        pieces = []
        for m1 in range(0, Mtot, max_m):
            m2 = min(m1 + max_m, Mtot)
            pieces.append(interp(spatial_input, crds_flt[m1:m2], order))
        out = torch.cat(pieces, dim=1)

    out = out.reshape((N, *coords.shape[:-1]))  # Reshape to (N, *crds_size)
    return out

def spatial_resize_poly(x: torch.Tensor,
                        im_size: tuple,
                        order: Optional[int] = 3,
                        mode: Optional[str] = 'nearest') -> torch.Tensor:
    """
    Resize a spatial tensor to a new spatial size.

    Parameters:
    -----------
    x : (torch.Tensor)
        The input tensor with shape (..., *inp_im_size)
    im_size : (tuple)
        The size of the image to resize to
    order : int, optional
        The order of the spline interpolation (default is cubic, order=3).
    mode : str, optional
        How to handle points outside the boundaries (default 'nearest').

    Returns:
    --------
    x_rs : (torch.Tensor)
        The resized tensor with shape (..., *im_size)
    """
    # Handle batch dims
    if x.ndim == len(im_size):
        x = x[None,]
        squeeze = True
        oshape = (1, *im_size)
    else:
        oshape = (*x.shape[:-len(im_size)], *im_size)
        x = x.reshape((-1, *x.shape[-len(im_size):]))
        squeeze = False

    inp_size = x.shape[-len(im_size):]
    kwargs = {'order': order, 'mode': mode}
    K = len(im_size)
    N = x.shape[0]
    torch_dev = x.device
    M = 1
    for n in im_size:
        M *= int(n)

    # Same mapping as (gen_grd(im_size, balanced=True) + 0.5) * (inp_size - 1),
    # but as 1D axes so we never meshgrid the full output FOV.
    axis_crds = [
        torch.linspace(0, inp_size[i] - 1, im_size[i], device=torch_dev, dtype=torch.float32)
        for i in range(K)
    ]

    max_m = 1 << 20
    x_rs_flt = torch.empty((N, M), dtype=x.dtype, device=torch_dev)
    for m1 in range(0, M, max_m):
        m2 = min(m1 + max_m, M)
        lin = torch.arange(m1, m2, device=torch_dev)
        ijk = torch.unravel_index(lin, im_size)
        spatial_crds = torch.stack([axis_crds[d][ijk[d]] for d in range(K)], dim=-1)
        x_rs_flt[:, m1:m2] = spatial_interp(x, spatial_crds, **kwargs)
    x_rs = x_rs_flt.reshape(oshape)

    if squeeze:
        return x_rs[0]
    else:
        return x_rs

def reduce_spatial(spatial_data: torch.Tensor,
                   im_size_low: tuple,
                   order: int = 3) -> torch.Tensor:
    """
    Reduce the spatial dimensions of the data using
    grid interpolation. Data must lie on a regular grid.

    Args
    ----
    data : torch.Tensor
        Data to reduce with shape (..., *im_size)
    im_size_low : tuple
        Low resolution spatial size
    order : int, optional
        Order of the polynomial interpolation

    Returns
    -------
    data_low : torch.Tensor
        Reduced data with shape (..., *im_size_low)
    """
    return spatial_resize_poly(spatial_data,
                               im_size=im_size_low,
                               order=order,
                               mode='nearest')

def expand_spatial(spatial_data: torch.Tensor,
                   im_size_high: tuple,
                   order: int = 3) -> torch.Tensor:
    """
    Expand the spatial dimensions of the data using
    grid interpolation. Data must lie on a regular grid.

    Args
    ----
    spatial_data : torch.Tensor
        Spatial data to expand with shape (..., *im_size)
    im_size_high : tuple
        High resolution spatial size
    order : int, optional
        Order of the polynomial interpolation

    Returns
    -------
    spatial_data_high : torch.Tensor
        Expanded spatial data with shape (..., *im_size_high)
    """
    return spatial_resize_poly(spatial_data,
                               im_size=im_size_high,
                               order=order,
                               mode='nearest')

def reduce_temporal(temporal_data: torch.Tensor,
                    num_time_low: int,
                    dim: int = -1,
                    order: int = 3) -> torch.Tensor:
    """
    Downsample the last (temporal) dimension by ``ds_factor`` using the same
    align-corners polynomial resize as :func:`reduce_spatial`. Endpoints are
    preserved, so ``expand_temporal(reduce_temporal(x), ...)`` approximately
    recovers smooth signals.

    Args
    ----
    temporal_data : torch.Tensor
        Temporal data to reduce with arb shape
    num_time_low : int
        Number of time points to reduce to
    order : int, optional
        Order of the polynomial interpolation (0, 1, or 3)
        
    Returns
    -------
    temporal_data_low : torch.Tensor
        Reduced temporal data 
    """
    temporal_data_low = temporal_data.moveaxis(dim, -1)
    temporal_data_low = spatial_resize_poly(temporal_data_low, im_size=(num_time_low,), order=order)
    temporal_data_low = temporal_data_low.moveaxis(-1, dim)
    return temporal_data_low

def expand_temporal(temporal_data: torch.Tensor,
                    num_time_high: int,
                    dim: int = -1,
                    order: int = 3) -> torch.Tensor:
    """
    Upsample the last (temporal) dimension by ``num_time_high`` using the same
    align-corners polynomial resize as :func:`expand_spatial`.

    Args
    ----
    temporal_data : torch.Tensor
        Temporal data to expand with arb shape
    num_time_high : int
        Integer upsampling factor
    order : int, optional
        Order of the polynomial interpolation (0, 1, or 3)

    Returns
    -------
    temporal_data_high : torch.Tensor
        Expanded temporal data
    """
    temporal_data_high = temporal_data.moveaxis(dim, -1)
    temporal_data_high = spatial_resize_poly(temporal_data_high, im_size=(num_time_high,), order=order)
    temporal_data_high = temporal_data_high.moveaxis(-1, dim)
    return temporal_data_high

# ---------------- Quantization and Sampling ----------------
def kmeans_centroids(data: torch.Tensor,
                     K: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find K centroids of the data using K-means clustering.
    
    Args
    ----
    data : torch.Tensor
        Data to find the centroids of with shape (N, d)
    K : int
        Number of centroids to find

    Returns
    -------
    centroids : torch.Tensor
        Centroids with shape (K, d)
    idxs : torch.Tensor
        Indices of cluster membership for each data sample with shape (N,) in [0, K-1]
    idx_nearest : torch.Tensor
        Indices of the nearest data sample for each centroid with shape (K,) in [0, N-1]
    """
    # Consts
    N, d = data.shape
    
    # ------------ Fast PyTorch KMeans ------------
    kmeans = KMeans(n_clusters=K)
    idxs = kmeans.fit_predict(data)
    centroids = kmeans.centroids
    
    
    # ------------ Flash KMeans ------------
    # # Pad dimensions to power of 2 -- annoying triton requirement
    # dpad = 1 << (d - 1).bit_length()   # 32 for D=23
    # if dpad < 16:
    #     dpad = 16
    # data_padded = torch.zeros((N, dpad), dtype=data.dtype, device=data.device)
    # data_padded[..., :d] = data
    
    # # Find centroids
    # idxs, centroids, _ = batch_kmeans_Euclid(data_padded[None,], n_clusters=K)
    # centroids = centroids[0, :, :d]
    # idxs = idxs[0]
    
    # Find nearest data sample for each centroid
    idx_nearest = torch.zeros(K, dtype=torch.long, device=data.device)
    for k in range(K):
        active_set = data[idxs == k]
        if active_set.shape[0] > 0:
            idx_nearest[k] = torch.argmin((centroids[k] - active_set).norm(dim=-1))
        else:
            idx_nearest[k] = torch.randint(0, N, (1,), device=data.device)
    
    # Return
    return centroids, idxs, idx_nearest

def maxmin_indices(vectors: torch.Tensor,
                   K: int,
                   seed: Optional[int] = None) -> torch.Tensor:
    """
    Greedy farthest-point (maxmin) index selection

    Args
    ----
    data : torch.Tensor
        Data to find the centroids indices of with shape (N, d)
    K : int
        Number of centroids to find
    seed : int, optional
        Random seed for the initial point

    Returns
    -------
    idx_nearest : torch.Tensor
        Indices of the data sample for each centroid with shape (K,) in [0, N-1]
    """
    # Consts
    N = vectors.shape[0]
    gen = torch.Generator(device=vectors.device)
    if seed is not None:
        gen.manual_seed(seed)
        
    # Keep indices on device: int(argmax)/int(randint) would sync every pivot.
    picked = torch.empty(K, dtype=torch.long, device=vectors.device)
    picked[0] = torch.randint(0, N, (), generator=gen, device=vectors.device)
    dist = torch.linalg.norm(vectors - vectors[picked[0]], dim=-1)
    for k in range(1, K):
        nxt = dist.argmax()
        picked[k] = nxt
        dist = torch.minimum(dist, torch.linalg.norm(vectors - vectors[nxt], dim=-1))
    return picked

def maxmin_centroids(data: torch.Tensor,
                     K: int,
                     seed: Optional[int] = None) -> torch.Tensor:
    """
    Find K centroids of the data using the maxmin algorithm.
    
    Args
    ----
    data : torch.Tensor
        Data to find the centroids of with shape (N, d)
    K : int
        Number of centroids to find 
    seed : int, optional
        Random seed for the initial point

    Returns
    -------
    centroids : torch.Tensor
        Centroids with shape (K, d)
    idxs : torch.Tensor
        Indices of cluster membership for each data sample with shape (N,) in [0, K-1]
    idx_nearest : torch.Tensor
        Indices of the nearest data sample for each centroid with shape (K,) in [0, N-1]
    """
    # Consts
    N = data.shape[0]
    
    # Find indices of nearest data samples
    idxs_nearest = maxmin_indices(data, K, seed)
    centroids = data[idxs_nearest]
    
    # Assign nearest data sample to each centroid
    idxs = torch.zeros(N, dtype=torch.long, device=data.device)
    nbs = 2 ** 10
    for n1 in range(0, N, nbs):
        n2 = min(n1 + nbs, N)
        idxs[n1:n2] = torch.argmin(torch.cdist(data[n1:n2], centroids), dim=-1)
    
    # Return
    return centroids, idxs_nearest

def fps_multi_center_indices(vectors: torch.Tensor,
                             K: int,
                             P: int = 1,
                             seed: Optional[int] = None) -> torch.Tensor:
    """
    Greedy farthest-point selection for the multi-center covering objective

        min_{centroids}  max_n  sum_{p=1}^P d_{n,p}

    where d_{n,p} is the p-th smallest of {||v_n - centroid_k||}_{k=1}^K.
    P = 1 reduces to standard maxmin / FPS: each new pick is the point whose
    nearest selected center is farthest away.

    Args
    ----
    vectors : torch.Tensor
        Data with shape (N, d)
    K : int
        Number of centroids to find
    P : int
        Number of nearest centers that enter the per-point cost. Clamped to [1, K].
    seed : int, optional
        Random seed for the initial point

    Returns
    -------
    idx_nearest : torch.Tensor
        Indices of the selected samples, shape (K,) in [0, N-1]
    """
    # Consts
    N = vectors.shape[0]
    K = min(K, N)
    P = max(1, min(P, K))
    gen = torch.Generator(device=vectors.device)
    if seed is not None:
        gen.manual_seed(seed)

    # dists_p[n, :p] = p smallest distances to the centers picked so far
    # (unused slots are +inf and do not contribute to the score)
    dists_p = torch.full((N, P), torch.inf, device=vectors.device, dtype=vectors.dtype)
    picked_mask = torch.zeros(N, dtype=torch.bool, device=vectors.device)
    picked = []

    nxt = int(torch.randint(0, N, (1,), generator=gen, device=vectors.device))
    for _ in range(K):
        picked.append(nxt)
        picked_mask[nxt] = True
        d_new = torch.linalg.norm(vectors - vectors[nxt], dim=-1)
        dists_p = torch.cat([dists_p, d_new[:, None]], dim=-1).sort(dim=-1).values[:, :P]

        finite = torch.isfinite(dists_p)
        score = torch.where(finite, dists_p, torch.zeros_like(dists_p)).sum(dim=-1)
        score = score.masked_fill(picked_mask, torch.finfo(vectors.dtype).min)
        nxt = int(score.argmax())

    return torch.tensor(picked, dtype=torch.long, device=vectors.device)

def fps_multi_center_centroids(data: torch.Tensor,
                               K: int,
                               P: int = 1,
                               seed: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find K centroids by greedy FPS on the multi-center covering objective

        min_{centroids}  max_n  sum_{p=1}^P d_{n,p}

    where d_{n,p} is the p-th smallest of {||v_n - centroid_k||}_{k=1}^K.
    P = 1 is exactly :func:`maxmin_centroids`.

    Args
    ----
    data : torch.Tensor
        Data to find the centroids of with shape (N, d)
    K : int
        Number of centroids to find
    P : int
        Number of nearest centers in the per-point cost
    seed : int, optional
        Random seed for the initial point

    Returns
    -------
    centroids : torch.Tensor
        Centroids with shape (K, d)
    idxs : torch.Tensor
        Indices of cluster membership for each data sample with shape (N,) in [0, K-1]
    idx_nearest : torch.Tensor
        Indices of the selected data samples, shape (K,) in [0, N-1]
    """
    # Consts
    N = data.shape[0]

    # Greedy multi-center FPS picks
    idx_nearest = fps_multi_center_indices(data, K, P=P, seed=seed)
    centroids = data[idx_nearest]

    # Assign each sample to its nearest selected center
    idxs = torch.zeros(N, dtype=torch.long, device=data.device)
    nbs = 2 ** 10
    for n1 in range(0, N, nbs):
        n2 = min(n1 + nbs, N)
        idxs[n1:n2] = torch.argmin(torch.cdist(data[n1:n2], centroids), dim=-1)

    return centroids, idxs, idx_nearest