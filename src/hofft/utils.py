"""
Contains tools for reducing the spatial dimension size.
"""

import itertools

import torch
import torch.nn.functional as F
from typing import Optional

__all__ = [
    'resize',
    'gen_grd',
    'normalize',
    'spatial_interp',
    'spatial_resize_poly',
    'reduce_spatial',
    'expand_spatial',
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

def lin_solve(AHA: torch.Tensor, 
              AHb: torch.Tensor, 
              lamda: Optional[float] = 0.0, 
              solver: Optional[int] = 'solve') -> torch.Tensor:
    """
    Solves (AHA + lamda I) @ x = AHb for x

    Args
    ----
    AHA : torch.Tensor
        square matrix with shape (..., n, n)
    AHb : torch.Tensor
        matrix with shape (..., n, m)
    lamda : float
        optional L2 regularization 
    solver : str
        'pinv' - pseudo inverse 
        'solve' - torch.linalg.solve
        'lstsq' - least squares
        'inv' - regular inverse
    
    Returns
    -------
    x : torch.Tensor
        solution with shape (..., n, m)
    """
    solver = solver.lower()
    if lamda > 0:
        I = torch.eye(AHA.shape[-1], dtype=AHA.dtype, device=AHA.device)
        tup = (AHA.ndim - 2) * (None,) + (slice(None),) * 2
        AHA += lamda * I[tup]
    if solver == 'lstsq':
        x = torch.linalg.lstsq(AHA, AHb).solution
    elif solver == 'solve':
        x = torch.linalg.solve(AHA, AHb)
    elif solver == 'pinv':
        x = torch.linalg.pinv(AHA, hermitian=True) @ AHb
    elif solver == 'pinv_noherm':
        x = torch.linalg.pinv(AHA) @ AHb
    elif solver == 'inv':
        x = torch.linalg.inv(AHA) @ AHb
    else:
        raise NotImplementedError
    return x

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
    if can_grid_sample:
        out = _interp_grid_sample(spatial_input, crds_flt, order)
    else:
        out = _interp_general_nd(spatial_input, crds_flt, order)

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

    # Call spatial interpolation
    inp_size = x.shape[-len(im_size):]
    kwargs = {'order': order, 'mode': mode}
    inp_size_tensor = torch.tensor(inp_size).to(x.device)
    spatial_crds = (gen_grd(im_size, balanced=True).to(x.device) + 0.5) * (inp_size_tensor - 1)
    x_rs = spatial_interp(x, spatial_crds, **kwargs).reshape(oshape)

    # Reshape to original batch dims
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
