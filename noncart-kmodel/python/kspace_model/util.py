import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fft, ifft
import scipy.sparse as sp

def ift2(X, shift=True, norm=False):
    """
    Computes the centered inverse 2D Fast Fourier Transform.

    Parameters
    ----------
    X : np.ndarray
        Input k-space data (complex). Can be 2D or multidimensional (transform applied to last 2 axes).
    shift : bool, optional
        If True, applies ifftshift before and fftshift after the transform
        to handle centered k-space data. Default is True.
    norm : bool, optional
        If True, scales the output by sqrt(H*W) to preserve energy (unitary transform).
        Default is False.

    Returns
    -------
    np.ndarray
        The reconstructed image domain data.
    """
    if shift:
        X = ifftshift(X)

    x = ifft2(X)

    if shift:
        x = fftshift(x)

    if norm:
        x = x*np.sqrt(X.shape[-2]*X.shape[-1])

    return x

def ft2(X, shift=True, norm=False):
    """
    Computes the centered forward 2D Fast Fourier Transform.

    Parameters
    ----------
    X : np.ndarray
        Input image domain data. Can be 2D or multidimensional (transform applied to last 2 axes).
    shift : bool, optional
        If True, applies ifftshift before and fftshift after the transform
        to output centered k-space data. Default is True.
    norm : bool, optional
        If True, scales the output by 1/sqrt(H*W) to preserve energy (unitary transform).
        Default is False.

    Returns
    -------
    np.ndarray
        The k-space data (complex).
    """
    if shift:
        X = ifftshift(X)

    x = fft2(X)

    if shift:
        x = fftshift(x)

    if norm:
        x = x/np.sqrt(X.shape[-2]*X.shape[-1])

    return x

def rsos(x, axis=0):
    """
    Calculates the root-sum-of-squares along a specified axis.

    Parameters
    ----------
    x : np.ndarray
        Input data array (usually complex multi-channel data).
    axis : int, optional
        The axis along which to perform the combination. Default is 0.

    Returns
    -------
    np.ndarray
        The combined magnitude image.
    """
    return np.sqrt(np.sum(np.abs(x)**2, axis=axis))
    return np.sqrt(np.sum(np.abs(x)**2, axis=axis))

def to_even(x):
    """"
    Rounds the input value up to the nearest even number.

    Parameters
    ----------
    x : int or float
        Input number.

    Returns
    -------
    int or float
        The nearest equal or larger even number.
    """
    return x + (x % 2)

def get_grids(L, N, rho):
    """
    Generates coordinate grid vectors for k-space and image domains based on dimensions and oversampling.

    Parameters
    ----------
    L : list or tuple
        Coefficient grid size.
    N : list or tuple
        Image matrix size.
    rho : list or tuple
        Oversampling factors for each dimension.

    Returns
    -------
    tuple
        (grid, igrid) where:
        - grid: List of arrays representing coefficient coordinates
        - igrid: List of arrays representing image domain coordinates
    """
    def make_grid_vec(L, rho):            
        neg = np.arange(-np.floor(L/2), 0) / rho
        pos = np.arange(1, np.floor(L/2 - (1 - L%2)) + 1) / rho
        return np.concatenate([neg, [0], pos])

    def make_igrid_vec(L, N):
        # Similar logic but denominator is N
        neg = np.arange(-np.floor(L/2), 0) / N
        pos = np.arange(1, np.floor(L/2 - (1 - L%2)) + 1) / N
        return np.concatenate([neg, [0], pos])

    grid = [make_grid_vec(L_, rho_) for L_, rho_ in zip(L, rho)]
    igrid = [make_igrid_vec(L_, N_) for L_, N_ in zip(L, N)]
    return grid, igrid

def imagesc(img, vlim=None, **opts):
    """
    Display an image with flexible options similar to MATLAB imagesc.

    Parameters
    ----------
    img : np.ndarray
        Image to display.
    vlim : scalar or [vmin, vmax]
        Value limits for display.
    opts : dict
        Optional keyword arguments:
            title, cmap, cbar, flipy, clim, xlabel, ylabel,
            xticks, yticks, xticklabels, yticklabels, fname, raw
    """
    plt.figure(opts.get('fnum', None))
    
    # compute vmin, vmax
    if vlim is not None and hasattr(vlim, '__len__') and len(vlim) == 2:
        vmin, vmax = vlim
        if np.isnan(vmin):
            vmin = np.nanmin(img)
        if np.isnan(vmax):
            vmax = np.nanmax(img)
    elif isinstance(vlim, (int, float)):
        vmin = np.nanmin(img)
        vmax = np.nanmax(img) / vlim
    else:
        vmin, vmax = np.nanmin(img), np.nanmax(img)

    plt.imshow(img, cmap=opts.get('cmap', 'gray'),
               vmin=vmin, vmax=vmax, aspect='equal', origin='upper')

    if opts.get('flipy', False):
        plt.gca().invert_yaxis()

    if 'clim' in opts:
        plt.clim(opts['clim'])
    if opts.get('cbar', False):
        plt.colorbar()

    if 'title' in opts:
        plt.title(opts['title'])
    if 'xlabel' in opts:
        plt.xlabel(opts['xlabel'])
    if 'ylabel' in opts:
        plt.ylabel(opts['ylabel'])
    if 'xticks' in opts:
        plt.xticks(opts['xticks'], opts.get('xticklabels', None))
    if 'yticks' in opts:
        plt.yticks(opts['yticks'], opts.get('yticklabels', None))

    # plt.axis('tight')

    plt.draw()
    plt.pause(0.1)
    
    # optional save
    if 'fname' in opts:
        ax = plt.gca()
        ax.set_axis_off()
        ax.title.set_visible(False)
        plt.savefig(opts['fname'], bbox_inches='tight', pad_inches=0, dpi=300)
        ax.set_axis_on()
        ax.title.set_visible(True)

    # plt.close()

    plt.pause(0.05)

def imagesc3d(imgs, vlim=None, **opts):
    """
    Displays a 3D volume of images as a 2D tiled montage.

    Parameters
    ----------
    imgs : np.ndarray
        Input 3D array of shape (N_slices, Height, Width).
    vlim : scalar or [vmin, vmax], optional
        Value limits for display windowing.
    **opts : dict
        Keyword arguments passed to `imagesc`.
        Must include 'ncol' to specify number of columns in the montage.

    Returns
    -------
    None
        Displays the figure using matplotlib.
    """
    n_imgs = imgs.shape[0]
    n_col = opts['ncol']
    n_row = int(np.ceil(n_imgs/n_col))

    [Ni, Nr, Nc] = imgs.shape
    
    if vlim is not None and hasattr(vlim, '__len__') and len(vlim) == 2:
        vmin, vmax = vlim
        if np.isnan(vmin):
            vmin = np.nanmin(imgs)
        if np.isnan(vmax):
            vmax = np.nanmax(imgs)
    elif isinstance(vlim, (int, float)):
        vmin = np.nanmin(imgs)
        vmax = np.nanmax(imgs) / vlim
    else:
        vmin, vmax = np.nanmin(imgs), np.nanmax(imgs)

    assem_img = np.ones((n_row*Nr, n_col*Nc))*vmin

    for r_idx in range(n_row):
        for c_idx in range(n_col):
            r_start = r_idx*Nr
            r_end = r_start + Nr
            c_start = c_idx*Nc
            c_end = c_start + Nc
            i_idx = r_idx*n_col + c_idx
            if i_idx >= Ni:
                break
            assem_img[r_start:r_end, c_start:c_end] = imgs[i_idx, :, :]
    
    imagesc(assem_img, [vmin, vmax], **opts)

# CG
import numpy as np
def cg_np(A, b, tol=1e-6, maxit=200, x0=None, verbose=True):
    """
    Solves a linear system Ax = b using the Conjugate Gradient method (NumPy version).

    Parameters
    ----------
    A : callable
        The system matrix. Can be a dense matrix or a function/operator that applies A(v).
    b : np.ndarray
        The right-hand side vector
    tol : float, optional
        Tolerance for convergence (relative residual). Default is 1e-6.
    maxit : int, optional
        Maximum number of iterations. Default is 200.
    x0 : np.ndarray, optional
        Initial guess for x. If None, zeros are used.
    verbose : bool, optional
        If True, prints convergence progress. Default is True.

    Returns
    -------
    np.ndarray
        The solution x.
    """
    n = b.shape[0]
    
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()
        
    def apply_A(v):
        if callable(A):
            return A(v)
        return A @ v

    n2b = np.linalg.norm(b)
    r = b - apply_A(x)
    normr = np.linalg.norm(r)
    tolb = tol * n2b

    # if residual is too small, return immediately
    if normr <= tolb:
        print('Initial guess satisfies the tolerance criterion. Return the initial guess')
        return x

    rho = 1.0    
    for ii in range(maxit):
        rho_old = rho
        rho = np.dot(r.flatten().conj(), r.flatten())
            
        if ii == 0:
            p = r.copy()
        else:
            beta = rho / rho_old
            p = r + beta * p
            
        q = apply_A(p)
        pq = np.dot(p.flatten().conj(), q.flatten())
                    
        alpha = rho / pq

        x = x + alpha * p
        r = r - alpha * q
        normr = np.linalg.norm(r)
        
        if verbose:
            print(f'    [CG] iter={ii+1}, residual={normr/n2b}')

        if normr <= tolb:
            break

    if ii == maxit-1:
        print(f'PCG fails to converge and stops at maximum iteration  {maxit}.')
    else:
        print(f'PCG returned at iter {ii+1}.')
    
    return x

# Pytorch version of conjugate gradient
try:
    import torch
except:
    print("Warning: PyTorch installation not found. The demo code will not be able to use the PyTorch backend")

def cg_pytorch(A, b, tol=1e-6, maxit=200, x0=None, verbose=True):
    """
    Solves a linear system Ax = b using the Conjugate Gradient method (PyTorch version).

    Parameters
    ----------
    A : callable
        The system matrix. Can be a tensor or a function/operator that applies A(v).
    b : torch.Tensor
        The right-hand side vector/tensor.
    tol : float, optional
        Tolerance for convergence (relative residual). Default is 1e-6.
    maxit : int, optional
        Maximum number of iterations. Default is 200.
    x0 : torch.Tensor, optional
        Initial guess for x. If None, zeros are used.
    verbose : bool, optional
        If True, prints convergence progress. Default is True.

    Returns
    -------
    torch.Tensor
        The approximate solution x.
    """
    n = b.shape[0]
    
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
        
    def apply_A(v):
        if callable(A):
            return A(v)
        return A @ v

    n2b = torch.norm(b)
    r = b - apply_A(x)
    normr = torch.norm(r)
    tolb = tol * n2b

    # If residual is too small, return immediately
    if normr <= tolb:
        print('Initial guess satisfies the tolerance criterion. Return the initial guess')
        return x

    rho = 1.0    
    # Initialize p to avoid UnboundLocalError if loop doesn't run (though unlikely given checks above)
    p = r.clone() 

    for ii in range(maxit):
        rho_old = rho
        rho = torch.dot(r.flatten().conj(), r.flatten())
            
        if ii == 0:
            p = r.clone()
        else:
            beta = rho / rho_old
            p = r + beta * p
            
        q = apply_A(p)
        pq = torch.dot(p.flatten().conj(), q.flatten())
                    
        alpha = rho / pq

        x = x + alpha * p
        r = r - alpha * q
        normr = torch.norm(r)
        
        if verbose:
            print(f'    [CG] iter={ii+1}, residual={(normr/n2b).item():.6f}')

        if normr <= tolb:
            print(f'PCG returned at iter {ii+1}.')
            return x

    print(f'PCG fails to converge and stops at maximum iteration  {maxit}.')
    
    return x


import math

def zero_pad(x, target_sizes):
    """
    Zero-pad the last 2 dimensions of x (H, W) using NumPy.
    
    Args:
        x (np.ndarray): Input array of shape (..., H, W)
        target_sizes (tuple): Target size (H_tar, W_tar)
        
    Returns:
        np.ndarray: Padded array
    """
    # Get current spatial dimensions (last two)
    H, W = x.shape[-2:]
    H_tar, W_tar = target_sizes
    
    if H_tar < H or W_tar < W:
        raise ValueError('target size must be larger than the original size')

    diff_h = H_tar - H
    diff_w = W_tar - W

    pad_top = math.floor(diff_h / 2) + (H % 2) * (diff_h % 2)
    pad_left = math.floor(diff_w / 2) + (W % 2) * (diff_w % 2)
    
    pad_bottom = diff_h - pad_top
    pad_right = diff_w - pad_left
    
    # Construct padding config for np.pad
    # Shape is (..., H, W), so we need (0,0) for all dimensions except the last two
    pad_width = [(0, 0)] * (x.ndim - 2)
    pad_width.append((pad_top, pad_bottom))  # Height dim
    pad_width.append((pad_left, pad_right))  # Width dim
    
    return np.pad(x, pad_width, mode='constant', constant_values=0)

def center_crop(x, target_sizes):
    """
    Center-crop the last 2 dimensions of x (H, W) using NumPy.
    Functions as the inverse of zero_pad.
    
    Args:
        x (np.ndarray): Input array of shape (..., H, W)
        target_sizes (tuple): Target size (H_tar, W_tar)
        
    Returns:
        np.ndarray: Cropped array
    """
    H, W = x.shape[-2:]
    H_tar, W_tar = target_sizes
    
    if H_tar > H or W_tar > W:
        raise ValueError('target size must be smaller than the original size')

    diff_h = H - H_tar
    diff_w = W - W_tar

    start_h = math.floor(diff_h / 2) + (H % 2) * (diff_h % 2)
    start_w = math.floor(diff_w / 2) + (W % 2) * (diff_w % 2)

    return x[..., start_h : start_h + H_tar, start_w : start_w + W_tar]

def get_sparse_basis_entries(basis_func, x):
    """
    Internal helper for function `get_forward_matrix`. This helper function evaluates the `basis_func` at `x`.`
    """
    M, N = x.shape
    x_flat = x.flatten()
    
    # Evaluate the basis function
    values = basis_func(x_flat)
    nzidx = np.flatnonzero(values)
    values = values[nzidx]

    # Unravel linear indices to (row, col)
    rows, cols = np.unravel_index(nzidx, (M, N))

    return rows, cols, values

def get_forward_matrix(Psi, t, x, a):
    """
    Get sparse forward and adjoint matrices for a 2D trajectory.
    Calculates Psi[0](a[0]*(x[0]-t[0])) * Psi[1](a[1]*(x[1]-t[1])).

    Parameters
    ----------
    Psi : list of functions
        Function handles for the basis functions in each dimension.
    t : list of array_like
        Shifts of basis functions (centers) for each dimension.
    x : list of array_like
        Points where basis functions are evaluated.
    a : list of float
        Scaling factors for each dimension.

    Returns
    -------
    H : scipy.sparse.csr_matrix
        Forward matrix.
    Hh : scipy.sparse.csr_matrix
        Adjoint matrix.
    """
    # only support 2D for now
    if len(Psi) != 2:
        raise ValueError('only 2D supported now')

    # Ensure inputs are 1D arrays
    cvt = lambda arr: np.asarray(arr).flatten()
    t1, t2 = [cvt(tt) for tt in t]
    x1, x2 = [cvt(xx) for xx in x]
    
    Psi1, Psi2 = Psi[0], Psi[1]
    a1, a2 = a[0], a[1]

    M = x1.size
    N1 = t1.size
    N2 = t2.size

    # Calculate arguments for basis function 
    axt1 = a1 * (x1[:, None] - t1[None, :])
    axt2 = a2 * (x2[:, None] - t2[None, :])

    # Basis evaluations
    rows1, cols1, vals1 = get_sparse_basis_entries(Psi1, axt1)
    rows2, cols2, vals2 = get_sparse_basis_entries(Psi2, axt2)

    # Construct intermediate sparse matrices to group by row efficiently
    S1 = sp.csr_matrix((vals1, (rows1, cols1)), shape=(M, N1))
    S2 = sp.csr_matrix((vals2, (rows2, cols2)), shape=(M, N2))

    # Pre-allocate arrays for the final result
    max_nnz1 = np.diff(S1.indptr).max() if S1.nnz > 0 else 0
    max_nnz2 = np.diff(S2.indptr).max() if S2.nnz > 0 else 0
    est_nnz = M * max(1, max_nnz1) * max(1, max_nnz2)
    
    ind_array = np.zeros((int(est_nnz), 3), dtype=np.int32)
    val_array = np.zeros((int(est_nnz), 1))
    cur_pos = 0

    # Iterate over rows
    for i in range(M):
        ptr1_start, ptr1_end = S1.indptr[i], S1.indptr[i+1]
        c1 = S1.indices[ptr1_start:ptr1_end]
        v1 = S1.data[ptr1_start:ptr1_end]

        ptr2_start, ptr2_end = S2.indptr[i], S2.indptr[i+1]
        c2 = S2.indices[ptr2_start:ptr2_end]
        v2 = S2.data[ptr2_start:ptr2_end]

        if c1.size == 0 or c2.size == 0:
            continue

        tt2, tt1 = np.meshgrid(c2, c1)
        vv = (v1[:, None] * v2[None, :]).flatten()
        size = tt1.size
        
        ind_array[cur_pos:cur_pos+size,0] = i
        ind_array[cur_pos:cur_pos+size,1] = tt1.flatten()
        ind_array[cur_pos:cur_pos+size,2] = tt2.flatten() 
        val_array[cur_pos:cur_pos+size,0] = vv.flatten() 
        cur_pos += size

    ind_array = ind_array[0:cur_pos, :]
    val_array = val_array[0:cur_pos, :]

    rr, cc = np.unravel_index(np.ravel_multi_index((ind_array[:,0], ind_array[:,1], ind_array[:,2]), (M,N1,N2)), (M,N1*N2))
    
    # Construct the sparse matrix H
    H = sp.csr_matrix((val_array.flatten(), (rr, cc)), shape=(M, N1 * N2))

    # Adjoint is simply the transpose for real matrices
    Hh = H.T

    return H, Hh
