import torch
import numpy as np

from mr_recon.utils import gen_grd
from einops import einsum

def _gen_kern_vectors(kern_size: tuple,
                      oversamp: float) -> torch.Tensor:
    """
    From kernel size, create vector pointing to each kernel point

    Args:
    -----
    kern_size : tuple
        Size of the kernel 
    oversamp : float
        Oversampling factor.

    Returns:
    -------
    kern : torch.Tensor
        Kernel vectors with shape (prod(kern_size), d)
    """
    d = len(kern_size)
    kerns_1d = []
    for i in range(d):
        kerns_1d.append(torch.arange(kern_size[i], dtype=torch.float32) - (kern_size[i] - 1) // 2)
    kerns = torch.meshgrid(*kerns_1d, indexing='ij')
    kern = torch.stack(kerns, dim=-1).reshape((-1, d)) / oversamp
    # kern = gen_grd(kern_size, kern_size).reshape((-1, d)) / oversamp
    kern -= kern.mean(dim=0)
    return kern

def _gen_kern_bases(im_size: tuple,
                    kern_size: tuple,
                    oversamp: float) -> torch.Tensor:
    """
    Generate the kernel bases for the Kaiser Bessel kernel.

    Parameters:
    -----------
    im_size : tuple
        Size of the image 
    kern_size : tuple
        Size of the kernel (same length as im_size).
    oversamp : float
        Oversampling factor.

    Returns:
    --------
    bases : torch.Tensor
        Kernel bases with shape (prod(kern_size), *im_size).
    """
    # Make kernel
    kern = _gen_kern_vectors(kern_size, oversamp)
    
    # Bases
    rs = gen_grd(im_size)
    bases = torch.exp(-2j * np.pi * einsum(kern, rs, 'K d, ... d -> K ...'))
    return bases

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
    eps = 1e-6
    apod = (
        beta**2 - (np.pi * width * (x + eps)) ** 2
    ) ** 0.5
    apod /= torch.sinh(apod)
    return apod
 
def sample_kb_kernel(kdevs: torch.Tensor,
                     kern_size: tuple,
                     os: float,
                     beta: float) -> torch.Tensor:
    """
    Sample the Kaiser Bessel kernel
    
    Parameters:
    -----------
    kdevs : torch.Tensor
        kernel deviations in [-1/2/os, 1/2/os] with shape (..., d)
    kern_size : tuple
        kernel width in each axis, len(kern_size) = d
    os : float
        oversampling factor
    beta : float
        beta parameter for kaiser bessel kernel
        
    Returns:
    --------
    weights : torch.Tensor <float>
        sampled kaiser bessel kernel with shape (*kern_size, ...)
    """
    torch_dev = kdevs.device
    d = kdevs.shape[-1]
    
    # Points to sample kernel at
    kern_size_tensor = torch.tensor(kern_size, dtype=torch.float32, device=torch_dev)
    kern = _gen_kern_vectors(kern_size, os).to(torch_dev)
    pts = os * (kern - kdevs[..., None, :]) / (kern_size_tensor / 2)
    
    # Sample kernel
    weights = torch.prod(kb_weights_1d(pts, beta), dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0) # ... K
    
    # Reshape
    weights = weights.moveaxis(-1, 0)
    weights = weights.reshape((*kern_size, *kdevs.shape[:-1]))
    
    return weights