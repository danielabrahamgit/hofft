import torch
import torch.nn as nn
import torch.fft as fft_torch

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from torch.special import i0
from math import ceil
from einops import einsum
from typing import Optional
from tqdm import tqdm


def ravel(x, shape, dim):
    """
    x: torch.LongTensor, arbitrary shape,
    shape: Shape of the array that x indexes into
    dim: dimension of x that is the "indexing" dimension

    Returns:
    torch.LongTensor of same shape as x but with indexing dimension removed
    """
    out = 0
    shape_shifted = tuple(shape[1:]) + (1,)
    for s, s_next, i in zip(shape, shape_shifted, range(x.shape[dim])):
        out += torch.select(x, dim, i) % s
        out *= s_next
    return out

def multi_grid(x: torch.Tensor, 
               idx: torch.Tensor, 
               final_size: tuple, 
               raveled: bool = False) -> torch.Tensor:
    """Grid values in x to im_size with indices given in idx
    x: [N... I...]
    ndims: number of dimensions from the end of the tensor to grid (length of I)
    idx: [I... ndims] or [I...] if raveled=True
    raveled: Whether the idx still needs to be raveled or not

    Returns:
    Tensor with shape [N... final_size]

    Notes:
    Adjoint of multi_index
    """
    if not raveled:
        assert len(final_size) == idx.shape[-1], f'final_size should be of dimension {idx.shape[-1]}'
        idx = ravel(idx, final_size, dim=-1)
    ndims = len(idx.shape)
    assert x.shape[-ndims:] == idx.shape, f'x and idx should correspond in last {ndims} dimensions'
    x_flat = torch.flatten(x, start_dim=-ndims, end_dim=-1) # [N... (I...)]
    idx_flat = torch.flatten(idx)

    batch_dims = x_flat.shape[:-1]
    y = torch.zeros((*batch_dims, *final_size), dtype=x_flat.dtype, device=x_flat.device)
    y = y.reshape((*batch_dims, -1))
    y = y.index_add_(-1, idx_flat, x_flat)
    y = y.reshape(*batch_dims, *final_size)
    return y

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

def gen_grd(im_size: tuple, 
            fovs: tuple = None,
            balanced: bool = False) -> torch.Tensor:
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

def fft(x, dim=None, oshape=None, norm='ortho'):
    """Matches Sigpy's fft, but in torch"""
    if oshape is not None:
        x = resize(x, oshape)
    x = fft_torch.ifftshift(x, dim=dim)
    x = fft_torch.fftn(x, dim=dim, norm=norm)
    x = fft_torch.fftshift(x, dim=dim)
    return x

def ifft(x, dim=None, oshape=None, norm='ortho'):
    """Matches Sigpy's fft adjoint, but in torch"""
    if oshape is not None:
        x = resize(x, oshape)
    x = fft_torch.ifftshift(x, dim=dim)
    x = fft_torch.ifftn(x, dim=dim, norm=norm)
    x = fft_torch.fftshift(x, dim=dim)
    return x

class nufft(nn.Module):
    
    def __init__(self, 
                 sig_size: tuple, 
                 oversamp: float, 
                 width: int,
                 param: Optional[float] = None,
                 kern_type: str = 'kb'):
        """
        Args
        ----
        sig_size : tuple
            The size of the signal.
        oversamp : float
            The oversampling factor.
        width : int
            The width of the kernel.
        param : float
            The parameter of the kernel.
        kern_type : str
            The type of kernel to use
            'kb' - Kaiser-Bessel kernel
            'gauss' - Gaussian kernel
        """
        super().__init__()
        if kern_type == 'kb':
            self.kernel = self._kb_kernel
            self.apod = self._kb_apod
            if param is None:
                os = oversamp
                param = torch.pi * (((width / os) * (os - 0.5))**2 - 0.8)**0.5
                if (((width / os) * (os - 0.5))**2 - 0.8) < 0:
                    param = 1.0
        elif kern_type == 'gauss':
            self.kernel = self._gauss_kernel
            self.apod = self._gauss_apod
            if param is None:
                param = 0.37
        elif kern_type == 'linear':
            self.kernel = self._linear_kernel
            self.apod = self._linear_apod
            assert width == 2
            param = -1.0
        elif kern_type == 'nearest':
            self.kernel = self._nearest_kernel
            self.apod = self._nearest_apod
            assert width == 1
            param = -1.0
        else:
            raise ValueError(f"Invalid kernel type: {kern_type}")
    
        self.sig_size = sig_size
        self.oversamp = oversamp
        self.width = width
        self.param = param
        self.kern_type = kern_type
        
        # Initialize kernel grid
        d = len(sig_size)
        kern_grd = gen_grd((self.width,)*d, (self.width,)*d).reshape(-1, d)
        kern_grd -= kern_grd.mean(dim=0, keepdim=True)
        self.kern_grd = kern_grd
        
        # Initialize spatial grid
        self.rs = gen_grd(sig_size)
        
    def _kb_kernel(self, k):
        """
        |k| <= width//2
        """
        arg = self.param * (1 - (k / self.width * 2) ** 2) ** 0.5
        return i0(arg)
    
    def _kb_apod(self, r):
        eps = 1e-12
        arg = (self.param**2 - (torch.pi * self.width * r) ** 2)
        apod_pos = arg.clamp(min=0).sqrt()
        apod_pos /= torch.sinh(apod_pos) + eps
        apod_neg = (-arg.clamp(max=0)).sqrt()
        apod_neg /= torch.sin(apod_neg) + eps
        return self.width * (apod_pos + apod_neg)
    
    def _gauss_kernel(self, k):
        return torch.exp(-((k / self.width * 2) / self.param) ** 2 / 2)
    
    def _gauss_apod(self, r):
        return torch.exp(2 * (torch.pi * (r * self.width / 2) * self.param) ** 2)

    def _linear_kernel(self, k):
        return 1 - torch.abs(k / self.width * 2)
    
    def _linear_apod(self, r):
        # return torch.ones_like(r)
        return 1 / torch.sinc(r * self.width / 2) ** 2
    
    def _nearest_kernel(self, k):
        return torch.ones_like(k)
    
    def _nearest_apod(self, r):
        # return torch.ones_like(r)
        return 1 / torch.sinc(r * self.width / 2)
    
    def apod_sig(self, 
                 sig: torch.Tensor,
                 conj: bool = False) -> torch.Tensor:
        """
        Apodize the signal and applies linear phase correction
        
        Args
        ----
        sig : torch.Tensor
            The signal to be apodized, shape (*sig_size)
        conj : bool
            Whether to conjugate the apodization function
        
        Returns
        -------
        sig_apod : torch.Tensor
            The apodized signal with shape (*sig_size)
        """
        # Consts
        sig_size = self.sig_size
        d = len(sig_size)
        sgn = 2 * conj - 1
        
        # Apodize
        self.rs = self.rs.to(sig.device)
        apod = self.apod(self.rs / self.oversamp).prod(dim=-1)
        phz = torch.exp((self.width % 2 == 0) * sgn * 2j * torch.pi * rs.sum(dim=-1) / self.oversamp / 2)
        
        return sig * apod * phz
    
    def plan(self,
             ks: torch.Tensor):
        """
        Plan the NUFFT.
        
        Args
        ----
        ks : torch.Tensor
            The k-space trajectory, shape (*ks_size, d)
        """
        # Store kernel weights
        self.kern_grd = self.kern_grd.to(torch_dev)
        kdevs = ks - (ks * self.oversamp).round() / self.oversamp # in [-1/2/os, +1/2/os]
        pts = self.kern_grd - self.oversamp * kdevs[..., None, :]
        weights = self.kernel(pts).prod(dim=-1) # *ks_size K
        weights = torch.nan_to_num(weights, nan=0.0)
        self.weights = weights
        
        # Interpolate k-space data
        sig_size_os = [round(self.oversamp * i) for i in self.sig_size]
        sig_size_os_tensor = torch.tensor(sig_size_os, device=torch_dev)
        idx_kerns = (ks * self.oversamp).round() + sig_size_os_tensor // 2
        idx_kerns = (idx_kerns[..., None, :] + self.kern_grd).type(torch.long)
        idx_kerns = (idx_kerns % sig_size_os_tensor).type(torch.long)
        self.idx_kerns = idx_kerns
        
        # Store spatial grid on correct device
        self.rs = self.rs.to(torch_dev)
    
    def forward(self, 
                x: torch.Tensor,
                ks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NUFFT.
        
        Args
        ----
        x : torch.Tensor
            The signal to be transformed, shape (N, *sig_size)
        ks : torch.Tensor
            The k-space trajectory, shape (*ks_size, d)
        
        Returns
        -------
        y : torch.Tensor
            The k-space data with shape (N, *ks_size)
        """
        # Consts
        N = x.shape[0]
        sig_size = self.sig_size
        torch_dev = x.device
        d = len(sig_size)
        
        # Apodize and apply linear phase correction
        x_apod = self.apod_sig(x, conj=False)
        
        # Zero Pad
        sig_size_os = [round(self.oversamp * i) for i in sig_size]
        x_zp = resize(x_apod, [N] + sig_size_os)
        
        # FFT
        y_os = fft(x_zp, dim=tuple(range(-d, 0)))
        
        # # Grab kernel weights
        # self.kern_grd = self.kern_grd.to(torch_dev)
        # kdevs = ks - (ks * self.oversamp).round() / self.oversamp # in [-1/2/os, +1/2/os]
        # pts = self.kern_grd - self.oversamp * kdevs[..., None, :]
        # weights = self.kernel(pts).prod(dim=-1) # *ks_size K
        # weights = torch.nan_to_num(weights, nan=0.0)
        
        # # Interpolate k-space data
        # sig_size_os_tensor = torch.tensor(sig_size_os, device=torch_dev)
        # idx_kerns = (ks * self.oversamp).round() + sig_size_os_tensor // 2
        # idx_kerns = (idx_kerns[..., None, :] + self.kern_grd).type(torch.long)
        # idx_kerns = (idx_kerns % sig_size_os_tensor).type(torch.long)
        tup = (slice(None),) + tuple(self.idx_kerns.moveaxis(-1, 0))
        y_blocks = y_os[tup] # *ks_size K
        y = (y_blocks * self.weights).sum(dim=-1) # *ks_size
        
        return y
    
    def adjoint(self, 
                y: torch.Tensor,
                ks: torch.Tensor) -> torch.Tensor:
        """
        Adjoint pass of the NUFFT.
        
        Args
        ----
        y : torch.Tensor
            The k-space data to be transformed, shape (N, *ks_size)
        ks : torch.Tensor
            The k-space trajectory, shape (*ks_size, d)
        
        Returns
        -------
        x : torch.Tensor
            The signal with shape (N, *sig_size)
        """
        # Consts
        N = y.shape[0]
        sig_size = self.sig_size
        torch_dev = y.device
        d = len(sig_size)
        sig_size_os = [round(self.oversamp * i) for i in sig_size]
        
        # # Grab kernel weights
        # self.kern_grd = self.kern_grd.to(torch_dev)
        # kdevs = ks - (ks * self.oversamp).round() / self.oversamp
        # pts = self.kern_grd - self.oversamp * kdevs[..., None, :]
        # weights = self.kernel(pts).prod(dim=-1) # *ks_size K
        # weights = torch.nan_to_num(weights, nan=0.0)
        
        # Gridding
        # sig_size_os_tensor = torch.tensor(sig_size_os, device=torch_dev)
        # idx_kerns = (ks * self.oversamp).round() + sig_size_os_tensor // 2
        # idx_kerns = (idx_kerns[..., None, :] + self.kern_grd).type(torch.long)
        # idx_kerns = (idx_kerns % sig_size_os_tensor).type(torch.long)
        y_os_weighted = einsum(y, self.weights.conj(), 'N ..., ... K -> N ... K')
        y_os = multi_grid(y_os_weighted, self.idx_kerns, sig_size_os)
        
        # iFFT
        x_zp = ifft(y_os, dim=tuple(range(-d, 0)))
        
        # Crop
        x_zp = resize(x_zp, (N, *sig_size))
        
        # Apod
        x_apod = self.apod_sig(x_zp, conj=True)
        
        return x_apod
        
    def opt_param(self, 
                  torch_dev: torch.device) -> torch.Tensor:
        """
        Optimize the parameter of the kernel.
        
        Args
        ----
        torch_dev : torch.device
            The device to optimize the parameter on.
            
        Returns
        -------
        param : torch.Tensor
            The optimized parameter.
        """
        if self.kern_type == 'linear' or self.kern_type == 'nearest':
            return -1.0
        
        # Discretize over kdevs, rs
        N = max(self.sig_size)
        Nparams = 100
        num_sigs = 1000
        kdevs = torch.linspace(-0.5, 0.5, N, device=torch_dev) / self.oversamp
        
        if self.kern_type == 'kb':
            params = torch.linspace(0.5, 20, Nparams, device=torch_dev)
        elif self.kern_type == 'gauss':
            params = torch.linspace(0.01, 1.0, Nparams, device=torch_dev)
        
        # Make a few random 1d signals
        rnd_sigs = torch.randn((num_sigs, N), dtype=torch.complex64, device=torch_dev)
        
        # Ground truth values
        rs = gen_grd((N,)).to(torch_dev)[:, 0]
        gt = torch.exp(-2j * torch.pi * (kdevs[:, None] @ rs[None, :]))
        vals_gt = torch.zeros((num_sigs, N), dtype=torch.complex64, device=torch_dev)
        for i in range(num_sigs):
            vals_gt[i] = gt @ rnd_sigs[i]
            
        # Make nufft
        nft = nufft((N,), self.oversamp, self.width, kern_type=self.kern_type)
        
        # See which beta is closest to the ground truth
        errs = []
        for param in params:
            
            # Compare beta-nufft to ground truth
            nft.param = param
            nft.plan(kdevs[:, None])
            vals = nft.forward(rnd_sigs, kdevs[:, None])
            scale = (vals.conj() * vals_gt).sum() / (vals.conj() * vals).sum()
            vals = vals * scale
            err = (vals - vals_gt).norm()
            errs.append(err)
        
        # Return best Beta
        errs = torch.stack(errs, dim=0)
        imin = errs.argmin(dim=0)
        # import matplotlib.pyplot as plt
        # plt.plot(params.cpu(), errs.cpu())
        # plt.axvline(param_old, color='r', linestyle='--', label='self.beta')
        # plt.legend()
        # plt.show()
        # quit()
        return params[imin].item()

# Params
torch_dev = torch.device(5)
im_size = (220,)*3
os = 1.2
width = 1
nrep = 100

# Create kooshball trajectory
Nspokes = 10_000
Nread = im_size[0] * 2
phis = torch.arccos(2 * torch.rand(Nspokes, device=torch_dev) - 1)
thetas = torch.rand(Nspokes, device=torch_dev) * torch.pi * 2
vecs = torch.stack([torch.cos(thetas) * torch.sin(phis), 
                    torch.sin(thetas) * torch.sin(phis), 
                    torch.cos(phis)], dim=-1)
rs = torch.linspace(-1, 1, Nread, device=torch_dev)
trj = vecs[None, :, :] * rs[:, None, None]

# Make delta function image
img = torch.zeros(im_size, device=torch_dev, dtype=torch.complex64)
img[tuple(im_size[i]//2 for i in range(len(im_size)))] = 1

# Sigpy
from mr_recon.fourier import sigpy_nufft
nft = sigpy_nufft(im_size, os, width)
nft.beta = nft.optimal_beta(torch_dev=torch_dev)
for _ in tqdm(range(nrep), 'Sigpy NUFFT'):
    y = nft.forward(img[None,], trj[None,])[0]
    img_nufft = nft.adjoint(y[None,], trj[None,])[0]

# Apply NUFFT and adjoint
nft = nufft(im_size, os, width, kern_type='kb')
nft.param = nft.opt_param(torch_dev)
nft.plan(trj)
for _ in tqdm(range(nrep), 'My NUFFT'):
    y = nft.forward(img[None,], trj)[0]
    img_nufft = nft.adjoint(y[None,], trj)[0]

