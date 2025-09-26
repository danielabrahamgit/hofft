import torch
import numpy as np

from mr_recon.linops import linop, batching_params
from mr_recon.utils import gen_grd, batch_iterator, resize
from mr_recon.fourier import fft, ifft
from mr_recon._func.indexing import multi_index, multi_grid
from mr_recon._func.pad import PadLast
from .kb import _gen_kern_vectors

from typing import Optional, Union
from einops import einsum, rearrange
from dataclasses import dataclass

__all__ = [
    'multi_apod_kern_linop', 
    'hofft_params'
]

@dataclass
class hofft_params:
    kern_size: tuple
    os: float
    L: int
    apods_init: Union[torch.Tensor, str] = 'seg'
    use_type3: bool = False
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
    apods_init : Union[torch.Tensor, str]
        If string:
        'seg' - uses segmentation method to initialize apodization functions
        'eigen' - uses eigen-decomposition method to initialize apodization functions
        'k_alphas' - uses K representative alphas to initialize apodization functions
        If torch.Tensor:
        Initial apodization functions with shape (L, *solve_size)
    use_type3 : Optional[bool]
        If True, uses type3 nufft for the forward and adjoint operations.
    verbose : Optional[bool]
        If True, prints progress
    """

class multi_apod_kern_linop(linop):
    """
    Linear operator for multi-apodized kernels.
    """

    def __init__(self, 
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 weights: torch.Tensor,
                 apods: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 os_grid: Optional[float] = 1.0,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Initialize the HOFFT linear operator.
        
        Args:
        -----
        trj : torch.Tensor
            Trajectory of the k-space samples with shape (*trj_size, D)
        mps : torch.Tensor
            Sensitivity maps with shape (C, *im_size)
        weights : torch.Tensor
            the kernel weights with shape (L, *kern_size, *trj_size)
        apods : torch.Tensor
            the apodization functions with shape (L, *im_size)
        dcf : Optional[torch.Tensor]
            Density compensation function with shape (*trj_size)
        os_grid : Optional[float]
            Oversampling factor for the grid
        bparams : Optional[batching_params]
            Batching parameters for the linear operator
        """
        im_size = mps.shape[1:]
        trj_size = trj.shape[:-1]
        kern_size = weights.shape[1:-len(trj_size)]
        oshape = (mps.shape[0], *trj_size)
        super().__init__(im_size, oshape)
        
        # Consts
        D = trj.shape[-1]
        L = weights.shape[0]
        torch_dev = trj.device
        assert mps.device == torch_dev
        assert weights.device == torch_dev
        assert apods.device == torch_dev
        assert apods.shape[0] == L
        
        # Make sure trajectory is on an oversampled grid
        assert torch.allclose(trj, (trj * os_grid).round() / os_grid), \
            f"Trajectory is not on an oversampled grid. os_grid: {os_grid}"
        
        # Default dcf
        if dcf is None:
            dcf = torch.ones(trj.shape[:-1], dtype=torch.float32, device=torch_dev)
        else:
            assert dcf.device == torch_dev
        
        # Trajectory of kernels
        if np.prod(kern_size) == 1:
            kern_vecs = torch.zeros(D, device=torch_dev)
        else:
            kern_vecs = gen_grd(kern_size, kern_size).reshape((-1, D)).to(torch_dev)
        
        # Convert to index units
        im_size_os = [round(im_size[i] * os_grid) for i in range(len(im_size))]
        im_size_os_tensor = torch.tensor(im_size_os, device=torch_dev)
        idx_kerns = (trj * os_grid).round() + im_size_os_tensor // 2
        idx_kerns = (idx_kerns[..., None, :] + kern_vecs).type(torch.int32) # (*trj_size, K, d)
        idx_kerns = idx_kerns % im_size_os_tensor.type(torch.int32)
        
        # Store params
        self.padder = PadLast(im_size_os, list(im_size))
        self.im_size_os = im_size_os
        self.im_size = im_size
        self.mps = mps
        self.os_grid = os_grid
        self.dcf = dcf
        self.idx_kerns = idx_kerns
        self.bparams = bparams
        self.weights = weights.reshape((L, -1, *trj_size))
        self.apods = apods

    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        """
        Applies forward model to image to get k-space data.
        
        Parameters
        ----------
        img : torch.Tensor
            The image to be transformed with shape (*im_size)
        
        Returns
        -------
        torch.Tensor
            The k-space data with shape (C, *trj_size)
        """
        # Consts
        D = self.idx_kerns.shape[-1]
        C = self.mps.shape[0]
        cbs = self.bparams.coil_batch_size
        
        # Output tensor
        ksp = torch.zeros(self.oshape, device=img.device, dtype=torch.complex64)
        
        # Batch over coils
        for c1, c2 in batch_iterator(C, cbs):
            
            # Apply sensitivity maps to image
            Sx = self.mps[c1:c2] * img
            
            # Apply apods to image
            MSx = einsum(Sx, self.apods,
                         'C ..., L ... -> C L ...')
            
            # Oversampled FFT
            MSx = self.padder(MSx)
            FMSx = fft(MSx, dim=tuple(range(-D, 0))) / torch.tensor(self.im_size).prod().sqrt()
            
            # Extract blocks of k-space data
            blocks = multi_index(FMSx, D, self.idx_kerns) # (C, L, *trj_size, K)
            blocks = blocks.moveaxis(-1, 2) # (C, L, K, *trj_size)
            
            # Apply kernels
            KFSx = einsum(blocks, self.weights, 'C L K ..., L K ... -> C ...')
            ksp[c1:c2] = KFSx
            
        return ksp
    
    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        """
        Applies adjoint model to k-space data to get image.
        
        Parameters
        ----------
        ksp : torch.Tensor
            The k-space data with shape (C, *trj_size)
        
        Returns
        -------
        torch.Tensor
            The image with shape (*im_size)
        """
        # Consts
        D = self.idx_kerns.shape[-1]
        C = self.mps.shape[0]
        cbs = self.bparams.coil_batch_size
        
        # Output tensor
        img = torch.zeros(self.ishape, device=ksp.device, dtype=torch.complex64)
        
        # Batch over coils
        for c1, c2 in batch_iterator(C, cbs):
            
            # Get Kernels
            y = ksp[c1:c2] * self.dcf
            Ky = einsum(y, self.weights.conj(), 'C ..., L K ... -> C L ... K')
            
            # Gridding 
            Ky = multi_grid(Ky, self.idx_kerns, self.im_size_os) # (C, L, *im_size_os)
            FKy = ifft(Ky, dim=tuple(range(-D, 0))) / torch.tensor(self.im_size_os).prod().sqrt()
            FKy = self.padder.adjoint(FKy) # (C, L, *im_size)
            
            # Apply adjoint sensitivity maps
            SFKy = (self.mps[c1:c2, None,].conj() * FKy).sum(dim=0) # L, *im_size
            
            # Apply adjoint source maps
            MSFKy = (SFKy * self.apods.conj()).sum(dim=0)
            
            # Update image
            img += MSFKy
        
        return img
    
    def normal(self,
               img: torch.Tensor) -> torch.Tensor:
        """
        Applies forward model and adjoint model to image to get normal operator.
        
        Parameters
        ----------
        img : torch.Tensor
            The image to be transformed with shape (*im_size)
            
        Returns
        -------
        torch.Tensor
            The response image with shape (*im_size)
        """
        
        return self.adjoint(self.forward(img))