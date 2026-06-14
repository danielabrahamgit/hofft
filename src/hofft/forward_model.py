import torch
import numpy as np

from mr_recon.linops import linop, batching_params
from mr_recon.utils import gen_grd, batch_iterator, resize
from mr_recon.fourier import fft, ifft
from mr_recon._func.indexing import multi_index, multi_grid, ravel
from mr_recon._func.pad import PadLast

from .triton_forward import hofft_forward_fused, hofft_adjoint_fused

from typing import Optional
from einops import einsum
from tqdm import tqdm

__all__ = [
    'hofft_linop', 
]

class hofft_linop(linop):

    def __init__(self, 
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 kern_weights: torch.Tensor,
                 spatial_factors: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 os_grid: float = 1.0,
                 bparams: batching_params = batching_params()):
        """
        Initialize the HOFFT linear operator.
        
        Args:
        -----
        trj : torch.Tensor
            Trajectory of the k-space samples with shape (*trj_size, D)
        mps : torch.Tensor
            Sensitivity maps with shape (C, *im_size)
        kern_weights : torch.Tensor
            the kernel weights with shape (L, *kern_size, *trj_size)
        spatial_factors : torch.Tensor
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
        kern_size = kern_weights.shape[1:-len(trj_size)]
        oshape = (mps.shape[0], *trj_size)
        super().__init__(im_size, oshape)
        
        # Consts
        D = trj.shape[-1]
        L = kern_weights.shape[0]
        torch_dev = trj.device
        assert mps.device == torch_dev
        assert kern_weights.device == torch_dev
        assert spatial_factors.device == torch_dev
        assert spatial_factors.shape[0] == L
        
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
        self.kern_weights = kern_weights.reshape((L, -1, *trj_size))
        self.spatial_factors = spatial_factors

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
        L = self.kern_weights.shape[0]
        cbs = self.bparams.coil_batch_size
        fbs = self.bparams.field_batch_size
        
        # Output tensor
        ksp = torch.zeros(self.oshape, device=img.device, dtype=torch.complex64)
        
        # Batch over coils
        for c1, c2 in batch_iterator(C, cbs):
            
            # Apply sensitivity maps to image
            Sx = self.mps[c1:c2] * img
            
            # Batch over field/basis terms
            for l1, l2 in batch_iterator(L, fbs):
            
                # Apply apods to image
                MSx = einsum(Sx, self.spatial_factors[l1:l2],
                            'C ..., L ... -> C L ...')
                
                # Oversampled FFT
                MSx = self.padder(MSx)
                FMSx = fft(MSx, dim=tuple(range(-D, 0)))
                FMSx *= np.prod(FMSx.shape[-D:])**0.5 / np.prod(self.im_size)**0.5
                
                # Extract blocks of k-space data
                blocks = multi_index(FMSx, D, self.idx_kerns) # (C, L, *trj_size, K)
                blocks = blocks.moveaxis(-1, 2) # (C, L, K, *trj_size)
                
                # Apply kernels
                KFSx = einsum(blocks, self.kern_weights[l1:l2], 'C L K ..., L K ... -> C ...')
                ksp[c1:c2] += KFSx
                
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
        L = self.kern_weights.shape[0]
        cbs = self.bparams.coil_batch_size
        fbs = self.bparams.field_batch_size
        
        # Output tensor
        img = torch.zeros(self.ishape, device=ksp.device, dtype=torch.complex64)
        
        # Batch over coils
        for c1, c2 in batch_iterator(C, cbs):
            
            # Apply dcf
            y = ksp[c1:c2] * self.dcf
            
            # Batch over field/basis terms to keep the (C, L, *trj_size, K) tensor small
            for l1, l2 in batch_iterator(L, fbs):
                
                # Get Kernels
                Ky = einsum(y, self.kern_weights[l1:l2].conj(), 'C ..., L K ... -> C L ... K')
                
                # Gridding 
                Ky = multi_grid(Ky, self.idx_kerns, self.im_size_os) # (C, L, *im_size_os)
                FKy = ifft(Ky, dim=tuple(range(-D, 0)))
                FKy *= np.prod(FKy.shape[-D:])**0.5 / np.prod(self.im_size)**0.5
                FKy = self.padder.adjoint(FKy) # (C, L, *im_size)
                
                # Apply adjoint sensitivity maps
                SFKy = (self.mps[c1:c2, None,].conj() * FKy).sum(dim=0) # L, *im_size
                
                # Apply adjoint source maps
                MSFKy = (SFKy * self.spatial_factors[l1:l2].conj()).sum(dim=0)
                
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
    
class hofft_compressed_linop(linop):
    
    def __init__(self,
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 compressed_kernels: torch.Tensor,
                 sparse_idxs: torch.Tensor,
                 sparse_coeffs: torch.Tensor,
                 spatial_factors: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 os_grid: float = 1.0,
                 bparams: batching_params = batching_params()):
        """
        Initialize the compressed HOFFT linear operator.
        
        Args
        ----
        trj : torch.Tensor
            Trajectory of the k-space samples with shape (*trj_size, D)
        mps : torch.Tensor
            Sensitivity maps with shape (C, *im_size)
        compressed_kernels : torch.Tensor
            Compressed kernels with shape (L, *kern_size, Q)
        sparse_idxs : torch.Tensor
            Sparse indices with shape (S, *trj_size) in [0, Q) S is sparsity level (at most Q)
        sparse_coeffs : torch.Tensor
            sparse coefficients with shape (S, *trj_size) 
        spatial_factor : torch.Tensor
            Spatial factor with shape (L, *im_size)
        dcf : Optional[torch.Tensor]
            Density compensation function with shape (*trj_size)
        os_grid : Optional[float]
            Oversampling factor for the grid
        bparams : Optional[batching_params]
            Batching parameters for the linear operator
        """
        im_size = mps.shape[1:]
        trj_size = trj.shape[:-1]
        kern_size = compressed_kernels.shape[1:-1]
        oshape = (mps.shape[0], *trj_size)
        super().__init__(im_size, oshape)
        
        # Consts
        D = trj.shape[-1]
        L = compressed_kernels.shape[0]
        Q = compressed_kernels.shape[-1]
        torch_dev = trj.device
        assert mps.device == torch_dev
        assert compressed_kernels.device == torch_dev
        assert sparse_coeffs.device == torch_dev
        assert spatial_factors.device == torch_dev
        assert sparse_idxs.device == torch_dev
        assert spatial_factors.shape[0] == L
        assert sparse_idxs.shape[0] == sparse_coeffs.shape[0]
        
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
        self.compressed_kernels = compressed_kernels.reshape((L, -1, Q))
        self.sparse_idxs = sparse_idxs
        self.sparse_coeffs = sparse_coeffs
        self.bparams = bparams
        self.spatial_factors = spatial_factors

        # Precompute flattened indices/tensors for the fused forward kernel.
        # idx_lin: (T, K) raveled linear indices into prod(im_size_os).
        T = int(np.prod(trj_size))
        K = idx_kerns.shape[-2]
        self.trj_size = trj_size
        self.idx_lin = ravel(idx_kerns, im_size_os, dim=-1).reshape(T, K).to(torch.int32)
        self.sparse_idxs_flat = sparse_idxs.reshape(sparse_idxs.shape[0], T)
        self.sparse_coeffs_flat = sparse_coeffs.reshape(sparse_coeffs.shape[0], T)
        
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
        L = self.compressed_kernels.shape[0]
        cbs = self.bparams.coil_batch_size
        fbs = self.bparams.field_batch_size
        
        # Output tensor
        ksp = torch.zeros(self.oshape, device=img.device, dtype=torch.complex64)
        
        # Batch over coils
        for c1, c2 in batch_iterator(C, cbs):
            
            # Apply sensitivity maps to image
            Sx = self.mps[c1:c2] * img
            
            # Batch over field/basis terms
            for l1, l2 in batch_iterator(L, fbs):
            
                # Apply apods to image
                MSx = einsum(Sx, self.spatial_factors[l1:l2],
                            'C ..., L ... -> C L ...')
                
                # Oversampled FFT
                MSx = self.padder(MSx)
                FMSx = fft(MSx, dim=tuple(range(-D, 0)))
                FMSx *= np.prod(FMSx.shape[-D:])**0.5 / np.prod(self.im_size)**0.5
                
                # Fused block-extraction + sparse-kernel apply (avoids materializing
                # the (C,L,K,*trj) blocks and (L,K,S,*trj) gather intermediates).
                FMSx = FMSx.reshape(FMSx.shape[0], FMSx.shape[1], -1) # (C, L, NPIX)
                KFSx = hofft_forward_fused(FMSx,
                                        self.idx_lin,
                                        self.compressed_kernels[l1:l2],
                                        self.sparse_idxs_flat,
                                        self.sparse_coeffs_flat)
                ksp[c1:c2] += KFSx.reshape(c2 - c1, *self.trj_size)
            
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
        L = self.compressed_kernels.shape[0]
        cbs = self.bparams.coil_batch_size
        fbs = self.bparams.field_batch_size
        
        # Output tensor
        img = torch.zeros(self.ishape, device=ksp.device, dtype=torch.complex64)
        
        # Batch over coils
        for c1, c2 in batch_iterator(C, cbs):
            
            # Apply dcf
            y = ksp[c1:c2] * self.dcf
            
            # Batch over field/basis terms to keep the (C, L, *trj_size, K) tensor small
            for l1, l2 in batch_iterator(L, fbs):
                
                # Fused sparse-kernel apply + gridding (transpose of the forward fusion;
                # avoids the (L,K,S,*trj) gather and (C,L,*trj,K) Ky intermediates).
                Ky = hofft_adjoint_fused(y.reshape(c2 - c1, -1),
                                         self.idx_lin,
                                         self.compressed_kernels[l1:l2],
                                         self.sparse_idxs_flat,
                                         self.sparse_coeffs_flat,
                                         NPIX=int(np.prod(self.im_size_os)))
                Ky = Ky.reshape(c2 - c1, l2 - l1, *self.im_size_os) # (C, L, *im_size_os)
                FKy = ifft(Ky, dim=tuple(range(-D, 0)))
                FKy *= np.prod(FKy.shape[-D:])**0.5 / np.prod(self.im_size)**0.5
                FKy = self.padder.adjoint(FKy) # (C, L, *im_size)
                
                # Apply adjoint sensitivity maps
                SFKy = (self.mps[c1:c2, None,].conj() * FKy).sum(dim=0) # L, *im_size
                
                # Apply adjoint source maps
                MSFKy = (SFKy * self.spatial_factors[l1:l2].conj()).sum(dim=0)
                
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
    