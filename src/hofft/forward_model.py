import torch
import numpy as np

from mr_recon.linops import linop, batching_params
from mr_recon.utils import gen_grd, batch_iterator, resize
from mr_recon.fourier import fft, ifft
from mr_recon._func.indexing import multi_index, multi_grid, ravel
from mr_recon._func.pad import PadLast

from .triton_forward import hofft_forward_fused, hofft_adjoint_fused

from typing import Optional, Tuple
from einops import einsum
from tqdm import tqdm

__all__ = [
    'hofft_linop',
    'hofft_compressed_linop',
    'densify_sparse_kernels',
    'expanded_encoding',
]

def densify_sparse_kernels(compressed_kernels: torch.Tensor,
                           sparse_idxs: torch.Tensor,
                           sparse_coeffs: torch.Tensor,
                           bias_kernel: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Expands a compressed HOFFT kernel representation (as produced by
    `als_hofft_sparse_lstsq`/`als_hofft_sparse_smooth`) into the dense
    per-trajectory-point kernel-weight tensor that `hofft_linop` expects.

    The compressed representation exists to make ALS fitting cheap (fitting
    Q representative kernels instead of one dense kernel field); it isn't
    required to stay compressed at application time. This is the same size
    tensor the full/dense HOFFT model already materializes internally, so if
    that model fits in memory, so does this one -- and reusing `hofft_linop`'s
    forward/adjoint avoids `hofft_compressed_linop`'s fused Triton kernel,
    whose serial (L, K, S) loop and (in the adjoint) atomic-add scatter are
    far slower per-point than the dense path's vectorized gather + einsum.

    Args
    ----
    compressed_kernels : (L, *kern_size, Q)
        Compressed kernels.
    sparse_idxs : (S, *trj_size) int
        Sparse indices into Q for each (sparsity term, trajectory point).
    sparse_coeffs : (S, *trj_size) complex
        Sparse coefficients.
    bias_kernel : (L, *kern_size), optional
        Per-(field, kernel offset) bias, constant across trajectory points.

    Returns
    -------
    kern_weights : (L, *kern_size, *trj_size) complex64
        Dense kernel weights, ready to pass to `hofft_linop`.
    """
    L = compressed_kernels.shape[0]
    kern_size = compressed_kernels.shape[1:-1]
    S = sparse_idxs.shape[0]
    trj_size = sparse_idxs.shape[1:]

    comp_flat = compressed_kernels.reshape(L, -1, compressed_kernels.shape[-1]) # (L, K, Q)
    idx_flat = sparse_idxs.reshape(S, -1) # (S, T)
    coef_flat = sparse_coeffs.reshape(S, -1) # (S, T)
    K = comp_flat.shape[1]
    T = idx_flat.shape[1]

    kern_weights = torch.zeros((L, K, T), dtype=comp_flat.dtype, device=comp_flat.device)
    for s in range(S):
        kern_weights += comp_flat[:, :, idx_flat[s]] * coef_flat[s]

    if bias_kernel is not None:
        kern_weights = kern_weights + bias_kernel.reshape(L, K, 1)

    return kern_weights.reshape(L, *kern_size, *trj_size)

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
                 bias_kernel: Optional[torch.Tensor] = None,
                 temporal_factors: Optional[torch.Tensor] = None,
                 dcf: Optional[torch.Tensor] = None,
                 os_grid: float = 1.0,
                 bparams: batching_params = batching_params(),
                 auto_tune: bool = False):
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
        bias_kernel : Optional[torch.Tensor]
            Bias kernel with shape (L, *kern_size)
        temporal_factors : Optional[torch.Tensor]
            Per-trajectory-point phase (e.g. phase-midpoint correction) with
            shape (*trj_size). Applied to the k-space output (and its conjugate
            in the adjoint). Because it multiplies every (L, kern) weight at a
            given trajectory point equally, it factors out of the kernel sum --
            so it must be applied here, not folded into sparse_coeffs (folding
            into sparse_coeffs silently drops the bias_kernel's phase).
        dcf : Optional[torch.Tensor]
            Density compensation function with shape (*trj_size)
        os_grid : Optional[float]
            Oversampling factor for the grid
        bparams : Optional[batching_params]
            Batching parameters for the linear operator
        auto_tune : bool
            If True spends some time selecting optimal triton GPU compute parameters for the fused forward kernel.
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
        self.auto_tune = auto_tune
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
        if bias_kernel is not None:
            self.bias_kernel = bias_kernel.reshape((L, -1))
        else:
            self.bias_kernel = None
        if temporal_factors is not None:
            assert temporal_factors.shape == trj_size, \
                f"temporal_factors shape {tuple(temporal_factors.shape)} != {tuple(trj_size)}"
            assert temporal_factors.device == torch_dev
            self.temporal_factors = temporal_factors.type(torch.complex64)
        else:
            self.temporal_factors = None

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
                bias_l = None if self.bias_kernel is None else self.bias_kernel[l1:l2]
                KFSx = hofft_forward_fused(FMSx,
                                        self.idx_lin,
                                        self.compressed_kernels[l1:l2],
                                        self.sparse_idxs_flat,
                                        self.sparse_coeffs_flat,
                                        bias_l,
                                        auto_tune=self.auto_tune)
                ksp[c1:c2] += KFSx.reshape(c2 - c1, *self.trj_size)
        
        # Per-trajectory-point phase factors out of the (L, kern) kernel sum
        if self.temporal_factors is not None:
            ksp = ksp * self.temporal_factors
        
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
            
            # Adjoint of the per-trajectory-point phase applied in forward
            if self.temporal_factors is not None:
                y = y * self.temporal_factors.conj()
            
            # Batch over field/basis terms to keep the (C, L, *trj_size, K) tensor small
            for l1, l2 in batch_iterator(L, fbs):
                
                # Fused sparse-kernel apply + gridding (transpose of the forward fusion;
                # avoids the (L,K,S,*trj) gather and (C,L,*trj,K) Ky intermediates).
                bias_l = None if self.bias_kernel is None else self.bias_kernel[l1:l2]
                Ky = hofft_adjoint_fused(y.reshape(c2 - c1, -1),
                                         self.idx_lin,
                                         self.compressed_kernels[l1:l2],
                                         self.sparse_idxs_flat,
                                         self.sparse_coeffs_flat,
                                         NPIX=int(np.prod(self.im_size_os)),
                                         bias_kernel=bias_l,
                                         auto_tune=self.auto_tune)
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

class expanded_encoding(linop):
    """
    Linop for naive encoding matrix construction.

    When ``spatial_mask`` is provided, the encoding matrix is only built and
    applied over the non-zero mask voxels (exact match to the full operator
    when the image/mps are zero outside the mask).
    """

    def __init__(self,
                 mps: torch.Tensor,
                 phis: torch.Tensor,
                 alphas: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 spatial_funcs: Optional[torch.Tensor] = None,
                 temporal_funcs: Optional[torch.Tensor] = None,
                 spatial_mask: Optional[torch.Tensor] = None,
                 temporal_batch_size: Optional[int] = None,
                 bparams: Optional[batching_params] = batching_params(),
                 verbose: Optional[bool] = False):
        """
        Initialize the expanded encoding linear operator.
        
        Args
        ----
        mps : torch.Tensor
            Sensitivity maps with shape (C, *im_size)
        phis : torch.Tensor
            Spatial basis functions with shape (B, *im_size)
        alphas : torch.Tensor
            Temporal basis functions with shape (B, *trj_size)
        dcf : Optional[torch.Tensor]
            Density compensation function with shape (*trj_size)
        spatial_funcs : Optional[torch.Tensor]
            Spatial basis functions with shape (L, *im_size)
        temporal_funcs : Optional[torch.Tensor]
            Temporal basis functions with shape (L, *trj_size)
        spatial_mask : Optional[torch.Tensor]
            Spatial support mask with shape (*im_size). Encoding is restricted
            to non-zero voxels (mask values are used only for support, not as
            soft weights).
        temporal_batch_size : Optional[int]
            Batch size for the temporal dimension
        bparams : Optional[batching_params]
            Batching parameters for the linear operator
        verbose : Optional[bool]
            If True prints verbose output
        """
        im_size = mps.shape[1:]
        trj_size = alphas.shape[1:]
        B = phis.shape[0]
        C = mps.shape[0]
        super().__init__(im_size, (C, *trj_size))

        # Consts
        torch_dev = alphas.device
        assert mps.device == torch_dev
        self.R_full = int(np.prod(im_size))

        # Default params
        if dcf is None:
            dcf = torch.ones(trj_size, dtype=torch.float32, device=torch_dev)

        # Restrict to non-zero spatial support (keeps full-FOV normalization)
        phis_flt = phis.reshape((B, -1))
        mps_flt = mps.reshape((C, -1))
        if spatial_mask is not None:
            assert spatial_mask.shape == im_size, (
                f"spatial_mask must have shape {im_size}, got {tuple(spatial_mask.shape)}"
            )
            mask_flt = spatial_mask.reshape(-1)
            mask_idx = mask_flt.nonzero(as_tuple=False).squeeze(-1)
            phis_flt = phis_flt[:, mask_idx]
            mps_flt = mps_flt[:, mask_idx]
            self.mask_idx = mask_idx
        else:
            self.mask_idx = None

        # additional spatial-temporal functions
        self.b = None
        self.h = None
        if spatial_funcs is not None:
            assert temporal_funcs is not None, "spatial_funcs and temporal_funcs must be provided together"
            assert spatial_funcs.shape[0] == temporal_funcs.shape[0], (
                f"spatial_funcs and temporal_funcs must have the same number of bases, "
                f"but got {spatial_funcs.shape[0]} and {temporal_funcs.shape[0]}"
            )
            b = spatial_funcs.reshape((spatial_funcs.shape[0], -1))
            if self.mask_idx is not None:
                b = b[:, self.mask_idx]
            self.b = b
            self.h = temporal_funcs.reshape((temporal_funcs.shape[0], -1))

        self.bparams = bparams
        self.alphas_flt = alphas.reshape((B, -1))
        self.phis_flt = phis_flt
        self.dcf_flt = dcf.flatten()
        self.mps_flt = mps_flt
        self.tbs = temporal_batch_size
        self.verbose = verbose

    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        # Consts
        T = self.alphas_flt.shape[1]
        C = self.mps_flt.shape[0]
        cbs = self.bparams.coil_batch_size
        tbs = self.tbs

        ksp = torch.zeros((C, T), dtype=torch.complex64, device=img.device)
        img_flt = img.flatten()
        if self.mask_idx is not None:
            img_flt = img_flt[self.mask_idx]
        for c1, c2 in batch_iterator(C, cbs):

            # Apply coil maps
            Sx = (self.mps_flt[c1:c2] * img_flt) # cbs R

            for t1, t2 in batch_iterator(T, tbs):

                # Grab a temporal batch
                alphas_batch = self.alphas_flt[:, t1:t2] # B tbs
                phis_batch = self.phis_flt # B R

                # Apply encoding matrix
                enc_mx = torch.exp(-2j * torch.pi * (phis_batch.T @ alphas_batch)) # R tbs

                # additional weighting
                if self.b is not None:
                    weighting = (self.b.T @ self.h[:, t1:t2]) # R tbs
                    enc_mx *= weighting

                ksp[c1:c2, t1:t2] += (Sx @ enc_mx)

        return ksp.reshape(self.oshape) / (self.R_full ** 0.5) # mimick orthogonal FFT

    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        # Consts
        R = self.phis_flt.shape[1]
        T = self.alphas_flt.shape[1]
        C = self.mps_flt.shape[0]
        cbs = self.bparams.coil_batch_size
        tbs = self.tbs

        img = torch.zeros(R, dtype=torch.complex64, device=ksp.device)
        ksp_flt = ksp.reshape((C, T))
        t_batches = list(batch_iterator(T, tbs))
        for c1, c2 in batch_iterator(C, cbs):

            for t1, t2 in tqdm(t_batches, disable=not self.verbose):

                # Grab a temporal batch
                alphas_batch = self.alphas_flt[:, t1:t2] # B tbs
                phis_batch = self.phis_flt # B R

                # Apply adjoint encoding matrix
                enc_mx = torch.exp(2j * torch.pi * (alphas_batch.T @ phis_batch)) # tbs R

                # additional weighting
                if self.b is not None:
                    weighting = (self.h[:, t1:t2].T @ self.b) # tbs R
                    enc_mx *= weighting.conj()

                coil_imgs = (ksp_flt[c1:c2, t1:t2] * self.dcf_flt[t1:t2]) @ enc_mx # cbs R

                # Apply adjoint coils
                img += (self.mps_flt[c1:c2].conj() * coil_imgs).sum(dim=0)

        if self.mask_idx is not None:
            img_full = torch.zeros(self.R_full, dtype=img.dtype, device=img.device)
            img_full[self.mask_idx] = img
            img = img_full

        return img.reshape(self.ishape) / (self.R_full ** 0.5) # mimick orthogonal FFT

    def normal(self,
               img: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(img))
