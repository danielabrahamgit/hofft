"""
This file contains the matrix-vector operations for high order phase integrals. 

The forward matrix-vector operation is given by:
y[t] = sum_r x[r] * exp(-j 2pi phi[r] * alpha[t])
and the adjoint matrix-vector operation is given by:
x[r] = sum_t y[t] * exp(j 2pi phi[r] * alpha[t])
"""

import torch
import sigpy as sp
import numpy as np

from mr_recon.utils import torch_to_np, np_to_torch, resize, quantize_data, pick_K_vectors
from mr_recon.fourier import sigpy_nufft, fft, ifft
from mr_recon.spatial import spatial_resize_poly
from mr_recon.algs import svd_operator
from mr_recon.imperfections.field import rescale_phis_alphas
from typing import Optional
from einops import einsum
from .phase_coeffs import uniform_quantization, kmeans_quantization

class matvec(torch.nn.Module):
    """
    Matrix-vector operation for the HOFFT model.
    """

    def __init__(self,
                 phis: torch.Tensor,
                 alphas: torch.Tensor,
                 spatial_batch_size: Optional[int] = None,
                 temporal_batch_size: Optional[int] = None,
                ):
        """
        Args
        ----
        phis : torch.Tensor
            Spatial phase maps with shape (B, *im_size)
        alphas : torch.Tensor
            Temporal phase coefficients with shape (B, *trj_size)
        spatial_batch_size : Optional[int]
            Spatial batch size for the matrix-vector operation.
        temporal_batch_size : Optional[int]
            Temporal batch size for the matrix-vector operation.
        spatial_weights : Optional[torch.Tensor]
            Spatial weights with shape (*im_size)
        temporal_weights : Optional[torch.Tensor]
            Temporal weights with shape (*trj_size)
        """
        super(matvec, self).__init__()
        self.phis = phis.to(torch.float32)
        self.alphas = alphas.to(torch.float32)
        self.im_size = self.phis.shape[1:]
        self.trj_size = self.alphas.shape[1:]
        self.ishape = self.im_size
        self.oshape = self.trj_size
        self.R = np.prod(self.im_size)
        self.T = np.prod(self.trj_size)
        self.B = self.phis.shape[0]
        assert self.B == self.alphas.shape[0], 'Number of spatial and temporal phase maps must match.'
        
        # Batch sizes
        if spatial_batch_size is None:
            self.spatial_batch_size = np.prod(self.phis.shape[1:])
        else:
            self.spatial_batch_size = spatial_batch_size
        if temporal_batch_size is None:
            self.temporal_batch_size = np.prod(self.alphas.shape[1:])
        else:
            self.temporal_batch_size = temporal_batch_size
            
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix-vector operation.
        
        Args
        ----
        x : torch.Tensor
            input signal to be multiplied with the matrix with shape (N, *im_size)
        
        Returns
        -------
        out : torch.Tensor
            Output signal with shape (N, *trj_size)
        """
        raise NotImplementedError('Forward pass not implemented.')
    
    def adjoint(self,
                y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint pass of the matrix-vector operation.
        
        Args
        ----
        y : torch.Tensor
            output signal to be multiplied with the matrix with shape (N, *trj_size)
        
        Returns
        -------
        x : torch.Tensor
            Input signal with shape (N, *im_size)
        """
        raise NotImplementedError('Adjoint pass not implemented.')
    
    def normal(self,
               x: torch.Tensor) -> torch.Tensor:
        """
        Normal pass of the matrix-vector operation.
        
        Args
        ----
        x : torch.Tensor
            Input signal to be multiplied with the matrix with shape (N, *im_size)
        
        Returns
        -------
        y : torch.Tensor
            Output signal with shape (N, *trj_size)
        """
        return self.adjoint(self.forward(x))

class matvec_naive(matvec):
    """
    Matrix-vector operation for the HOFFT model using naive implementation.
    """

    def __init__(self,
                 phis: torch.Tensor,
                 alphas: torch.Tensor,
                 spatial_batch_size: Optional[int] = None,
                 temporal_batch_size: Optional[int] = None):
        """
        Args
        ----
        phis : torch.Tensor
            Spatial phase maps with shape (B, *im_size)
        alphas : torch.Tensor
            Temporal phase coefficients with shape (B, *trj_size)
        spatial_batch_size : Optional[int]
            Spatial batch size for the matrix-vector operation.
        temporal_batch_size : Optional[int]
            Temporal batch size for the matrix-vector operation.
        """
        super(matvec_naive, self).__init__(phis, alphas, spatial_batch_size, temporal_batch_size)
        
        # Flatten phis and alphas
        B = phis.shape[0]
        self.phis = self.phis.reshape((B, -1))
        self.alphas = self.alphas.reshape((B, -1))
        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix-vector operation.
        
        Args
        ----
        x : torch.Tensor
            Input signal to be multiplied with the matrix with shape (N, *im_size)
        
        Returns
        -------
        y : torch.Tensor
            Output signal with shape (N, *trj_size)
        """
        # consts
        N = x.shape[0]
        T = self.T
        R = self.R
        
        # Perform the matrix-vector operation
        x_flt = x.reshape((N, -1)) # N R
        y_flt = torch.zeros((N, T), dtype=x.dtype, device=x.device)
        for r1 in range(0, R, self.spatial_batch_size):
            r2 = min(r1 + self.spatial_batch_size, R)
            for t1 in range(0, T, self.temporal_batch_size):
                t2 = min(t1 + self.temporal_batch_size, T)
                enc_mat = torch.exp(-2j * torch.pi * (self.phis[:, r1:r2].T @ self.alphas[:, t1:t2])) # R T
                y_flt[:, t1:t2] += x_flt[:, r1:r2] @ enc_mat # N T
        
        # Reshape output
        y = y_flt.reshape((N, *self.trj_size))
        
        return y
    
    def adjoint(self,
                y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint pass of the matrix-vector operation.
        
        Args
        ----
        y : torch.Tensor
            Output signal to be multiplied with the matrix with shape (N, *trj_size)
        
        Returns
        -------
        x : torch.Tensor
            Input signal with shape (N, *im_size)
        """
        # consts
        N = y.shape[0]
        R = self.R    
        T = self.T
        
        # Perform the matrix-vector operation
        y_flt = y.reshape((N, -1)) # N T
        x_flt = torch.zeros((N, R), dtype=y.dtype, device=y.device)
        for r1 in range(0, R, self.spatial_batch_size):
            r2 = min(r1 + self.spatial_batch_size, R)
            for t1 in range(0, T, self.temporal_batch_size):
                t2 = min(t1 + self.temporal_batch_size, T)
                enc_mat = torch.exp(2j * torch.pi * (self.alphas[:, t1:t2].T @ self.phis[:, r1:r2])) # T R
                x_flt[:, r1:r2] += y_flt[:, t1:t2] @ enc_mat
        
        # Reshape output
        x = x_flt.reshape((N, *self.im_size))
        
        return x

class matvec_type3(matvec):

    def __init__(self,
                 phis: torch.Tensor,
                 alphas: torch.Tensor,
                 oversamp: float = 1.25,
                 width: float = 4.0,
                 use_toep: Optional[bool] = False):
        """
        KB implimentation of the type-3 nufft as described in:
        "A PARALLEL NONUNIFORM FAST FOURIER TRANSFORM LIBRARY 
        BASED ON AN ``EXPONENTIAL OF SEMICIRCLE" KERNEL - Barnett et. al.
        "https://epubs.siam.org/doi/pdf/10.1137/18M120885X

        Args
        -----
        phis : torch.Tensor
            The spatial phase maps, shape (B, *im_size)
        alphas : torch.Tensor
            The temporal phase coefficients, shape (B, *trj_size)
        oversamp : float
            The oversampling factor for spatial gridding
        width : float
            The width of the gridding kernel
        use_toep : bool
            toggles Topelitz for gram/normal/AHA operator
        """    
        # Consts
        B, *im_size = phis.shape
        B_, *trj_size = alphas.shape
        R = np.prod(im_size)
        T = np.prod(trj_size)
        torch_dev = phis.device
        assert B == B_, "phis and alphas must have same number of bases (B)"
        assert phis.device == alphas.device, "phis and alphas must be on same device"
        assert phis.dtype == alphas.dtype, "phis and alphas must have same dtype"
        super(matvec_type3, self).__init__(phis, alphas, None, None)
     
        # Flatten everything
        phis_flt = phis.reshape((B, R))
        alphas_flt = alphas.reshape((B, T))
        
        # Center alphas and phis
        phis_mp = (phis_flt.min(dim=1).values + phis_flt.max(dim=1).values)/2
        alphas_mp = (alphas_flt.min(dim=1).values + alphas_flt.max(dim=1).values)/2
        phis_flt_cent = phis_flt - phis_mp[:, None]
        alphas_flt_cent = alphas_flt - alphas_mp[:, None]
        
        # Rescale phis to be between [-1/2, 1/2]
        scales = phis_flt_cent.abs().max(dim=1).values * 2
        phis_flt_cent /= scales[:, None]
        phis_mp /= scales
        alphas_flt_cent *= scales[:, None]
        alphas_mp *= scales
        
        # Store phis, alphas, other constants
        self.phis = phis_flt_cent
        self.alphas = alphas_flt_cent
        self.phis_mp = phis_mp
        self.alphas_mp = alphas_mp
        self.torch_dev = torch_dev
        self.B = B
        
        # Consts for gridding
        self.grd_S = self.alphas.abs().max(dim=1).values
        self.grd_W = width
        self.grd_N_os = torch.ceil(2 * self.grd_S * oversamp + self.grd_W).long()
        self.grd_os = self.grd_N_os / (self.grd_N_os / oversamp).round() # FIXME?
        self.grd_N = self.grd_N_os / self.grd_os
        self.grd_gamma = self.grd_N_os / (2 * self.grd_os * self.grd_S)
        self.grd_beta = np.pi * (((self.grd_W / self.grd_os) * (self.grd_os - 0.5))**2 - 0.8)**0.5
        self.grd_N_os = tuple((self.grd_N * self.grd_os).ceil().long().tolist())
        self.nft = sigpy_nufft(self.grd_N_os)
        
        # Optimize beta
        self.nft.beta = self.nft.optimal_beta(torch_dev=torch_dev) # Better beta calculation
        if use_toep:
            # print('Computing toeplitz Kernels ... ', end='')
            apod_weights = self.apod(self.alphas.T * self.grd_gamma / self.grd_N / self.grd_os, self.grd_beta, self.grd_W).prod(dim=-1)
            self.kerns = self.nft.calc_teoplitz_kernels(trj=self.alphas.T[None,] * self.grd_gamma, weights=apod_weights[None,] ** 2)
            # print('done.')
        else:
            self.kerns = None
        
    @staticmethod
    def apod(x, beta, width):
        eps = 1e-12
        arg = (beta**2 - (np.pi * width * x) ** 2)
        apod_pos = arg.clamp(min=0).sqrt()
        apod_pos /= torch.sinh(apod_pos) + eps
        apod_neg = (-arg.clamp(max=0)).sqrt()
        apod_neg /= torch.sin(apod_neg) + eps
        return apod_pos + apod_neg
        
    def forward(self,
                x: torch.Tensor,) -> torch.Tensor:
        """
        Forward type 3 nufft.
        
        Args:
        -----
        x : torch.Tensor
            The image to be transformed, shape (N, *im_size)
        
        Returns:
        --------
        y : torch.Tensor
            The output with shape (N, *trj_size)
        """
        # Consts
        N = x.shape[0]
        
        # ----------------- Step 0: Apply spatial midpoints -----------------
        phz = self.phis.T @ self.alphas_mp
        x_mp = x.reshape(N, -1) * torch.exp(-2j * torch.pi * phz)

        # ----------------- Step 1: gridding in the image domain -----------------
        # Compute shifts and scales
        scales = (self.grd_os * self.grd_N).ceil() / self.grd_N
        shifts = (self.grd_os * self.grd_N).ceil() // 2
        
        # Define output matrix size and betas
        betas = tuple(self.grd_beta.tolist())
        
        # Move to cupy
        x_cp_flt = torch_to_np(x_mp)
        dev = sp.get_device(x_cp_flt)
        with dev:
            
            # Rescale trajectory
            trj = scales * (self.phis.T * self.grd_N / self.grd_gamma) + shifts
            trj_cp = torch_to_np(trj)
            
            # Gridding
            output = sp.interp.gridding(x_cp_flt, trj_cp, (N,) + self.grd_N_os,
                                        kernel='kaiser_bessel', width=self.grd_W, param=betas)
            x_grid = np_to_torch(output)
            x_grid /= self.grd_W ** self.B
            
        # ----------------- Step 2: Call NUFFT on gridded data -----------------
        alphas_rep = torch.repeat_interleave(self.alphas.T[None,], N, dim=0) # N T B
        y_pre_apod = self.nft.forward(x_grid, alphas_rep * self.grd_gamma) # N T
        y_pre_apod *= np.prod(self.grd_N_os) ** 0.5
        
        # ----------------- Step 3: Apodize -----------------
        y = y_pre_apod * self.apod(alphas_rep * self.grd_gamma / self.grd_N / self.grd_os, self.grd_beta, self.grd_W).prod(dim=-1)
        
        # ----------------- Step 4: Apply temporal midpoints -----------------
        phz = (self.alphas.T + self.alphas_mp) @ (self.phis_mp)
        y = y * torch.exp(-2j * torch.pi * phz)
        
        # ----------------- Step 5: Reshape and Pray -----------------
        return y.reshape((N, *self.trj_size))

    def adjoint(self,
                y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint type 3 nufft.
        
        Args:
        -----
        y : torch.Tensor
            The k-space data to be transformed, shape (N, *trj_size)
        
        Returns:
        --------
        x : torch.Tensor
            The output with shape (N, *im_size)
        """
        # Consts
        N = y.shape[0]
        
        # ----------------- Step 0: Apply temporal midpoints -----------------
        phz = self.alphas.T @ self.phis_mp
        y_mp = y.reshape((N, -1)) * torch.exp(2j * torch.pi * phz)
        
        # ----------------- Step 1: Apodize -----------------
        alphas_rep = torch.repeat_interleave(self.alphas.T[None,], N, dim=0) # N T B
        y_apod = y_mp * self.apod(alphas_rep * self.grd_gamma / self.grd_N / self.grd_os, self.grd_beta, self.grd_W).prod(dim=-1)
        
        # ----------------- Step 2: Call Adjoint NUFFT -----------------
        x_grid = self.nft.adjoint(y_apod, alphas_rep * self.grd_gamma) # N *self.grd_N_os
        x_grid *= np.prod(self.grd_N_os) ** 0.5
        
        # ----------------- Step 3: Interpolation in the image domain -----------------
        # Compute shifts and scales
        scales = (self.grd_os * self.grd_N).ceil() / self.grd_N
        shifts = (self.grd_os * self.grd_N).ceil() // 2
        
        # Define output matrix size and betas
        betas = tuple(self.grd_beta.tolist())

        # Move to cupy
        x_grid_cp = torch_to_np(x_grid)
        dev = sp.get_device(x_grid_cp)
        with dev:
            
            # Rescale trajectory
            trj = scales * (self.phis.T * self.grd_N / self.grd_gamma) + shifts
            trj_cp = torch_to_np(trj)
            
            # Interpolate
            output = sp.interp.interpolate(x_grid_cp, trj_cp,
                                           kernel='kaiser_bessel', width=self.grd_W, param=betas)
            x = np_to_torch(output)
            x /= self.grd_W ** self.B

        # ----------------- Step 4: Apply spatial midpoints -----------------
        phz = (self.phis.T + self.phis_mp) @ self.alphas_mp
        x = x * torch.exp(2j * torch.pi * phz)
        
        # ----------------- Step 5: Reshape and pray -----------------
        return x.reshape((N, *self.im_size))

    def normal(self,
               x: torch.Tensor) -> torch.Tensor:
        """
        Applies normal operator
        
        Args:
        -----
        x : torch.Tensor
            The image to be transformed, shape (N, *im_size)
            
        Returns:
        --------
        torch.Tensor
            The output with shape (N, *im_size)
        """
        if self.kerns is None:
            return self.adjoint(self.forward(x))
        else:
            # Consts
            N = x.shape[0]
            B = self.phis.shape[0]
            
            # ----------------- Step 0: Apply spatial midpoints -----------------
            phz = (self.phis.T + self.phis_mp) @ self.alphas_mp
            x_mp = x.reshape(N, -1) * torch.exp(-2j * torch.pi * phz)

            # ----------------- Step 1: gridding in the image domain -----------------
            # Compute shifts and scales
            scales = (self.grd_os * self.grd_N).ceil() / self.grd_N
            shifts = (self.grd_os * self.grd_N).ceil() // 2
            
            # Define output matrix size and betas
            betas = tuple(self.grd_beta.tolist())

            # Move to cupy
            x_cp_flt = torch_to_np(x_mp)
            dev = sp.get_device(x_cp_flt)
            with dev:
                
                # Rescale trajectory
                trj = scales * (self.phis.T * self.grd_N / self.grd_gamma) + shifts
                trj_cp = torch_to_np(trj)
                
                # Gridding
                output = sp.interp.gridding(x_cp_flt, trj_cp, (N,) + self.grd_N_os,
                                                kernel='kaiser_bessel', width=self.grd_W, param=betas)
                x_grid = np_to_torch(output)
                x_grid /= self.grd_W ** self.B
                
            # ----------------- Step 2: Apply toeplitz kernels -----------------
            x_zp = resize(x_grid, [N,] + [self.grd_N_os[i] * 2 for i in range(B)])
            x_ft = fft(x_zp, dim=tuple(range(-B, 0)))
            x_tp = x_ft * self.kerns
            x_ift = ifft(x_tp, dim=tuple(range(-B, 0)))
            x_crp = resize(x_ift, (N, *self.grd_N_os))
            x_grid = x_crp
            x_grid *= np.prod(self.grd_N_os)
            
            # ----------------- Step 3: Interpolation in the image domain -----------------
            # Compute shifts and scales
            scales = (self.grd_os * self.grd_N).ceil() / self.grd_N
            shifts = (self.grd_os * self.grd_N).ceil() // 2
            
            # Define output matrix size and betas
            betas = tuple(self.grd_beta.tolist())

            # Move to cupy
            x_grid_cp = torch_to_np(x_grid)
            dev = sp.get_device(x_grid_cp)
            with dev:
                
                # Rescale trajectory
                trj = scales * (self.phis.T * self.grd_N / self.grd_gamma) + shifts
                trj_cp = torch_to_np(trj)
                
                # Interpolate
                output = sp.interp.interpolate(x_grid_cp, trj_cp,
                                               kernel='kaiser_bessel', width=self.grd_W, param=betas)
                x = np_to_torch(output)
                x /= self.grd_W ** self.B

            # ----------------- Step 4: Apply spatial midpoints -----------------
            phz = (self.phis.T + self.phis_mp) @ self.alphas_mp
            x = x * torch.exp(2j * torch.pi * phz)
            
            # ----------------- Step 5: Reshape and pray -----------------
            return x.reshape((N, *self.ishape))

class matvec_svd(matvec):
    
    def __init__(self,
                 phis: torch.Tensor,
                 alphas: torch.Tensor,
                 svd_rank: int = 100,
                 svd_method: str = 'torch',
                 num_iter: int = 15,
                 im_size_low: Optional[tuple] = None,
                 trj_size_low: Optional[tuple] = None,
                 poly_order_img: int = 3,
                 poly_order_trj: int = 3,
                 spatial_batch_size: Optional[int] = None,
                 temporal_batch_size: Optional[int] = None):
        """
        Args
        ----
        phis : torch.Tensor
            Spatial phase maps with shape (B, *im_size)
        alphas : torch.Tensor
            Temporal phase coefficients with shape (B, *trj_size)
        svd_rank : Optional[int]
            Number of singular values to keep.
        svd_method : Optional[str]
            Method to use for SVD.
            'torch' - uses torch.svd
            'lobpcg' - uses iterative lobpcg algorithm, spatial/temporal batching supported
            'power' - uses power iteration algorithm, spatial/temporal batching supported
        num_iter : int
            Number of iterations for svd iterative methods.
        im_size_low : Optional[tuple]
            low resolution image size for faster interpolation.
        trj_size_low : Optional[tuple]
            low resolution trajectory size for faster interpolation.
        poly_order_img : Optional[int]
            Polynomial order for image interpolation.
        poly_order_trj : Optional[int]
            Polynomial order for trajectory interpolation.
        spatial_batch_size : Optional[int]
            Spatial batch size for the matrix-vector operation.
        temporal_batch_size : Optional[int]
            Temporal batch size for the matrix-vector operation.
        """
        super(matvec_svd, self).__init__(phis, alphas, spatial_batch_size, temporal_batch_size)
        self.svd_rank = svd_rank
        
        # Downsample if needed
        if im_size_low is not None:
            phis = spatial_resize_poly(phis, im_size_low, order=poly_order_img)
        if trj_size_low is not None:
            alphas = spatial_resize_poly(alphas, trj_size_low, order=poly_order_trj)
            
        # Build flattened encoding matrix
        phis_flt = phis.reshape((self.B, -1))
        alphas_flt = alphas.reshape((self.B, -1))
        enc_mat = torch.exp(-2j * torch.pi * (alphas_flt.T @ phis_flt)) # T R
            
        # Perform SVD
        if svd_method == 'torch':
            U, S, Vh = torch.linalg.svd(enc_mat, full_matrices=False)
        else:
            A = lambda x: einsum(x, enc_mat, 'N R, T R -> N T')
            enc_normal = enc_mat.H @ enc_mat
            AHA = lambda x: einsum(x, enc_normal, 'N Ri, Ro Ri -> N Ro')
            inp_vec = torch.randn(enc_mat.shape[1], dtype=enc_mat.dtype, device=enc_mat.device)
            if svd_method == 'lobpcg':
                U, S, Vh = svd_operator(A, AHA, inp_vec, rank=svd_rank, lobpcg=True, num_iter=num_iter)
            elif svd_method == 'power':
                U, S, Vh = svd_operator(A, AHA, inp_vec, rank=svd_rank, lobpcg=False, num_iter=num_iter)
        
        # Extract spatial and temporal factors
        self.spatial_factors = (S[:svd_rank, None] ** 0.5) * Vh[:svd_rank, :]
        self.spatial_factors = self.spatial_factors.reshape((svd_rank, *phis.shape[1:]))
        self.temporal_factors = (S[:svd_rank, None] ** 0.5) * U[:, :svd_rank].T
        self.temporal_factors = self.temporal_factors.reshape((svd_rank, *alphas.shape[1:]))
        
        # Normal operator
        self.normal_factor = einsum(self.spatial_factors, S[:svd_rank] ** 0.5, 'L ..., L -> L ...')
        
        # Upsample if needed
        if im_size_low is not None:
            self.spatial_factors = spatial_resize_poly(self.spatial_factors, self.im_size, order=poly_order_img)
        if trj_size_low is not None:
            self.temporal_factors = spatial_resize_poly(self.temporal_factors, self.trj_size, order=poly_order_trj)
            
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix-vector operation.
        
        Args
        ----
        x : torch.Tensor
            Input signal to be multiplied with the matrix with shape (N, *im_size)
        
        Returns
        -------
        y : torch.Tensor
            Output signal with shape (N, *trj_size)
        """
        # consts
        N = x.shape[0]
            
        # Perform the low-rank approximation
        coeffs = einsum(x, self.spatial_factors, 'N ..., L ... -> N L')
        y = einsum(coeffs, self.temporal_factors, 'N L, L ... -> N ...')
        
        return y

    def adjoint(self,
                y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint pass of the matrix-vector operation.
        
        Args
        ----
        y : torch.Tensor
            Output signal to be multiplied with the matrix with shape (N, *trj_size)
            
        Returns
        -------
        x : torch.Tensor
            Input signal with shape (N, *im_size)
        """
        # Consts
        N = y.shape[0]
        
        # Perform the low-rank approximation
        coeffs = einsum(y, self.temporal_factors.conj(), 'N ..., L ... -> N L')
        x = einsum(coeffs, self.spatial_factors.conj(), 'N L, L ... -> N ...')
        
        return x
    
    def normal(self,
               x: torch.Tensor) -> torch.Tensor:
        """
        Normal pass of the matrix-vector operation.
        
        Args
        ----
        x : torch.Tensor
            Input signal to be multiplied with the matrix with shape (N, *im_size)
        """
        coeffs = einsum(x, self.normal_factor, 'N ..., L ... -> N L')
        return einsum(coeffs, self.normal_factor.conj(), 'N L, L ... -> N ...')
    
class matvec_cur(matvec):

    def __init__(self,
                 phis: torch.Tensor,
                 alphas: torch.Tensor,
                 cur_rank: int = 100,
                 spatial_batch_size: Optional[int] = None,
                 temporal_batch_size: Optional[int] = None):
        """
        Args
        ----
        phis : torch.Tensor
            Spatial phase maps with shape (B, *im_size)
        alphas : torch.Tensor
            Temporal phase coefficients with shape (B, *trj_size)
        cur_rank : int
            Number of columns to sample for CUR decomposition.
        spatial_batch_size : Optional[int]
            Spatial batch size for the matrix-vector operation.
        temporal_batch_size : Optional[int]
            Temporal batch size for the matrix-vector operation.
        """
        super(matvec_cur, self).__init__(phis, alphas, spatial_batch_size, temporal_batch_size)
        
        # Normalize phis and alphas before clustering
        phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis.reshape((self.B, -1)), alphas.reshape((self.B, -1)))
        
        # Pick clusters
        method = 'kmeans'
        B = alphas.shape[0]
        alpha_clusts, _ = pick_K_vectors(vectors=alphas_nrm.reshape((B,-1)).T, K=cur_rank, sigma=0, method=method)
        alpha_clusts = alpha_clusts + alphas_mp
        phi_clusts, _ = pick_K_vectors(vectors=phis_nrm.reshape((B,-1)).T, K=cur_rank, sigma=0, method=method)
        phi_clusts = phi_clusts + phis_mp
        
        # Build R and C matrices
        R = torch.exp(-2j * torch.pi * (alpha_clusts @ (phis_nrm + phis_mp[:, None]))) # K R
        C = torch.exp(-2j * torch.pi * (phi_clusts @ (alphas_nrm + alphas_mp[:, None]))) # K T
        
        # Buiild U matrix
        W = torch.exp(-2j * torch.pi * (alpha_clusts @ phi_clusts.T)) # K K
        U = torch.linalg.pinv(W)
        
        # Reshape matrices and save
        self.R = R.reshape((cur_rank, *self.im_size))
        self.R = einsum(U, self.R, 'Ko K, K ... -> Ko ...')
        self.C = C.reshape((cur_rank, *self.trj_size))
        
        # Save stuff for normal operator
        self.normal_factor = einsum(self.C.conj(), self.C, 'Kl ..., Kr ... -> Kl Kr')
        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix-vector operation.
        
        Args
        ----
        x : torch.Tensor
            Input signal to be multiplied with the matrix with shape (N, *im_size)
        
        Returns
        -------
        y : torch.Tensor
            Output signal with shape (N, *trj_size)
        """
        # Consts
        N = x.shape[0]
        
        # Perform the CUR decomposition
        coeffs = einsum(x, self.R, 'N ..., K ... -> N K')
        y = einsum(coeffs, self.C, 'N K, K ... -> N ...')
        
        return y
    
    def adjoint(self,
                y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint pass of the matrix-vector operation.
        
        Args
        ----
        y : torch.Tensor
            Output signal to be multiplied with the matrix with shape (N, *trj_size)
            
        Returns
        -------
        x : torch.Tensor
            Input signal with shape (N, *im_size)
        """
        # Consts
        N = y.shape[0]
        
        # Perform the CUR decomposition
        coeffs = einsum(y, self.C.conj(), 'N ..., K ... -> N K')
        x = einsum(coeffs, self.R.conj(), 'N K, K ... -> N ...')
        
        return x
    
    def normal(self,
               x: torch.Tensor) -> torch.Tensor:
        """
        Normal pass of the matrix-vector operation.
        
        Args
        ----
        x : torch.Tensor
            Input signal to be multiplied with the matrix with shape (N, *im_size)
            
        Returns
        -------
        y : torch.Tensor
            Output signal with shape (N, *im_size)
        """
        # return self.adjoint(self.forward(x))
        coeffs = einsum(x, self.R, 'N ..., K ... -> N K')
        coeffs = einsum(coeffs, self.normal_factor, 'N K, Ko K -> N Ko')
        return einsum(coeffs, self.R.conj(), 'N K, K ... -> N ...')

class matvec_histogram(matvec):
    
    def __init__(self, 
                 phis: torch.Tensor, 
                 alphas: torch.Tensor,
                 dphi: Optional[float] = None,
                 dalpha: Optional[float] = None,
                 Kphi: Optional[int] = None,
                 Kalpha: Optional[int] = None,
                 **kwargs):
        """
        Args
        ----
        phis : torch.Tensor
            Spatial phase maps with shape (B, *im_size)
        alphas : torch.Tensor
            Temporal phase coefficients with shape (B, *trj_size)
        dphi : Optional[float]
            phi quantization step size
        dalpha : Optional[float]
            alpha quantization step size
        Kphi : Optional[int]
            number of phi quantization bins
        Kalpha : Optional[int]
            number of alpha quantization bins
        """
        super(matvec_histogram, self).__init__(phis, alphas, **kwargs)
        
        # Quantize phis
        phis_flt = phis.reshape((self.B, -1))
        if dphi is not None or Kphi is not None:
            if dphi is not None:
                phis_quant, phis_inds = uniform_quantization(phis_flt, dphi)
            elif Kphi is not None:
                phis_quant, phis_inds = kmeans_quantization(phis_flt, Kphi)
        else:
            phis_quant = phis_flt
            phis_inds = torch.arange(phis_quant.shape[1], device=phis_quant.device)
        self.phis_inds = phis_inds
        print(f'Phi reduction factor: {phis_flt.shape[1] / phis_quant.shape[1]:.2f}')
            
        # Quantize alphas
        alphas_flt = alphas.reshape((self.B, -1))
        if dalpha is not None or Kalpha is not None:
            if dalpha is not None:
                alphas_quant, alphas_inds = uniform_quantization(alphas_flt, dalpha)
            elif Kalpha is not None:
                alphas_quant, alphas_inds = kmeans_quantization(alphas_flt, Kalpha)
        else:
            alphas_quant = alphas_flt
            alphas_inds = torch.arange(alphas_quant.shape[1], device=alphas_quant.device)
        self.alphas_inds = alphas_inds
        print(f'Alpha reduction factor: {alphas_flt.shape[1] / alphas_quant.shape[1]:.2f}')
        
        # Make standard matvec on quantized coefficients
        assert phis_quant.ndim == 2
        assert alphas_quant.ndim == 2
        self.matvec_quant = matvec_naive(phis_quant, alphas_quant, **kwargs)
        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix-vector operation.
        
        Args
        ----
        x : torch.Tensor
            Input signal to be multiplied with the matrix with shape (N, *im_size)
        
        Returns
        -------
        y : torch.Tensor
            Output signal with shape (N, *trj_size)
        """
        # Consts
        N = x.shape[0]
        Kphi = self.matvec_quant.phis.shape[1]

        # Flatten input
        x_flt = x.reshape((N, -1))
        
        # Sum over voxels at equal quantization indices
        idxs_rep = torch.repeat_interleave(self.phis_inds[None,], N, dim=0)
        g = torch.zeros((N, Kphi), device=x.device, dtype=x.dtype)
        g.scatter_add_(dim=1, index=idxs_rep, src=x_flt) # N kphi
        
        # Perform matrix-vector operation on quantized coefficients
        y_quant = self.matvec_quant.forward(g) # N kalpha
        
        # Map back to trjsize with alpha indices
        y = y_quant[:, self.alphas_inds].reshape((N, *self.trj_size))
        
        return y
    
    def adjoint(self,
                y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint pass of the matrix-vector operation.
        
        Args
        ----
        y : torch.Tensor
            Output signal to be multiplied with the matrix with shape (N, *trj_size)
        
        Returns
        -------
        x : torch.Tensor
            Input signal with shape (N, *im_size)
        """
        # Consts
        N = y.shape[0]
        Kalpha = self.matvec_quant.alphas.shape[1]
        
        # Flatten input
        y_flt = y.reshape((N, -1))
        
        # Sum over alphas at equal quantization indices
        idxs_rep = torch.repeat_interleave(self.alphas_inds[None,], N, dim=0)
        g = torch.zeros((N, Kalpha), device=y.device, dtype=y.dtype)
        g.scatter_add_(dim=1, index=idxs_rep, src=y_flt) # N kalpha
        
        # Perform matrix-vector operation on quantized coefficients
        x_quant = self.matvec_quant.adjoint(g) # N kphi
        
        # Map back to imsize with phi indices
        x = x_quant[:, self.phis_inds].reshape((N, *self.im_size))
        
        return x
    
    # TODO can technically make normal a bit faster

class matvec_rnd(matvec):
    """
    On each call, randomly sample a subset of the phis and alphas to use for the matrix-vector operation.
    """
    
    def __init__(self,
                 phis: torch.Tensor,
                 alphas: torch.Tensor,
                 rnd_frac_phis: float = 0.1,
                 rnd_frac_alphas: float = 0.1,
                 **kwargs):
        super(matvec_rnd, self).__init__(phis, alphas, **kwargs)
        self.rnd_frac_phis = rnd_frac_phis
        self.rnd_frac_alphas = rnd_frac_alphas
        
        # Flatten phis and alphas
        B = phis.shape[0]
        self.phis = self.phis.reshape((B, -1))
        self.alphas = self.alphas.reshape((B, -1))
        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the matrix-vector operation.
        
        Args
        ----
        x : torch.Tensor
            Input signal to be multiplied with the matrix with shape (N, *im_size)
        
        Returns
        -------
        y : torch.Tensor
            Output signal with shape (N, *trj_size)
        """
        # Consts    
        N = x.shape[0]
        B = self.B
        R = self.phis.shape[1]
        
        # Flatten input
        x_flt = x.reshape((N, -1))
        
        # Randomly sample voxels
        voxel_inds = torch.randperm(R)[:int(R * self.rnd_frac_phis)]
        
        # Perform the matrix-vector operation
        enc_mat = torch.exp(-2j * torch.pi * (self.phis[:, voxel_inds].T @ self.alphas)) # R' T
        y_flt = x_flt[:, voxel_inds] @ enc_mat # N T
    
        # Reshape output
        y = y_flt.reshape((N, *self.trj_size))
        
        return y

    def adjoint(self,
                y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint pass of the matrix-vector operation.
        
        Args
        ----
        y : torch.Tensor
            Output signal to be multiplied with the matrix with shape (N, *trj_size)
        """
        # Consts
        N = y.shape[0]
        T = self.T
        
        # Flatten input
        y_flt = y.reshape((N, -1))
        
        # Randomly sample alphas
        temporal_inds = torch.randperm(T)[:int(T * self.rnd_frac_alphas)]
        
        # Perform the matrix-vector operation
        enc_mat = torch.exp(2j * torch.pi * (self.alphas[:, temporal_inds].T @ self.phis)) # T' R
        x_flt = y_flt[:, temporal_inds] @ enc_mat # N R
    
        # Reshape output
        x = x_flt.reshape((N, *self.im_size))
        
        return x