import numpy as np
from functools import reduce
import torch

from ..util import get_forward_matrix
from ..kspace_model import GenericBackend, register_backend

"""
This file implements the computational backend using PyTorch. It provides implementations 
of the mathematical operators required by the k-space model.

This backend supports GPU acceleration, which can be enabled by setting the 
`device` flag to 'cuda' during initialization.

NOTES ON EXTENSIBILITY:
  1. Other backend: Please refer to the GenericBackend class in kspace_model.py for 
     instructions on the abstract methods that must be implemented.


V1.0 Chin-Cheng Chan and Justin P. Haldar 05/19/2026

This software is Copyright ©2026 The University of Southern
California. All Rights Reserved. See the accompanying
license.txt for additional license information.
"""

@register_backend("pytorch")
class PytorchBackend(GenericBackend):
    def __init__(self, k, N, L, rho, grid, igrid, Psi, psi_img, **kwargs):
        """
        Initialize the PyTorch backend.
        """
        self.device = kwargs.get('device', 'cpu')
        self.N = N
        self.L = L
        
        # Cast inputs to numpy arrays temporarily to pass to the NumPy-only matrix generator
        k_np = [np.array(ki) if ki is not None else None for ki in k]
        grid_np = [np.array(g) if g is not None else None for g in grid]
        igrid_np = [np.array(ig) if ig is not None else None for ig in igrid]
        
        # Precompute the sparse interpolation matrices using numpy
        # get_forward_matrix returns Hmat [M, L] and Hmath [L, M]
        Hmat_np, Hmath_np = get_forward_matrix(Psi, grid_np, k_np, rho)
        
        # Convert them to PyTorch sparse CSC tensors and move to device
        self.Hmat = self._scipy_sparse_to_torch(Hmat_np).to_sparse_csc()
        self.Hmath = self._scipy_sparse_to_torch(Hmath_np).to_sparse_csc()
        
        self.norm_factor = np.prod(self.L)

        # Precompute apodization operators (Y and Yh)
        # Evaluate basis using NumPy to prevent compatibility issues
        psi_1d_np = [psi_img[ii](igrid_np[ii] / rho[ii]) for ii in range(2)]
        psi_np = reduce(np.multiply.outer, psi_1d_np)
        
        # Convert the evaluated basis to a PyTorch tensor
        psi_torch = torch.from_numpy(psi_np).to(self.device).to(torch.complex64)
        
        term1 = ft2_pt(psi_torch)
        self.Y_vec = ift2_pt(term1).flatten()

        # Precalculate crop/pad boundaries
        self.start = [int(l // 2 - n // 2) for l, n in zip(self.L, self.N)]
        self.end = [int(l // 2 + n // 2) for l, n in zip(self.L, self.N)]

    def _scipy_sparse_to_torch(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo()       
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape).to(self.device).to(torch.complex64)

    def H(self, c):
        c_reshaped = c.reshape([-1, self.norm_factor])
        out = torch.sparse.mm(self.Hmat, c_reshaped.T).T
        return out.reshape(-1)

    def Hh(self, d):
        d_reshaped = d.reshape([-1, self.Hmat.shape[0]])
        out = torch.sparse.mm(self.Hmath, d_reshaped.T).T
        return out.reshape(-1)

    def F(self, g):
        g_reshaped = g.reshape([-1] + list(self.L))
        out = ft2_pt(g_reshaped)
        return out.reshape(-1)

    def Fh(self, c):
        c_reshaped = c.reshape([-1] + list(self.L))
        out = ift2_pt(c_reshaped) * self.norm_factor
        return out.reshape(-1)

    def T_ext(self, c):
        Fh_c = self.Fh(c).reshape([-1, self.norm_factor])
        out = Fh_c * self.Y_vec
        return out.reshape(-1) / self.norm_factor

    def T_exth(self, y):
        y_reshaped = y.reshape([-1, self.norm_factor])
        Yh_y = y_reshaped * torch.conj(self.Y_vec)
        out = self.F(Yh_y)
        return out.reshape(-1) / self.norm_factor

    def crop_to_nominal(self, x_ext):
        x_ext_reshaped = x_ext.reshape([-1] + list(self.L))
        out = x_ext_reshaped[:, self.start[0]:self.end[0], self.start[1]:self.end[1]]
        return out.reshape(-1)

    def pad_to_extended(self, x_nom):
        x_nom_reshaped = x_nom.reshape([-1] + list(self.N))
        batch_size = x_nom_reshaped.shape[0]
        out = torch.zeros([batch_size] + list(self.L), dtype=x_nom_reshaped.dtype, device=x_nom_reshaped.device)
        out[:, self.start[0]:self.end[0], self.start[1]:self.end[1]] = x_nom_reshaped
        return out.reshape(-1)

# ==============================================================================
# PyTorch Helper Functions
# ==============================================================================

def ft2_pt(X, shift=True, norm=False):
    """
    2D Fourier Transform for PyTorch tensors.
    """
    norm_ = 'ortho' if norm else None
    if shift:
        x = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(X, dim=(-1, -2)),
                dim=(-1, -2),
                norm=norm_
            ),
            dim=(-1, -2)
        )
    else:
        x = torch.fft.fft2(X, dim=(-1, -2), norm=norm_)

    return x

def ift2_pt(X, shift=True, norm=False):
    """
    2D Inverse Fourier Transform for PyTorch tensors.
    """
    norm_ = 'ortho' if norm else None
    if shift:
        x = torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(X, dim=(-1, -2)),
                dim=(-1, -2),
                norm=norm_
            ),
            dim=(-1, -2)
        )
    else:
        x = torch.fft.ifft2(X, dim=(-1, -2), norm=norm_)

    return x
