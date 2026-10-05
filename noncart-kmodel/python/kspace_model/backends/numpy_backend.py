import numpy as np
from functools import reduce
from ..util import ft2, ift2, get_forward_matrix
from ..kspace_model import GenericBackend, register_backend

"""
This file implements the computational backend using NumPy. It provides implementations 
of the mathematical operators required by the k-space model.

NOTES ON EXTENSIBILITY:
  1. Other backend: Please refer to the GenericBackend class in kspace_model.py for 
     instructions on the abstract methods that must be implemented.

V1.0 Chin-Cheng Chan and Justin P. Haldar 05/19/2026

This software is Copyright ©2026 The University of Southern
California. All Rights Reserved. See the accompanying
license.txt for additional license information.
"""

@register_backend("numpy")
class NumpyBackend(GenericBackend):
    def __init__(self, k, N, L, rho, grid, igrid, Psi, psi_img, **kwargs):
        """
        Initialize the NumPy backend.

        Parameters:
        -----------
        k : Tuple or list of vectors. Represents trajectory coordinates.
        N : Tuple of int. Represents image sizes in the nominal field-of-view (FOV).
        L : Tuple of int. Dimensions of the coefficient grid.
        rho : Scalar or list. Represents the oversampling factor for each dimension.        
        grid : list of numpy.ndarray. Coordinates of the coefficient grid.
        igrid : list of numpy.ndarray. Coordinates of the reciprocal grid in the image domain.
        Psi : list of functions. Evaluating the basis function for each dimension.
        psi_img : list of functions. Evaluating the inverse Fourier transform of the basis.
        """
        self.N = N
        self.L = L
        
        # Cast inputs to native array types
        self.k = [np.array(ki) if ki is not None else None for ki in k]
        self.grid = [np.array(g) if g is not None else None for g in grid]
        self.igrid = [np.array(ig) if ig is not None else None for ig in igrid]
        
        # Precompute the sparse interpolation matrices
        Hmat, Hmath = get_forward_matrix(Psi, self.grid, self.k, rho)
        self.Hmat = Hmat.T
        self.Hmath = Hmath.T
        
        # Precompute apodization operators 
        psi_1d = [psi_img[ii](self.igrid[ii] / rho[ii]) for ii in range(2)]
        psi = reduce(np.multiply.outer, psi_1d)
        
        term1 = ft2(psi)
        self.Y_vec = ift2(term1).flatten()
        self.norm_factor = np.prod(self.L)

        # Precalculate crop/pad boundaries
        self.start = [int(l // 2 - n // 2) for l, n in zip(self.L, self.N)]
        self.end = [int(l // 2 + n // 2) for l, n in zip(self.L, self.N)]

    def H(self, c):
        c_reshaped = c.reshape([-1, self.norm_factor])
        out = c_reshaped @ self.Hmat
        return out.reshape(-1)

    def Hh(self, d):
        d_reshaped = d.reshape([-1, self.Hmath.shape[0]])
        out = d_reshaped @ self.Hmath
        return out.reshape(-1)

    def F(self, g):
        g_reshaped = g.reshape([-1] + list(self.L))
        out = ft2(g_reshaped)
        return out.reshape(-1)

    def Fh(self, c):
        c_reshaped = c.reshape([-1] + list(self.L))
        out = ift2(c_reshaped) * self.norm_factor
        return out.reshape(-1)

    def T_ext(self, c):
        Fh_c = self.Fh(c).reshape([-1, self.norm_factor])
        out = Fh_c * self.Y_vec
        return out.reshape(-1) / self.norm_factor

    def T_exth(self, y):
        y_reshaped = y.reshape([-1, self.norm_factor])
        Yh_y = y_reshaped * np.conj(self.Y_vec)
        out = self.F(Yh_y)
        return out.reshape(-1) / self.norm_factor

    def crop_to_nominal(self, x_ext):
        x_ext_reshaped = x_ext.reshape([-1] + list(self.L))
        # Extract the central nominal FOV
        out = x_ext_reshaped[:, self.start[0]:self.end[0], self.start[1]:self.end[1]]
        return out.reshape(-1)

    def pad_to_extended(self, x_nom):
        x_nom_reshaped = x_nom.reshape([-1] + list(self.N))
        batch_size = x_nom_reshaped.shape[0]
        # Allocate empty extended grid and inject nominal values into the center
        out = np.zeros([batch_size] + list(self.L), dtype=x_nom_reshaped.dtype)
        out[:, self.start[0]:self.end[0], self.start[1]:self.end[1]] = x_nom_reshaped
        return out.reshape(-1)
