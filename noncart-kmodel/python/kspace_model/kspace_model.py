from abc import ABC, abstractmethod
from typing import Dict, Type, Any
import numpy as np
from .util import to_even, get_grids
from .bspline import bspline 

# ==============================================================================
# This class provides the main access point to the functionality provided in this software.

# V1.0 Chin-Cheng Chan and Justin P. Haldar 05/31/2026

# This software is Copyright ©2026 The University of Southern
# California. All Rights Reserved. See the accompanying
# license.txt for additional license information.
#
# ==============================================================================

class KSpaceModel():
    def __init__(self, k, N, Psi=None, psi_img=None, rho=1.3, backend='numpy', **kwargs):
        """
        Initialize the KSpaceModel.

        Parameters:
        -----------
        k : tuple or list of array-like
            k-space coordinates of the trajectory samples.
            For 2D, this is a tuple (k1, k2), where each ki is a 1D array of length M  (number of samples).
        N : tuple or list of int
            Spatial dimensions of the nominal field of view (e.g., (N1, N2) for 2D).
            Must contain positive even integers.
        Psi : function or list of functions, optional
            Functions that evaluate the basis at arbitrary locations.
             - Defaults to a 3rd-degree B-spline if not provided.
             - If a single function is provided, the same function is used for all spatial dimensions.
             - If a list is provided, the length of the list must match the number of spatial dimensions.
        psi_img : function or list of functions, optional
             Functions that evaluate the inverse Fourier transform of the basis functions at arbitrary locations.
             - Defaults to the inverse Fourier transform of a 3rd-degree B-spline if not provided.
             - If a single function is provided, the same function is used for all spatial dimensions.
        rho : float or list of float, optional
            Oversampling factor for the coefficient grid. Default is 1.3.
        backend : str, optional
            The computation backend to use. Currently supported:
            - 'numpy': CPU-based computation using NumPy.
            - 'pytorch': GPU/CPU computation using PyTorch.
            Default is 'numpy'.
        **kwargs : dict
            Additional arguments passed to the backend constructor (e.g., 'device' for PyTorch).
        """
        ndim = len(N)           # number of spatial dimensions
        if ndim != 2:
            raise ValueError("Only 2D images are supported now")
            
        for n in N:
            if n % 2 != 0 or n <= 0:
                raise ValueError("Input N must contain positive even integers.")

        # --- Handle Optional Inputs and Defaults ---
        if Psi is None or psi_img is None:
            # Default to 3rd degree B-spline if missing
            Psi_def, psi_img_def = bspline(3)
            if Psi is None:
                Psi = Psi_def
            if psi_img is None:
                psi_img = psi_img_def

        # --- Input Handling ---
        rho = [rho]*ndim if np.isscalar(rho) else rho         
        Psi = [Psi]*ndim if not isinstance(Psi, list) else Psi
        psi_img = [psi_img]*ndim if not isinstance(psi_img, list) else psi_img

        # Size of the coefficient grid. The symbol L in the documentation.
        self.coeff_grid_size = [to_even(int(round(N[idx] * rho[idx]))) for idx in range(0, ndim)] 
        
        # `grid` is a grid of coordinates of the coefficients grid
        # `igrid` is the corresponding reciprocal grid in the image domain
        grid, igrid = get_grids(self.coeff_grid_size, N, rho) 

        # Initialize backend, providing low-level operations
        self.backend = get_backend(backend, k=k, N=N, L=self.coeff_grid_size,
                                   rho=rho, grid=grid, igrid=igrid,
                                   Psi=Psi, psi_img=psi_img,
                                   **kwargs)
    # ==============================================================================
    # Standard Operators
    # All operators  support batched/multi-channel processing. Input to these functions should be flattened vectors. ==============================================================================

    def H(self, c):
        """Forward operator mapping model coefficients to non-Cartesian k-space data."""
        return self.backend.H(c)

    def Hh(self, d):
        """Adjoint operator mapping non-Cartesian k-space data to model coefficients."""
        return self.backend.Hh(d)

    def T_ext(self, c):
        """Evaluation operator mapping model coefficients to the image evaluated on the extended FOV (L1 x L2)."""
        return self.backend.T_ext(c)

    def T_exth(self, y):
        """Adjoint operator mapping the extended FOV image back to model coefficients."""
        return self.backend.T_exth(y)

    def T(self, c):
        """Evaluation operator mapping model coefficients to the image evaluated on the nominal FOV (N1 x N2)."""
        c_ext = self.T_ext(c)
        return self.backend.crop_to_nominal(c_ext)

    def Th(self, x):
        """Adjoint operator mapping the nominal FOV image back to model coefficients."""
        x_ext = self.backend.pad_to_extended(x)
        return self.T_exth(x_ext)

    # ==============================================================================
    # Operators for the Image-Domain Reformulation
    # ==============================================================================
    def Htilde(self, g):
        """Forward operator mapping inverse DFT of coefficients to k-space data for the image-domain reformulation."""

        return self.H(self.backend.F(g))

    def Htildeh(self, d):
        """Adjoint operator mapping k-space data to inverse DFT of coefficients for the image-domain reformulation."""

        return self.backend.Fh(self.Hh(d))

    def Ttilde_ext(self, g):
        """Evaluation operator for the extended FOV for the image-domain reformulation."""
        return self.T_ext(self.backend.F(g))

    def Ttilde_exth(self, y):
        """Adjoint evaluation operator for the extended FOV for the image-domain reformulation."""
        return self.backend.Fh(self.T_exth(y))

    def Ttilde(self, g):
        """Evaluation operator for the nominal FOV for the image-domain reformulation."""
        return self.T(self.backend.F(g))

    def Ttildeh(self, x):
        """Adjoint evaluation operator for the nominal FOV for the image-domain reformulation."""
        return self.backend.Fh(self.Th(x))


# ==============================================================================
# Backend Registry and Dispatcher
# ==============================================================================

class GenericBackend(ABC):
    @abstractmethod
    def H(self, c):
        """Computes the forward model."""
        pass

    @abstractmethod
    def Hh(self, d):
        """Computes the adjoint of the forward model."""
        pass

    @abstractmethod
    def T_ext(self, c):
        """Computes the image evaluation on the extended FOV."""
        pass

    @abstractmethod
    def T_exth(self, y):
        """Computes the adjoint of the image evaluation on the extended FOV."""

        pass

    @abstractmethod
    def F(self, g):
        """Computes the 2D Fourier transform."""
        pass

    @abstractmethod
    def Fh(self, c):
        """Computes the adjoint of the 2D Fourier transform."""
        pass

    @abstractmethod
    def crop_to_nominal(self, x_ext):
        """Center-crops an extended FOV image to the nominal FOV."""
        pass

    @abstractmethod
    def pad_to_extended(self, x_nom):
        """Zero-pads a nominal FOV image to the extended FOV."""
        pass

_BACKEND_REGISTRY: Dict[str, Type[GenericBackend]] = {}

def register_backend(name: str):
    """Decorator to register a new backend class."""
    def decorator(cls):
        if not issubclass(cls, GenericBackend):
            raise TypeError(f"{cls.__name__} must inherit from GenericBackend")
        _BACKEND_REGISTRY[name] = cls
        return cls
    return decorator

def get_backend(name: str, **kwargs) -> GenericBackend:
    """Factory function to instantiate a backend by name."""
    if name not in _BACKEND_REGISTRY:
        raise ValueError(f"Backend '{name}' not found. Available: {list(_BACKEND_REGISTRY.keys())}")
    return _BACKEND_REGISTRY[name](**kwargs)
