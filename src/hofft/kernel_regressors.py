import torch
import torch.nn as nn

from einops import einsum
from mr_recon.utils import pick_K_vectors

class kernel_regressor(nn.Module):
    """
    Kernel regressors relate some vector value function as:
    v(x) = sum_n b_n K(x, x_n)
    to some features x.
    
    Different instances of this class will define different kernel functions K(x, x_n)
    """
    def __init__(self,
                 features: torch.Tensor,
                 N_bases: int = 1000,
                 kern_params: dict[str, torch.nn.Parameter] = {}):
        """
        Args
        ----
        features : torch.Tensor
            features with shape (..., f)
        N_bases : int
            number of kernel bases
        kern_params : dict[str, torch.nn.Parameter]
            parameters for the kernel function
        """
        super(kernel_regressor, self).__init__()
        
        # Default kernel centers
        if 'kernel_centers' not in kern_params:
            f = features.shape[-1]
            kernel_centers = pick_K_vectors(features.reshape(-1, f), N_bases, method='maxmin')[0]
            kern_params['kernel_centers'] = torch.nn.Parameter(data=kernel_centers, requires_grad=False)
            
        # Save consts
        self.features = features
        self.kern_params = kern_params
        self.N_bases = N_bases
        
    def get_kern_weights(self,
                         features: torch.Tensor,
                         **kwargs) -> torch.Tensor:
        """
        Get the kernel weights for the given features
        
        Args
        ----
        features : torch.Tensor
            features with shape (..., f)
        **kwargs : dict
            keyword arguments for the kernel function
        """
        raise NotImplementedError
        
    def get_sparse_kern_weights(self,
                                features: torch.Tensor,
                                sparsity: int = 20) -> torch.Tensor:
        """
        Sparsify the kernel weights
        
        Args
        ----
        features : torch.Tensor
            features with shape (..., f)
        sparsity : int
            number of kernel weights to keep
            
        Returns
        -------
        sparse_kern_weights : torch.Tensor
            sparse kernel weights with shape (..., sparsity)
        idxs : torch.Tensor
            query indices of the kernel bases with shape (..., sparsity) in [0, N_bases)
        """
        # Get kernel weights
        kern_weights = self.get_kern_weights(features)
        
        # Sparsify by picking the top K weights
        sparse_kern_weights, idxs = torch.topk(kern_weights, sparsity, dim=-1)
        
        return sparse_kern_weights, idxs
    
    @staticmethod
    def fast_kernel_query(basis_vectors: torch.Tensor,
                          sparse_kern_weights: torch.Tensor,
                          idxs: torch.Tensor,) -> torch.Tensor:
        """
        Fast kernel query
        
        Args
        ----
        basis_vectors : torch.Tensor
            Basis vectors with shape (N_bases, *vec_size)
        sparse_kern_weights : torch.Tensor
            sparse kernel weights with shape (..., sparsity)
        idxs : torch.Tensor
            indices of the sparse kernel weights with shape (..., sparsity) in [0, N_bases)
            
        Returns
        -------
        vecs : torch.Tensor
            vectors with shape (..., *vec_size)
        """
        # Consts
        N_bases = basis_vectors.shape[0]
        arb_shape = idxs.shape[:-1]
        vec_size = basis_vectors.shape[1:]
        
        # Apply sparse weights to kernel bases
        kern_bases_flt = basis_vectors.reshape((N_bases, -1))
        sparse_kern_bases = kern_bases_flt[idxs] # ... sparsity vec_size
        kern = einsum(sparse_kern_bases, sparse_kern_weights, '... S K, ... S -> ... K') # ... vec_size
        return kern.reshape(*arb_shape, *vec_size)

class gaussian_regressor(kernel_regressor):
    """
    Related some vector value function as:
    v(x) = sum_n b_n K(x, x_n)
    
    We use a Gaussian kernel function with a cholesky factorization of the covariance matrix:
    K(x, xn) = exp(-(x - xn)^T Sigma^-1 (x - xn) / 2)
    where Sigma = L L^T is the covariance matrix and L is the cholesky factor.
    """
    def __init__(self, 
                 features: torch.Tensor,
                 N_bases: int = 1000, 
                 kern_params: dict[str, torch.nn.Parameter] = {}):
        """
        Args
        ----
        features : torch.Tensor
            features with shape (..., f)
        N_bases : int
            number of kernel bases
        kern_params : dict[str, torch.nn.Parameter]
            parameters for the kernel function with keys:
            - 'kernel_centers' for kernel centers
            - 'Ls' cholesky factors of anisotropic Gaussian kernel shape (N_bases, f, f)
            - 'gaussian_power' power of the gaussian function
        """
        
        # Constants
        f = features.shape[-1]
        torch_dev = features.device
        
        # Default covariances
        if 'Ls' not in kern_params:
            I = torch.eye(f, dtype=torch.float32, device=torch_dev)
            Ls = torch.repeat_interleave(I[None, :, :], N_bases, dim=0)
            kern_params['Ls'] = torch.nn.Parameter(data=Ls, requires_grad=False)
        
        # Default gaussian power
        if 'gaussian_power' not in kern_params:
            kern_params['gaussian_power'] = 2.0
        
        super(gaussian_regressor, self).__init__(features, N_bases, kern_params)
        
    def get_kern_weights(self,
                         features: torch.Tensor) -> torch.Tensor:
        """
        Get the kernel weights for the given features
        
        Args
        ----
        features : torch.Tensor
            features with shape (..., f)
            
        Returns
        -------
        kern_weights : torch.Tensor
            kernel weights with shape (..., N_bases)
        """
        # Consts
        x = features[..., None, :]
        xn = self.kern_params['kernel_centers']
        Ls = self.kern_params['Ls']
        gaussian_power = self.kern_params['gaussian_power']
        
        # Lxs = einsum(x - xn, Ls, '... N f, N fo f -> ... N fo')
        Lxs = ((x - xn)[..., None, :] * Ls).sum(dim=-1)
        kern_weights = torch.exp(-(Lxs.norm(dim=-1) ** gaussian_power) / 2)
        return kern_weights

class wendland_regressor(kernel_regressor):
    """
    Related some vector value function as:
    v(x) = sum_n b_n K(x, x_n)
    
    We use a Wendland kernel function, which looks something like this:
    d = ||L(x - xn)||_2, L is a matrix for anisotropy/distance scaling
    K(x, xn) = phi(d)
    where possible phi functions are:   
    - phi(d) = (1 - d)^2_+ (C0)
    - phi(d) = (1 - d)^4_+ * (4d + 1) (C2)
    - phi(d) = (1 - d)^6_+ * (35d^2 + 18d + 3) (C4)
    """
    def __init__(self,
                 features: torch.Tensor,
                 N_bases: int = 1000,
                 kern_params: dict[str, torch.nn.Parameter] = {}):
        """
        Args
        ----
        features : torch.Tensor
            features with shape (..., f)
        N_bases : int
            number of kernel bases
        kern_params : dict[str, torch.nn.Parameter]
            parameters for the kernel function with keys:
            - 'kernel_centers' for kernel centers
            - 'Ls' matrix factors of anisotropic Gaussian kernel shape (N_bases, f, f)
            - 'C' integer in {0, 2, 4} for selecting the Wendland kernel function   
        """
         # Constants
        f = features.shape[-1]
        torch_dev = features.device
        
        # Default distance scaling matrix
        if 'Ls' not in kern_params:
            I = torch.eye(f, dtype=torch.float32, device=torch_dev)
            Ls = torch.repeat_interleave(I[None, :, :], N_bases, dim=0)
            kern_params['Ls'] = torch.nn.Parameter(data=Ls, requires_grad=False)
        
        # Default C value
        if 'C' not in kern_params:
            kern_params['C'] = 0
        
        # define phi functions
        if kern_params['C'] == 0:
            self.phi = lambda d: (1 - d).clamp(min=0) ** 2
        elif kern_params['C'] == 2:
            self.phi = lambda d: (1 - d).clamp(min=0) ** 4 * (4 * d + 1)
        elif kern_params['C'] == 4:
            self.phi = lambda d: (1 - d).clamp(min=0) ** 6 * (35 * d**2 + 18 * d + 3)
        else:
            raise ValueError(f"Invalid C value: {kern_params['C']}")
        
        super(wendland_regressor, self).__init__(features, N_bases, kern_params)
        
    def get_kern_weights(self,
                         features: torch.Tensor) -> torch.Tensor:
        """
        Get the kernel weights for the given features
        
        Args
        ----
        features : torch.Tensor
            features with shape (..., f)
        """
        # Consts
        x = features[..., None, :]
        xn = self.kern_params['kernel_centers']
        Ls = self.kern_params['Ls']
        
        # Lxs = einsum(x - xn, Ls, '... N f, N fo f -> ... N fo')
        Lxs = ((x - xn)[..., None, :] * Ls).sum(dim=-1)
        ds = Lxs.norm(dim=-1)
        kern_weights = self.phi(ds)
        return kern_weights