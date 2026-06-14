import torch
import torch.nn as nn

from typing import Optional
from einops import rearrange, einsum

class fixed_kernel_model(nn.Module):
    
    def __init__(self,
                 kern_size: tuple,
                 im_size: tuple,
                 source_maps: Optional[torch.Tensor] = None,
                 target_maps: Optional[torch.Tensor] = None):
        """
        Parameters:
        -----------
        kern_size : tuple
            size of kernel
        im_size : tuple
            size of image
        source_maps : torch.Tensor <complex64>
            source weighting functions with shape (L, *im_size)
        target_maps : torch.Tensor <complex64>
            target weighting functions with shape (M, *im_size)
        """
        super(fixed_kernel_model, self).__init__()
        
        # Default maps
        if source_maps is None:
            source_maps = torch.ones((1, *im_size), dtype=torch.complex64)
        if target_maps is None:
            target_maps = torch.ones((1, *im_size), dtype=torch.complex64)
            
        # Save consts
        self.source_maps = nn.Parameter(data=source_maps, requires_grad=source_maps.requires_grad)
        self.target_maps = nn.Parameter(data=target_maps, requires_grad=target_maps.requires_grad)
        self.kern_size = kern_size
        self.im_size = im_size
        self.L = source_maps.shape[0]
        self.M = target_maps.shape[0]
        self.K = torch.prod(torch.tensor(kern_size)).item()
        
    def forward_kernel(self, 
                       feature_vecs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of model to get kernels only

        Parameters:
        -----------
        feature_vecs : torch.tensor <float>
            features with shape (N, f)

        Returns:
        ----------
        kern : torch.tensor <complex>
            kernel with shape (N, M, L, K)
        """
        raise NotImplementedError

    def forward(self, 
                feature_vecs: torch.Tensor, 
                source_pts: torch.Tensor,
                L_slc: Optional[slice] = slice(None)) -> torch.Tensor:
        """
        Forward pass of model

        Parameters:
        -----------
        feature_vecs : torch.tensor <float>
            features with shape (..., f)
        source_pts : torch.tensor <complex>
            source points with shape (..., L, K)

        Returns:
        ----------
        target : torch.tensor <complex>
            output target points with shape (..., M)
        """

        # Get kernel
        kern = self.forward_kernel(feature_vecs)[..., :, L_slc, :] # ... M L K
        
        # Forward
        target = (kern * source_pts.unsqueeze(dim=-3)).sum(dim=[-2,-1])

        return target
    
    def adjoint(self,
                feature_vecs: torch.Tensor,
                target_pts: torch.Tensor,
                L_slc: Optional[slice] = slice(None)) -> torch.Tensor:
        """
        Adjoint pass of kernel application.
        
        Parameters:
        -----------
        feature_vecs : torch.tensor
            features with shape (..., f)
        target_pts : torch.tensor
            target points with shape (..., M)
            
        Returns:
        ----------
        source_pts : torch.tensor
            source points with shape (..., L, K)
        """
        
        # Get kernel
        kern = self.forward_kernel(feature_vecs)[..., L_slc, :] # ... M L K
        
        # Adjoint
        source_pts = (kern.conj() * target_pts[..., None, None]).sum(dim=-3)
        
        return source_pts
        
class learnable_kernels(fixed_kernel_model):
    
    def __init__(self, 
                 num_features: int,
                 kern_size: tuple,
                 im_size: tuple,
                 source_maps: Optional[torch.Tensor] = None,
                 target_maps: Optional[torch.Tensor] = None,
                 num_layers: Optional[int] = 3, 
                 hidden_width: Optional[int] = 256,
                 latent_width: Optional[int] = 256,
                 num_fourier: Optional[int] = 256):
        """
        Parameters:
        -----------
        num_features : int
            number of features
        kern_size : int
            size of kernel
        im_size : tuple
            size of image
        num_layers : int
            number of model layers
        hidden_width : int
            width of hidden layers
        latent_width : int
            width of final hidden layer
        num_fourier : int
            number of fourier feature for positional encoding
        """
        super().__init__(kern_size, im_size, source_maps, target_maps)

        # Save some consts
        self.num_fourier = num_fourier

        # Fourier features
        if num_fourier is not None:
            num_fourier = 2 * int((num_fourier - num_features)/2) + num_features
            B = torch.randn((num_features, (num_fourier - num_features)//2), dtype=torch.float32)
            self.B = nn.Parameter(
                data=B,
                requires_grad=False)
            num_features = num_fourier

        # Construct dimensions
        dimensions = [num_features] \
                    + [hidden_width] * (num_layers-1) \
                    + [latent_width, self.K * self.L * self.M]

        # Define layers from inputs
        self.feature_layers = nn.ModuleList()
        for k in range(0, len(dimensions) - 2):
            
            # Mat mul
            lin = nn.Linear(dimensions[k], dimensions[k+1], dtype=torch.float32)
            self.feature_layers.append(lin)

            # Activation
            self.feature_layers.append(nn.ReLU())
        
        # Last layer
        self.last_layer = nn.Linear(dimensions[-2], dimensions[-1], dtype=torch.complex64)
    
    def forward_kernel(self, 
                       feature_vecs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of model to get grappa kernel only

        Parameters:
        -----------
        feature_vecs : torch.tensor <float>
            features with shape (N, f)

        Returns:
        ----------
        kern : torch.tensor <complex>
            kernel with shape (N, M, L, K)
        """

        # Fourier features
        if self.num_fourier is not None:
            fourier_feats = torch.cat((torch.cos(feature_vecs @ self.B), torch.sin(feature_vecs @ self.B)), dim=-1)
            feature_vecs = torch.cat((feature_vecs, fourier_feats), dim=-1)
        
        # Feature extraction
        for layer in self.feature_layers:
            feature_vecs = layer(feature_vecs)

        # Linear comb kernel
        kern_flat = self.last_layer(feature_vecs.type(torch.complex64))
        
        # Reshape
        kern = rearrange(kern_flat, '... (M L K) -> ... M L K',
                         K=self.K, L=self.L, M=self.M)
        
        return kern

class sparse_network(nn.Module):
    
    def __init__(self,
                 num_features: int,
                 num_outputs: int,
                 sparsity: int,
                 num_layers: Optional[int] = 3, 
                 hidden_width: Optional[int] = 256,
                 latent_width: Optional[int] = 256,
                 num_fourier: Optional[int] = 256):
        """
        Initializes a sparse network, where each output vector is a 
        linear combination of at most K terms from the second to last layer.
        
        Args
        ----
        num_features : int
            number of features
        num_outputs : int
            number of outputs
        sparsity : int
            number of terms to use for each output
        num_layers : int
            number of model layers
        hidden_width : int
            width of hidden layers
        latent_width : int
            width of final hidden layer
        """
        super().__init__()
        
        # Save some consts
        self.O = num_outputs
        self.K = sparsity
        self.f = num_features
        
        # Fourier features
        if num_fourier is not None:
            num_fourier = 2 * int((num_fourier - num_features)/2) + num_features
            B = torch.randn((num_features, (num_fourier - num_features)//2), dtype=torch.float32)
            self.B = nn.Parameter(
                data=B,
                requires_grad=False)
            num_features = num_fourier
        
        # Construct dimensions
        dimensions = [num_features] \
                    + [hidden_width] * (num_layers-1) \
                    + [latent_width]
                    
        # Define layers
        self.feature_layers = nn.ModuleList()
        for k in range(0, num_layers):
            self.feature_layers.append(nn.Linear(dimensions[k], dimensions[k+1], dtype=torch.float32))
            self.feature_layers.append(nn.ReLU())
            
        # Last layer
        self.last_layer = nn.Linear(latent_width, num_outputs, dtype=torch.complex64)
        
    def forward(self,
                feature_vecs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of model

        Parameters:
        -----------
        feature_vecs : torch.tensor <float>
            features with shape (N, f)

        Returns:
        ----------
        outputs : torch.tensor
            outputs with shape (N, num_outputs)
        """
        # Get sparse weights
        weights, topk_inds = self.forward_sparse(feature_vecs)
        
        # Embed sparse weights
        c = torch.zeros_like(feature_vecs)
        c.scatter_(-1, topk_inds, weights) 
        
        # Apply to output layer 
        outputs = self.last_layer(c)
        
        return outputs
    
    def forward_sparse(self,
                       feature_vecs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of model

        Parameters:
        -----------
        feature_vecs : torch.tensor <float>
            features with shape (N, f)

        Returns:
        ----------
        weights : torch.tensor <float>
            weights with shape (N, K)
        topk_inds : torch.tensor <long>
            indices of the top K terms with shape (N, K) with values in [0, self.latent_width-1]
        """
        # Fourier features
        if self.num_fourier is not None:
            fourier_feats = torch.cat((torch.cos(feature_vecs @ self.B), torch.sin(feature_vecs @ self.B)), dim=-1)
            feature_vecs = torch.cat((feature_vecs, fourier_feats), dim=-1)
            
        # Feature extraction
        for layer in self.feature_layers:
            feature_vecs = layer(feature_vecs)

        # Force K with convex combination
        topk_vals, topk_inds = torch.topk(feature_vecs, dim=-1, k=self.K)
        weights = torch.softmax(topk_vals, dim=-1)         
        
        return weights, topk_inds
        
        