import torch
import torch.nn as nn
import numpy as np
import gc

from tqdm import tqdm
from einops import rearrange, einsum
from typing import Optional, Mapping
from dataclasses import dataclass

from mr_recon.utils import gen_grd
from .kernel_models import learnable_kernels, mlp
from .decomp import hofft_params
from .sparse_fit import sparse_params

@dataclass
class training_params:
    epochs: Optional[int] = 15
    batch_size: Optional[int] = 2 ** 14
    l2_reg: Optional[float] = 0.0
    loss: Optional[nn.Module] = nn.L1Loss()
    show_loss: Optional[bool] = False
    float_precision: Optional[str] = 'medium'
    lr: Optional[float] = 1e-3
    """
    epochs: int
        Number of training epochs
    batch_size: int
        batch size for training
    l2_reg: float
        L2 regularization on GRAPPA kernel output
    loss: torch.nn.Module
        Loss function for training
    show_loss: bool
        Whether to show loss during training
    lr: float
        Learning rate for training
    """

class phase_dataset(object):
    """
    Dataset for training fixed fourier kernels with arbitrary target phase
    and source/target weighting functions.
    """
    
    def __init__(self,
                 kern_vecs: torch.Tensor,
                 source_maps: torch.Tensor,
                 target_maps: torch.Tensor,
                 alphas_train: torch.Tensor,
                 phis_train: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters:
        -----------
        kern_vecs : torch.Tensor <float>
            Kernel position vectors with shape (K, d)
        source_maps : torch.Tensor <complex64>
            source weighting functions with shape (L, *im_size)
        target_maps : torch.Tensor <complex64>
            target weighting functions with shape (M, *im_size)
        alphas_train: torch.Tensor
            set of phase coefficients to train on with shape (B, ...)
        phis_train : torch.Tensor
            set of phase bases with shape (B, *im_size)
        mask : Optional[torch.Tensor]
            mask with shape (*im_size)
        """

        # Consts
        self.im_size = source_maps.shape[1:]
        self.device = kern_vecs.device
        self.Nvox = torch.prod(torch.tensor(self.im_size)).item()
        self.kern_vecs = kern_vecs
        assert target_maps.shape[1:] == self.im_size
        
        # Source maps
        self.source_maps = rearrange(source_maps, 'L ... -> (...) L')
        assert self.device == source_maps.device
        
        # Target maps
        self.target_maps = rearrange(target_maps, 'M ... -> (...) M')
        assert self.device == target_maps.device
        
        # Grids are for fourier kernel
        r = gen_grd(self.im_size, fovs=(1,) * len(self.im_size)).to(self.device)
        self.r = rearrange(r, '... d -> (...) d')
        
        # Temporal phase coefficients
        self.B = alphas_train.shape[0]
        self.alphas_train = rearrange(alphas_train, 'B ... -> (...) B')
        assert self.device == alphas_train.device
        
        # Spatial phase bases
        self.phis_train = rearrange(phis_train, 'B ... -> (...) B')
        assert self.device == phis_train.device
        assert self.B == self.phis_train.shape[1]
        
        # Mask
        if mask is None:
            self.valid_voxels = torch.arange(self.Nvox, device=self.device)
        else:
            self.valid_voxels = torch.argwhere(mask.flatten()).flatten()
       
    def __getitem__(self, 
                    batch_size: int) -> dict:
        """
        Creates random batch of features, source, and target points
        
        Parameters:
        -----------
        batch_size : int
            number of points to randomly sample
        
        Returns:
        --------
        data_dct: dictionary
        {
            'feature_vecs' : torch.Tensor <float> 
                Feature vectors with shape (N, f)
            'source_data' : torch.Tensor <complex64>
                source k-space data with shape (N, L, K)
            'target_data' : torch.Tensor <complex64>
                target k-space data with shape (N, M)
        }
        where
        - f is the number of features
        - N is the batch size
        - K is the number of kernel source points
        - L is the number of source image weights
        - M is the number of target image weights
        - B is the number of basis functions
        """
        # Random feature batch
        a_inds = torch.randint(0, self.alphas_train.shape[0], (batch_size,), device=self.device)
        a_batch = self.alphas_train[a_inds] # N B
        
        # Random voxel batch
        valid_inds = torch.randint(0, len(self.valid_voxels), (batch_size,), device=self.device)
        r_inds = self.valid_voxels[valid_inds]
        r_batch = self.r[r_inds] # N d

        # Apply Source
        source_batch = torch.exp(-2j * torch.pi * \
                                 einsum(r_batch, self.kern_vecs, 
                                        'N d, K d -> N K'))
        source_batch = einsum(self.source_maps[r_inds], source_batch, 
                              'N L, N K -> N L K')
        
        # Apply Target
        target_batch = torch.exp(-2j * torch.pi * \
                                 einsum(self.phis_train[r_inds], a_batch, 
                                        'N B, N B -> N'))
        target_batch = einsum(self.target_maps[r_inds], target_batch, 
                              'N M, N -> N M')

        # Feature vector is just basis coefficients
        feature_batch = a_batch

        data_dct = {
            'feature_vecs' : feature_batch,
            'source_data' : source_batch,
            'target_data' : target_batch
        }

        return data_dct

def train_sparse_net(phis: torch.Tensor, 
                     alphas: torch.Tensor, 
                     spatial_factors_init: torch.Tensor,
                     hparams: hofft_params,
                     sparams: sparse_params,
                     tparams: Optional[training_params] = training_params(),
                     spatial_mask: Optional[torch.Tensor] = None,
                     opt_spatial_factors: bool = True,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Train a neural network to learn the kernel weights, 
    and grid representation of spatial factors.

    Args:
    -----
    phis : torch.Tensor
        The spatial phase bases with shape (B, *im_size).
    alphas : torch.Tensor
        The spatial apodization bases with shape (B, *trj_size).
    spatial_factors_init : torch.Tensor
        The initial spatial factors with shape (L, *im_size).
    hparams : hofft_params
        The HOFFT parameters.
    sparams : sparse_params
        The sparse decomposition parameters.
    tparams : training_params
        The training parameters.
    spatial_mask : Optional[torch.Tensor]
        The spatial mask with shape (*im_size).
    opt_spatial_factors : bool
        If True, optimizes the spatial factors.
        
    Returns:
    --------
    spatial_factors : torch.Tensor
        The learned spatial factors with shape (L, *im_size).
    compressed_kernels : torch.Tensor
        The learned compressed kernel dictionary with shape (L, *kern_size, Q).
    bias_kern : torch.Tensor
        The learned bias kernel with shape (L, *kern_size).
    sparse_inds : torch.Tensor
        The learned sparse indices (long) with shape (S, *trj_size), values in [0, Q).
    sparse_coeffs : torch.Tensor
        The learned sparse coefficients (real softmax weights, cast to complex64)
        with shape (S, *trj_size).
    """
    # Consts
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    kern_size = hparams.kern_size
    os = hparams.os
    d = len(im_size)
    B = phis.shape[0]
    L = spatial_factors_init.shape[0]
    Q = sparams.Q
    W = np.prod(kern_size)
    S = sparams.S
    torch_dev = phis.device

    # Default spatial mask
    if spatial_mask is None:
        spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)

    # Initialize sparse kernel model
    sparse_mlp = mlp(num_features=B,
                     num_outputs=L * np.prod(kern_size),
                     sparsity=S,
                     latent_width=Q,
                     num_layers=4,
                     hidden_width=256,
                     num_fourier=256).to(torch_dev)
    
    # Make kernel bases vectors
    kern_vecs = gen_grd(kern_size, kern_size)
    kern_vecs = kern_vecs.to(torch_dev).reshape((-1, d)) / os
    
    # Make dataset and kernel model
    if opt_spatial_factors:
        source_maps = spatial_factors_init.clone().type(torch.complex64).requires_grad_(True)
    else:
        source_maps = spatial_factors_init.clone().type(torch.complex64).requires_grad_(False)
    target_maps = torch.ones((1, *im_size), device=torch_dev, dtype=torch.complex64)
    sparse_mlp.source_maps = source_maps
    sparse_mlp.target_maps = target_maps
    alphas_train = alphas.reshape((B, -1))
    dataset = phase_dataset(kern_vecs, sparse_mlp.source_maps, sparse_mlp.target_maps, 
                            alphas_train=alphas_train, 
                            phis_train=phis,
                            mask=spatial_mask)
    
    # Train the model
    sparse_mlp = hofft_sgd(sparse_mlp, dataset, hparams, tparams)
    
    # Get sparse weights and indices
    sparse_coeffs, sparse_inds = sparse_mlp.sparse_weights_idxs(alphas_train.T)
    sparse_coeffs = sparse_coeffs.T.reshape((S, *trj_size)).type(torch.complex64)
    sparse_inds = sparse_inds.T.reshape((S, *trj_size))
    
    # Compressed kernel weights and spatial factors
    bias_kern = sparse_mlp.last_layer.bias.reshape((L, W))
    bias_kern = bias_kern.reshape((L, *kern_size))
    compressed_kernels = rearrange(sparse_mlp.last_layer.weight, 
                                   '(L W) Q -> L W Q', L=L)
    compressed_kernels = compressed_kernels.reshape((L, *kern_size, Q))
    spatial_factors = sparse_mlp.source_maps.detach()

    return spatial_factors, compressed_kernels, bias_kern, sparse_inds, sparse_coeffs

def hofft_sgd(kernel_model: nn.Module,
              data_loader: Mapping[int, dict],
              hparams: hofft_params,
              tparams: training_params,) -> nn.Module:
    """
    Train a neural network to learn the kernel weights, 
    and grid representation of spatial factors.
    
    Args
    ----
    kernel_model : nn.Module
        Kernel model to train
    data_loader : Mapping[int, dict]
        Data loader for training
    hparams : hofft_params
        HOFFT parameters
    tparams : training_params
        Training parameters

    Returns
    -------
    kernel_model : nn.Module
        Trained kernel model
    """
    # Consts
    criterion = tparams.loss
    lr = tparams.lr
    epochs = tparams.epochs
    batch_size = tparams.batch_size
    verbose = hparams.verbose
    L = hparams.L
    K = np.prod(hparams.kern_size)

    # Training params
    optim = torch.optim.Adam(kernel_model.parameters(), lr=lr)

    # Set precision
    precision_old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(tparams.float_precision)

    # Train
    losses = []
    for _ in tqdm(range(epochs), 'Training Epochs', disable=not verbose):
            
        # Extract batch
        data_dct = data_loader[batch_size]
        feature_batch = data_dct['feature_vecs'] # B f
        source_batch = data_dct['source_data'] # B L K
        target_batch = data_dct['target_data'] # B M

        # Get kernel weights
        weights_batch = kernel_model(feature_batch) # B (L K M)
        
        # Apply weights to target
        weights_batch = rearrange(weights_batch, 
                                  'B (L K M) -> B L K M', L=L, K=K, M=1)
        pred_batch = einsum(weights_batch, source_batch, 
                            'B L K M, B L K -> B M')
        
        # Loss on prediction
        loss_batch = criterion(pred_batch, target_batch)

        # Update
        loss_batch.backward()
        optim.step()
        for param in kernel_model.parameters():
            param.grad = None
        losses.append(float(loss_batch) / batch_size)

    if tparams.show_loss:
        # Debug training loss
        import matplotlib.pyplot as plt
        plt.plot(torch.log10(torch.tensor(losses)))

    device = next(kernel_model.parameters()).device
    if 'cpu' not in str(device):
        kernel_model = kernel_model.to('cpu')
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()   
            
    # Eval mode
    kernel_model = kernel_model.to(device).eval()
    for param in kernel_model.parameters():
        param.detach_()

    torch.set_float32_matmul_precision(precision_old)
    return kernel_model 

def stochastic_train_fixed(kernel_model: nn.Module,
                           data_loader: Mapping[int, dict],
                           train_params: training_params,
                           keep_grad: Optional[bool] = False,
                           verbose: Optional[bool] = False,) -> nn.Module:
    """
    Stochastically trains a model to estimate target points using linear 
    combinations of source points, where the linear functions are learned
    via input features (usually coordinate vectors).
    
    Parmeters:
    ----------
    kernel_model : nn.Module
        model to train, takes in features and source points, outputs target points
    data_loader : indexable
        data loader for training
    train_params : training_params
        training parameters
    keep_grad : bool
        whether to keep gradients after training
    verbose : bool
        whether to print training progress

    Returns:
    --------
    kernel_model: nn.Module
        trained torch model
    """ 

    # Consts
    criterion = train_params.loss
    lr = train_params.lr
    epochs = train_params.epochs
    batch_size = train_params.batch_size

    # Training params
    optim = torch.optim.Adam(kernel_model.parameters(), lr=lr)

    # Set precision
    precision_old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(train_params.float_precision)

    # Train
    losses = []
    for _ in tqdm(range(epochs), 'Training Epochs', disable=not verbose):
            
        # Extract batch
        data_dct = data_loader[batch_size]
        feature_batch = data_dct['feature_vecs']
        source_batch = data_dct['source_data']
        target_batch = data_dct['target_data']

        # Forward Pass
        pred_batch = kernel_model(feature_batch, source_batch)
        
        # Loss on prediction
        loss_batch = criterion(pred_batch, target_batch)

        # Update
        loss_batch.backward()
        optim.step()
        for param in kernel_model.parameters():
            param.grad = None
        losses.append(float(loss_batch) / batch_size)

    if train_params.show_loss:
        # Debug training loss
        import matplotlib.pyplot as plt
        plt.plot(torch.log10(torch.tensor(losses)))

    device = next(kernel_model.parameters()).device
    if 'cpu' not in str(device):
        kernel_model = kernel_model.to('cpu')
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()   
    if not keep_grad:
        kernel_model = kernel_model.to(device).eval()
        for param in kernel_model.parameters():
            param.detach_()

    torch.set_float32_matmul_precision(precision_old)
    return kernel_model 
 
def gradient_descent(phis: torch.Tensor, 
                     alphas: torch.Tensor, 
                     apods_init: torch.Tensor,
                     kerns_init: torch.Tensor,
                     os: float,
                     epochs: Optional[int] = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Train a neural network to learn the kernel weights, 
    and grid representation of apodization functions.

    Args:
    -----
    phis : torch.Tensor
        The spatial phase bases with shape (B, *im_size).
    alphas : torch.Tensor
        The spatial apodization bases with shape (B, *trj_size).
    apods_init : torch.Tensor
        The initial apodization functions with shape (L, *im_size).
    kerns_init : torch.Tensor
        The initial kernel weights with shape (L, *kern_size, *trj_size).
    os : float
        The oversampling factor
    opt_apods : Optional[bool]
        If True, optimizes the apodization functions.
        If False, uses the initial apodization functions.
    epochs : Optional[int]
        The number of epochs to train the model. Default is 100.
        
    Returns:
    --------
    weights : torch.Tensor
        The learned kernel weights with shape (L, *kern_size, *trj_size).
    apods : torch.Tensor
        The learned apodization functions with shape (L, *im_size).
    kern_model : nn.Module
        The learned kernel model.
    """
    # Consts
    torch_dev = phis.device
    B = phis.shape[0]
    L = apods_init.shape[0]
    im_size = phis.shape[1:]
    trj_size = alphas.shape[1:]
    kern_size = kerns_init.shape[1:1+len(im_size)]
    d = len(im_size)
    
    # Make kernel bases vectors
    kern_vecs = gen_grd(kern_size, kern_size)
    kern_vecs = kern_vecs.to(torch_dev).reshape((-1, d)) / os
    
    raise NotImplementedError('Gradient descent not implemented yet')