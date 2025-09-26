import torch
import torch.nn as nn
import numpy as np
import gc

from tqdm import tqdm
from einops import rearrange, einsum
from typing import Optional, Mapping
from dataclasses import dataclass

from mr_recon.utils import gen_grd
from hofft.kernel_models import learnable_kernels

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

def train_net_apod(phis: torch.Tensor, 
                   alphas: torch.Tensor, 
                   apods_init: torch.Tensor,
                   kern_size: tuple, 
                   os: float, 
                   opt_apods: Optional[bool] = True,
                   epochs: Optional[int] = 100) -> tuple[torch.Tensor, torch.Tensor, nn.Module]:
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
    kern_size : tuple
        The kernel size, len(im_size) = len(kern_size).
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
    d = len(im_size)
    
    # Training parameters for the model
    tparams = training_params(epochs=epochs*100, 
                              batch_size=2**14, 
                              show_loss=False, 
                              loss=lambda x, y: (x-y).abs().square().sum(), 
                              lr=1e-3)
    
    # Make kernel bases vectors
    kern_vecs = gen_grd(kern_size, kern_size)
    kern_vecs = kern_vecs.to(torch_dev).reshape((-1, d)) / os
    
    # Make dataset and kernel model
    if opt_apods:
        source_maps = apods_init.clone().type(torch.complex64).requires_grad_(True)
    else:
        source_maps = apods_init.clone().type(torch.complex64).requires_grad_(False)
    target_maps = torch.ones((1, *im_size), device=torch_dev, dtype=torch.complex64)
    kern_model = learnable_kernels(B, kern_size, im_size, source_maps, target_maps).to(torch_dev)
    alphas_train = alphas.reshape((B, -1))
    dataset = phase_dataset(kern_vecs, kern_model.source_maps, kern_model.target_maps, 
                            alphas_train=alphas_train, 
                            phis_train=phis)
    
    # Train the model
    kern_model = stochastic_train_fixed(kern_model, dataset, tparams, verbose=True)
    
    # Query model for weights
    weights = kern_model.forward_kernel(alphas_train.T) # T 1 L K
    assert weights.shape[1] == 1
    weights = weights[:, 0].moveaxis(0, -1) # L K T
    weights = weights.reshape((L, *kern_size, *trj_size))
    
    # Get apodization functions
    apods = kern_model.source_maps.detach()
    
    return weights, apods, kern_model
 
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