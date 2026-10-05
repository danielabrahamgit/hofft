import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from kspace_model import KSpaceModel
from kspace_model.util import cg_np, get_grids, imagesc
try:
    from kspace_model.util import cg_pytorch
    import torch
except ImportError:
    pass
from kspace_model.bspline import bspline

"""
DESCRIPTION:
  This script demonstrates Demo 1: Single-Channel Reconstruction using 
  the novel k-space model described in Ref. [1]. 

       C.-C. Chan, J. P. Haldar, “A new k-space model for non-Cartesian
       Fourier imaging,” IEEE Trans. Comput. Imaging, 2026. In Press.  
  
  It solves the standard Tikhonov-regularized single-channel problem:
  c_hat = argmin_c || H c - d ||_2^2 + lambda || c ||_2^2

PARAMETERS:
  rho     (float) : k-space grid oversampling factor.
  lambda  (float) : Tikhonov regularization strength.
  backend (str)   : Computation backend to use. Options are 'numpy' or 'pytorch'.
  device  (str)   : Computation device if using PyTorch (e.g., 'cpu' or 'cuda').

  V1.0 Chin-Cheng Chan and Justin P. Haldar 05/19/2026

  This software is Copyright ©2026 The University of Southern
  California. All Rights Reserved. See the accompanying
  license.txt for additional license information.
"""

def demo1():
    # Close all open figures
    plt.close('all')
    
    # ==========================================
    # Parameters
    # ==========================================
    rho = 1.3
    lam = 1e-1
    N = (160, 160)            # Nominal image dimensions [N1, N2]
    backend = 'numpy'         # Backend to use: 'numpy' or 'pytorch'
    device = 'cpu'            # Device to use if backend is 'pytorch': 'cpu' or 'cuda'

    # ==========================================
    # Data loading
    # ==========================================    
    try:
        df = sio.loadmat('../data/demo1.mat', squeeze_me=True) # Load the demo data
        d, k = df['data_k'], (df['k1'], df['k2'])
    except FileNotFoundError:
        print("Error: Data file '../data/demo1.mat' not found.")
        return
    
    # ==========================================
    # Model Setup
    # ==========================================
    print('Constructing operators...')
    
    # Obtain functions for evaluating the 3rd-degree B-spline basis
    [Psi_1d, psi_img_1d] = bspline(3)
    Psi = [Psi_1d, Psi_1d]
    psi_img = [psi_img_1d, psi_img_1d]
    
    # Initialize the k-space model
    kmodel = KSpaceModel(k, N, Psi=Psi, psi_img=psi_img, backend=backend, device=device)
    
    # Calculate the oversampled grid size (L) 
    L1, L2 = kmodel.coeff_grid_size

    # ==========================================
    # Reconstruction
    # ==========================================
    print(f'Running conjugate gradient using {backend} backend...')
    
    # Define the normal operator
    A = lambda x: kmodel.Hh(kmodel.H(x)) + lam * x
    
    # Solve for the coefficients using PCG based on the selected backend
    if backend == 'numpy':
        rhs = kmodel.Hh(d.flatten())
        c_hat = cg_np(A, rhs, tol=1e-9, maxit=500, verbose=True)
        
    elif backend == 'pytorch':
        d_t = torch.from_numpy(d.flatten().astype(np.complex64)).to(device)
        rhs = kmodel.Hh(d_t)
        c_hat_t = cg_pytorch(A, rhs, tol=1e-9, maxit=500, verbose=True)
        c_hat = c_hat_t.cpu().numpy()

    # ==========================================
    # Image Evaluation
    # ==========================================
    print('Visualizing reconstruction...')
    
    def to_numpy(x):
        return x.cpu().numpy() if hasattr(x, 'cpu') else x

    # Re-wrap c_hat as a tensor if using PyTorch for the evaluation steps
    eval_input = torch.from_numpy(c_hat).to(device) if backend == 'pytorch' else c_hat

    # Evaluate the image on the nominal FOV using T
    img_nom = to_numpy(kmodel.T(eval_input)).reshape(N)
    
    # Evaluate the image on the extended FOV using T_ext
    img_ext = to_numpy(kmodel.T_ext(eval_input)).reshape((L1, L2))
    
    # Evaluate the Cartesian k-space signal
    # Construct the Cartesian k-space grid using util's grid generator
    kcart1_vec, kcart2_vec = get_grids((L1, L2), N, (rho, rho))[0]
    kcart1, kcart2 = np.meshgrid(kcart1_vec, kcart2_vec, indexing='ij')
    
    # Generate the forward operator Hcart mapping the model coefficients to the Cartesian k-space grid
    Hcart = KSpaceModel((kcart1.flatten(), kcart2.flatten()), N, Psi=Psi, psi_img=psi_img, rho=rho, backend=backend, device=device)
    recon_k = to_numpy(Hcart.H(eval_input)).reshape((L1, L2))

    # ==========================================
    # Visualization and Display
    # ==========================================
    
    # 1. k-Space trajectory 
    plt.figure()
    k1_shots = k[0].reshape((-1, 17), order='F')
    k2_shots = k[1].reshape((-1, 17), order='F')
    plt.plot(k1_shots, k2_shots, '-', color=[0, 0.4470, 0.7410], linewidth=0.5)
    plt.axis('square')
    plt.title('Sampling Trajectory')
    plt.xlabel('k_x')
    plt.ylabel('k_y')
    plt.show(block=False)
    
    # 2. Reconstruction in nominal FOV
    imagesc(np.abs(img_nom), vlim=[0, 8.5e-5], 
            title='Reconstruction (Nominal FOV)', 
            fname='demo1_reconstruction_nominal_FOV.png')
    
    # 3. Reconstruction in extended FOV
    imagesc(np.abs(img_ext), vlim=[0, 8.5e-5], 
            title='Reconstruction (Extended FOV)', 
            fname='demo1_reconstruction_extended_FOV.png')
    
    # 4. K-Space Visualization
    imagesc(np.abs(recon_k), vlim=[0, 3.5e-3], 
            title='Reconstruction (Cartesian k-Space)')
    
    print('Reconstruction complete. Images saved.')
    input('End of the demo. Press Enter to exit...')

if __name__ == '__main__':
    demo1()
