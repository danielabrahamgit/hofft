import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from kspace_model import KSpaceModel
from kspace_model.util import cg_np, imagesc
try:
    from kspace_model.util import cg_pytorch
    import torch
except ImportError:
    pass

"""
DESCRIPTION:
  This script demonstrates Demo 2: Multi-Channel Reconstruction using 
  the novel k-space model described in

       C.-C. Chan, J. P. Haldar, “A new k-space model for non-Cartesian
       Fourier imaging,” IEEE Trans. Comput. Imaging, 2026. In Press.  
  
  It solves the following problem:
  c_hat = argmin_c || H c - d ||_2^2 + lambda || P c - b ||_2^2

PARAMETERS:
  rho     (float) : k-space grid oversampling factor.
  lambda  (float) : Regularization strength.
  backend (str)   : Computation backend to use. Options are 'numpy' or 'pytorch'.
  device  (str)   : Computation device if using PyTorch (e.g., 'cpu' or 'cuda').

  V1.0 Chin-Cheng Chan and Justin P. Haldar 05/19/2026

  This software is Copyright ©2026 The University of Southern
  California. All Rights Reserved. See the accompanying
  license.txt for additional license information.
"""

def demo2():
    # Close all open figures
    plt.close('all')
    
    # ==========================================
    # Parameters
    # ==========================================
    rho = 1.3                           # k-space grid oversampling factor
    lam = 1e-3                          # Regularization strength
    N = (256, 256)                      # Image dimensions [N1, N2]
    backend = 'numpy'                   # Backend to use: 'numpy' or 'pytorch'
    device = 'cpu'                     # Device to use if backend is 'pytorch'

    # ==========================================
    # Data Loading 
    # ==========================================
    try:
        # demo2_py.mat is a copy of demo2.mat with `PhP` and `b` reformatted to match the python dimension convention
        df = sio.loadmat('../data/demo2_py.mat', squeeze_me=True) 
        d, k = df['data_k'].T, (df['k1'], df['k2'])
        PhP, b = df['PhP'].flatten(), df['b'].flatten()
    except FileNotFoundError:
        print("Error: Data file '../data/demo2.mat' not found.")
        return
    
    
    Q = d.shape[0]                      # Number of channels

    # ==========================================
    # Model Setup
    # ==========================================
    print('Constructing operators...')
    
    # Initialize the k-space model. We leave Psi and psi_img empty to use 
    # the defaults (3rd-degree B-splines).
    kmodel = KSpaceModel(k, N, rho=rho, backend=backend, device=device)
    
    # Calculate the oversampled grid size (L) 
    L1, L2 = kmodel.coeff_grid_size

    # ==========================================
    # Reconstruction
    # ==========================================
    print(f'Running reconstruction using {backend} backend...')

    if backend == 'numpy':
        d_flat = d.flatten()
        
        # Normal operator
        def A_np(c):
            return kmodel.Hh(kmodel.H(c)) + lam * PhP * c
        
        # RHS of the normal equation
        bigVect = kmodel.Hh(d_flat) + lam * np.sqrt(PhP) * b
        
        # Solve for the coefficients using PCG
        c_hat = cg_np(A_np, bigVect, tol=1e-6, maxit=500, verbose=True)

    elif backend == 'pytorch':
        # Cast arrays to PyTorch tensors
        d_t = torch.from_numpy(d.flatten().astype(np.complex64)).to(device)
        PhP_t = torch.from_numpy(PhP.astype(np.
                                            float32)).to(device)
        b_t = torch.from_numpy(b.astype(np.complex64)).to(device)
        
        def A_pt(c):
            return kmodel.Hh(kmodel.H(c)) + lam * PhP_t * c
        
        bigVect_t = kmodel.Hh(d_t) + lam * torch.sqrt(PhP_t) * b_t
        
        # Solve for the coefficients using PCG
        c_hat_t = cg_pytorch(A_pt, bigVect_t, tol=1e-6, maxit=500, verbose=True)
        c_hat = c_hat_t.cpu().numpy()

    # ==========================================
    # Image Evaluation
    # ==========================================
    print('Visualizing reconstruction...')

    # In case the backend requires/returns PyTorch tensors for evaluation
    eval_input = torch.from_numpy(c_hat).to(device) if backend == 'pytorch' else c_hat
    
    def to_numpy(x):
        return x.cpu().numpy() if hasattr(x, 'cpu') else x

    # Evaluate the image on the nominal FOV using T
    img_multichan = to_numpy(kmodel.T(eval_input)).reshape((Q, N[0], N[1]))
    
    # Do a root-sum-of-squares coil combination across the channel dimension (axis 0)
    img = np.sqrt(np.sum(np.abs(img_multichan)**2, axis=0))

    # ==========================================
    # Visualization and Display
    # ==========================================
    
    # 1. k-Space trajectory 
    plt.figure()
    k1_shots = k[0].reshape((-1, 9), order='F')
    k2_shots = k[1].reshape((-1, 9), order='F')
    plt.plot(k1_shots, k2_shots, '-', color=[0, 0.4470, 0.7410], linewidth=0.5)
    plt.axis('square')
    plt.title('Sampling Trajectory')
    plt.xlabel('k_x')
    plt.ylabel('k_y')
    plt.show(block=False)

    # 2. Reconstruction in nominal FOV
    imagesc(np.abs(img), vlim=[0, 3e-5], 
            title='Reconstruction (Nominal FOV)', 
            fname='demo2_reconstruction.png')

    print('Reconstruction complete. Images saved.')
    input('End of the demo. Press Enter to exit...')

if __name__ == "__main__":
    demo2()
