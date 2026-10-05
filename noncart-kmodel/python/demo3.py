import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from kspace_model import KSpaceModel
from kspace_model.util import cg_np, imagesc, zero_pad
try:
    from kspace_model.util import cg_pytorch
    import torch
except ImportError:
    pass

"""
DESCRIPTION:
  This script demonstrates Demo 3: Multi-Channel Reconstruction (SENSE)
  using the novel k-space model described in Ref. [1]. 

       C.-C. Chan, J. P. Haldar, “A new k-space model for non-Cartesian
       Fourier imaging,” IEEE Trans. Comput. Imaging, 2026. In Press.  
  
  It performs the SENSE reconstruction using the image-domain reformulation
  of the new k-space model:

  g_hat = argmin_g sum_{q=1}^{Q} || Htilde S_q g - d_q ||_2^2 + lambda ||g||_2^2

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

def demo3():
    # Close all open figures
    plt.close('all')
    
    # ==========================================
    # Parameters
    # ==========================================
    rho = 1.3                           # k-space grid oversampling factor
    lam = 1e3                           # Tikhonov regularization strength
    N = (256, 256)                      # Nominal image dimensions [N1, N2]
    backend = 'numpy'                   # Backend to use: 'numpy' or 'pytorch'
    device = 'cpu'                     # Device to use if backend is 'pytorch'

    # ==========================================
    # Data Loading 
    # ==========================================
    try:        
        df = sio.loadmat('../data/demo3.mat', squeeze_me=True) # Load the demo data
        d, k = df['data_k'].T, (df['k1'], df['k2'])
        smaps = df['smaps'].transpose([2,0,1]) # (Q, N1, N2)
    except FileNotFoundError:
        print("Error: Data file '../data/demo3.mat' not found.")
        return

    Q = d.shape[0]                         # Number of channels

    # ==========================================
    # Model Setup
    # ==========================================
    print('Constructing operators...')
    
    # Initialize the k-space model. We leave Psi and psi_img empty to use 
    # the defaults (3rd-degree B-splines). 
    kmodel = KSpaceModel(k, N, rho=rho, backend=backend, device=device)
    
    # Calculate the oversampled grid size L
    L1, L2 = kmodel.coeff_grid_size
    L = L1 * L2

    # Pad the sensitivity maps to the oversampled grid size [L1, L2]
    S_padded = zero_pad(smaps, (L1, L2))   
    S = S_padded.reshape((Q, L))           

    # ==========================================
    # Reconstruction
    # ==========================================
    print(f'Running SENSE reconstruction using {backend} backend...')

    if backend == 'numpy':
        d_flat = d.flatten()
        
        # Normal operator
        def A_np(g):
            # S * g utilizes NumPy's implicit expansion: (Q, L) * (L,) -> (Q, L)
            S_g = S * g
            Htilde_S_g = kmodel.Htilde(S_g.flatten())
            Htildeh_Htilde_S_g = kmodel.Htildeh(Htilde_S_g).reshape((Q, L))
            return np.sum(S.conj() * Htildeh_Htilde_S_g, axis=0) + lam * g
        
        # RHS of the normal equation
        Htildeh_d = kmodel.Htildeh(d_flat).reshape((Q, L))
        bigVect = np.sum(S.conj() * Htildeh_d, axis=0)
        
        # Solve for g_hat using PCG
        g_hat = cg_np(A_np, bigVect, tol=1e-6, maxit=500, verbose=True)

    elif backend == 'pytorch':
        # Cast arrays to PyTorch tensors
        d_t = torch.from_numpy(d.flatten().astype(np.complex64)).to(device)
        S_t = torch.from_numpy(S.astype(np.complex64)).to(device)
        
        def A_pt(g):
            S_g = S_t * g
            Htilde_S_g = kmodel.Htilde(S_g.flatten())
            Htildeh_Htilde_S_g = kmodel.Htildeh(Htilde_S_g).reshape((Q, L))
            return torch.sum(S_t.conj() * Htildeh_Htilde_S_g, dim=0) + lam * g
        
        Htildeh_d = kmodel.Htildeh(d_t).reshape((Q, L))
        bigVect_t = torch.sum(S_t.conj() * Htildeh_d, dim=0)
        
        # Solve for g_hat using PCG
        g_hat_t = cg_pytorch(A_pt, bigVect_t, tol=1e-6, maxit=500, verbose=True)
        g_hat = g_hat_t.cpu().numpy()

    # ==========================================
    # Image Evaluation
    # ==========================================
    print('Visualizing reconstruction...')

    eval_input = torch.from_numpy(g_hat).to(device) if backend == 'pytorch' else g_hat
    
    def to_numpy(x):
        return x.cpu().numpy() if hasattr(x, 'cpu') else x

    # Evaluate the true image f(x) on the nominal FOV by passing g_hat through Ttilde
    img = to_numpy(kmodel.Ttilde(eval_input)).reshape(N)

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
            fname='demo3_reconstruction.png')

    print('Reconstruction complete. Images saved.')
    input('End of the demo. Press Enter to exit...')

if __name__ == "__main__":
    demo3()
