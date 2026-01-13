import torch 

import matplotlib as mpl
mpl.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.algs import density_compensation
from mr_recon.fourier import sigpy_nufft
from mr_recon.utils import gen_grd, np_to_torch
from mr_recon.linops import batching_params, sense_linop
from mr_recon.recons import CG_SENSE_recon

from hofft.pipelines import als_nufft, kb_nufft
from hofft.decomp import hofft_params
from hofft.forward_model import hofft_linop

# Parameters\
kern_size = (3, 3)
os = 1.2
torch_dev = torch.device(1)

# Load the data
fpath = '/local_mount/space/tiger/1/users/abrahamd/hofft/data/sim_spiral'
img = torch.load(f'{fpath}/img.pt', map_location=torch_dev)
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
dcf = torch.load(f'{fpath}/dcf.pt', map_location=torch_dev)
im_size = img.shape
C = mps.shape[0]

# Compare with ground truth
kwargs = {'bparams': batching_params(coil_batch_size=C), 'use_toeplitz': False}
nufft_gt = sigpy_nufft(im_size, oversamp=2.0, width=6)
A_gt = sense_linop(trj, mps, dcf, nufft=nufft_gt, **kwargs)
ksp_gt = A_gt(img)

# Compare with sigpy
nufft_sp = sigpy_nufft(im_size, oversamp=os, width=kern_size[0])
A_sp = sense_linop(trj, mps, dcf, nufft=nufft_sp, **kwargs)

# Compare with hofft
hparams = hofft_params(kern_size=kern_size, os=os, L=1, spatial_init='eigen')
kern_weights, spatial_factor = als_nufft(trj, im_size, hparams, num_als_iter=1000, im_size_low=(50,)*2)
trj_grd = (trj * os).round() / os
A_hofft = hofft_linop(trj_grd, mps, kern_weights, spatial_factor, dcf=dcf, os_grid=os)

# Recon all
max_iter = 30
max_eigen = None
img_gt = CG_SENSE_recon(A_gt, ksp_gt, max_iter=max_iter, max_eigen=max_eigen).cpu().rot90()
img_sp = CG_SENSE_recon(A_sp, ksp_gt, max_iter=max_iter, max_eigen=max_eigen).cpu().rot90()
img_hofft = CG_SENSE_recon(A_hofft, ksp_gt, max_iter=max_iter, max_eigen=max_eigen).cpu().rot90()

# Show results real part
plt.figure(figsize=(14, 7))
plt.subplot(131)
plt.imshow(img_gt.abs(), cmap='gray', vmin=0, vmax=0.75)
plt.axis('off')
plt.subplot(132)
plt.imshow(img_sp.abs(), cmap='gray', vmin=0, vmax=0.75)
plt.axis('off')
plt.subplot(133)
plt.imshow(img_hofft.abs(), cmap='gray', vmin=0, vmax=0.75)
plt.axis('off')
plt.tight_layout()
plt.show()