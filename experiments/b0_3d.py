import torch
import numpy as np

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from hofft.decomp import hofft_params
from hofft.pipelines import als_hofft
from hofft.reduce import reduce_params
from hofft.phase_coeffs import rescale_phis_alphas, apply_phase_midpoints
from hofft.forward_model import hofft_linop, hofft_compressed_linop


from mr_recon.imperfections.field import b0_to_phis_alphas, alpha_segementation
from mr_recon.linops import sense_linop, batching_params
from mr_recon.recons import CG_SENSE_recon
from mr_recon.fourier import sigpy_nufft
from mr_recon.utils import normalize, cvplot

from scipy.ndimage import gaussian_filter

# Load 3D MRF data
torch_dev = torch.device(2)
fdir = '/local_mount/space/tiger/1/users/abrahamd/mr_data/mrf_b0/data'
trj = torch.from_numpy(np.load(f'{fdir}/trj.npy')).to(torch_dev)
dcf = torch.from_numpy(np.load(f'{fdir}/dcf.npy')).to(torch_dev)
ksp = torch.from_numpy(np.load(f'{fdir}/ksp.npy')).to(torch_dev)
mps = torch.from_numpy(np.load(f'{fdir}/mps.npy')).to(torch_dev)
im_size = mps.shape[1:]
C = mps.shape[0]
dcf /= dcf.max()

# Filter b0 map
b0 = np.load(f'{fdir}/b0.npy')
# b0 = gaussian_filter(b0, sigma=3)
b0 = torch.from_numpy(b0).to(torch_dev)

# Params
R = 3
bparams = batching_params(coil_batch_size=1,
                          field_batch_size=1)
hparams = hofft_params(kern_size=(3,)*3,
                       os=1.25,
                       L=8,)
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0] # Round to nearest even

# Undersample groups
# grps = slice(None, None, R)
grps = torch.arange(0, trj.shape[1]//R, device=torch_dev)
trj = trj[:, grps]
dcf = dcf[:, grps]
ksp = ksp[:, :, grps]
trj_size = dcf.shape

# Print shapes
print(f'Trj shape: {trj.shape}')
print(f'Dcf shape: {dcf.shape}')
print(f'Ksp shape: {ksp.shape}')
print(f'Mps shape: {mps.shape}')

# HOFT Recon
trj_grd = (trj * hparams.os).round() / hparams.os
spatial_factors = torch.randn((hparams.L, *im_size), device=torch_dev, dtype=torch.complex64)
# kern_weights = torch.randn((hparams.L, *hparams.kern_size, *trj_size), device=torch_dev, dtype=torch.complex64)
# A = hofft_linop(trj_grd, mps, kern_weights, spatial_factors, dcf, os_grid=hparams.os, bparams=bparams)
S = 8
Q = 1000
compressed_kernels = torch.randn((hparams.L, *hparams.kern_size, Q), device=torch_dev, dtype=torch.complex64)
sparse_idxs = torch.randint(0, Q, (S, *trj_size), device=torch_dev, dtype=torch.long)
sparse_coeffs = torch.randn((S, *trj_size), device=torch_dev, dtype=torch.complex64)
A = hofft_compressed_linop(trj_grd, mps, dcf=dcf, 
                           compressed_kernels=compressed_kernels, 
                           spatial_factors=spatial_factors,
                           sparse_idxs=sparse_idxs, 
                           sparse_coeffs=sparse_coeffs, 
                           os_grid=hparams.os, bparams=bparams)
img_hofft = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0).cpu()

# B0 correction
nufft = sigpy_nufft(im_size, oversamp=hparams.os, width=hparams.kern_size[0])
nufft.beta = nufft.optimal_beta(torch_dev=torch_dev)
phis, alphas = b0_to_phis_alphas(b0, trj_size, 0, 2e-6)
b, h, _ = alpha_segementation(phis, alphas, L=hparams.L, interp_type='lstsq')
A = sense_linop(trj, mps, dcf, 
                spatial_funcs=b, 
                temporal_funcs=h,
                nufft=nufft,
                bparams=bparams)
img_b0 = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0).cpu()

# Naive recon
nufft = sigpy_nufft(im_size, oversamp=hparams.os, width=hparams.kern_size[0])
nufft.beta = nufft.optimal_beta(torch_dev=torch_dev)
A = sense_linop(trj, mps, dcf, 
                nufft=nufft,
                bparams=bparams)
img_naive = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0).cpu()

plt.figure(figsize=(14, 7))
imgs = [img_naive, img_b0, img_hofft]
titles = ['Naive', 'B0 Corrected', 'HOFT']
for i in range(len(imgs)):
    plt.subplot(1, len(imgs), i+1)
    plt.imshow(imgs[i][..., 70].abs().rot90(), cmap='gray')
    plt.axis('off')
    plt.title(titles[i])
plt.tight_layout()
plt.show()
