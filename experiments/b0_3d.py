import gc
import torch
import numpy as np

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from hofft.reduce import reduce_params, reduce_spatial, expand_spatial
from hofft.forward_model import hofft_compressed_linop, hofft_linop
from hofft.sparse_decomp import sparse_params
from hofft.decomp import hofft_params
from hofft.matvec import matvec_naive, matvec_type3
from hofft.pipelines import (
  als_hofft_compressed, 
  als_hofft, 
  sparse_fit_hofft_omp,
  sparse_fit_hofft_known_coeffs,
)
from hofft.phase_coeffs import (
  trj_dev_to_phis_alphas,
  compress_phis_alphas,
  rescale_phis_alphas,
  apply_phase_midpoints,
)
from mr_recon.imperfections.field import b0_to_phis_alphas, alpha_segementation
from mr_recon.linops import sense_linop, batching_params
from mr_recon.recons import CG_SENSE_recon
from mr_recon.fourier import sigpy_nufft
from mr_recon.utils import normalize, cvplot, clear_gpu_memory

from scipy.ndimage import gaussian_filter

# Params
R = 3
B_compressed = None
torch_dev = torch.device(1)
hparams = hofft_params(kern_size=(5,)*3,
                       os=1.25,
                       L=1,
                       reduced_im_size=(120,)*3,
                       spatial_init='seg',
                       solver='pinv',
                       lamda=1e-3*0,
                       verbose=True)
sparams = sparse_params(Q=300, S=16,
                        spatial_subsample=2**15,
                        temporal_batch_size=2**15, 
                        cur_rank_phi=256,
                        cur_rank_alpha=128,
                        )
bparams = batching_params(coil_batch_size=1)

# Load 3D MRF data
fdir = '/local_mount/space/tiger/1/users/abrahamd/mr_data/mrf_b0/data'
trj = torch.from_numpy(np.load(f'{fdir}/trj.npy')).to(torch_dev)
dcf = torch.from_numpy(np.load(f'{fdir}/dcf.npy')).to(torch_dev)
ksp = torch.from_numpy(np.load(f'{fdir}/ksp.npy')).to(torch_dev)
mps = torch.from_numpy(np.load(f'{fdir}/mps.npy')).to(torch_dev)
im_size = mps.shape[1:]
C = mps.shape[0]
dcf /= dcf.max()
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0] # Round to nearest even integer

# Filter b0 map
b0 = np.load(f'{fdir}/b0.npy')
# b0 = gaussian_filter(b0, sigma=3)
b0 = torch.from_numpy(b0).to(torch_dev)

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

# ----------------- Process phase coefficients -----------------
# Stack phase coefficients, compress, rescale
phis_b0, alphas_b0 = b0_to_phis_alphas(b0, trj_size, 0, 2e-6, repeat_empty_dims=True)
phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, hparams.os)
phis_stack = torch.cat([phis_b0, phis_dev], dim=0)
alphas_stack = torch.cat([alphas_b0, alphas_dev], dim=0)
if B_compressed is not None:
    phis_stack, alphas_stack = compress_phis_alphas(phis_stack, alphas_stack, 
                                                    B_compressed=B_compressed)
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack)

# ----------------- Compressed HOFFT Decomposition -----------------
# ALS decomp and apply phase midpoints
# rets = als_hofft_compressed(phis_nrm, alphas_nrm, hparams=hparams, sparams=sparams,
#                             num_als_iter=10)
rets = sparse_fit_hofft_omp(phis_nrm, alphas_nrm, 
                            hparams=hparams, sparams=sparams,
                            num_als_iter=100)
spatial_factors, compressed_kernels, sparse_inds, sparse_coeffs = rets
spatial_factors, sparse_coeffs = apply_phase_midpoints(phis_nrm, alphas_nrm,
                                                       phis_mp, alphas_mp,
                                                       spatial_factors, sparse_coeffs)
clear_gpu_memory(torch_dev)

# Build compressed HOFFT linear operator
trj_grd = (trj * hparams.os).round() / hparams.os
A_comp = hofft_compressed_linop(trj_grd, mps, 
                                dcf=dcf,
                                compressed_kernels=compressed_kernels,
                                sparse_idxs=sparse_inds, sparse_coeffs=sparse_coeffs,
                                spatial_factors=spatial_factors, os_grid=hparams.os,
                                bparams=bparams)
img_hofft = CG_SENSE_recon(A_comp, ksp, max_iter=10, max_eigen=1.0).cpu()

# ----------------- Tme Segmented B0 Correction -----------------
nufft = sigpy_nufft(im_size, oversamp=hparams.os, width=hparams.kern_size[0])
nufft.beta = nufft.optimal_beta(torch_dev=torch_dev)
phis_reduced = reduce_spatial(phis_b0, im_size_low=hparams.reduced_im_size, order=3)
b, h, _ = alpha_segementation(phis_reduced, alphas_b0, L=hparams.L, interp_type='lstsq')
b = expand_spatial(b, im_size_high=im_size, order=3)
A = sense_linop(trj, mps, dcf, 
                spatial_funcs=b, 
                temporal_funcs=h,
                nufft=nufft,
                bparams=bparams)
img_b0 = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0).cpu()

# ----------------- Naive Recon -----------------
nufft = sigpy_nufft(im_size, oversamp=hparams.os, width=hparams.kern_size[0])
nufft.beta = nufft.optimal_beta(torch_dev=torch_dev)
A = sense_linop(trj, mps, dcf, 
                nufft=nufft,
                bparams=bparams)
img_naive = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0).cpu()

plt.figure(figsize=(14, 7))
imgs = [img_naive, img_b0, img_hofft]
titles = ['Naive', 'B0 Corrected', 'HOFFT Corrected']
for i in range(len(imgs)):
    plt.subplot(1, len(imgs), i+1)
    plt.imshow(imgs[i][..., 70].abs().rot90(), cmap='gray')
    plt.axis('off')
    plt.title(titles[i])
plt.tight_layout()
plt.show()
