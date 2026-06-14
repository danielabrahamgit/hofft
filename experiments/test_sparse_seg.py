import torch

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from hofft.reduce import reduce_params, reduce_spatial, expand_spatial
from hofft.forward_model import hofft_compressed_linop, hofft_linop
from hofft.sparse_decomp import sparse_params
from hofft.decomp import hofft_params
from hofft.pipelines import als_hofft_compressed, als_hofft
from hofft.phase_coeffs import (
  trj_dev_to_phis_alphas,
  compress_phis_alphas,
  rescale_phis_alphas,
  apply_phase_midpoints,
)
from mr_recon.imperfections.field import alpha_segementation
from mr_recon.recons import CG_SENSE_recon
from mr_recon.linops import sense_linop, batching_params
from mr_recon.fourier import sigpy_nufft

# params
torch.manual_seed(0)
R = 3
B_compressed = 4
# B_compressed = None
torch_dev = torch.device(4)
hparams = hofft_params(kern_size=(5,)*2,
                       os=1.25,
                       L=5,
                       reduced_im_size=(120,120),
                       spatial_init='seg',
                       verbose=True)
sparams = sparse_params(Q=100, K=4, 
                        beta_method='kmeans', 
                        # beta_method='grid', 
                        # interp_method='lstsq', 
                        interp_method='rbf', 
                        # interp_method='grid', 
                        # grid_spacing=0.25,
                        # grid_width=2,
                        # spatial_subsample=2**13,
                        temporal_batch_size=2**10, 
                        lamda=1e-3, 
                        normalize_coeffs=True)

# Load data
fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_spiral'
ksp = torch.load(f'{fpath}/ksp.pt', map_location=torch_dev)
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
dcf = torch.load(f'{fpath}/dcf.pt', map_location=torch_dev)
alphas = torch.load(f'{fpath}/alphas.pt', map_location=torch_dev)
phis = torch.load(f'{fpath}/phis.pt', map_location=torch_dev)
im_size = mps.shape[1:]
C = mps.shape[0]
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0] # Round to nearest even integer
bparams = batching_params(coil_batch_size=C)

# Undersample
trj = trj[..., ::R, :].type(torch.float32)
dcf = dcf[..., ::R].type(torch.float32)
ksp = ksp[..., ::R].type(torch.complex64)
trj_size = trj.shape[:-1]

# ----------------- Process phase coefficients -----------------
# Remove small energy alphas or phis
B = phis.shape[0]
phi_energy = phis.reshape((B, -1)).abs().mean(dim=1)
alpha_energy = alphas.reshape((B, -1)).abs().mean(dim=1)
energy = phi_energy * alpha_energy
idxs = torch.argwhere(energy > 1e-6)[:, 0]
phis = phis[idxs]
alphas = alphas[idxs]
B = phis.shape[0]

# Stack phase coefficients, compress, rescale
phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, hparams.os)
phis_stack = torch.cat([phis, phis_dev], dim=0)
alphas_stack = torch.cat([alphas, alphas_dev], dim=0)
# if B_compressed is not None:
#     phis_stack, alphas_stack = compress_phis_alphas(phis_stack, alphas_stack, 
#                                                     B_compressed=B_compressed)
# phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack, norm_dists=True)
# spat, temp = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp)
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack, norm_dists=True)
spat, temp = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp)
if B_compressed is not None:
    phis_nrm, alphas_nrm = compress_phis_alphas(phis_nrm, alphas_nrm, 
                                                B_compressed=B_compressed)

# ----------------- Standard HOFFT Decomposition -----------------
# ALS decomp and apply phase midpoints
rets = als_hofft(phis_nrm, alphas_nrm, hparams=hparams,
                 rparams=reduce_params(spatial_reduce_size=hparams.reduced_im_size,),
                 num_als_iter=100)
spatial_factors, kernel_weights = rets
spatial_factors *= spat
kernel_weights *= temp

# Build standard HOFFT linear operator
trj_grd = (trj * hparams.os).round() / hparams.os
A_hofft = hofft_linop(trj_grd, mps, kernel_weights, spatial_factors, dcf, 
                      os_grid=hparams.os, bparams=bparams)
img_hofft = CG_SENSE_recon(A_hofft, ksp, max_iter=10, max_eigen=1.0).cpu()

# ----------------- Compressed HOFFT Decomposition -----------------
# ALS decomp and apply phase midpoints
rets = als_hofft_compressed(phis_nrm, alphas_nrm, hparams=hparams, sparams=sparams,
                            num_als_iter=100)
spatial_factors, compressed_kernels, sparse_inds, sparse_coeffs = rets
spatial_factors *= spat
sparse_coeffs *= temp

# Build compressed HOFFT linear operator
trj_grd = (trj * hparams.os).round() / hparams.os
A_comp = hofft_compressed_linop(trj_grd, mps, 
                                dcf=dcf,
                                compressed_kernels=compressed_kernels,
                                sparse_idxs=sparse_inds, sparse_coeffs=sparse_coeffs,
                                spatial_factors=spatial_factors, os_grid=hparams.os,
                                bparams=bparams)
img_comp = CG_SENSE_recon(A_comp, ksp, max_iter=10, max_eigen=1.0).cpu()

# ----------------- Spatio Temporal Splitting Decomposition -----------------
phis_reduced = reduce_spatial(phis, im_size_low=hparams.reduced_im_size, order=3)
bs, cs, _ = alpha_segementation(phis_reduced, alphas, 
                                L=hparams.L, 
                                L_batch_size=1,
                                interp_type='lstsq',
                                method='maxmin',
                                use_type3=False)
bs = expand_spatial(bs, im_size_high=im_size, order=3)
nft = sigpy_nufft(im_size, oversamp=hparams.os, width=hparams.kern_size[0])
nft.beta = nft.optimal_beta(torch_dev=torch_dev)
A_split = sense_linop(trj, mps, dcf, 
                      spatial_funcs=bs,
                      temporal_funcs=cs,
                      nufft=nft,
                      bparams=bparams)
img_split = CG_SENSE_recon(A_split, ksp, max_iter=10, max_eigen=1.0).cpu()

# ----------------- Compare ----------------
plt.figure(figsize=(14, 7))
imgs = [img_comp, img_hofft, img_split]
titles = ['Compressed HOFFT', 'Standard HOFFT', 'Spatio Temporal Splitting']
for i in range(len(imgs)):
    plt.subplot(1, len(imgs), i+1)
    plt.imshow(imgs[i].abs().rot90(), cmap='gray')
    plt.axis('off')
    plt.title(titles[i])
plt.tight_layout()
plt.show()