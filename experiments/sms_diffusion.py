import torch
import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.fourier import sigpy_nufft
from mr_recon.recons import CG_SENSE_recon
from mr_recon.utils import gen_grd, normalize
from mr_recon.linops import batching_params, encoding_matrix, sense_linop
from mr_recon.imperfections.field import alpha_segementation, phi_alpha_svd
from mr_recon.spatial import spatial_resize_poly

from hofft.pipelines import als_hofft
from hofft.decomp import hofft_params
from hofft.forward_model import hofft_linop
from hofft.reduce import reduce_params, reduce_spatial, expand_spatial
from hofft.phase_coeffs import (
  coco_to_phis_alphas, 
  b0_to_phis_alphas, 
  rescale_phis_alphas, 
  apply_phase_midpoints,
  trj_dev_to_phis_alphas,
  compress_phis_alphas,
)

# HOFFT params
num_als_iter = 0
hparams = hofft_params(kern_size=(5,5,2),
                       os=2.0,
                       L=6*2,
                       spatial_init='100_alphas_10'
                       )
rparams = reduce_params(alpha_reduce_use_apod=True,
                        alpha_reduce_width=2,
                        spatial_reduce_size=(160,160,2),
                        alpha_reduce_grid_spacing=0.5,
                        alpha_interp_batch_size=2**10)

# Load data
d = 0
torch_dev = torch.device(5)
fpath = '/local_mount/space/mayday/data/users/zachs/festive/analysis/20260113_hofft_benchmarks/data/raw/spi_dwi_mb2_z12_D4/'
data = torch.load(f'{fpath}/data.pt', map_location=torch_dev)
ksp = data['ksp'][d]
dcf = data['dcf']
trj = data['trj']
mps = data['mps']
phase_data = torch.load(f'{fpath}/imperfection_nufft.pt', map_location=torch_dev)
alphas = phase_data['alphas'][:, d]
phis = phase_data['phis']
spatial_phase = phase_data['phis_mean']
temporal_phase = phase_data['alphas_mean'][d]
im_size = mps.shape[1:]
trj_size = trj.shape[:-1]
C = mps.shape[0]
print(im_size)

# # ------------------------- TS-NUFFT -------------------------
# nufft = sigpy_nufft(im_size, oversamp=1.25, width=3)
# nufft.beta = nufft.optimal_beta(torch_dev=torch_dev)
# bparams = batching_params(coil_batch_size=C)
# # b, h = phi_alpha_svd(phis, alphas, L=30)
# phis_rs = reduce_spatial(phis, im_size_low=rparams.spatial_reduce_size)
# b, h = alpha_segementation(phis_rs, alphas, L=30, interp_type='lstsq', use_type3=False)
# b = expand_spatial(b, im_size_high=im_size)
# b = b * spatial_phase.conj()
# h = h * temporal_phase.conj()
# A = sense_linop(trj, mps, dcf,
#                 spatial_funcs=b, temporal_funcs=h,
#                 nufft=nufft, bparams=bparams,)

# ------------------------- HOFFT -------------------------
# Stack phase coefficients and normalize
phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, hparams.os)
phis = torch.cat([phis, phis_dev], dim=0)
alphas = torch.cat([alphas, alphas_dev], dim=0)

# SVD and rescale phase coefficients
phis, alphas = compress_phis_alphas(phis, alphas, B_compressed=5)
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)

# Perform high order phase decomposition
kern_weights, spatial_factors = als_hofft(trj, phis_nrm, alphas_nrm, hparams, rparams,
                                          num_als_iter=num_als_iter)
kern_weights, spatial_factors = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp, kern_weights, spatial_factors)
kern_weights *= temporal_phase.conj()
spatial_factors *= spatial_phase

# Build linear operator
bparams = batching_params(C*0+1)
trj_grd = (hparams.os * trj).round()/hparams.os
A = hofft_linop(trj_grd, mps, kern_weights, spatial_factors, dcf, os_grid=hparams.os, bparams=bparams)
img = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0)

img_gt = torch.load(f'{fpath}recon_matrix.pt', weights_only=True)[0]

# plt.figure(figsize=(14, 7))
# plt.subplot(121)
# plt.imshow(img_gt[..., 0].abs().rot90().cpu(), cmap='gray')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(img_gt[..., 1].abs().rot90().cpu(), cmap='gray')
# plt.axis('off')
# plt.tight_layout()

# Show image
plt.figure(figsize=(14, 7))
plt.subplot(121)
plt.imshow(img[..., 0].abs().rot90().cpu(), cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(img[..., 1].abs().rot90().cpu(), cmap='gray')
plt.axis('off')
plt.tight_layout()

plt.show()