import torch
import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.recons import CG_SENSE_recon
from mr_recon.utils import gen_grd, normalize
from mr_recon.linops import batching_params, encoding_matrix, sense_linop
from mr_recon.fourier import sigpy_nufft

from hofft.pipelines import als_hofft
from hofft.decomp import hofft_params
from hofft.forward_model import hofft_linop
from hofft.reduce import reduce_params
from hofft.phase_coeffs import (
  coco_to_phis_alphas, 
  b0_to_phis_alphas, 
  rescale_phis_alphas, 
  apply_phase_midpoints,
  trj_dev_to_phis_alphas,
  compress_phis_alphas,
)

# Params
R = 3
fov = 0.22
torch_dev = torch.device(4)
torch.manual_seed(0)
num_als_iter = 100*0
hparams = hofft_params(kern_size=(6,)*2,
                       os=1.25,
                       L=5,
                       spatial_init='100_alphas_10',
                      #  anderson_order=10,
                      #  spatial_init='seg',
                      #  spatial_init='ones',
                      #  spatial_init='eigen',
                       )
rparams = reduce_params(alpha_reduce_use_apod=True,
                        # alpha_reduce_width=2,
                        spatial_reduce_size=(120,120),
                        alpha_reduce_grid_spacing=0.12,
                        alpha_interp_batch_size=2**10)

# Load data
fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_spiral'
b0 = torch.load(f'{fpath}/b0.pt', map_location=torch_dev)
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
dcf = torch.load(f'{fpath}/dcf.pt', map_location=torch_dev)
ksp = torch.load(f'{fpath}/ksp.pt', map_location=torch_dev)
img_gt = torch.load(f'{fpath}/img_gt.pt', map_location=torch_dev)
im_size = b0.shape
trj_size = trj.shape[:-1]
C = mps.shape[0]
print(f'img size: {im_size}')
print(f'trj Size: {trj_size}')
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0] # makes sure we get an even integer

# Undersample data and typecast
trj = trj[..., ::R, :].type(torch.float32)
dcf = dcf[..., ::R].type(torch.float32)
ksp = ksp[..., ::R].type(torch.complex64)
trj_size = dcf.shape

# Get phase coefficients from b0
phis_b0, alphas_b0 = b0_to_phis_alphas(b0, dcf.shape, ro_dim=0, dt=2e-6, repeat_empty_dims=True)

# Get phase coefficients from coco
trj_3d = torch.stack([trj[..., 0], trj[..., 0] * 0, trj[..., 1]], dim=-1) / fov
spatial_crds = gen_grd(im_size, (fov,)*2).to(torch_dev)
spatial_crds = torch.stack([spatial_crds[..., 0], spatial_crds[..., 0] * 0, spatial_crds[..., 1]], dim=-1)
phis_coco, alphas_coco = coco_to_phis_alphas(trj_3d, spatial_crds, field_strength=3, ro_dim=0, dt=2e-6)
idxs = [0, 1] # only two of the coco terms are actually relevant for a x-z plane spiral
phis_coco = phis_coco[idxs]
alphas_coco = alphas_coco[idxs]

# Phase coefficients from trajectory deviation
phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, hparams.os)

# Stack phase coefficients and normalize
if hparams.kern_size == (1,)*2:
  phis = torch.cat([phis_b0, phis_coco], dim=0)
  alphas = torch.cat([alphas_b0, alphas_coco], dim=0)
else:  
  phis = torch.cat([phis_b0, phis_coco, phis_dev], dim=0)
  alphas = torch.cat([alphas_b0, alphas_coco, alphas_dev], dim=0)

# SVD and rescale phase coefficients
# phis, alphas = compress_phis_alphas(phis, alphas, B_compressed=5)
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)

# Perform high order phase decomposition
spatial_factors, kern_weights = als_hofft(phis_nrm, alphas_nrm, hparams, rparams,
                                          num_als_iter=num_als_iter)
spatial_factors, kern_weights = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp, spatial_factors, kern_weights)

# Build linear operator
bparams = batching_params(C*0+1)
if hparams.kern_size == (1,)*2:
  nufft = sigpy_nufft(im_size, oversamp=1.25, width=4)
  A = sense_linop(trj, mps, dcf, 
                  nufft=nufft,
                  spatial_funcs=spatial_factors,
                  temporal_funcs=kern_weights[:, 0, 0, ...],
                  bparams=bparams)
else:
  trj_grd = (hparams.os * trj).round()/hparams.os
  A = hofft_linop(trj_grd, mps, kern_weights, spatial_factors, dcf, os_grid=hparams.os, bparams=bparams)

# Recon
img_recon = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0).cpu()

# Expanded encoding model as ground truth
# phis_dev = gen_grd(im_size).to(torch_dev).moveaxis(-1, 0)
# alphas_dev = trj.moveaxis(-1, 0)
# phis = torch.cat([phis, phis_dev], dim=0)
# alphas = torch.cat([alphas, alphas_dev], dim=0)
# A = encoding_matrix(mps, phis, alphas, dcf, temporal_batch_size=2**10)
# img_gt = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0)
# torch.save(img_gt.cpu(), f'./coco_spiral_expanded_encoding_recon.pt')
img_gt = torch.load(f'./experiments/coco_spiral_expanded_encoding_recon.pt', weights_only=True)

# Normalize data
img_recon = normalize(img_recon, img_gt, mag=True, ofs=False)

# Show
vmax = img_gt.abs().max()
plt.figure(figsize=(14, 7))
nrmse = (img_recon.abs() - img_gt.abs()).norm() / img_gt.norm()
plt.suptitle(f'NRMSE = {100*nrmse:.2f}%')
plt.subplot(131)
plt.imshow(img_gt.abs().rot90(), cmap='gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(img_recon.abs().rot90(), cmap='gray')
plt.axis('off')
plt.subplot(133)
plt.imshow((img_recon.abs() - img_gt.abs()).abs().rot90(), cmap='gray', vmax=vmax/5)
plt.axis('off')
plt.tight_layout()
plt.show()