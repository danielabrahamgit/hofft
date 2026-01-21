import torch
import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.recons import CG_SENSE_recon
from mr_recon.utils import gen_grd, normalize
from mr_recon.linops import batching_params, encoding_matrix

from hofft.pipelines import als_hofft
from hofft.decomp import hofft_params
from hofft.forward_model import hofft_linop
from hofft.reduce import reduce_params
from hofft.phase_coeffs import (
  coco_to_phis_alphas, 
  b0_to_phis_alphas, 
  rescale_phis_alphas, 
  apply_phase_midpoints
)

# Params
R = 3
fov = 0.22
torch_dev = torch.device(6)
torch.manual_seed(0)
num_als_iter = 10
hparams = hofft_params(kern_size=(5,5),
                       os=1.5,
                       L=6*2,)
rparams = reduce_params(alpha_reduce_width=2,
                        spatial_reduce_size=(120,120),
                        alpha_reduce_grid_spacing=0.3,
                        alpha_reduce_use_apod=True,
                        alpha_interp_batch_size=2**10)

# Load data
fpath = '/local_mount/space/tiger/1/users/abrahamd/hofft/data/coco_spiral'
b0 = torch.load(f'{fpath}/b0.pt', map_location=torch_dev)
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
dcf = torch.load(f'{fpath}/dcf.pt', map_location=torch_dev)
ksp = torch.load(f'{fpath}/ksp.pt', map_location=torch_dev)
img_gt = torch.load(f'{fpath}/img_gt.pt', map_location=torch_dev)
im_size = b0.shape
C = mps.shape[0]

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

# Stack phase coefficients and normalize
phis = torch.cat([phis_b0, phis_coco], dim=0)
alphas = torch.cat([alphas_b0, alphas_coco], dim=0)
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)

# Perform high order phase decomposition
kern_weights, spatial_factors = als_hofft(trj, phis_nrm, alphas_nrm, hparams, rparams,
                                          num_als_iter=num_als_iter)
kern_weights, spatial_factors = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp, kern_weights, spatial_factors)

# Build linear operator
bparams = batching_params(C*0+1)
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
img_gt = torch.load(f'./coco_spiral_expanded_encoding_recon.pt', weights_only=True)

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