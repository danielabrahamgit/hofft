import torch
import gc

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from time import perf_counter
from tqdm import tqdm
from itertools import product

from mr_recon.fourier import sigpy_nufft, ifft
from mr_recon.recons import CG_SENSE_recon
from mr_recon.utils import gen_grd, normalize, cvplot
from mr_recon.algs import density_compensation
from mr_recon.linops import batching_params, encoding_matrix, sense_linop
from mr_recon.imperfections.field import alpha_segementation, phi_alpha_svd
from mr_recon.multi_coil.calib import synth_cal
from mr_recon.multi_coil.coil_est import csm_from_espirit
from mr_recon.spatial import spatial_resize_poly

from hofft.pipelines import als_hofft, kb_nufft, sparse_fit_hofft_omp
from hofft.decomp import hofft_params
from hofft.matvec import matvec_cur
from hofft.forward_model import hofft_linop, hofft_compressed_linop
from hofft.reduce import reduce_params, reduce_spatial, expand_spatial
from hofft.sparse_decomp import sparse_params
from hofft.phase_coeffs import (
  coco_to_phis_alphas, 
  b0_to_phis_alphas, 
  rescale_phis_alphas, 
  apply_phase_midpoints,
  trj_dev_to_phis_alphas,
  compress_phis_alphas,
)

# Params
torch.manual_seed(0)
torch_dev = torch.device('cuda:0')

# Load full data
trj = torch.load('./data/coco_7t_spi/trj.pt', weights_only=True, map_location=torch_dev)
dcf = torch.load('./data/coco_7t_spi/dcf.pt', weights_only=True, map_location=torch_dev)
ksp = torch.load('./data/coco_7t_spi/ksp.pt', weights_only=True, map_location=torch_dev)
evals = torch.load('./data/coco_7t_spi/evals.pt', weights_only=True, map_location=torch_dev)
mps = torch.load('./data/coco_7t_spi/mps.pt', weights_only=True, map_location=torch_dev)
b0 = torch.load('./data/coco_7t_spi/b0.pt', weights_only=True, map_location=torch_dev)
im_size = b0.shape
C = mps.shape[0]

# Print shapes
print(f'trj shape: {trj.shape}')
print(f'dcf shape: {dcf.shape}')
print(f'ksp shape: {ksp.shape}')
print(f'evals shape: {evals.shape}')
print(f'mps shape: {mps.shape}')
print(f'b0 shape: {b0.shape}')

# Subsample read
# ros = slice(0, 4_000)
M = 4
ros = slice(None, 10_000, M)
ksp = ksp[:, ros]
trj = trj[ros]
dcf = dcf[ros]

# Build linop
ts = torch.arange(trj.shape[0], device=torch_dev, dtype=torch.float32)[None, :, None, None] * 1e-6 * M
b0_ds = spatial_resize_poly(b0, (150,)*3, order=3)
# bs, hs, _ = alpha_segementation(b0_ds[None,], ts, L=25, L_batch_size=1, use_type3=True, interp_type='lstsq')
# bs = spatial_resize_poly(bs, im_size, order=3)
nufft = sigpy_nufft(im_size, width=3)
nufft.beta = nufft.optimal_beta(torch_dev=torch_dev)
bparams = batching_params(coil_batch_size=1, field_batch_size=3)
A = sense_linop(trj, mps, dcf, 
                # spatial_funcs=bs,
                # temporal_funcs=hs,
                bparams=bparams,
                nufft=nufft)
img_recon = CG_SENSE_recon(A, ksp, max_iter=5, max_eigen=1.0).cpu()


plt.figure(figsize=(14, 7))
zs = [im_size[2]//2 - 20, im_size[2]//2, im_size[2]//2 + 20]
for i, z in enumerate(zs):
  plt.subplot(1, 3, i+1)
  # plt.imshow(img_recon.cpu()[..., z].abs().rot90().cpu(), cmap='gray')
  plt.imshow(b0.cpu()[..., z].rot90().cpu(), cmap='jet')
  plt.axis('off')
plt.tight_layout()
plt.show()