import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from einops import einsum, rearrange
from tqdm import tqdm

from mr_recon.utils import np_to_torch, gen_grd, quantize_data
from mr_recon.fourier import sigpy_nufft, triton_nufft
from mr_recon.linops import batching_params, sense_linop
from mr_recon.recons import CG_SENSE_recon
from mr_recon.spatial import spatial_resize_poly
from mr_recon.imperfections.field import (
    alpha_phi_svd, 
    b0_to_phis_alphas, 
    isotropic_cluster_alphas
)

from hofft import kb_nufft, als_nufft, als_hofft, multi_apod_kern_linop

torch_dev = torch.device(3)

# Load
fpath = '/local_mount/space/tiger/1/users/abrahamd/mr_data/sevenT/data/'
ksp = np_to_torch(np.load(f'{fpath}ksp.npy')).type(torch.complex64).to(torch_dev)
phi = np_to_torch(np.load(f'{fpath}phi.npy')).type(torch.complex64).to(torch_dev)
mps = np_to_torch(np.load(f'{fpath}mps.npy')).type(torch.complex64).to(torch_dev)
b0  = np_to_torch(np.load(f'{fpath}b0.npy')).type(torch.float32).to(torch_dev)
trj = np_to_torch(np.load(f'{fpath}trj.npy')).type(torch.float32).to(torch_dev)
dcf = np_to_torch(np.load(f'{fpath}dcf.npy')).type(torch.float32).to(torch_dev)
dcf /= dcf.max()
b0 = -b0.rot90(k=-1).flip(dims=[2]) / (2 * np.pi)
ro = trj.shape[0]
d = trj.shape[-1]
C = ksp.shape[0]
im_size = mps.shape[1:]

# Start with a smaller problem
grps = slice(None, None, 2)
# grps = slice(0, 4)
ksp = ksp[..., grps, :].reshape((C, ro, -1))
trj = trj[:, grps].reshape((ro, -1, d))
dcf = dcf[:, grps].reshape((ro, -1))
print(f'ksp: {ksp.shape}')
print(f'trj: {trj.shape}')
print(f'dcf: {dcf.shape}')
print(f'phi: {phi.shape}')
print(f'mps: {mps.shape}')
print(f'b0: {b0.shape}')

# HOFFT Params
L = 1
solve_size = (50,)*3
os = 1.2
K = 1000
kern_size = (4,)*3

# # Build phis alphas 
# phis, alphas = b0_to_phis_alphas(b0, dcf.shape, 0, dt=2e-6)
# rs = gen_grd(im_size).to(torch_dev).moveaxis(-1, 0)
# kdevs = (trj - (trj * os).round() / os).moveaxis(-1, 0)
# alphas_tiled = alphas.tile((1, 1, dcf.shape[-1]))
# alphas = torch.cat([alphas_tiled, kdevs], dim=0)
# phis = torch.cat([phis, rs], dim=0)

# # Downsize phis and alphas
# phis_small = spatial_resize_poly(phis, solve_size, order=3)
# alphas_small, _ = isotropic_cluster_alphas(alphas, phis, K)
# alphas_small = alphas

# Decomposition on smaller problem
# weights, apods = als_hofft(phis_small, alphas_small, im_size, kern_size,
#                            os, L, num_als_iter=100, init_method='seg',
#                            use_type3=False)
# weights, apods = als_nufft(trj, im_size, kern_size, os, L, 
#                            num_als_iter=100, init_method='seg',
#                            solve_size=solve_size)
weights, apods = kb_nufft(trj, im_size, kern_size, os)

# Make linop
trj_grd = (os * trj).round()/os
bparams = batching_params()
A = multi_apod_kern_linop(trj_grd, mps, weights, apods, dcf, os, bparams)
# A = sense_linop(trj, mps, dcf, 
#                 nufft=triton_nufft(im_size, os, kern_size[0]),
#                 bparams=bparams)
# ---------------------------------------------------------------

# CG SENSE recon
vol = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0)

# Plot
img = vol[..., 220//2].abs().rot90(1).cpu()
vmin = 0
vmax = img.median() + 3 * img.std()
plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.tight_layout()

plt.figure()
img = b0[..., 220//2].cpu().rot90(1)
vmin = -200
vmax = 200
plt.imshow(img, cmap='jet')#, vmin=vmin, vmax=vmax)
plt.axis('off')
plt.tight_layout()
plt.show()
