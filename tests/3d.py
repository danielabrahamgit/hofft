import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from einops import einsum, rearrange
from tqdm import tqdm

from mr_recon.utils import np_to_torch, gen_grd, quantize_data
from mr_recon.fourier import sigpy_nufft
from mr_recon.linops import batching_params, experimental_sense
from mr_recon.recons import CG_SENSE_recon
from mr_recon.spatial import (
    spatial_interp, 
    spatial_resize, 
    spatial_resize_poly,
    apply_interp_kernel,
    get_interp_kernel
)
from mr_recon.imperfections.field import (
    alpha_phi_svd, 
    b0_to_phis_alphas, 
    alpha_segementation, 
    rescale_phis_alphas
)

from igrog.kernel_linop import fixed_kern_naive_linop
from hofftify import multi_apod_nufft, multi_apod_hofft

torch_dev = torch.device(1)

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

# Params
L = 3
solve_size = (50,)*3
phis, alphas = b0_to_phis_alphas(b0, dcf.shape, 0, dt=2e-6)

# --------------------- Build efficient nufft model ---------------------
bparams = batching_params()
os = 1.2
kern_size = (2,)*3
K = 1000
# weights, apods = multi_apod_nufft(trj, im_size, 
#                                   kern_size=kern_size, 
#                                   os=os, 
#                                   L=2, 
#                                   num_als_iter=100, 
#                                   solve_size=solve_size)

# Incorporate b0 too
rs = gen_grd(im_size).to(torch_dev).moveaxis(-1, 0)
kdevs = (trj - (trj * os).round() / os).moveaxis(-1, 0)
alphas_tiled = alphas.tile((1, 1, dcf.shape[-1]))
alphas = torch.cat([alphas_tiled, kdevs], dim=0)
phis = torch.cat([phis, rs], dim=0)

# Rescale for kmeans
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)
betas, idxs = quantize_data(alphas_nrm.moveaxis(0, -1)[::10], K, method='cluster')

# SVD on smaller problem
phis_small = spatial_resize_poly(phis_nrm, solve_size, order=3)
weights, apods = multi_apod_hofft(phis_small, betas.T, im_size, 
                                  kern_size=(2,)*3,
                                  os=os,
                                  L=L,
                                  num_als_iter=100,
                                  use_type3=False)

# Interpolate back up via RBF
kern_weights, kern_func = get_interp_kernel(betas, weights.reshape((-1, K)).T, kern_param=.4)
weights_interp = apply_interp_kernel(betas,  alphas_nrm.moveaxis(0, -1), 
                                     kern_weights=kern_weights, 
                                     batch_size=1000,
                                     kern_func=kern_func)
weights = weights_interp.moveaxis(-1, 0).reshape((*weights.shape[:-1], *alphas.shape[1:]))
weights = weights.reshape((L, *kern_size, *dcf.shape))

# Apply midpoints
phz_0 = torch.exp(-2j * np.pi * (phis_mp @ alphas_mp))
phz_spat = torch.exp(-2j * torch.pi * einsum(phis, alphas_mp, 'B ... , B -> ...')) * phz_0
phz_temp = torch.exp(-2j * np.pi * einsum(phis_mp, alphas, 'B, B ... -> ...'))
weights *= phz_temp
apods *= phz_spat

# Make linop
trj_grd = (os * trj).round()/os
A = fixed_kern_naive_linop(weights, apods, mps, trj_grd, dcf, 
                           os_grid=os, 
                           bparams=bparams)
# ---------------------------------------------------------------

# # --------------------- Regular nufft model ---------------------
# phis = spatial_resize_poly(phis, solve_size, order=3)
# # b, h = alpha_segementation(phis, alphas, L=L, L_batch_size=1, interp_type='lstsq', use_type3=True)
# b, h = alpha_phi_svd(phis, alphas, L=L, L_batch_size=1, use_type3=True)
# b = spatial_resize_poly(b, im_size, order=3)
# bparams = batching_params(field_batch_size=L)
# nufft = sigpy_nufft(im_size, width=3)
# A = experimental_sense(trj, mps, dcf,
#                        spatial_funcs=b,
#                        temporal_funcs=h, 
#                        nufft=nufft, 
#                        use_toeplitz=False, 
#                        bparams=bparams)
# # ---------------------------------------------------------------

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
