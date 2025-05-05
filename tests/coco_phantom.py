import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from einops import einsum, rearrange

from mr_recon.linops import experimental_sense, batching_params, type3_nufft_naive, type3_nufft
from mr_recon.recons import CG_SENSE_recon
from mr_recon.imperfections.field import b0_to_phis_alphas, coco_to_phis_alphas, alpha_phi_svd, alpha_segementation, isotropic_cluster_alphas
from mr_recon.multi_coil.calib import synth_cal
from mr_recon.fourier import gridded_nufft, ifft, sigpy_nufft
from mr_recon.utils import gen_grd, np_to_torch  
from mr_recon.spatial import spatial_resize_poly
from mr_recon.spatial import (
    spatial_interp, 
    spatial_resize, 
    spatial_resize_poly,
    apply_interp_kernel,
    get_interp_kernel
)
from mr_recon.gp import gp_model, optimize_hyper

from igrog.kernel_linop import fixed_kern_naive_linop

# from hofft import als_hofft, als_nufft, also_this_one, mlp_hofft, kb_nufft, idonttrustthis
from hofft import als_hofft, als_iterations, als_nufft

# Set seeds
torch.manual_seed(0)
np.random.seed(0)

# load data
torch_dev = torch.device(5)
# torch_dev = torch.device('cpu')
R = 3
dt = 2e-6
FOV = 0.22
fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/coco_spiral/best_slice/'
kwargs = {'map_location': torch_dev, 'weights_only': True}
b0 = torch.load(f'{fpath}b0.pt', **kwargs).type(torch.float32)
mps = torch.load(f'{fpath}mps.pt', **kwargs).type(torch.complex64)
img_gt = torch.load(f'{fpath}img_gt.pt', **kwargs).type(torch.complex64)
ksp_fs = torch.load(f'{fpath}ksp.pt', **kwargs).type(torch.complex64)
trj_fs = torch.load(f'{fpath}trj.pt', **kwargs).type(torch.float32)
dcf_fs = torch.load(f'{fpath}dcf.pt', **kwargs).type(torch.float32)
ksp = ksp_fs[..., ::R]
trj = trj_fs[:, ::R]
dcf = dcf_fs[:, ::R]
C = ksp.shape[0]
im_size = img_gt.shape
msk = 1 * (mps.abs().sum(dim=0) > 0).to(torch_dev)
# msk = None

# Calib data
ksp_cal = synth_cal(ksp_fs, (32, 32), trj_fs, dcf_fs, num_iter=1)
img_cal = ifft(ksp_cal, dim=[-2,-1], oshape=mps.shape)

# Get phis alphas from field imperfections
xyz = gen_grd(im_size).to(torch_dev)
xyz = torch.stack([xyz[..., 0], xyz[..., 0]*0, xyz[..., 1]], dim=-1)
trj_physical = torch.stack([trj[..., 0], trj[..., 0]*0, trj[..., 1]], dim=-1) * FOV
phis_coco, alphas_coco = coco_to_phis_alphas(trj_physical, xyz, 3, 0, dt)
phis_b0, alphas_b0 = b0_to_phis_alphas(b0, dcf.shape, 0, dt)
phis = torch.cat([phis_coco[:-2], phis_b0], dim=0)
alphas = torch.cat([alphas_coco[:-2], alphas_b0], dim=0)

# HOFFT Params
# os = 1.5
os = round(276*1.211)/276
kern_size = (6, 6)
solve_size = (120, 120)
L = 4

# Add on non-cartesian term
rs = gen_grd(im_size).to(torch_dev).moveaxis(-1, 0)
kdevs = (trj - (os * trj).round()/os).moveaxis(-1, 0)
phis = torch.cat([phis, rs], dim=0)
alphas = torch.cat([alphas, kdevs], dim=0)

phis_small = spatial_resize_poly(phis, solve_size, order=3)
alphas_small = alphas
M = 1000
alphas_small = isotropic_cluster_alphas(alphas, phis, M)[0].T
# rnd_inds = np_to_torch(np.random.choice(alphas.shape[1], M, replace=True)).to(torch_dev)
# alphas_small = alphas.reshape((alphas.shape[0], -1))[:, rnd_inds]

# # Decomp
# b, h = alpha_segementation(phis_small, alphas, L=L, interp_type='lstsq', use_type3=False)
# # b, h = alpha_phi_svd(phis_small, alphas, L=L, use_type3=False)
# b = spatial_resize_poly(b, im_size, order=3)

print(f'Matrix size (T, R) = ({alphas_small[0].numel()}, {phis_small[0].numel()})')
weights, apods = als_hofft(phis_small, alphas_small, im_size, kern_size, os=os, L=L,
                           num_als_iter=100, 
                           use_type3=False,
                           init_method='seg',
                           verbose=True)    

# RBF kernel
if (alphas_small.shape != alphas.shape) and \
    alphas_small.ndim == 2:
        
    # Build dataset
    weights_flt = weights.reshape((L * np.prod(kern_size), -1)).T
    X = alphas_small.T
    Y = torch.cat([weights_flt.real, weights_flt.imag], dim=-1)
    # Y = torch.cat([weights_flt[:, 5:6].real, weights_flt[:, 5:6].imag], dim=-1)
    X_pred = alphas.moveaxis(0, -1).reshape((-1, alphas.shape[0]))
    
    # # ---------------- Fit GP Model ----------------
    # model = gp_model(X, Y).to(torch_dev)
    # model = optimize_hyper(model, X, Y, num_iter=80, lr=1e-1)
    # model.eval()
    # model.likelihood.eval()
    # Y_pred = torch.zeros((X_pred.shape[0], Y.shape[1])).to(torch_dev)
    # bs = 100
    # for n1 in range(0, X_pred.shape[0], bs):
    #     n2 = min(n1 + bs, X_pred.shape[0])
    #     Y_pred[n1:n2] = model.predict(X_pred[n1:n2]).detach()
    
    # ---------------- Fit RBF Model ----------------
    kern_weights, kern_func = get_interp_kernel(X, Y, 
                                                kern_param=0.5,
                                                lamda=1e-4)
    Y_pred = apply_interp_kernel(X, X_pred, 
                                 kern_weights=kern_weights, 
                                 batch_size=1000,
                                 kern_func=kern_func)
    
    # Reshape
    weights = (Y_pred[:, :Y_pred.shape[1]//2] + \
               1j*Y_pred[:, Y_pred.shape[1]//2:]).T
    
    weights = weights.reshape((L, *kern_size, *dcf.shape))

# weights, apods = kb_nufft(trj, im_size, kern_size, os)
# apods = b * apods
# weights = h[:, None, None] * weights


# Build linop
bparams = batching_params(mps.shape[0]*0+1,)
trj_grd = (os * trj).round()/os
# A = experimental_sense(mps=mps, dcf=dcf,
#                        spatial_funcs=b, 
#                        temporal_funcs=h,
#                        trj=trj_grd,
#                        nufft=gridded_nufft(im_size, os),
#                     #    trj=trj,
#                     #    nufft=sigpy_nufft(im_size, oversamp=os, width=kern_size[0]),
#                        bparams=bparams)
A = fixed_kern_naive_linop(weights, apods, mps, trj_grd, dcf, os_grid=os, bparams=bparams)

# Recon
img = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0).rot90()

# Show
plt.imshow(img.abs().cpu(), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.figure()
plt.imshow(img.angle().cpu(), cmap='jet')
plt.axis('off')
plt.tight_layout()

plt.figure(figsize=(14,7))
for l in range(L):
    plt.subplot(2, L, l+1)
    plt.imshow(apods[l].abs().cpu(), cmap='gray')
    plt.axis('off')
    plt.subplot(2, L, l+1+L)
    plt.imshow(apods[l].angle().cpu(), cmap='jet')
    plt.axis('off')
plt.tight_layout()
plt.show()




