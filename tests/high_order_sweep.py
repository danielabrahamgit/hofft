import torch
import numpy as np
import time

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from einops import einsum, rearrange
from tqdm import tqdm

from mr_recon.spatial import spatial_resize, spatial_interp
from mr_recon.linops import experimental_sense, batching_params, type3_nufft_naive, type3_nufft
from mr_recon.recons import CG_SENSE_recon, coil_combine
from mr_recon.imperfections.field import b0_to_phis_alphas, coco_to_phis_alphas, alpha_phi_svd, alpha_segementation
from mr_recon.fourier import gridded_nufft, ifft, fft, lr_nufft, matrix_nufft, sigpy_nufft, svd_nufft
from mr_recon.utils import gen_grd, normalize, np_to_torch  
from mr_recon.spatial import spatial_resize
from mr_recon.calib import synth_cal
from mr_recon.spatial import spatial_resize_poly

from igrog.kernel_linop import fixed_kern_naive_linop
# from als import init_apods, als_iterations
from hofftify import kb_nufft, mlp_hofft, als_nufft, als_hofft


# Params
torch_dev = torch.device(5)
# torch_dev = torch.device('cpu')
R = 3
os = round(276*1.211)/276
cal_size = (100, 100)
num_als_iter = 100

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

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

# non-cartesian deviation bases
rs = gen_grd(im_size).to(torch_dev).moveaxis(-1, 0)
kdevs = (trj - (os * trj).round()/os).moveaxis(-1, 0)
phis = torch.cat([phis, rs], dim=0)
alphas = torch.cat([alphas, kdevs], dim=0)

# Smaller problem
phis_small = spatial_resize_poly(phis, cal_size, order=3)

# Loop over kernel sizes and number of apodizations
imgs = []
Ls = torch.arange(1, 16, 2)
ks = torch.arange(3, 10, 2)
idxs = torch.meshgrid(torch.arange(len(Ls)), 
                      torch.arange(len(ks)),
                      torch.arange(len(ks)), indexing='ij')
idxs = torch.stack(idxs, dim=-1).reshape((-1, 3))
times = torch.zeros(len(idxs))
for n in tqdm(range(len(idxs))):
    
    # Set hofft params
    kern_size = (ks[idxs[n, 1]].item(), ks[idxs[n, 2]].item())
    L = Ls[idxs[n, 0]].item()
    K = np.prod(kern_size)
    if kern_size[0] != kern_size[1]:
        imgs.append(torch.zeros(im_size, dtype=torch.complex64))
        continue
    print(L, kern_size)
    
    torch.manual_seed(0) # Deterministic clustering/svd
    
    # HOFFT
    weights, apods = als_hofft(phis_small, alphas, im_size, kern_size, os=os, L=L,
                               num_als_iter=num_als_iter, 
                               use_type3=False,
                               init_method='seg',
                               verbose=True,)
    bparams = batching_params(mps.shape[0], field_batch_size=L)
    trj_grd = (os * trj).round()/os
    A = fixed_kern_naive_linop(weights, apods, mps, trj_grd, dcf, os_grid=os, bparams=bparams)
    
    # # Classical
    # # b, h = alpha_segementation(phis_small[:-2], alphas[:-2], L=L, interp_type='zero', use_type3=True)
    # b, h = alpha_phi_svd(phis_small[:-2], alphas[:-2], L=L, L_batch_size=L, use_type3=True)
    # b = spatial_resize_poly(b, im_size, order=3)
    # bparams = batching_params(mps.shape[0], field_batch_size=L)
    # A = experimental_sense(trj, mps, dcf, 
    #                        nufft=sigpy_nufft(im_size, oversamp=os, width=kern_size[0]),
    #                        spatial_funcs=b,
    #                        temporal_funcs=h,
    #                        bparams=bparams)
    

    # Recon
    torch.cuda.synchronize()  
    start = time.perf_counter()
    rep = 1
    for _ in range(rep):
        img = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0, verbose=False, tolerance=0.0).cpu()
    torch.cuda.synchronize()
    end = time.perf_counter()
    times[n] = (end - start) / rep
    
    # Save image
    imgs.append(img)

# Save data
data = {
    'imgs': torch.stack(imgs, dim=0).cpu(),
    'Ls': Ls.cpu(),
    'ks': ks.cpu(),
    'idxs': idxs.cpu(), 
    'num_als_iter': num_als_iter
}
torch.save(data, f'./high_order_experiments/svd.pt')
# torch.save(data, f'./high_order_experiments/{num_als_iter}_iter.pt')
quit()

# Save ground truth
nufft = sigpy_nufft(im_size, 2.0, 6)
b, h = alpha_segementation(phis[:-2], alphas[:-2], L=50, interp_type='lstsq', use_type3=True)
# nufft = matrix_nufft(im_size, spatial_batch_size=220 * 10)
A = experimental_sense(trj, mps, dcf, nufft=nufft,
                       spatial_funcs=b,
                       temporal_funcs=h,)
img_gt = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0, verbose=True).cpu()
# torch.save(img_gt, './high_order_experiments/img_gt.pt')
# quit()

nufft = sigpy_nufft(im_size)
A = experimental_sense(trj, mps, dcf, nufft=nufft)
img_nufft = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0, verbose=True).cpu()
img_nufft = normalize(img_nufft, img_gt, mag=False, ofs=False)

# Show
img = normalize(img, img_gt, mag=False, ofs=False)
vmin = 0
vmax = img.abs().median() + 3 * img.abs().std()
M = 20
plt.figure(figsize=(14, 7))
plt.subplot(231)
plt.imshow(img.abs(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(232)
plt.imshow(img_nufft.abs(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(233)
plt.imshow(img_gt.abs(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(234)
plt.imshow((img_gt - img).abs(), cmap='gray', vmin=vmin, vmax=vmax/M)
plt.axis('off')
plt.subplot(235)
plt.imshow((img_gt - img_nufft).abs(), cmap='gray', vmin=vmin, vmax=vmax/M)
plt.axis('off')
plt.subplot(236)
plt.imshow((img_gt - img_gt).abs(), cmap='gray', vmin=vmin, vmax=vmax/M)
plt.axis('off')
plt.tight_layout()
# plt.show()

# Show apods
plt.figure(figsize=(14, 7))
for l in range(L):
    plt.subplot(2, L, l+1)
    plt.imshow(apods[l].abs().cpu(), cmap='gray')
    plt.axis('off')
    plt.subplot(2, L, l+1+L)
    plt.imshow(apods[l].angle().cpu(), cmap='jet')
    plt.axis('off')
plt.tight_layout()
plt.show()




