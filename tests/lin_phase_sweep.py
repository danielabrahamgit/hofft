import torch
import numpy as np
import time

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from einops import einsum, rearrange
from tqdm import tqdm

from mr_recon.spatial import spatial_resize, spatial_interp
from mr_recon.linops import sense_linop, batching_params, type3_nufft_naive, type3_nufft
from mr_recon.recons import CG_SENSE_recon, coil_combine
from mr_recon.imperfections.field import b0_to_phis_alphas, coco_to_phis_alphas, alpha_phi_svd, alpha_segementation
from mr_recon.fourier import gridded_nufft, ifft, fft, matrix_nufft, sigpy_nufft, svd_nufft
from mr_recon.utils import gen_grd, normalize, np_to_torch  
from mr_recon.spatial import spatial_resize

from igrog.kernel_linop import fixed_kern_naive_linop
# from als import init_apods, als_iterations
from hofft import kb_nufft, mlp_hofft, als_nufft


# Params
torch_dev = torch.device(5)
# torch_dev = torch.device('cpu')
R = 3

np.random.seed(0)
torch.manual_seed(0)
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

# load data
fpath = '/local_mount/space/mayday/data/users/abrahamd/60_shot/data/'
ksp_fs = np_to_torch(np.load(f'{fpath}ksp.npy')).type(torch.complex64).to(torch_dev)
trj_fs = np_to_torch(np.load(f'{fpath}trj.npy')).type(torch.float32).to(torch_dev)
dcf_fs = np_to_torch(np.load(f'{fpath}dcf.npy')).type(torch.float32).to(torch_dev)
mps = np_to_torch(np.load(f'{fpath}mps.npy')).type(torch.complex64).to(torch_dev)
evals = np_to_torch(np.load(f'{fpath}evals.npy')).type(torch.float32).to(torch_dev)
img_cal = np_to_torch(np.load(f'{fpath}img_cal.npy')).type(torch.complex64).to(torch_dev)
ksp = ksp_fs[..., ::R]
trj = trj_fs[:, ::R]
dcf = dcf_fs[:, ::R]

# Consts
C = ksp.shape[0]
im_size = mps.shape[1:]
trj_size = trj.shape[:-1]
cal_size = (50,)*2
# cal_size = im_size
num_als_iter = 10000
# cal_size = im_size

# Masking
img_cal = spatial_resize(img_cal, cal_size, method='fourier')
mask = 1.0 * (evals > 0.95)
mps *= mask

# non-cartesian deviation bases
os = 1.2
rs = gen_grd(cal_size).to(torch_dev).moveaxis(-1, 0)
kdevs = gen_grd(cal_size).to(torch_dev).moveaxis(-1, 0) / os

# Loop over kernel sizes and number of apodizations
imgs = []
Ls = torch.arange(1, 7)
ks = torch.arange(1, 7)
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
    
    # Alternating least squares for weights
    # weights, apods = als_nufft(trj, im_size, kern_size, os, L,
    #                            num_als_iter=num_als_iter,
    #                         #    init_method='seg',
    #                            solve_size=cal_size, verbose=False)
    weights, apods = kb_nufft(trj, im_size, kern_size, os)
    weights = torch.repeat_interleave(weights, L, dim=0)
    apods = torch.repeat_interleave(apods, L, dim=0)

    # Build linop
    bparams = batching_params(mps.shape[0], field_batch_size=L)
    trj_grd = (os * trj).round()/os
    A = fixed_kern_naive_linop(weights, apods, mps, trj_grd, dcf, os_grid=os, bparams=bparams)
    # A = experimental_sense(trj, mps, dcf, nufft=sigpy_nufft(im_size, oversamp=os, width=kern_size[0]), bparams=bparams)

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
    
    break

# torch.save(times, 'times_gpu_rep20.pt')
# quit()

# # Save data
# data = {
#     'imgs': torch.stack(imgs, dim=0).cpu(),
#     'Ls': Ls.cpu(),
#     'ks': ks.cpu(),
#     'idxs': idxs.cpu(), 
#     'num_als_iter': num_als_iter
# }
# torch.save(data, f'./lin_phase_experiments/kb.pt')
# torch.save(data, f'./lin_phase_experiments/{num_als_iter}_iter.pt')
# quit()

# Save ground truth
nufft = sigpy_nufft(im_size, 2.0, 6)
# nufft = matrix_nufft(im_size, spatial_batch_size=220 * 10)
A = sense_linop(trj, mps, dcf, nufft=nufft)
img_gt = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0, verbose=True).cpu()
torch.save(img_gt, './lin_phase_experiments/img_gt.pt')
quit()

nufft = sigpy_nufft(im_size, os, kern_size[0])
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




