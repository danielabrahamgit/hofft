import torch

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.utils import gen_grd, resize
from mr_recon.fourier import sigpy_nufft, svd_nufft
from hofft import kb_nufft, multi_apod_kern_linop
from hofft.kb import _gen_kern_bases, sample_kb_kernel
from einops import rearrange, einsum

# Params
torch_dev = torch.device(4)
im_size = (256, 256)
os = 1.25
width = 4
kern_size = (width, width)
L = width ** 2

# # Holy shit what is going on 
# nkdev = 9
# kdev_size = (nkdev, nkdev)
# kdevs = 2 * gen_grd(kdev_size).to(torch_dev) / os
# weights, apods = kb_nufft(kdevs, im_size, kern_size, os)
# kern_bases = _gen_kern_bases(im_size, kern_size, os).to(torch_dev)
# weights = weights.reshape((-1, *kdev_size))

# imgs = einsum(apods * kern_bases, weights, 'L X Y, L Kx Ky -> Kx Ky X Y').cpu()
# M = 3
# # imgs = imgs * 0 + 1
# plt.figure(figsize=(7, 7))
# rs = gen_grd(im_size).to(torch_dev)
# for i in range(nkdev):
#     for j in range(nkdev):
#         plt.subplot(nkdev, nkdev, i * nkdev + j + 1)
#         kd = kdevs[i,j] - (kdevs[i,j] * os).round() / os
#         phz_gt = torch.exp(-2j * torch.pi * einsum(rs, kd, 
#                                                    '... d, d -> ...')).cpu()
#         mag = (imgs[i, j] * phz_gt.conj()).real
#         plt.imshow(mag, cmap='bwr', vmin=0.0, vmax=2.0)
#         plt.axis('off')
# # plt.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.figure(figsize=(7, 7))
# rs = gen_grd(im_size).to(torch_dev)
# for i in range(nkdev):
#     for j in range(nkdev):
#         plt.subplot(nkdev, nkdev, i * nkdev + j + 1)
#         kd = kdevs[i,j] - (kdevs[i,j] * os).round() / os
#         phz_gt = torch.exp(-2j * torch.pi * einsum(rs, kd, 
#                                                    '... d, d -> ...')).cpu()
#         mag = (imgs[i, j] * phz_gt.conj()).imag
#         plt.imshow(mag, cmap='bwr', vmin=-1, vmax=+1)
#         plt.axis('off')
# # plt.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()
# quit()

# First get our predicted apod and kb weights
trj = 2 * gen_grd(im_size).to(torch_dev)
weights, apods = kb_nufft(trj, im_size, kern_size, os)

# Build our nufft
nufft = svd_nufft(im_size, os, n_svd=L)
trj_grd = nufft.rescale_trajectory(trj) # don't really need this
kern_bases = _gen_kern_bases(im_size, kern_size, os).to(torch_dev)
nufft.spatial_funcs = kern_bases * apods
nufft.temporal_funcs = weights.reshape((-1, *im_size))

# Compare to sigpy
sp_nufft = sigpy_nufft(im_size, os, width)
img = torch.ones(im_size, device=torch_dev, dtype=torch.complex64)
ksp = nufft(img[None,], trj_grd[None,])[0]
ksp_sp = sp_nufft(img[None,], trj[None],)[0]

plt.figure()
plt.imshow(ksp_sp.abs().cpu(), cmap='gray')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.figure()
plt.imshow(ksp.abs().cpu(), cmap='gray')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.show()