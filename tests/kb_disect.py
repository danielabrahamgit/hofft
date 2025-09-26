import torch

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.utils import gen_grd, resize
from mr_recon.fourier import sigpy_nufft, ifft
from hofft import kb_nufft
from hofft.kb import _gen_kern_bases
from einops import rearrange, einsum

# Disect sigpy's apod and kbs
torch_dev = torch.device(4)
im_size = (256, 256)
os = 2
width = 2
kern_size = (width, width)
L = width ** 2

# First get our predicted apod and kb weights
kdevs = gen_grd(im_size).to(torch_dev) / os
# kdevs /= 2
# kdevs = 0 * kdevs
# kdevs += 1/os/2
weights, apods = kb_nufft(kdevs, im_size, kern_size, os)
weights = weights[0]
kern_bases = _gen_kern_bases(im_size, kern_size, os).to(torch_dev)
kern_bases = kern_bases.reshape((*kern_size, *im_size))
h_hofft = weights
b_hofft = kern_bases * apods[None,]

# # Now probe sigpy nufft to get actual kb and apods
# # F(img(r), kdev) ---> sum_r img(r) * sum_l b_l(r) h_l(kdev)
# # F(img_k(r), kdev) ---> sum_l (sum_r img_k(r) * b_l(r)) h_l(kdev)
# # b_k(kdev) ---> sum_l A_kl h_l(kdev)
# nufft = sigpy_nufft(im_size, os, width)
# bl = kern_bases * apods
# bl = bl.reshape((-1, *im_size))
# K = 100
# imgs = torch.randn((K, *im_size), device=torch_dev, dtype=torch.complex64)
# A = einsum(imgs, bl, 'K ..., L ... -> K L')
# b = nufft.forward(imgs[None,], kdevs[None,])[0] # K ...
# AHA_inv = torch.linalg.inv(A.H @ A) # L L
# AHb = einsum(A.H, b, 'L K, K ... -> L ...')
# hl = einsum(AHA_inv, AHb, 'Lo L, L ... -> Lo ...')
# hl = hl.reshape((*kern_size, *im_size))
# bl = bl.reshape((*kern_size, *im_size))

# Other probing method
nufft = sigpy_nufft(im_size, os, width)
delta = torch.ones((1, *im_size), device=torch_dev, dtype=torch.complex64)
ksp_os = nufft.forward_FT_only(delta)
apod = ifft(ksp_os, dim=(-2, -1))
apod = resize(apod, im_size)
weights_sp = torch.zeros_like(weights)
im_size_os = ksp_os.shape[1:]
for i in range(0, width):
    for j in range(0, width):
        ksp_os *= 0
        ksp_os[0, (im_size_os[0] // 2) + i, (im_size_os[1] // 2) + j] = 1.0
        weights_sp[i, j] = nufft.forward_interp_only(ksp_os, kdevs[None,])[0]
h_sp = weights_sp
b_sp = kern_bases * apod[None,]

# Re-arrange for plot
def plot(hs, bs, title=''):
    plt.figure(figsize=(14, 7))
    plt.suptitle(title)
    vmax = hs.abs().max().item()
    for i in range(width):
        for j in range(width):
            plt.subplot(width, width, i * width + j + 1)
            plt.imshow(hs[i, j].cpu().abs().flip(dims=[-2,-1]), cmap='gray', vmin=0, vmax=vmax)
            plt.colorbar()
            plt.axis('off')
            # plt.subplot(3,width**2,i*width + j + 1)
            # plt.imshow(hs[i, j].cpu().abs(), cmap='gray', vmin=0, vmax=vmax)
            # plt.axis('off')
        
            # plt.subplot(3, width**2, width**2 + i*width + j + 1)
            # plt.imshow(bs[i, j].cpu().angle(), cmap='jet', vmin=-3.14, vmax=3.14)
            # plt.axis('off')
            
            # plt.subplot(3, width**2, 2 * width**2 + i*width + j + 1)
            # plt.imshow(bs[i, j].abs().cpu(), cmap='gray')
            # plt.axis('off')
    plt.tight_layout()
plot(h_sp, b_sp, title='Sigpy Nufft Kernels and Apods')
plot(h_hofft, b_hofft, 'HOFFT')
plt.show()