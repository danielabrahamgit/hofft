import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.linops import type3_nufft, type3_nufft_naive
from mr_recon.fourier import fft, ifft
from mr_recon.utils import gen_grd
from mr_recon.algs import lin_solve
from einops import einsum   

from igrog.fixed_kernel_models import kaiser_bessel_model

from einops import rearrange, einsum
from pyeyes import ComparativeViewer
from tqdm import tqdm

from als import init_apods, lstsq_spatial, lstsq_temporal, als_iterations

# Simple linear phase testing
os = 1.25
rs_size = (50,50)
kdevs_size = (20, 20)
rs = gen_grd(rs_size).reshape((-1, 2))
phis = torch.stack([
    rs[..., 0] ** 2 + rs[..., 1] ** 2,
    rs[..., 0] * rs[..., 1],
], dim=-1)
# phis = rs
kdevs = 2*gen_grd(kdevs_size).reshape((-1, 2)) / os
alphas = kdevs

# Build kernel
kern_size = (3, 3)
kern = gen_grd(kern_size, kern_size).reshape((-1, 2)) / os
kern -= kern.mean(dim=0)

# Create type3 nufft to model high order phase
# t3n = type3_nufft(phis.T, alphas.T)
t3n = type3_nufft_naive(phis.T, alphas.T)
K = kern.shape[0]
T = alphas.shape[0]
R = rs.shape[0]
L = 2

# Create targets
Targ = torch.exp(-2j * torch.pi * (phis @ alphas.T)) # (R, T)
kern_bases = torch.exp(-2j * torch.pi * (rs @ kern.T)) # (R, K)

def forward(weights, apods, kern_bases):
    # weights -> (L, K, T)
    # apods -> (R, L)
    # kern_bases -> (R, K)
    bases = einsum(kern_bases, apods, 'R K, R L -> R L K')
    return bases.reshape((R, -1)) @ weights.reshape((-1, T))

# ALS
apods = init_apods(t3n, L, method='rnd').T
weights, apods = als_iterations(t3n, kern_bases.T, apods.T, max_iter=100)
apods = apods.T

# Show difference
Targ = Targ.reshape((*rs_size, *kdevs_size))
T_est = forward(weights, apods, kern_bases).reshape((*rs_size, *kdevs_size))

plt.figure(figsize=(14, 7))
plt.subplot(221)
vmin = 0.95
vmax = 1.05
plt.imshow(Targ.abs()[..., 0, 0], cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(222)
plt.imshow(T_est.abs()[..., 0, 0], cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')

vmin = -np.pi
vmax = np.pi
plt.subplot(223)
plt.imshow(Targ.angle()[..., 0, 0], cmap='jet', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(224)
plt.imshow(T_est.angle()[..., 0, 0], cmap='jet', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.tight_layout()

# plt.figure()
# plt.plot(loss)

plt.figure(figsize=(7, 7))
err = (T_est - Targ).abs().reshape((-1, *kdevs_size)).max(dim=0).values
plt.imshow(err)
plt.colorbar()
plt.axis('off')
plt.tight_layout()

def weights_to_img(weights):
    kerns = rearrange(weights.abs(), '(kx ky) (Tx Ty) -> kx ky Tx Ty', 
                    kx=kern_size[0], ky=kern_size[1],
                    Tx=kdevs_size[0], Ty=kdevs_size[1])
    kerns = kerns.flip(dims=[0, 1])
    kerns = rearrange(kerns, 'kx ky Tx Ty -> (kx Tx) (ky Ty)')
    return kerns
for l in range(L):
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.imshow(apods[:, l].abs().reshape(rs_size), cmap='gray')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(weights_to_img(weights[l]), cmap='gray')
    plt.axis('off')
plt.show()
