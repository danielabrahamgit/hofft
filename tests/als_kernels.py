import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.fourier import fft, ifft
from mr_recon.utils import gen_grd
from mr_recon.algs import lin_solve
from einops import einsum   

from igrog.fixed_kernel_models import kaiser_bessel_model

from einops import rearrange, einsum
from pyeyes import ComparativeViewer
from tqdm import tqdm

# Simple linear phase testing
os = 1.25
rs_size = (50,50)
kdevs_size = (20, 20)
rs = gen_grd(rs_size).reshape((-1, 2))
# phis = torch.stack([
#     rs[..., 0] ** 2 + rs[..., 1] ** 2,
#     rs[..., 0] * rs[..., 1],
# ], dim=-1)
phis = rs
kdevs = gen_grd(kdevs_size).reshape((-1, 2)) / os

# Build kernel
kern_size = (3, 3)
kern = gen_grd(kern_size, kern_size).reshape((-1, 2)) / os
kern -= kern.mean(dim=0)
K = kern.shape[0]

# Optimal soln
kb = kaiser_bessel_model(kern_size, rs_size, os)
apod_opt = kb.source_maps.flatten()
weights_opt = kb.forward_kernel(kdevs).squeeze().T / np.prod(kern_size)

# Initialize apod
apod = torch.ones(rs.shape[0], dtype=torch.complex64)
# kb = kaiser_bessel_model(kern_size, rs_size, os)
# apod = kb.source_maps.flatten()

# Initialize matrices
T = torch.exp(-2j * torch.pi * (phis @ kdevs.T)) # R T
S = torch.exp(-2j * torch.pi * (rs @ kern.T)) # R K

# ALS
loss = []
S_apod = S * apod[:, None]
for _ in tqdm(range(1), 'Solving'):
    
    # Solve for weights
    weights = lin_solve(S_apod.H @ S_apod, S_apod.H @ T, solver='solve')
    
    # Update loss
    T_est = S_apod @ weights
    loss.append(torch.linalg.norm(T - T_est).square())

    # Solve for apodization
    T_est = S @ weights
    denom = (T_est.conj() * T_est).sum(dim=-1)
    numer = (T_est.conj() * T).sum(dim=-1)
    apod = numer / denom
    
    # Update loss
    S_apod = S * apod[:, None]
    T_est = S_apod @ weights
    loss.append(torch.linalg.norm(T - T_est).square())
    
# SVD solution
Tk = einsum(T, 1/S, 'R T, R K -> R T K').reshape((rs.shape[0], -1))
U, sig, Vh = torch.linalg.svd(Tk, full_matrices=False)
apod = U[:, 0]
weights_svd = sig[0] * Vh[0]
weights = rearrange(weights_svd, '(T K) -> K T', T=kdevs.shape[0])
S_apod = S * apod[:, None]
T_est = S_apod @ weights
loss.append(torch.linalg.norm(T - T_est).square())

# Sub in optimal apodization and kernel
T_opt = (S * apod_opt[:, None]) @ weights_opt
scale = T.norm() / T_opt.norm()
T_opt = T_opt * scale
loss_opt = torch.linalg.norm(T - T_opt)

# Show difference
T = T.reshape((*rs_size, *kdevs_size))
S_apod = S * apod[:, None]
T_est = S_apod @ weights
T_est = T_est.reshape((*rs_size, *kdevs_size))
T_opt = T_opt.reshape((*rs_size, *kdevs_size))
# T_opt = T_est

plt.figure(figsize=(14, 7))
plt.subplot(231)
vmin = 0.95
vmax = 1.05
plt.imshow(T.abs()[..., 0, 0], cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(232)
plt.imshow(T_est.abs()[..., 0, 0], cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')

plt.subplot(233)
plt.imshow(T_opt.abs()[..., 0, 0], cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
vmin = -np.pi
vmax = np.pi
plt.subplot(234)
plt.imshow(T.angle()[..., 0, 0], cmap='jet', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(235)
plt.imshow(T_est.angle()[..., 0, 0], cmap='jet', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.subplot(236)
plt.imshow(T_opt.angle()[..., 0, 0], cmap='jet', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.tight_layout()

plt.figure()
plt.plot(loss)
# plt.axhline(loss_opt, color='r', linestyle='--')

plt.figure(figsize=(14, 7))
plt.subplot(121)
err = (T_est - T).abs().reshape((-1, *kdevs_size)).max(dim=0).values
plt.imshow(err)
plt.colorbar()
plt.axis('off')
plt.subplot(122)
err_opt = (T_opt - T).abs().reshape((-1, *kdevs_size)).max(dim=0).values
plt.imshow(err_opt)
plt.colorbar()
plt.axis('off')
plt.tight_layout()

plt.figure(figsize=(14, 7))
plt.subplot(221)
plt.imshow(apod.abs().reshape(rs_size), cmap='gray')
plt.axis('off')
plt.subplot(222)
plt.imshow(apod_opt.abs().reshape(rs_size), cmap='gray')
plt.axis('off')
def weights_to_img(weights):
    kerns = rearrange(weights.abs(), '(kx ky) (Tx Ty) -> kx ky Tx Ty', 
                    kx=kern_size[0], ky=kern_size[1],
                    Tx=kdevs_size[0], Ty=kdevs_size[1])
    kerns = kerns.flip(dims=[0, 1])
    kerns = rearrange(kerns, 'kx ky Tx Ty -> (kx Tx) (ky Ty)')
    return kerns
plt.subplot(223)
plt.imshow(weights_to_img(weights), cmap='gray')
plt.axis('off')
plt.subplot(224)
plt.imshow(weights_to_img(weights_opt), cmap='gray')
plt.axis('off')
plt.show()
