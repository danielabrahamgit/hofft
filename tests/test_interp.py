import torch
from torch.special import bessel_j0

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from hofft.reduce import reduce_temporal, alpha_interp_kerns, expand_temporal

torch.manual_seed(0)

# consts
os = 1.13
W = 3
im_size = (220,)
trj_size = (300,)
# torch_dev = torch.device('cpu')
torch_dev = torch.device(6)

# Generate phis alphas
rs = torch.arange(-(im_size[0]//2), (im_size[0]//2), device=torch_dev) / im_size[0]
rs = torch.stack(
    torch.meshgrid([rs for _ in range(len(im_size))], indexing='ij'), dim=0)
phis = rs ** 1
phis /= phis.abs().max() * 2
alphas = torch.linspace(-5, 5, trj_size[0], device=torch_dev)[None,]

# Try to interpolate from grid 
enc_mat = torch.exp(-2j * torch.pi * (alphas.T @ phis)) # T R
alphas_grid = torch.arange(alphas.min() - W/os, alphas.max() + W/os, 1/os, device=torch_dev)[None, :]
enc_mat_grid = torch.exp(-2j * torch.pi * (alphas_grid.T @ phis)) # G R

# --------------- Zero Order Interpolation ---------------
enc_mat_zero = enc_mat * 0
for i in range(enc_mat.shape[0]):
    k = torch.argmin((alphas_grid - alphas[0, i]).abs())
    enc_mat_zero[i] = enc_mat_grid[k] + 0.0

# --------------- First Order Interpolation ---------------
enc_mat_linear = enc_mat * 0
for i in range(enc_mat.shape[0]):
    k = torch.argmin((alphas_grid - alphas[0, i]).abs())
    delta = alphas[0, i] - alphas_grid[0, k]
    if delta > 0:
        # |* -a----- |
        kleft = k
        kright = k+1
    else:
        # | -----a- |*
        kleft = k-1
        kright = k
    p = (alphas[0, i] - alphas_grid[0, kleft]) / (alphas_grid[0, kright] - alphas_grid[0, kleft])
    enc_mat_linear[i] = enc_mat_grid[kleft] * (1-p) + enc_mat_grid[kright] * p
enc_mat_linear *= enc_mat.abs().max() / enc_mat_linear.abs().max()
    
# --------------- Least Squares Interpolation ---------------
# | ---- | ---- | ---- |
# 0      1      2      3
alphas_bases = alphas_grid[:, :W].clone()
alphas_bases -= alphas_bases.mean(dim=-1, keepdim=True)
dalpha = 1 / os
N = 100
ns = torch.arange(N+1, device=torch_dev)
dalphas = dalpha * (ns[None,] - N // 2) / N 
B = torch.exp(-2j * torch.pi * (phis.T @ dalphas)) # R A
A = torch.exp(-2j * torch.pi * (phis.T @ alphas_bases)) # R L
weights = torch.linalg.solve(A.H @ A, A.H @ B) # L A

# | ---- | ---- |
# 0      1      2

# Interp
enc_mat_lstsq = enc_mat * 0
if W % 2 == 0:
    ofs = dalpha/2
else:
    ofs = 0
for i in range(enc_mat.shape[0]):
    k = torch.argmin((alphas_grid - ofs - alphas[0, i]).abs())
    delta = alphas[:, i] - alphas_grid[:, k] + ofs
    n = torch.argmin((dalphas - delta).abs())
    ws = weights[:, n]
    idxs = torch.arange(k-W//2, k+W//2 + (W%2), device=torch_dev)
    enc_mat_lstsq[i] = (ws[:, None] * enc_mat_grid[idxs, :]).sum(dim=0)
    
# --------------- Experimental Interpolation ---------------
alphas += ofs
dalphas = (1/os,)
dalphas_tensor = torch.tensor(dalphas, device=torch_dev)
weights, delta_alphas, apod = alpha_interp_kerns(phis, W=W, dalphas=dalphas, solve_apod=True)
alphas_unq, alpha_kern, alpha_to_unq_idx = reduce_temporal(alphas, W=W, dalphas=dalphas)
enc_mat_grid = torch.exp(-2j * torch.pi * (alphas_unq @ phis)) * apod # G R
enc_mat_exp = expand_temporal(enc_mat_grid.T, alphas, dalphas, weights, delta_alphas, alpha_kern, alpha_to_unq_idx)
enc_mat_exp = enc_mat_exp.T

# Compare
plt.figure(figsize=(14, 7))
plt.suptitle(f'Interpolation Error OS = {os}, W = {W}')
encs = [enc_mat, enc_mat_zero, enc_mat_linear, enc_mat_lstsq, enc_mat_exp]
titles = ['True', 'Zero', 'Linear', 'LS', 'Experimental']
I = len(encs)
for i, enc in enumerate(encs):
    plt.subplot(2,I,i + 1)
    plt.title(titles[i])
    plt.imshow(enc.angle().cpu(), cmap='jet', vmin=-torch.pi, vmax=torch.pi)
    # plt.imshow(enc.abs(), cmap='RdBu', vmin=0.8, vmax=1.2)
    plt.axis('off')
    plt.subplot(2,I,i + I + 1)
    plt.imshow((enc - enc_mat).abs().cpu(), cmap='gray', vmin=0, vmax=0.2)
    plt.axis('off')
    print((enc - enc_mat).norm())
plt.tight_layout()
plt.show()
