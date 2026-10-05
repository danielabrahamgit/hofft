"""
Paper demo: farthest-point sampling (FPS) on 2D-compressed phase coefficients
from tilt_spi. Compresses via the joint SVD of A^T Φ, then runs FPS with K=100
in that plane. Background points are grey; FPS centers are lime-green.

  python paper_experiments/matvec_analysis/show_fps.py
"""
import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from hofft.phase_coeffs import whiten_phis_alphas, rescale_phis_alphas, remove_linear_terms
from hofft.utils import fps_multi_center_indices, kmeans_centroids, expand_spatial, expand_temporal, reduce_spatial, reduce_temporal
from hofft.cur_ops import _pivot_indices


K = 100
seed = 0
# fpath = './data/highres_spiral'
# fpath = './data/tilt_spi_invivo'
fpath = './data/axial_spi_invivo'
# fpath = './data/tilt_spi'
out_png = './paper_experiments/matvec_analysis/fps_k100_2d.png'
out_pdf = './paper_experiments/matvec_analysis/fps_k100_2d.pdf'

torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
kwargs = {'weights_only': True, 'map_location': torch_dev}

phis = torch.load(f'{fpath}/phis.pt', **kwargs)
alphas = torch.load(f'{fpath}/alphas.pt', **kwargs).float()[..., :1]
evals = torch.load(f'{fpath}/evals.pt', **kwargs).float()
im_size = phis.shape[1:]

# Reduce
Rn = 1
Rm = 1
im_size_low = (im_size[0]//Rn, im_size[1]//Rn)
phis = reduce_spatial(phis, im_size_low)
evals = reduce_spatial(evals, im_size_low)
alphas = reduce_temporal(alphas, (alphas.shape[1]//Rm), dim=1)
trj_size_low = alphas.shape[1:]
mask = (evals > 0.99).float()

# Remove first order terms
phis, kdev, zero = remove_linear_terms(phis, alphas, mask)

B = phis.shape[0]
energy = phis.reshape(B, -1).abs().mean(1) * alphas.reshape(B, -1).abs().mean(1)
keep = torch.argwhere(energy > 1e-4)[:, 0]
phis, alphas = phis[keep], alphas[keep]
B = phis.shape[0]


# Joint 2D compression: Φ_w = V^T[:, :2], A_w = Σ U^T[:2]
phis, alphas = whiten_phis_alphas(phis, alphas,
                                  B_compressed=2)
phis_plt = phis.clone()
B = phis.shape[0]


# phis_plt, phis_mean, alphas_plt, alphas_mean = rescale_phis_alphas(phis, alphas,
#                                                                    quantiles=(0.0, 1.0),
#                                                                    mask=mask)
# # for b in range(B):
# #     alphas_plt[b] += alphas_mean[b]
# #     phis_plt[b] += phis_mean[b]

# ts = torch.arange(alphas_plt.shape[1]) * 1e-3
# plt.figure(figsize=(12, 5.6))
# for b in range(0,2):
#     for k in range(alphas_plt.shape[2]):
#         plt.plot(ts, alphas_plt[b, :, k].cpu(), color='black', alpha=0.6)
# plt.ylim(-8, 8)
# plt.show()
# quit()
# for b in range(0,1):
#     for k in range(alphas_plt.shape[2]):
#         plt.plot(ts, alphas_plt[b, :, k].cpu(), color='C0', alpha=0.6)
# for b in range(1,5):
#     for k in range(alphas_plt.shape[2]):
#         plt.plot(ts, alphas_plt[b, :, k].cpu(), color='C1', alpha=0.6)
# for b in range(5,17):
#     for k in range(alphas_plt.shape[2]):
#         plt.plot(ts, alphas_plt[b, :, k].cpu(), color='C2', alpha=0.6)
# plt.ylim(-8, 8)
# plt.show()
# quit()

# slc = (slice(None, -40), slice(30, -30))
# def proc_img(img):
#     return (img / mask)[slc].cpu()
# for b in range(2):
#     plt.figure()
#     plt.imshow(proc_img(phis_plt[b]), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
#     plt.axis('off')
#     plt.tight_layout()
# plt.show()
# quit()

# plt.figure()
# plt.imshow(proc_img(phis_plt[0]), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
# plt.axis('off')
# plt.tight_layout()

# plt.figure(figsize=(8,8))
# for b in range(1,5):
#     plt.subplot(2,2,b)
#     plt.imshow(proc_img(phis_plt[b]), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
#     plt.axis('off')
# plt.subplots_adjust(wspace=0.00, hspace=0.00)

# plt.figure(figsize=(8,8*3/4))
# for b in range(5,17):
#     plt.subplot(3,4,b - 4)
#     plt.imshow(proc_img(phis_plt[b]), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
#     plt.axis('off')
# plt.subplots_adjust(wspace=0.00, hspace=0.00)
# plt.show()
# quit()

# ------------------ Pick indices via CUR routine ------------------
# flatten
cluster_method = 'maxmin'
phis = phis.reshape(B, -1)
alphas = alphas.reshape(B, -1)

# Demean
phi0 = phis.mean(dim=1)
alpha0 = alphas.mean(dim=1)
phis_demean = phis - phi0[:, None]
alphas_demean = alphas - alpha0[:, None]

# Accumulate mean related terms
spat = phis_demean.T @ alpha0
temp = alphas_demean.T @ phi0
temp += alpha0 @ phi0
spat = torch.exp(-2j * torch.pi * spat)
temp = torch.exp(-2j * torch.pi * temp)

# Whiten, figure out auto cutoff
phis_nrm, alphas_nrm, S = whiten_phis_alphas(phis_demean, alphas_demean, 
                                                B_compressed=B, return_singular_values=True)
# phis_nrm = phis_demean * -1
# alphas_nrm = alphas_demean * -1
# S = torch.ones(B, device=phis_nrm.device)

# Cluster in Whitened space
phis_full = phis_nrm
alphas_full = (alphas_nrm.T * S).T
phi_idxs = _pivot_indices(phis_nrm.T * S, K, cluster_method, seed=seed)
alpha_idxs = _pivot_indices(alphas_nrm.T * S, K, cluster_method, seed=seed)
# ------------------------------------------------------------------

# Plotting in normalized space
phis_full, _, alphas_full, _ = rescale_phis_alphas(phis_full, alphas_full,
                                                 quantiles=(0.0, 1.0))
S = S * 0 + 1
phi_cent = phis_full.T[phi_idxs] * S 
alpha_cent = alphas_full.T[alpha_idxs]
phis_plt = phis_full.reshape((B, *im_size_low))

N = phis_nrm.shape[1]
T = alphas_nrm.shape[1]
npts_bg = 200_000
n_bg = min(npts_bg, N, T)
i_phi = torch.randperm(N, device=phis_nrm.device)[:N]
i_alpha = torch.randperm(T, device=alphas_nrm.device)[:T]

phi_bg = (phis_full.T[i_phi] * S).cpu().numpy() 
alpha_bg = alphas_full.T[i_alpha].cpu().numpy()

mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.linewidth': 1.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
panels = [
    (axes[0], phi_bg, phi_cent.cpu().numpy(), r'$\varphi(\mathbf{r})$', 'PC 1', 'PC 2'),
    (axes[1], alpha_bg, alpha_cent.cpu().numpy(), r'$\alpha(t)$', 'PC 1', 'PC 2'),
]
for ax, bg, cents, title, xlab, ylab in panels:
    ax.scatter(
        bg[:, 0], bg[:, 1], s=6, c='black',
        linewidths=0, rasterized=True, zorder=1,
        alpha=0.03
    )
    ax.scatter(
        cents[:, 0], cents[:, 1],
        c='red', s=28, edgecolors='k', linewidths=1, zorder=3,
        marker='x'
    )
    ax.axhline(0, color='k', linewidth=0.8, zorder=2)
    ax.axvline(0, color='k', linewidth=0.8, zorder=2)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal', adjustable='datalim')
    # ax.grid(True, alpha=0.25)

fig.savefig(out_png, dpi=200, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(f'Saved {out_png}')
print(f'Saved {out_pdf}')

plt.figure(figsize=(12, 5.6))
for b in range(2):
    plt.subplot(1,2, b+1)
    plt.imshow(phis_plt[b].cpu(), cmap='RdBu_r')
    plt.colorbar()
    plt.axis('off')
plt.subplots_adjust(wspace=0.00, hspace=0.00)
plt.show()
