import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter
from einops import einsum

from mr_recon.utils import gen_grd, pick_K_vectors
from mr_recon.spatial import spatial_resize_poly
from mr_recon.imperfections.field import alpha_segementation


# ---------------------------------------------------------------------------
# Spatial bases (already normalized so |alpha| = wraps at the peak of phi)
# ---------------------------------------------------------------------------
slc = 0
fpath = '/local_mount/space/tiger/1/users/abrahamd/mr_data/congyu_spiral_2d/data'
b0 = np.load(f'{fpath}/b0.npy')[..., slc]
mps = np.load(f'{fpath}/mps.npy')[..., slc]
mask = 1.0 * (np.mean(np.abs(mps), axis=0) > 1e-6)

b0 = gaussian_filter(b0, 0.5)
b0 = torch.from_numpy(b0).type(torch.float32)
b0 = spatial_resize_poly(b0, mps.shape[1:], order=3).numpy()
phi0 = b0 * mask
phi0 /= np.abs(phi0).max()
crds = gen_grd(mps.shape[1:]).numpy()
phi1 = (crds[..., 0] ** 2 + crds[..., 1] ** 2) * mask
phi1 /= np.abs(phi1).max()

# Sinusoidal eddy + linear B0
t = np.linspace(0, 1, 100)
f = 2
alpha0 = t * 2
alpha1 = np.sin(2 * np.pi * t * f)
alphas = np.stack([alpha0, alpha1], axis=0)
phis = np.stack([phi0, phi1], axis=0)

# Cluster alpha space -> beta vectors
alphas = torch.from_numpy(alphas).type(torch.float32)
phis = torch.from_numpy(phis).type(torch.float32)
betas, _ = pick_K_vectors(alphas.T, K=4, method='maxmin')
spatial_funcs = torch.exp(-2j * torch.pi * einsum(phis, betas,
                                                  'B ..., K B -> K ...'))

# Alpha segmentation
L = 4
_, temporal_funcs_zero, _ = alpha_segementation(
    phis, alphas, L=L, interp_type='zero', use_type3=True,
    manual_spatial_funcs=spatial_funcs,
)
_, temporal_funcs_lstsq, _ = alpha_segementation(
    phis, alphas, L=L, interp_type='lstsq', use_type3=True,
    manual_spatial_funcs=spatial_funcs,
)

# Ground truth and estimated phase factors
phz_gt = torch.exp(-2j * torch.pi * einsum(phis, alphas, 'B ..., B T -> T ...'))
phz_est_zero = einsum(spatial_funcs, temporal_funcs_zero, 'L ..., L T -> T ...')
phz_est_lstsq = einsum(spatial_funcs, temporal_funcs_lstsq, 'L ..., L T -> T ...')

# Numpy views for plotting
a0 = alphas[0].numpy()
a1 = alphas[1].numpy()
b0_vec = betas[:, 0].numpy()
b1_vec = betas[:, 1].numpy()
mask_np = mask.astype(bool)


def phase_img(phz_t):
    ang = np.angle(phz_t.numpy())
    ang[~mask_np] = np.nan
    return ang


n_frames = a0.shape[0]
phz_gt_imgs = [phase_img(phz_gt[i]) for i in range(n_frames)]
phz_zero_imgs = [phase_img(phz_est_zero[i]) for i in range(n_frames)]
phz_lstsq_imgs = [phase_img(phz_est_lstsq[i]) for i in range(n_frames)]


# ---------------------------------------------------------------------------
# Figure: alpha traj | GT | zero-order | lstsq
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
ax_trj, ax_gt, ax_zero, ax_lstsq = axes

# 1) Clean alpha-space trajectory + static betas
ax_trj.plot(a0, a1, color='gray', lw=1.8, alpha=0.55, zorder=1)
ax_trj.plot(
    b0_vec, b1_vec, 'x', color='green', ms=9, mew=2.0, zorder=3,
)
point, = ax_trj.plot([], [], 'o', color='black', ms=9, zorder=4, animated=True)
trail, = ax_trj.plot([], [], color='black', lw=2.0, alpha=0.9, zorder=2, animated=True)
pad = 0.12 * max(a0.max() - a0.min(), a1.max() - a1.min(), 1e-6)
ax_trj.set_xlim(min(a0.min(), b0_vec.min()) - pad, max(a0.max(), b0_vec.max()) + pad)
ax_trj.set_ylim(min(a1.min(), b1_vec.min()) - pad, max(a1.max(), b1_vec.max()) + pad)
ax_trj.set_aspect('equal')
ax_trj.axis('off')

# 2–4) Phase maps
im_kwargs = dict(cmap='jet', vmin=-np.pi, vmax=np.pi, animated=True)
im_gt = ax_gt.imshow(phz_gt_imgs[0], **im_kwargs)
im_zero = ax_zero.imshow(phz_zero_imgs[0], **im_kwargs)
im_lstsq = ax_lstsq.imshow(phz_lstsq_imgs[0], **im_kwargs)

for ax, title in [
    (ax_gt, 'ground truth'),
    (ax_zero, r'zero-order ($L=%d$)' % L),
    (ax_lstsq, r'least squares ($L=%d$)' % L),
]:
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    ax.set_aspect('equal')

fig.tight_layout()
fig.canvas.draw()
background = fig.canvas.copy_from_bbox(fig.bbox)

frames = []
for i in range(n_frames):
    fig.canvas.restore_region(background)
    point.set_data([a0[i]], [a1[i]])
    trail.set_data(a0[: i + 1], a1[: i + 1])
    im_gt.set_data(phz_gt_imgs[i])
    im_zero.set_data(phz_zero_imgs[i])
    im_lstsq.set_data(phz_lstsq_imgs[i])
    ax_trj.draw_artist(trail)
    ax_trj.draw_artist(point)
    ax_gt.draw_artist(im_gt)
    ax_zero.draw_artist(im_zero)
    ax_lstsq.draw_artist(im_lstsq)
    fig.canvas.blit(fig.bbox)
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

out_path = 'alpha_seg_demo.gif'
duration_ms = 70
imageio.mimsave(out_path, frames, duration=duration_ms, loop=0)
print(f'Saved {out_path}  ({len(frames)} frames)')

# Save spatial functions
for i in range(spatial_funcs.shape[0]):
    plt.figure()
    plt.axis('off')
    plt.imshow(spatial_funcs[i].angle() / mask, vmin=-np.pi, vmax=np.pi, cmap='jet')
    plt.savefig(f'spatial_func_{i}.png', bbox_inches='tight', pad_inches=0)
