import torch
import numpy as np
import matplotlib as mpl
mpl.use('webAgg')
import matplotlib.pyplot as plt

from mr_sim.phantoms import shepp_logan
from mr_sim.trj_lib import trj_lib
from mr_sim.coil_maps import surface_coil_maps

from hofft.forward_model import hofft_linop
from hofft.decomp import hofft_params
from hofft.sparse_decomp import sparse_params
from hofft.pipelines import (
    als_nufft, 
    kb_nufft, 
    als_hofft, 
    sparse_fit_hofft_lstsq_support,
    sparse_fit_hofft_smooth_interp
)

from mr_recon.linops import sense_linop
from mr_recon.spatial import spatial_resize_poly
from mr_recon.fourier import sigpy_nufft, matrix_nufft
from mr_recon.utils import gen_grd, normalize
from mr_recon.algs import density_compensation
from mr_recon.recons import CG_SENSE_recon

from einops import rearrange
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Load data
W = 3
save = True
pth = f'./paper_experiments/nufft_2d/'
dct = torch.load(pth + f'W{W}.pt')
pth += 'figs/'
os_list = dct['os_list']
# img_gt = dct['img_gt']
img_gt = dct['img_highacc']

# Pick a specific oversampling and show
os_plt = 1.185 * 0 + 1.3
os_idx = torch.argmin(torch.abs(os_list - os_plt)).item()
os_plt = os_list[os_idx]
print(f'Using OS: {os_plt}')

# Compute masked and normalized images and errors
plt_mask = 1.0 * (img_gt.abs() > 0.0)
img_gt *= plt_mask
msk_and_nrm = lambda x : normalize(x * plt_mask, img_gt, mag=False) * plt_mask
for i in range(len(os_list)):
    for key in ['imgs_nufft', 'imgs_hofft', 'imgs_hofft_mskd']:
        if key + '_err' not in dct:
            dct[key + '_err'] = torch.zeros_like(dct[key])
            dct[key + '_nrmse'] = torch.zeros(len(os_list))
        dct[key][i] = msk_and_nrm(dct[key][i])
        dct[key + '_err'][i] = (dct[key][i] - img_gt).abs()
        dct[key + '_nrmse'][i] = dct[key + '_err'][i].norm() / img_gt.norm()
dct['img_highacc'] = msk_and_nrm(dct['img_highacc'])
dct['img_highacc_err'] = (dct['img_highacc'] - img_gt).abs()
dct['img_highacc_nrmse'] = dct['img_highacc_err'].norm() / img_gt.norm()

# Plot lines
plt.figure()
for key in ['imgs_nufft', 'imgs_hofft', 'imgs_hofft_mskd']:
    plt.plot(os_list, dct[key + '_nrmse'], label=key)
plt.axhline(dct['img_highacc_nrmse'], color='red', label='High-Acc')
plt.legend()

# -------------- Show Recon Images --------------
fig, axes = plt.subplots(1, 3, figsize=(9, 4))
imgs = [dct['imgs_nufft'][os_idx], dct['imgs_hofft'][os_idx], dct['imgs_hofft_mskd'][os_idx]]
vmax = 0.4
M = 10
slc_img = (slice(None), slice(30, -30))
# slc_img = slice(None)
for k, (img, ax) in enumerate(zip(imgs, axes)):
    err = (img - img_gt).abs()
    nrmse = err.norm() / img_gt.norm()
    recon = img.abs().rot90()
    error = err.rot90() * M  # scale error so both halves share vmax
    half = recon.shape[1] // 2 + 15
    combined = recon.clone()
    combined[:, half:] = error[:, half:]
    ax.imshow(combined[slc_img], cmap='gray', vmin=0, vmax=vmax)
    ax.set_facecolor('black')
    ax.axis('off')
    ax.text(0.98, 0.02, f'NRMSE: {100 * nrmse:.1f}%', transform=ax.transAxes,
            color='white', fontsize=8, ha='right', va='bottom')
fig.subplots_adjust(wspace=0, hspace=0)
if save:
    fig.savefig(pth + f'W{W}_imgs.png', bbox_inches='tight', pad_inches=0)

# -------------- Show Kernels & Apodization --------------
kb_kern = dct['kerns_kb'][os_idx]
hft_kern = dct['kerns_hft'][os_idx]
hft_kern_mskd = dct['kerns_hft_mskd'][os_idx]
kb_apod = dct['apods_kb'][os_idx]
hft_apod = dct['apods_hft'][os_idx]
hft_apod_mskd = dct['apods_hft_mskd'][os_idx]
labels = ['NUFFT', 'HOFFT', 'HOFFT_mskd']
kbmax = kb_kern.abs().max()
kb_kern /= kbmax
kb_apod *= kbmax
hftmax = hft_kern.abs().max()
hft_kern /= hftmax
hft_apod *= hftmax
hftmax_mskd = hft_kern_mskd.abs().max()
hft_kern_mskd /= hftmax_mskd
hft_apod_mskd *= hftmax_mskd
kerns = [kb_kern.abs(), hft_kern.abs(), hft_kern_mskd.abs()]
apods = [kb_apod.abs(), hft_apod.abs(), hft_apod_mskd.abs()]
fig, axes = plt.subplots(1, 3, figsize=(9, 4))
for apod, label, ax in zip(apods, labels, axes):
    ax.imshow(apod[0].cpu().rot90(), cmap='gray',
              vmin=0, vmax=1.5)
    # ax.set_title(label)
    ax.axis('off')
fig.subplots_adjust(wspace=0, hspace=0)
if save:
    fig.savefig(pth + f'W{W}_apods.png', bbox_inches='tight', pad_inches=0)
plt.figure(facecolor='black')
colors = ['red', 'orange', 'green']
lw = 4
for apod, label, color in zip(apods, labels, colors):
    plt.plot(apod[0, :, apod.shape[-1]//2], label=label, linewidth=lw, color=color)
plt.axis('off')
plt.legend()
if save:
    plt.savefig(pth + f'W{W}_apods_line.png', bbox_inches='tight', pad_inches=0)

fig, axes = plt.subplots(1, 3, figsize=(9, 4))
for kern, label, ax in zip(kerns, labels, axes):
    kern = rearrange(kern[0].flip(dims=[0, 1]), 'Wx Wy Nx Ny -> (Wx Nx) (Wy Ny)')
    ax.imshow(kern.cpu().rot90(), cmap='gray', vmin=0, vmax=1.0)
    # ax.set_title(label)
    ax.axis('off')
fig.subplots_adjust(wspace=0, hspace=0)
if save:
    fig.savefig(pth + f'W{W}_kerns.png', bbox_inches='tight', pad_inches=0)

plt.figure(facecolor='black')
for kern, label, color in zip(kerns, labels, colors):
    kern = rearrange(kern[0].flip(dims=[-1, -2]), 'Wx Wy Nx Ny -> (Wx Nx) (Wy Ny)')
    plt.plot(kern[:, kern.shape[-1]//2], label=label, linewidth=lw, color=color)
plt.axis('off')
plt.legend()
if save:
    plt.savefig(pth + f'W{W}_kerns_line.png', bbox_inches='tight', pad_inches=0)
if not save:
    plt.show()