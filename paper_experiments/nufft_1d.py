import torch

import matplotlib as mpl
mpl.use('webAgg')
import matplotlib.pyplot as plt

from mr_sim.phantoms import shepp_logan
from mr_sim.trj_lib import trj_lib

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

# Params
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 1024
L = 1
os = 1.25
W = 4
d = 1
num_als_iter = 100
im_size = (N,)*d
kern_size = (W,)*d
mask = False

# Build test signal
img_gt = shepp_logan(torch_dev).img(im_size)
trj = torch.linspace(-N/8, N/8, N*5, device=torch_dev)[:, None]
trj_size = trj.shape[:-1]
dcf = density_compensation(trj, im_size)

# Mask
if mask:
    mask = 1.0 * (img_gt.abs() > 0.0)
else:
    mask = torch.ones_like(img_gt)

# GT
nft_gt = matrix_nufft(im_size, spatial_batch_size=2**10)
ksp_gt = nft_gt.forward(img_gt[None,], trj[None,])

# -------------- Build High accuracy linop --------------
nft_gt = sigpy_nufft(im_size, oversamp=2.0, width=6)
mps = img_gt[None, :] * 0 + 1
Agt = sense_linop(trj, mps, dcf, nufft=nft_gt)

# -------------- Build NUFFT linop --------------
# Get optimal beta
nft = sigpy_nufft(im_size, oversamp=os, width=W)
nft.beta = nft.optimal_beta()

# Get HOFFT style tensors using kb nufft funcs
kb_apod, kb_weights = kb_nufft(trj, im_size, kern_size, os, nft.beta)

# Build linop
trj_grd = (trj * os).round() / os
Anufft = hofft_linop(trj_grd, mps, kb_weights, kb_apod, 
                     dcf=dcf,
                     os_grid=os)

# -------------- Build HOFFT linop --------------
# HOFFT setup
hparams = hofft_params(kern_size, os, L=L,
                       solver='pinv',
                       lamda=0.0,
                       spatial_init='ones')
phis = gen_grd(im_size).moveaxis(-1, 0).to(torch_dev)
alphas = (trj - trj_grd).moveaxis(-1, 0)
im_size_ds = (180,)*d
phis_ds = spatial_resize_poly(phis, im_size_ds, order=3)
mask_ds = spatial_resize_poly(mask, im_size_ds, order=3)

# HOFFT Decomp
def hofft_decomp(phis, alphas, num_als_iter, spatial_mask=None, htype='smooth'):
    # Full HOFFT model
    if htype == 'full':
        spatial_factor, kern_weights = als_hofft(phis, alphas, 
                                                spatial_mask=spatial_mask,
                                                hparams=hparams, 
                                                num_als_iter=num_als_iter)
        return spatial_factor, kern_weights
    
    # Sparse HOFFT model
    sparams = sparse_params(Q=500, S=10)
    if htype == 'lstsq':
        ret = sparse_fit_hofft_lstsq_support(phis, alphas, 
                                         spatial_mask=spatial_mask,
                                         hparams=hparams, 
                                         sparams=sparams, 
                                         ls_lamda=1e-3,
                                         num_als_iter=num_als_iter*10)
    elif htype == 'smooth':
        ret = sparse_fit_hofft_smooth_interp(phis, alphas, 
                                            spatial_mask=spatial_mask,
                                            hparams=hparams, 
                                            sparams=sparams, 
                                            kernel='rbf',
                                            #  kernel='inv_dist',
                                            auto_tune=True,
                                            num_val=30,
                                            num_als_iter=num_als_iter*10)
        
    # Convert sparse to full
    spatial_factor, compressed_kernels, sparse_inds, sparse_coeffs = ret
    kern_weights = torch.zeros((*compressed_kernels.shape[:-1], *alphas.shape[1:]),
                               device=torch_dev, dtype=torch.complex64)
    for s in range(sparse_inds.shape[0]):
        kern_weights += compressed_kernels[..., sparse_inds[s]] * sparse_coeffs[s]
    
    return spatial_factor, kern_weights
spatial_factor, kern_weights = hofft_decomp(phis_ds, alphas, num_als_iter, mask_ds)
spatial_factor = spatial_resize_poly(spatial_factor, im_size, order=3)

# Build HOFFT linop
mps = img_gt[None, :] * 0 + 1
Ahofft = hofft_linop(trj_grd, mps, kern_weights, spatial_factor, 
                     dcf=dcf,
                     os_grid=os)

# -------------- Extract Kernel and Apod funcs --------------
trj_dev = gen_grd((50,)*d).to(torch_dev) / os
kb_apod, kb_kern = kb_nufft(trj_dev, im_size, kern_size, os, nft.beta)
hft_apod, hft_kern = hofft_decomp(phis_ds, trj_dev.moveaxis(-1, 0), num_als_iter, mask_ds)
hft_apod = spatial_resize_poly(hft_apod, im_size, order=3)

# -------------- Recon --------------
kwargs = {'max_iter': 10, 'max_eigen': 1.0}
mask = mask.cpu()
# img_gt = CG_SENSE_recon(Agt, ksp_gt, **kwargs).cpu() * mask
img_gt /= img_gt.abs().max()
img_nufft = CG_SENSE_recon(Anufft, ksp_gt, **kwargs).cpu() * mask
img_hofft = CG_SENSE_recon(Ahofft, ksp_gt, **kwargs).cpu() * mask
img_gt = img_gt.cpu() * mask

# -------------- Show Recon Images --------------
plt.figure(figsize=(14, 7))
imgs = [img_gt, img_nufft, img_hofft]
labels = ['GT', 'NUFFT', 'HOFFT']
vmax = 0.4
M = 5
for k, img in enumerate(imgs):
    img = normalize(img, img_gt, mag=False) * mask
    err = (img - img_gt).abs()
    nrmse = err.norm() / img_gt.norm()
    plt.subplot(2, len(imgs), k+1)
    plt.imshow(img.abs().rot90(), cmap='gray', vmin=0, vmax=vmax)
    plt.axis('off')
    plt.title(labels[k])
    plt.subplot(2, len(imgs), k+1  + len(imgs))
    plt.imshow(err.rot90(), cmap='gray', vmin=0, vmax=vmax / M)
    plt.axis('off')
    plt.title(labels[k] + f' (NRMSE: {100 * nrmse:.2f}%, Max: {100 * err.max():.2f}%)')
plt.tight_layout()

# -------------- Show Kernels & Apodization --------------
labels = ['KB Magnitude', 'KB Phase', 'HOFFT Magnitude', 'HOFFT Phase']
kerns = [kb_kern.abs(), kb_kern.angle(), hft_kern.abs(), hft_kern.angle()]
apods = [kb_apod.abs(), kb_apod.angle(), hft_apod.abs(), hft_apod.angle()]
for apod, label in zip(apods, labels):
    plt.figure()
    plt.title('Apodization ' + label)
    if d == 1:
        plt.plot(apod[0].cpu(), label=label)
    elif d == 2:
        cmap = 'gray' if 'mag' in label.lower() else 'jet'
        plt.imshow(apod[0].cpu().rot90(), cmap=cmap)
        plt.axis('off')
    plt.tight_layout()
for kern, label in zip(kerns, labels):
    plt.figure()
    plt.title('Kernel' + label)
    if d == 1:
        kern = rearrange(kern[0].flip(dims=[-1]), 'W N -> (W N)')
        plt.plot(kern.cpu())
    elif d == 2:
        kern = rearrange(kern[0].flip(dims=[-1, -2]), 'Wx Wy Nx Ny -> (Wx Nx) (Wy Ny)')
        cmap = 'gray' if 'mag' in label.lower() else 'jet'
        plt.imshow(kern.cpu().rot90(), cmap=cmap)
        plt.axis('off')
    plt.tight_layout()
plt.show()