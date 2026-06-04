import torch 

import matplotlib as mpl
mpl.use('Webagg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from mr_recon.utils import normalize, gen_grd
from mr_recon.fourier import sigpy_nufft, mr_recon_nufft, matrix_nufft
from mr_recon.linops import batching_params, sense_linop
from mr_recon.recons import CG_SENSE_recon

from hofft.pipelines import als_nufft, als_hofft
from hofft.decomp import hofft_params
from hofft.forward_model import hofft_linop
from hofft.reduce import reduce_params
from hofft.phase_coeffs import rescale_phis_alphas, apply_phase_midpoints

# Parameters
torch_dev = torch.device(5)
max_iter = 100
max_eigen = None
cg_sense_kwargs = {'max_iter': max_iter, 'max_eigen': max_eigen, 'verbose': False}

# Load the data
fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/sim_spiral'
evals = torch.load(f'{fpath}/evals.pt', map_location=torch_dev)
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
img = torch.load(f'{fpath}/img.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
dcf = torch.load(f'{fpath}/dcf.pt', map_location=torch_dev)
im_size = img.shape
C = mps.shape[0]

# Mask
spatial_mask = 1.0 * (evals > 0.95)
mps = mps * spatial_mask

# Compare with ground truth
kwargs = {'bparams': batching_params(coil_batch_size=C), 'use_toeplitz': False}
nufft_gt = sigpy_nufft(im_size, oversamp=2.0, width=6)
# nufft_gt = matrix_nufft(im_size, spatial_batch_size=2**10)
A_gt = sense_linop(trj, mps, dcf, nufft=nufft_gt, **kwargs)
ksp_gt = A_gt(img)
img_gt = CG_SENSE_recon(A_gt, ksp_gt, **cg_sense_kwargs).cpu().rot90()

# Loop over different kernel sizes and oversampling factors
kern_widths = [1, 2, 3, 4]
os_factors = [1.1, 1.2, 1.3]
Ls = [1,]
errs_sp = torch.zeros(len(kern_widths), len(os_factors), dtype=torch.float32)
errs_hofft = torch.zeros(len(kern_widths), len(os_factors), len(Ls), dtype=torch.float32)
for k in tqdm(range(len(kern_widths)), 'Kernel widths'):
    kern_width = kern_widths[k]
    for o in range(len(os_factors)):
        os = os_factors[o]
            
        # Sigpy recon
        nufft_sp = sigpy_nufft(im_size, oversamp=os, width=kern_width)
        nufft_sp.beta = nufft_sp.optimal_beta(torch_dev=torch_dev)
        # nufft_sp = mr_recon_nufft(im_size, oversamp=os, width=kern_width)
        # nufft_sp.param = nufft_sp.opt_param(torch_dev=torch_dev)
        # nufft_sp.plan(trj[None,])
        A_sp = sense_linop(trj, mps, dcf, nufft=nufft_sp, **kwargs)
        img_sp = CG_SENSE_recon(A_sp, ksp_gt, **cg_sense_kwargs).cpu().rot90()
        img_sp = normalize(img_sp, img_gt, mag=False, ofs=True)
        errs_sp[k, o] = (img_sp - img_gt).norm() / img_gt.norm()
        
        # HOFFT recon
        for l in range(len(Ls)):
            L = Ls[l]
            hparams = hofft_params(kern_size=(kern_width,)*2, 
                                    # spatial_init='100_alphas_10',
                                    spatial_init='ones',
                                    verbose=False,
                                    # anderson_order=3,
                                    os=os, L=L)
            kern_weights, spatial_factor = als_nufft(trj, im_size, hparams,
                                                    spatial_mask=spatial_mask,
                                                    num_als_iter=1000, 
                                                    #  im_size_low=(70,)*2,
                                                    )
            trj_grd = (trj * os).round() / os
            A_hofft = hofft_linop(trj_grd, mps, kern_weights, spatial_factor, 
                                  dcf=dcf, os_grid=os, bparams=kwargs['bparams'])
            img_hofft = CG_SENSE_recon(A_hofft, ksp_gt, **cg_sense_kwargs).cpu().rot90()
            img_hofft = normalize(img_hofft, img_gt, mag=False, ofs=True)
            errs_hofft[k, o, l] = (img_hofft - img_gt).norm() / img_gt.norm()

# Show results
xticks = [f'{w}' for w in kern_widths]
yticks = [f'{o}' for o in os_factors][::-1]
zticks = [f'{L}' for L in Ls]
errs = [errs_sp.cpu().rot90() * 100, errs_hofft.cpu().rot90() * 100]
titles = ['Sigpy', 'HOFFT']
plt.figure(figsize=(14, 7))
for i in range(len(errs)):
    plt.subplot(1, 2, i+1)
    plt.imshow(errs[i], vmin=0, vmax=10, cmap='plasma')
    plt.xticks(range(len(xticks)), xticks)
    plt.yticks(range(len(yticks)), yticks)
    plt.colorbar()
    plt.title(titles[i])
plt.tight_layout()
plt.show()
quit()

# Show results
plt.figure(figsize=(14, 7))
imgs = [img_gt, img_sp, img_hofft]
titles = ['Ground Truth', 'Sigpy', 'HOFFT']
vmax = 0.5
for i in range(len(imgs)):
    # image
    plt.subplot(2, 3, i+1)
    plt.imshow(imgs[i].abs(), cmap='gray', vmin=0, vmax=vmax)
    plt.axis('off')
    plt.title(titles[i])
    
    # error
    nrmse = (imgs[i] - img_gt).norm() / img_gt.norm()
    M = 10
    plt.subplot(2, 3, i+4)
    plt.title(f'Error({M}x), NRMSE = {100*nrmse:.2f}%')
    plt.imshow((imgs[i].abs() - img_gt.abs()).abs(), cmap='gray', vmin=0, vmax=vmax/M)
    plt.axis('off')
    
    
plt.tight_layout()
plt.show()