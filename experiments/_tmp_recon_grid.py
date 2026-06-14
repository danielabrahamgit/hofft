"""
Replicates test_sparse_seg.py recons for 'lstsq' vs 'grid' sparse decomps and
saves images + error maps for comparison.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hofft.reduce import reduce_params
from hofft.forward_model import hofft_compressed_linop, hofft_linop
from hofft.sparse_decomp import sparse_params
from hofft.decomp import hofft_params
from hofft.pipelines import als_hofft_compressed, als_hofft
from hofft.phase_coeffs import (
  trj_dev_to_phis_alphas,
  compress_phis_alphas,
  rescale_phis_alphas,
  apply_phase_midpoints,
)
from mr_recon.recons import CG_SENSE_recon
from mr_recon.linops import batching_params

torch.manual_seed(0)
R = 3
B_compressed = 4
torch_dev = torch.device(1)
hparams = hofft_params(kern_size=(5,)*2, os=1.25, L=5, reduced_im_size=(120,120),
                       spatial_init='seg', verbose=False)

fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_spiral'
ksp = torch.load(f'{fpath}/ksp.pt', map_location=torch_dev)
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
dcf = torch.load(f'{fpath}/dcf.pt', map_location=torch_dev)
alphas = torch.load(f'{fpath}/alphas.pt', map_location=torch_dev)
phis = torch.load(f'{fpath}/phis.pt', map_location=torch_dev)
im_size = mps.shape[1:]
C = mps.shape[0]
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0]
bparams = batching_params(coil_batch_size=C)

trj = trj[..., ::R, :].type(torch.float32)
dcf = dcf[..., ::R].type(torch.float32)
ksp = ksp[..., ::R].type(torch.complex64)

B = phis.shape[0]
phi_energy = phis.reshape((B, -1)).abs().mean(dim=1)
alpha_energy = alphas.reshape((B, -1)).abs().mean(dim=1)
idxs = torch.argwhere(phi_energy * alpha_energy > 1e-6)[:, 0]
phis, alphas = phis[idxs], alphas[idxs]

phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, hparams.os)
phis_stack = torch.cat([phis, phis_dev], dim=0)
alphas_stack = torch.cat([alphas, alphas_dev], dim=0)
phis_stack, alphas_stack = compress_phis_alphas(phis_stack, alphas_stack, B_compressed=B_compressed)
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack)

# Reference: standard HOFFT
rets = als_hofft(phis_nrm, alphas_nrm, hparams=hparams,
                 rparams=reduce_params(spatial_reduce_size=hparams.reduced_im_size,),
                 num_als_iter=100)
sf_h, kw_h = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp, *rets)
trj_grd = (trj * hparams.os).round() / hparams.os
A_hofft = hofft_linop(trj_grd, mps, kw_h, sf_h, dcf, os_grid=hparams.os, bparams=bparams)
img_ref = CG_SENSE_recon(A_hofft, ksp, max_iter=10, max_eigen=1.0, verbose=False).cpu()


def run_comp(label, sparams):
    rets = als_hofft_compressed(phis_nrm, alphas_nrm, hparams=hparams, sparams=sparams,
                                num_als_iter=100)
    sf, ck, si, sc = rets
    sf, sc = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp, sf, sc)
    A = hofft_compressed_linop(trj_grd, mps, dcf=dcf, compressed_kernels=ck,
                               sparse_idxs=si, sparse_coeffs=sc,
                               spatial_factors=sf, os_grid=hparams.os, bparams=bparams)
    img = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0, verbose=False).cpu()
    nrmse = (img - img_ref).norm() / img_ref.norm()
    print(f'{label:24s} Q={ck.shape[-1]:5d} K={si.shape[0]:3d} NRMSE vs hofft: {nrmse:.4f}')
    return img


common = dict(temporal_batch_size=2**10, lamda=0.0, normalize_coeffs=True)
imgs = {'standard hofft': img_ref}
imgs['lstsq'] = run_comp('lstsq', sparse_params(Q=200, K=5, beta_method='maxmin',
                                                interp_method='lstsq', **common))
imgs['grid'] = run_comp('grid', sparse_params(interp_method='grid', grid_spacing=0.25,
                                            grid_width=2, **common))

# No spatial reduction: isolates the 120 -> full-res expansion step
hparams.reduced_im_size = None
imgs['lstsq full-res'] = run_comp('lstsq full-res', sparse_params(Q=200, K=5, beta_method='maxmin',
                                                                  interp_method='lstsq', **common))
imgs['grid full-res'] = run_comp('grid full-res', sparse_params(interp_method='grid', grid_spacing=0.25,
                                                                grid_width=2, **common))

fig, axs = plt.subplots(2, len(imgs), figsize=(5*len(imgs), 10))
vmax = imgs['standard hofft'].abs().max().item()
for i, (k, v) in enumerate(imgs.items()):
    axs[0, i].imshow(v.abs().rot90(), cmap='gray', vmin=0, vmax=vmax)
    axs[0, i].set_title(k); axs[0, i].axis('off')
    diff = (v - imgs['standard hofft']).abs().rot90()
    axs[1, i].imshow(diff, cmap='gray', vmin=0, vmax=vmax*0.1)
    axs[1, i].set_title(f'{k} - ref (10x)'); axs[1, i].axis('off')
plt.tight_layout()
plt.savefig('experiments/_tmp_recon_grid.png', dpi=120)
print('saved experiments/_tmp_recon_grid.png')
