"""
Staged diagnostic comparing 'lstsq' vs 'grid' sparse decompositions on the
coco_spiral data, replicating test_sparse_seg.py preprocessing.

Stage 1: phase error of the sparse interpolation alone
Stage 2: HOFFT (als_compressed) factorization error of the spatial bases
Stage 3: end-to-end phase error of the compressed HOFFT model
"""
import torch
import numpy as np
from einops import einsum

from hofft.decomp import hofft_params, als_compressed, build_kern_bases
from hofft.spatial_init import choose_init
from hofft.sparse_decomp import sparse_alpha_segmentation, sparse_params
from hofft.phase_coeffs import trj_dev_to_phis_alphas, compress_phis_alphas, rescale_phis_alphas
from mr_recon.spatial import spatial_resize_poly

torch.manual_seed(0)
R_under = 3
B_compressed = 4
torch_dev = torch.device(1)
hparams = hofft_params(kern_size=(5,)*2, os=1.25, L=5, reduced_im_size=(120,120),
                       spatial_init='seg', verbose=False)

fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_spiral'
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
alphas = torch.load(f'{fpath}/alphas.pt', map_location=torch_dev)
phis = torch.load(f'{fpath}/phis.pt', map_location=torch_dev)
im_size = mps.shape[1:]
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0]

trj = trj[..., ::R_under, :].type(torch.float32)
trj_size = trj.shape[:-1]

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

# Reduce spatial size like als_hofft_compressed does
red_size = hparams.reduced_im_size
phis_red = spatial_resize_poly(phis_nrm, im_size=red_size, order=3)
kern_bases = build_kern_bases(hparams.kern_size, red_size, os=hparams.os).to(torch_dev)
B = phis_red.shape[0]
R = int(np.prod(red_size))
T = int(np.prod(trj_size))
phis_flt = phis_red.reshape((B, R))
alphas_flt = alphas_nrm.reshape((B, T))

# Random time subsample for error evals
t_eval = torch.randperm(T, device=torch_dev)[:2000]
truth = torch.exp(-2j * np.pi * einsum(phis_flt, alphas_flt[:, t_eval], 'B R, B T -> T R'))

spatial_init = choose_init(phis_red, alphas_nrm, hparams=hparams,
                           spatial_mask=torch.ones(red_size, dtype=torch.complex64, device=torch_dev),
                           spatial_init='seg')

configs = [
    ('lstsq Q=200 K=5', sparse_params(Q=200, K=5, interp_method='lstsq')),
    ('grid db=.25', sparse_params(Q=1, K=4, interp_method='grid', grid_spacing=0.25, grid_width=2)),
]

for label, sp in configs:
    sp.temporal_batch_size = 2 ** 10
    bases, inds, coeffs = sparse_alpha_segmentation(phis_red, alphas_nrm, sp, verbose=False)
    Q, K = bases.shape[0], inds.shape[0]
    bases_flt = bases.reshape((Q, R))
    inds_flt = inds.reshape((K, T))[:, t_eval]
    coeffs_flt = coeffs.reshape((K, T))[:, t_eval]

    # Stage 1: sparse interpolation error
    approx = einsum(bases_flt[inds_flt], coeffs_flt, 'K T R, K T -> T R')
    err1 = ((approx - truth).norm(dim=-1) / truth.norm(dim=-1))

    # apodization stats
    mag = bases_flt.abs().mean(dim=0)
    print(f'{label:22s} Q={Q:5d} K={K:3d} | stage1 err mean {err1.mean():.2e} '
          f'max {err1.max():.2e} | |bases| min {mag.min():.2f} max {mag.max():.2f}')

    # Stage 2: HOFFT factorization of the spatial bases
    sf, ck = als_compressed(spatial_bases=bases_flt.reshape((Q, *red_size)),
                            kern_bases=kern_bases,
                            spatial_factors_init=spatial_init.clone(),
                            spatial_batch_size=None, mask=None,
                            max_iter=100, lamda=hparams.lamda,
                            solver=hparams.solver, verbose=False)
    # reconstruct bases: bases_q(r) ~= sum_l sf_l(r) * sum_w kern_w(r) ck[l,w,q]
    sf_flt = sf.reshape((hparams.L, R))
    kb_flt = kern_bases.reshape((-1, R))
    recon = einsum(sf_flt, kb_flt, ck, 'L R, W R, L W Q -> Q R')
    err2_q = (recon - bases_flt).norm(dim=-1) / bases_flt.norm(dim=-1)

    # Stage 3: end-to-end
    approx3 = einsum(recon[inds_flt], coeffs_flt, 'K T R, K T -> T R')
    err3 = ((approx3 - truth).norm(dim=-1) / truth.norm(dim=-1))
    print(f'{"":22s} stage2 per-basis err mean {err2_q.mean():.2e} max {err2_q.max():.2e} | '
          f'stage3 err mean {err3.mean():.2e} max {err3.max():.2e}')
