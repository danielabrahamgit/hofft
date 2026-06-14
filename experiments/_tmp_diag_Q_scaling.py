"""How does grid_spacing (hence Q) affect sparse interp vs HOFFT factorization?"""
import torch
import numpy as np
from einops import einsum

from hofft.decomp import hofft_params, als_compressed, build_kern_bases
from hofft.spatial_init import choose_init
from hofft.sparse_decomp import sparse_alpha_segmentation, sparse_params
from hofft.phase_coeffs import trj_dev_to_phis_alphas, compress_phis_alphas, rescale_phis_alphas
from mr_recon.spatial import spatial_resize_poly

torch.manual_seed(0)
R_under, B_compressed = 3, 4
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
idxs = torch.argwhere(
    phis.reshape(B, -1).abs().mean(1) * alphas.reshape(B, -1).abs().mean(1) > 1e-6
)[:, 0]
phis, alphas = phis[idxs], alphas[idxs]
phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, hparams.os)
phis_stack = torch.cat([phis, phis_dev], dim=0)
alphas_stack = torch.cat([alphas, alphas_dev], dim=0)
phis_stack, alphas_stack = compress_phis_alphas(phis_stack, alphas_stack, B_compressed=B_compressed)
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack)

red_size = hparams.reduced_im_size
phis_red = spatial_resize_poly(phis_nrm, im_size=red_size, order=3)
kern_bases = build_kern_bases(hparams.kern_size, red_size, os=hparams.os).to(torch_dev)
R = int(np.prod(red_size))
T = int(np.prod(trj_size))
phis_flt = phis_red.reshape((phis_red.shape[0], R))
alphas_flt = alphas_nrm.reshape((alphas_nrm.shape[0], T))
t_eval = torch.randperm(T, device=torch_dev)[:2000]
truth = torch.exp(-2j * np.pi * einsum(phis_flt, alphas_flt[:, t_eval], 'B R, B T -> T R'))
spatial_init = choose_init(phis_red, alphas_nrm, hparams=hparams,
                           spatial_mask=torch.ones(red_size, dtype=torch.complex64, device=torch_dev),
                           spatial_init='seg')

print(f'{"db":>6s} {"Q":>6s} {"K":>4s}  {"s1":>8s} {"s2":>8s} {"s3":>8s}')
for db in [0.5, 0.25, 0.125, 0.0625]:
    sp = sparse_params(Q=1, K=4, interp_method='grid', grid_spacing=db, grid_width=2)
    bases, inds, coeffs = sparse_alpha_segmentation(phis_red, alphas_nrm, sp, verbose=False)
    Q, K = bases.shape[0], inds.shape[0]
    bases_flt = bases.reshape((Q, R))
    inds_flt = inds.reshape((K, T))[:, t_eval]
    coeffs_flt = coeffs.reshape((K, T))[:, t_eval]
    approx = einsum(bases_flt[inds_flt], coeffs_flt, 'K T R, K T -> T R')
    s1 = ((approx - truth).norm(dim=-1) / truth.norm(dim=-1)).mean().item()
    sf, ck = als_compressed(bases_flt.reshape((Q, *red_size)), kern_bases,
                            spatial_init.clone(), max_iter=100,
                            lamda=hparams.lamda, solver=hparams.solver, verbose=False)
    recon = einsum(sf.reshape(hparams.L, R), kern_bases.reshape(-1, R), ck,
                   'L R, W R, L W Q -> Q R')
    s2 = ((recon - bases_flt).norm(dim=-1) / bases_flt.norm(dim=-1)).mean().item()
    approx3 = einsum(recon[inds_flt], coeffs_flt, 'K T R, K T -> T R')
    s3 = ((approx3 - truth).norm(dim=-1) / truth.norm(dim=-1)).mean().item()
    print(f'{db:6.3f} {Q:6d} {K:4d}  {s1:8.3e} {s2:8.3e} {s3:8.3e}')
