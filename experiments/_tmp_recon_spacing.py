"""Recon NRMSE vs standard HOFFT for different grid_spacing values."""
import torch
from hofft.reduce import reduce_params
from hofft.forward_model import hofft_compressed_linop, hofft_linop
from hofft.sparse_decomp import sparse_params
from hofft.decomp import hofft_params
from hofft.pipelines import als_hofft_compressed, als_hofft
from hofft.phase_coeffs import trj_dev_to_phis_alphas, compress_phis_alphas, rescale_phis_alphas, apply_phase_midpoints
from mr_recon.recons import CG_SENSE_recon
from mr_recon.linops import batching_params

torch_dev = torch.device(1)
R, B_compressed = 3, 4
hparams = hofft_params(kern_size=(5,)*2, os=1.25, L=5, reduced_im_size=(120,120),
                       spatial_init='seg', verbose=False)
common = dict(temporal_batch_size=2**10, lamda=0.0, normalize_coeffs=True)

fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_spiral'
ksp = torch.load(f'{fpath}/ksp.pt', map_location=torch_dev)
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
dcf = torch.load(f'{fpath}/dcf.pt', map_location=torch_dev)
alphas = torch.load(f'{fpath}/alphas.pt', map_location=torch_dev)
phis = torch.load(f'{fpath}/phis.pt', map_location=torch_dev)
im_size = mps.shape[1:]
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0]
bparams = batching_params(coil_batch_size=mps.shape[0])
trj = trj[..., ::R, :].type(torch.float32)
dcf, ksp = dcf[..., ::R].type(torch.float32), ksp[..., ::R].type(torch.complex64)

B = phis.shape[0]
idxs = torch.argwhere(phis.reshape(B,-1).abs().mean(1)*alphas.reshape(B,-1).abs().mean(1)>1e-6)[:,0]
phis, alphas = phis[idxs], alphas[idxs]
pd, ad = trj_dev_to_phis_alphas(trj, im_size, hparams.os)
ps = torch.cat([phis, pd]); as_ = torch.cat([alphas, ad])
ps, as_ = compress_phis_alphas(ps, as_, B_compressed=B_compressed)
pn, pm, an, am = rescale_phis_alphas(ps, as_)
trj_grd = (trj * hparams.os).round() / hparams.os

sf, kw = als_hofft(pn, an, hparams=hparams, rparams=reduce_params(spatial_reduce_size=hparams.reduced_im_size), num_als_iter=100)
sf, kw = apply_phase_midpoints(pn, an, pm, am, sf, kw)
img_ref = CG_SENSE_recon(hofft_linop(trj_grd, mps, kw, sf, dcf, os_grid=hparams.os, bparams=bparams),
                         ksp, max_iter=10, max_eigen=1.0, verbose=False).cpu()

def nrmse(sparams):
    sf, ck, si, sc = als_hofft_compressed(pn, an, hparams=hparams, sparams=sparams, num_als_iter=100)
    sf, sc = apply_phase_midpoints(pn, an, pm, am, sf, sc)
    A = hofft_compressed_linop(trj_grd, mps, dcf=dcf, compressed_kernels=ck,
                               sparse_idxs=si, sparse_coeffs=sc, spatial_factors=sf,
                               os_grid=hparams.os, bparams=bparams)
    img = CG_SENSE_recon(A, ksp, max_iter=10, max_eigen=1.0, verbose=False).cpu()
    return (img - img_ref).norm() / img_ref.norm(), ck.shape[-1], si.shape[0]

configs = [
    ('lstsq', sparse_params(Q=200, K=5, beta_method='maxmin', interp_method='lstsq', **common)),
]
for db in [0.5, 0.25, 0.125]:
    configs.append((f'grid db={db}',
                    sparse_params(interp_method='grid', grid_spacing=db, grid_width=2,
                                  **common)))

for label, sp in configs:
    e, Q, K = nrmse(sp)
    print(f'{label:24s}  Q={Q:5d} K={K:3d}  NRMSE={e:.4f}')
