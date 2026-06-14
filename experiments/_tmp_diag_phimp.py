"""Check if passing phi_mp into grid 1D solves hurts accuracy."""
import torch
import numpy as np
from einops import einsum

from hofft.sparse_decomp import _structured_grid_interp
from hofft.phase_coeffs import trj_dev_to_phis_alphas, compress_phis_alphas, rescale_phis_alphas
from mr_recon.spatial import spatial_resize_poly

torch.manual_seed(0)
torch_dev = torch.device(1)
R_under, B_compressed = 3, 4
red_size = (120, 120)

fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_spiral'
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
alphas = torch.load(f'{fpath}/alphas.pt', map_location=torch_dev)
phis = torch.load(f'{fpath}/phis.pt', map_location=torch_dev)
im_size = mps.shape[1:]
os = 2 * round(1.25 * im_size[0] / 2) / im_size[0]
trj = trj[..., ::R_under, :].type(torch.float32)

B = phis.shape[0]
idxs = torch.argwhere(
    phis.reshape(B, -1).abs().mean(1) * alphas.reshape(B, -1).abs().mean(1) > 1e-6
)[:, 0]
phis, alphas = phis[idxs], alphas[idxs]
phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, os)
phis_stack = torch.cat([phis, phis_dev], dim=0)
alphas_stack = torch.cat([alphas, alphas_dev], dim=0)
phis_stack, alphas_stack = compress_phis_alphas(phis_stack, alphas_stack, B_compressed=B_compressed)
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack)
phis_red = spatial_resize_poly(phis_nrm, im_size=red_size, order=3)

R = int(np.prod(red_size))
T = int(np.prod(alphas_nrm.shape[1:]))
phis_flt = phis_red.reshape((phis_red.shape[0], R))
alphas_flt = alphas_nrm.reshape((alphas_nrm.shape[0], T))
midpoint_factor = torch.exp(-2j * np.pi * einsum(phis_flt, alphas_mp, 'B R, B -> R'))

t_eval = torch.randperm(T, device=torch_dev)[:2000]
truth = torch.exp(-2j * np.pi * einsum(phis_flt, alphas_flt + alphas_mp[:, None],
                                         'B R, B T -> T R'))

kw = dict(grid_spacing=0.25, grid_width=2, num_apod_iter=5, num_dalpha=128, lamda=0.0, verbose=False)

for label, phis_in in [
    ('phis_nrm only', phis_flt),
    ('phis_nrm + phi_mp (current)', phis_flt + phis_mp[:, None]),
]:
    betas, apod, sparse_inds, coeffs = _structured_grid_interp(phis_in, alphas_flt, **kw)
    Q = betas.shape[0]
    bases = torch.exp(-2j * np.pi * einsum(phis_flt, betas, 'B R, Q B -> Q R'))
    bases = bases * (apod * midpoint_factor)
    approx = einsum(bases[sparse_inds[t_eval]], coeffs[t_eval], 'Te K R, Te K -> Te R')
    err = (approx - truth[t_eval]).norm(dim=-1) / truth[t_eval].norm(dim=-1)
    print(f'{label:30s}  mean {err.mean():.3e}  max {err.max():.3e}  Q={Q}')

print(f'phi_mp values: {phis_mp.cpu().numpy()}')
print(f'alphas_mp values: {alphas_mp.cpu().numpy()}')
