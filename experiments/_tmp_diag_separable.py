"""
Isolate the separable (tensor-product) approximation in the grid method.

For the SAME stencil indices/betas that grid selects, compare:
  - grid separable coeffs (current implementation)
  - joint lstsq coeffs over the full K=16 stencil (coupled across B dims)
"""
import torch
import numpy as np
from einops import einsum
from tqdm import tqdm

from hofft.sparse_decomp import sparse_alpha_segmentation, sparse_params, _structured_grid_interp
from hofft.phase_coeffs import trj_dev_to_phis_alphas, compress_phis_alphas, rescale_phis_alphas
from mr_recon.spatial import spatial_resize_poly
from mr_recon.dtypes import complex_dtype

torch.manual_seed(0)
R_under = 3
B_compressed = 4
torch_dev = torch.device(1)
red_size = (120, 120)

fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_spiral'
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
alphas = torch.load(f'{fpath}/alphas.pt', map_location=torch_dev)
phis = torch.load(f'{fpath}/phis.pt', map_location=torch_dev)
im_size = mps.shape[1:]
os = 2 * round(1.25 * im_size[0] / 2) / im_size[0]

trj = trj[..., ::R_under, :].type(torch.float32)
trj_size = trj.shape[:-1]

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
B = phis_red.shape[0]
R = int(np.prod(red_size))
T = int(np.prod(trj_size))
phis_flt = phis_red.reshape((B, R))
alphas_flt = alphas_nrm.reshape((B, T))

# Full target phase (re-add midpoint like lstsq path)
truth = torch.exp(-2j * np.pi * einsum(phis_flt, alphas_flt + alphas_mp[:, None],
                                         'B R, B T -> T R'))

# Grid decomposition
betas, apod, sparse_inds, grid_coeffs = _structured_grid_interp(
    phis_flt, alphas_flt, grid_spacing=0.25, grid_width=2,
    num_apod_iter=5, num_dalpha=128, lamda=0.0, verbose=False)
Q, K = betas.shape[0], sparse_inds.shape[1]
midpoint_factor = torch.exp(-2j * np.pi * einsum(phis_flt, alphas_mp, 'B R, B -> R'))
spatial_bases = torch.exp(-2j * np.pi * einsum(phis_flt, betas, 'B R, Q B -> Q R'))
spatial_bases = spatial_bases * (apod * midpoint_factor)

t_eval = torch.randperm(T, device=torch_dev)[:2000]
inds = sparse_inds[t_eval]  # Te K
gc = grid_coeffs[t_eval]    # Te K
assert inds.max() < Q, f'index OOB: max={inds.max()} Q={Q}'

selected = spatial_bases[inds]  # Te K R
approx_grid = einsum(selected, gc, 'Te K R, Te K -> Te R')
truth_eval = truth[t_eval]
err_grid = (approx_grid - truth_eval).norm(dim=-1) / truth_eval.norm(dim=-1)

# Joint lstsq on the SAME stencil indices, batched over time
lamda_I = 0.0
BHB = einsum(spatial_bases.conj(), spatial_bases, 'Q1 R, Q2 R -> Q1 Q2')
approx_joint = torch.zeros((len(t_eval), R), dtype=complex_dtype, device=torch_dev)
batch = 256
for t1 in tqdm(range(0, len(t_eval), batch), desc='joint lstsq'):
    sl = slice(t1, min(t1 + batch, len(t_eval)))
    Tb = sl.stop - sl.start
    target = truth[t_eval[sl]]
    A = spatial_bases[inds[sl]]  # Tb K R
    AHA = BHB[inds[sl, :, None], inds[sl, None, :]]
    AHb = einsum(A.conj(), target, 'Tb K R, Tb R -> Tb K')
    c = torch.linalg.solve(AHA + lamda_I, AHb[..., None])[..., 0]
    approx_joint[sl] = einsum(A, c, 'Tb K R, Tb K -> Tb R')

err_joint = (approx_joint - truth_eval).norm(dim=-1) / truth_eval.norm(dim=-1)

print(f'Q={Q} K={K} B={B}')
print(f'grid separable:  mean {err_grid.mean():.3e}  max {err_grid.max():.3e}')
print(f'joint lstsq:     mean {err_joint.mean():.3e}  max {err_joint.max():.3e}')
print(f'improvement factor (mean): {(err_grid.mean() / err_joint.mean()).item():.2f}x')

# Also compare lstsq unstructured baseline
sp = sparse_params(Q=200, K=5, interp_method='lstsq', temporal_batch_size=2**10)
bases, inds_ls, coeffs_ls = sparse_alpha_segmentation(phis_red, alphas_nrm, sp, verbose=False)
approx_ls = einsum(bases.reshape(-1, R)[inds_ls.reshape(-1, T)[:, t_eval]],
                   coeffs_ls.reshape(-1, T)[:, t_eval],
                   'K Te R, K Te -> Te R')
err_ls = (approx_ls - truth_eval).norm(dim=-1) / truth_eval.norm(dim=-1)
print(f'lstsq maxmin:    mean {err_ls.mean():.3e}  max {err_ls.max():.3e}')
