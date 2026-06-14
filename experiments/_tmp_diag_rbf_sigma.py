"""
Compare lstsq, RBF, and grid sparse interpolation on coco_spiral.

Usage:
    PYTHONPATH=src python experiments/_tmp_diag_rbf_sigma.py
"""
import time

import torch
import numpy as np
from einops import einsum

from hofft.sparse_decomp import sparse_alpha_segmentation, sparse_params
from hofft.phase_coeffs import trj_dev_to_phis_alphas, compress_phis_alphas, rescale_phis_alphas
from mr_recon.spatial import spatial_resize_poly

R_UNDER = 3
B_COMPRESSED = 4
Q, K = 200, 5
T_EVAL = 2000
torch_dev = torch.device(0 if torch.cuda.is_available() else 'cpu')

fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_spiral'
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
alphas = torch.load(f'{fpath}/alphas.pt', map_location=torch_dev)
phis = torch.load(f'{fpath}/phis.pt', map_location=torch_dev)
im_size = mps.shape[1:]
os_grid = 2 * round(1.25 * im_size[0] / 2) / im_size[0]
red_size = (120, 120)

trj = trj[..., ::R_UNDER, :].type(torch.float32)
trj_size = trj.shape[:-1]

B = phis.shape[0]
idxs = torch.argwhere(
    (phis.reshape(B, -1).abs().mean(1) * alphas.reshape(B, -1).abs().mean(1)) > 1e-6)[:, 0]
phis, alphas = phis[idxs], alphas[idxs]

phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, os_grid)
phis_stack, alphas_stack = compress_phis_alphas(
    torch.cat([phis, phis_dev]), torch.cat([alphas, alphas_dev]), B_compressed=B_COMPRESSED)
phis_nrm, _, alphas_nrm, _ = rescale_phis_alphas(phis_stack, alphas_stack)

phis_red = spatial_resize_poly(phis_nrm, im_size=red_size, order=3)
R = int(np.prod(red_size))
T = int(np.prod(trj_size))
t_eval = torch.randperm(T, device=torch_dev)[:T_EVAL]


def compute_truth(t_idx: torch.Tensor) -> torch.Tensor:
    phis_flt = phis_red.reshape((phis_red.shape[0], R))
    alphas_flt = alphas_nrm.reshape((alphas_nrm.shape[0], T))
    phis_nrm_in, phis_mp_in, alphas_in, alphas_mp_in = rescale_phis_alphas(phis_flt, alphas_flt)
    phis_full = phis_nrm_in + phis_mp_in[:, None]
    return torch.exp(-2j * np.pi * einsum(
        phis_full, alphas_in[:, t_idx] + alphas_mp_in[:, None], 'B R, B T -> T R'))


truth = compute_truth(t_eval)
common = dict(Q=Q, K=K, beta_method='maxmin', temporal_batch_size=2**10,
              lamda=0.0, normalize_coeffs=True)

configs = [
    ('lstsq', sparse_params(interp_method='lstsq', **common)),
    ('rbf', sparse_params(interp_method='rbf', **common)),
    ('grid', sparse_params(interp_method='grid', grid_spacing=0.25, grid_width=2, **common)),
]

print(f'{"method":>8s}  {"mean err":>10s}  {"max err":>10s}  {"time":>8s}')
for label, sp in configs:
    t0 = time.perf_counter()
    bases, inds, coeffs = sparse_alpha_segmentation(phis_red, alphas_nrm, sp, verbose=False)
    elapsed = time.perf_counter() - t0
    approx = einsum(
        bases.reshape(-1, R)[inds.reshape(-1, T)[:, t_eval]],
        coeffs.reshape(-1, T)[:, t_eval],
        'K T R, K T -> T R')
    err = (approx - truth).norm(dim=-1) / truth.norm(dim=-1)
    print(f'{label:>8s}  {err.mean().item():10.4f}  {err.max().item():10.4f}  {elapsed:8.2f}s')
