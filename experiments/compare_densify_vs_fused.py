"""
Compares `hofft_compressed_linop` (fused Triton kernel) against
`hofft_linop` fed a densified version of the SAME compressed weights
(via `densify_sparse_kernels`), on the highres_spiral (Q=500, S=1, lstsq)
config. Both are built from identical spatial_factors/compressed_kernels/
sparse_idxs/sparse_coeffs/temp, so this isolates the fused-kernel-vs-dense
application cost specifically (not a difference in what was learned).

The profiler showed ~89% of total CUDA time inside `_hofft_forward_kernel`/
`_hofft_adjoint_kernel` (adjoint ~2.6x worse than forward, likely from its
atomic_add scatter). Since the compressed representation only needs to stay
compressed for cheap ALS fitting -- not for linop application -- this tests
whether just expanding it once and reusing the already-fast dense
`hofft_linop` sidesteps the slow kernel entirely.
"""
import torch

from time import perf_counter

from mr_recon.linops import batching_params
from mr_recon.algs import density_compensation
from mr_recon.recons import CG_SENSE_recon

from hofft.decomp import hofft_params
from hofft.matvec import matvec_cur
from hofft.sparse_fit import sparse_params
from hofft.pipelines import als_hofft_sparse_lstsq
from hofft.forward_model import hofft_compressed_linop, hofft_linop, densify_sparse_kernels
from hofft.phase_coeffs import (
    trj_dev_to_phis_alphas,
    compress_phis_alphas,
    rescale_phis_alphas,
    apply_phase_midpoints,
)


class GPUTimer:
    def __init__(self, torch_dev):
        self.is_cuda = torch_dev.type == 'cuda'

    def __enter__(self):
        if self.is_cuda:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
            self._start_evt.record()
        else:
            self._t0 = perf_counter()
        return self

    def __exit__(self, *exc_info):
        if self.is_cuda:
            self._end_evt.record()
            torch.cuda.synchronize()
            self.elapsed = self._start_evt.elapsed_time(self._end_evt) / 1000
        else:
            self.elapsed = perf_counter() - self._t0
        return False


def time_repeated(fn, torch_dev, n_reps=10):
    times = []
    result = None
    for _ in range(n_reps):
        with GPUTimer(torch_dev) as t:
            result = fn()
        times.append(t.elapsed)
    return torch.tensor(times).median().item(), result


torch.manual_seed(0)
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------ Params (matching paper_experiments/highres_spiral/run.py) ------------
os = 1.25
W = 5
L = 15
B_compressed = 8
num_als_iter = 0
mask_thresh = 0.9
R = 2

# ------------ Load data ------------
fpath = './data/highres_spiral'
kwargs = {'weights_only': True, 'map_location': torch_dev}
trj = torch.load(f'{fpath}/trj.pt', **kwargs)
mps = torch.load(f'{fpath}/mps.pt', **kwargs)
ksp = torch.load(f'{fpath}/ksp.pt', **kwargs)
evals = torch.load(f'{fpath}/evals.pt', **kwargs)
phis = torch.load(f'{fpath}/phis.pt', **kwargs)
alphas = torch.load(f'{fpath}/alphas.pt', **kwargs)
im_size = mps.shape[1:]
C = mps.shape[0]
os = 2 * round(os * im_size[0] / 2) / im_size[0]
bparams = batching_params(coil_batch_size=C)

trj = trj[:, ::R]
ksp = ksp[:, :, ::R]
alphas = alphas[:, :, ::R]
dcf = density_compensation(trj, im_size)

mask = (evals > mask_thresh).float()
mps = mps * mask

B = phis.shape[0]
phi_energy = phis.reshape((B, -1)).abs().mean(dim=1)
alpha_energy = alphas.reshape((B, -1)).abs().mean(dim=1)
energy = phi_energy * alpha_energy
idxs = torch.argwhere(energy > 1e-6)[:, 0]
phis = phis[idxs]
alphas = alphas[idxs]

hparams = hofft_params((W,) * 2, os, L,
                       reduced_im_size=(150, 150),
                       spatial_init='seg',
                       matvec_type=matvec_cur,
                       matvec_kwargs={'rank_phi': 500, 'rank_alpha': 500},
                       verbose=False)
sparams = sparse_params(Q=500, S=1, interp_type='lstsq')

phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, hparams.os)
trj_grd = (hparams.os * trj).round() / hparams.os
phis_stack = torch.cat([phis_dev, phis], dim=0)
alphas_stack = torch.cat([alphas_dev, alphas], dim=0)
phis_stack, alphas_stack = compress_phis_alphas(phis_stack, alphas_stack, B_compressed=B_compressed)
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack, whiten=True)
spat, temp = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp)

spatial_factors, compressed_kernels, sparse_idxs, sparse_coeffs = als_hofft_sparse_lstsq(
    phis_nrm, alphas_nrm, spatial_mask=mask, hparams=hparams, sparams=sparams,
    num_als_iter=num_als_iter)
spatial_factors = spatial_factors * spat

# ------------ Build both linops from identical decomposition output ------------
A_fused = hofft_compressed_linop(trj=trj_grd, mps=mps, dcf=dcf,
                                 compressed_kernels=compressed_kernels,
                                 sparse_idxs=sparse_idxs,
                                 sparse_coeffs=sparse_coeffs,
                                 spatial_factors=spatial_factors,
                                 temporal_factors=temp,
                                 os_grid=hparams.os, bparams=bparams)

t0 = perf_counter()
kern_weights = densify_sparse_kernels(compressed_kernels, sparse_idxs, sparse_coeffs)
kern_weights = kern_weights * temp  # phase-midpoint correction: same math as A_fused's temporal_factors
if torch_dev.type == 'cuda':
    torch.cuda.synchronize()
print(f'densify_sparse_kernels: {perf_counter() - t0:.3f}s, kern_weights shape {tuple(kern_weights.shape)}, '
      f'{kern_weights.element_size() * kern_weights.nelement() / 1e9:.2f} GB')

A_dense = hofft_linop(trj=trj_grd, mps=mps, dcf=dcf,
                      kern_weights=kern_weights,
                      spatial_factors=spatial_factors,
                      os_grid=hparams.os, bparams=bparams)

# ------------ Correctness ------------
img_rand = torch.randn(im_size, dtype=torch.complex64, device=torch_dev)

ksp_fused = A_fused.forward(img_rand)
ksp_dense = A_dense.forward(img_rand)
fwd_relerr = (torch.linalg.norm(ksp_fused - ksp_dense) / torch.linalg.norm(ksp_fused)).item()
print(f'\nforward relative error: {fwd_relerr:.3e}')

img_fused = A_fused.adjoint(ksp)
img_dense = A_dense.adjoint(ksp)
adj_relerr = (torch.linalg.norm(img_fused - img_dense) / torch.linalg.norm(img_fused)).item()
print(f'adjoint relative error: {adj_relerr:.3e}')

# ------------ Speed ------------
n_reps = 10
t_fwd_fused, _ = time_repeated(lambda: A_fused.forward(img_rand), torch_dev, n_reps)
t_fwd_dense, _ = time_repeated(lambda: A_dense.forward(img_rand), torch_dev, n_reps)
t_adj_fused, _ = time_repeated(lambda: A_fused.adjoint(ksp), torch_dev, n_reps)
t_adj_dense, _ = time_repeated(lambda: A_dense.adjoint(ksp), torch_dev, n_reps)

cg_kwargs = {'max_iter': 10, 'max_eigen': 1.0, 'verbose': False}
CG_SENSE_recon(A_fused, ksp, **cg_kwargs)  # warm-up
CG_SENSE_recon(A_dense, ksp, **cg_kwargs)  # warm-up
t_recon_fused, img_recon_fused = time_repeated(lambda: CG_SENSE_recon(A_fused, ksp, **cg_kwargs), torch_dev, 3)
t_recon_dense, img_recon_dense = time_repeated(lambda: CG_SENSE_recon(A_dense, ksp, **cg_kwargs), torch_dev, 3)
recon_relerr = (torch.linalg.norm(img_recon_fused - img_recon_dense) / torch.linalg.norm(img_recon_fused)).item()

print(f'\n{"":>20} {"fused":>12} {"densify+dense":>14} {"speedup":>8}')
print(f'{"forward":>20} {t_fwd_fused:>12.4f} {t_fwd_dense:>14.4f} {t_fwd_fused / t_fwd_dense:>8.2f}x')
print(f'{"adjoint":>20} {t_adj_fused:>12.4f} {t_adj_dense:>14.4f} {t_adj_fused / t_adj_dense:>8.2f}x')
print(f'{"full CG recon":>20} {t_recon_fused:>12.4f} {t_recon_dense:>14.4f} {t_recon_fused / t_recon_dense:>8.2f}x')
print(f'\nrecon relative error: {recon_relerr:.3e}')
