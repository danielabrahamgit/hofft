"""
Compares `hofft_compressed_linop` against `hofft_compressed_linop_test`
(the presplit-real/imag diagnostic variant) on the highres_spiral dataset,
using the exact (Q=500, S=1, lstsq) sparse config from
paper_experiments/highres_spiral/run.py.

Both linops are built from the *same* decomposition result (same
spatial_factors/compressed_kernels/sparse_idxs/sparse_coeffs/temp), so any
difference in output is a bug, and any difference in speed isolates the cost
of resplitting the invariant tensors (compressed_kernels, sparse_idxs,
sparse_coeffs, bias_kernel) and `y` on every forward/adjoint call.
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
from hofft.forward_model import hofft_compressed_linop, hofft_compressed_linop_test
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


def time_repeated(fn, torch_dev, n_reps=5):
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

# ------------ Reproduce hofft_decomp_linop's sparse branch up through the ------------
# ------------ ALS fit, so both linops are built from IDENTICAL weights.   ------------
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

linop_kwargs = dict(trj=trj_grd, mps=mps, dcf=dcf,
                    compressed_kernels=compressed_kernels,
                    sparse_idxs=sparse_idxs,
                    sparse_coeffs=sparse_coeffs,
                    spatial_factors=spatial_factors,
                    temporal_factors=temp,
                    os_grid=hparams.os, bparams=bparams)

A = hofft_compressed_linop(**linop_kwargs)
A_test = hofft_compressed_linop_test(**linop_kwargs)

# ------------ Correctness: forward + adjoint should match closely ------------
img_rand = torch.randn(im_size, dtype=torch.complex64, device=torch_dev)

ksp_A = A.forward(img_rand)
ksp_test = A_test.forward(img_rand)
fwd_relerr = (torch.linalg.norm(ksp_A - ksp_test) / torch.linalg.norm(ksp_A)).item()
print(f'forward relative error: {fwd_relerr:.3e}')

img_A = A.adjoint(ksp)
img_test = A_test.adjoint(ksp)
adj_relerr = (torch.linalg.norm(img_A - img_test) / torch.linalg.norm(img_A)).item()
print(f'adjoint relative error: {adj_relerr:.3e}')

# ------------ Speed: forward, adjoint, and a full CG_SENSE_recon ------------
n_reps = 10
t_fwd_A, _ = time_repeated(lambda: A.forward(img_rand), torch_dev, n_reps)
t_fwd_test, _ = time_repeated(lambda: A_test.forward(img_rand), torch_dev, n_reps)
t_adj_A, _ = time_repeated(lambda: A.adjoint(ksp), torch_dev, n_reps)
t_adj_test, _ = time_repeated(lambda: A_test.adjoint(ksp), torch_dev, n_reps)

cg_kwargs = {'max_iter': 10, 'max_eigen': 1.0, 'verbose': False}
CG_SENSE_recon(A, ksp, **cg_kwargs)  # warm-up
CG_SENSE_recon(A_test, ksp, **cg_kwargs)  # warm-up
t_recon_A, img_recon_A = time_repeated(lambda: CG_SENSE_recon(A, ksp, **cg_kwargs), torch_dev, 3)
t_recon_test, img_recon_test = time_repeated(lambda: CG_SENSE_recon(A_test, ksp, **cg_kwargs), torch_dev, 3)
recon_relerr = (torch.linalg.norm(img_recon_A - img_recon_test) / torch.linalg.norm(img_recon_A)).item()

print(f'\n{"":>20} {"original":>12} {"presplit":>12} {"speedup":>8}')
print(f'{"forward":>20} {t_fwd_A:>12.4f} {t_fwd_test:>12.4f} {t_fwd_A / t_fwd_test:>8.2f}x')
print(f'{"adjoint":>20} {t_adj_A:>12.4f} {t_adj_test:>12.4f} {t_adj_A / t_adj_test:>8.2f}x')
print(f'{"full CG recon":>20} {t_recon_A:>12.4f} {t_recon_test:>12.4f} {t_recon_A / t_recon_test:>8.2f}x')
print(f'\nrecon relative error: {recon_relerr:.3e}')
