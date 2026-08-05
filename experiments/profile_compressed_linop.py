"""
Profiles `hofft_compressed_linop` (Q=500, S=1, lstsq) on the highres_spiral
dataset to find where time actually goes inside the sparse forward/adjoint,
now that we've ruled out (a) redundant complex->real/imag resplitting
(hofft_compressed_linop_test showed no change) and (b) per-field-batch Triton
launch overhead (raising field_batch_size showed no change).

Reports:
  1. forward-only vs adjoint-only wall time (isolates whether the adjoint's
     tl.atomic_add scatter is the dominant cost)
  2. a torch.profiler breakdown by self-CUDA-time, so we can see directly how
     much time is spent inside `_hofft_forward_kernel` / `_hofft_adjoint_kernel`
     themselves vs. the shared FFT/padder/einsum apodization step.
"""
import torch

from time import perf_counter

from mr_recon.linops import batching_params
from mr_recon.algs import density_compensation

from hofft.decomp import hofft_params
from hofft.matvec import matvec_cur
from hofft.sparse_fit import sparse_params
from hofft.pipelines import als_hofft_sparse_lstsq
from hofft.forward_model import hofft_compressed_linop
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
    for _ in range(n_reps):
        with GPUTimer(torch_dev) as t:
            fn()
        times.append(t.elapsed)
    return torch.tensor(times).median().item()


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

A = hofft_compressed_linop(trj=trj_grd, mps=mps, dcf=dcf,
                           compressed_kernels=compressed_kernels,
                           sparse_idxs=sparse_idxs,
                           sparse_coeffs=sparse_coeffs,
                           spatial_factors=spatial_factors,
                           temporal_factors=temp,
                           os_grid=hparams.os, bparams=bparams)

img_rand = torch.randn(im_size, dtype=torch.complex64, device=torch_dev)

# ------------ 1. forward vs adjoint wall time ------------
A.forward(img_rand)  # warm-up (triton compile)
A.adjoint(ksp)
if torch_dev.type == 'cuda':
    torch.cuda.synchronize()

t_fwd = time_repeated(lambda: A.forward(img_rand), torch_dev, n_reps=10)
t_adj = time_repeated(lambda: A.adjoint(ksp), torch_dev, n_reps=10)
print(f'forward: {t_fwd*1000:.2f} ms')
print(f'adjoint: {t_adj*1000:.2f} ms')
print(f'adjoint / forward ratio: {t_adj / t_fwd:.2f}x')

# ------------ 2. torch.profiler breakdown ------------
if torch_dev.type == 'cuda':
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            ksp_out = A.forward(img_rand)
            _ = A.adjoint(ksp_out)
        torch.cuda.synchronize()

    print('\nTop ops by self CUDA time:')
    print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=25))
else:
    print('\nCUDA not available -- skipping torch.profiler breakdown.')
