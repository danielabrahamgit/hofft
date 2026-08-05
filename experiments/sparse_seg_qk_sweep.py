"""
Sweeps the compressed HOFFT sparse-fit strategies (see test_sparse_seg.py) over
the number of compressed kernels Q and the sparsity factor S, benchmarking:
  1. Reconstruction error (NRMSE)
  2. Decomposition time (ALS + sparse coefficient estimation)
  3. Reconstruction time (CG-SENSE)

Data loading / phase coefficient processing mirrors test_sparse_seg.py exactly.
Only the compressed HOFFT implementations are benchmarked (no standard HOFFT,
no spatio-temporal splitting).
"""
import time
import torch
import numpy as np

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from hofft.forward_model import hofft_compressed_linop
from hofft.sparse_decomp import sparse_params
from hofft.decomp import hofft_params
from hofft.pipelines import (
    sparse_fit_hofft_lstsq_support,
    sparse_fit_hofft_smooth_interp,
)
from hofft.phase_coeffs import (
    trj_dev_to_phis_alphas,
    compress_phis_alphas,
    rescale_phis_alphas,
    apply_phase_midpoints,
)
from mr_recon.recons import CG_SENSE_recon
from mr_recon.linops import batching_params
from mr_recon.utils import normalize

# ----------------- params (mirrors test_sparse_seg.py) -----------------
torch.manual_seed(0)
R = 1
B_compressed = 4
torch_dev = torch.device(0)
hparams = hofft_params(kern_size=(5,)*2,
                       os=1.25,
                       L=5,
                       reduced_im_size=(120,120),
                       spatial_init='seg',
                       kalpha_method='maxmin',
                       matvec_kwargs={'temporal_batch_size': 2**10},
                       verbose=False)
num_als_iter = 100

# ----------------- Load data (mirrors test_sparse_seg.py) -----------------
fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/highres_spiral'
ksp = torch.load(f'{fpath}/ksp.pt', map_location=torch_dev)
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
dcf = torch.load(f'{fpath}/dcf.pt', map_location=torch_dev)
alphas = torch.load(f'{fpath}/alphas.pt', map_location=torch_dev)
phis = torch.load(f'{fpath}/phis.pt', map_location=torch_dev)
img_gt = torch.load(f'{fpath}/img_gt.pt', map_location=torch_dev)
im_size = mps.shape[1:]
C = mps.shape[0]
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0] # Round to nearest even integer
bparams = batching_params(coil_batch_size=C)

# Undersample
trj = trj[..., ::R, :].type(torch.float32)
dcf = dcf[..., ::R].type(torch.float32)
ksp = ksp[..., ::R].type(torch.complex64)
alphas = alphas[..., ::R].type(torch.float32)
trj_size = trj.shape[:-1]

# ----------------- Process phase coefficients (mirrors test_sparse_seg.py) -----------------
# Remove small energy alphas or phis
B = phis.shape[0]
phi_energy = phis.reshape((B, -1)).abs().mean(dim=1)
alpha_energy = alphas.reshape((B, -1)).abs().mean(dim=1)
energy = phi_energy * alpha_energy
idxs = torch.argwhere(energy > 1e-6)[:, 0]
phis = phis[idxs]
alphas = alphas[idxs]
B = phis.shape[0]

# Stack phase coefficients, compress, rescale
phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, hparams.os)
phis_stack = torch.cat([phis, phis_dev], dim=0)
alphas_stack = torch.cat([alphas, alphas_dev], dim=0)
if B_compressed is not None:
    phis_stack, alphas_stack = compress_phis_alphas(phis_stack, alphas_stack,
                                                    B_compressed=B_compressed)
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack, norm_dists=True)
spat, temp = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp)
trj_grd = (trj * hparams.os).round() / hparams.os
img_gt = img_gt.cpu()

# ----------------- Q x S sweep over compressed HOFFT strategies -----------------
Q_values = [100, 200, 400, 800]
S_values = [1, 4, 8, 16, 32]

sparse_strategies = {
    'Compressed HOFFT (lstsq support)': (sparse_fit_hofft_lstsq_support, dict(ls_lamda=0.0)),
    'Compressed HOFFT (smooth RBF, auto-tuned)': (sparse_fit_hofft_smooth_interp,
                                                  dict(kernel='rbf', auto_tune=True, num_val=2**10)),
    'Compressed HOFFT (smooth inv-dist, auto-tuned)': (sparse_fit_hofft_smooth_interp,
                                                       dict(kernel='inv_dist', auto_tune=True, num_val=2**10)),
}

nQ, nS = len(Q_values), len(S_values)
metrics = ('nrmse', 'decomp_time', 'recon_time')
results = {name: {m: np.full((nQ, nS), np.nan) for m in metrics}
           for name in sparse_strategies}

for qi, Q in enumerate(Q_values):
    for si, S in enumerate(S_values):
        if S > Q:
            print(f'Skipping Q={Q}, S={S} (S must be <= Q)')
            continue
        sparams = sparse_params(Q=Q, S=S,
                                spatial_subsample=2**10,
                                temporal_batch_size=2**10)
        for name, (fit_fn, kwargs) in sparse_strategies.items():
            print(f'--- {name}: Q={Q}, S={S} ---')

            # Decomposition (ALS + sparse coefficient estimation)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rets = fit_fn(phis_nrm, alphas_nrm,
                         hparams=hparams, sparams=sparams,
                         num_als_iter=num_als_iter, **kwargs)
            torch.cuda.synchronize()
            decomp_time = time.perf_counter() - t0
            spatial_factors, compressed_kernels, sparse_inds, sparse_coeffs = rets
            spatial_factors = spatial_factors * spat

            # Build compressed HOFFT linear operator
            A_comp = hofft_compressed_linop(trj_grd, mps,
                                            dcf=dcf,
                                            compressed_kernels=compressed_kernels,
                                            sparse_idxs=sparse_inds, sparse_coeffs=sparse_coeffs,
                                            spatial_factors=spatial_factors, os_grid=hparams.os,
                                            bias_kernel=None,
                                            temporal_factors=temp,
                                            bparams=bparams)

            # Reconstruction
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            img = CG_SENSE_recon(A_comp, ksp, max_iter=10, max_eigen=1.0).cpu()
            torch.cuda.synchronize()
            recon_time = time.perf_counter() - t0

            im_nrm = normalize(img, img_gt)
            nrmse = ((im_nrm.abs() - img_gt.abs()).norm() / img_gt.abs().norm()).item()

            results[name]['nrmse'][qi, si] = 100 * nrmse
            results[name]['decomp_time'][qi, si] = decomp_time
            results[name]['recon_time'][qi, si] = recon_time

            print(f'    NRMSE = {100*nrmse:.2f}%, decomp = {decomp_time:.1f}s, recon = {recon_time:.1f}s')

# ----------------- Save raw results -----------------
save_path = f'{fpath}/sparse_seg_qk_sweep_results.pt'
torch.save({'results': results, 'Q_values': Q_values, 'S_values': S_values}, save_path)
print(f'Saved sweep results to {save_path}')

# ----------------- Plot -----------------
metric_labels = {
    'nrmse': 'NRMSE (%)',
    'decomp_time': 'Decomposition Time (s)',
    'recon_time': 'Reconstruction Time (s)',
}
fig, axes = plt.subplots(len(metric_labels), len(sparse_strategies),
                         figsize=(6 * len(sparse_strategies), 5 * len(metric_labels)),
                         squeeze=False)
for mi, (mkey, mlabel) in enumerate(metric_labels.items()):
    for si, name in enumerate(sparse_strategies):
        ax = axes[mi][si]
        data = results[name][mkey]
        for s_idx, S in enumerate(S_values):
            valid = ~np.isnan(data[:, s_idx])
            if valid.any():
                ax.plot(np.array(Q_values)[valid], data[valid, s_idx], marker='o', label=f'S={S}')
        ax.set_xlabel('Q')
        ax.set_ylabel(mlabel)
        if mi == 0:
            ax.set_title(name, fontsize=10)
        ax.set_xscale('log', base=2)
        if mkey != 'nrmse':
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
plt.tight_layout()
fig_path = f'{fpath}/sparse_seg_qk_sweep.png'
plt.savefig(fig_path, dpi=150)
print(f'Saved figure to {fig_path}')
plt.show()
