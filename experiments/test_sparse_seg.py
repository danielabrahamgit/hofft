import torch

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from hofft.sgd import training_params
from hofft.utils import reduce_spatial, expand_spatial, normalize
from hofft.forward_model import hofft_compressed_linop, hofft_linop
from hofft.sparse_fit import sparse_params
from hofft.decomp import hofft_params
from hofft.matvec import matvec_cur
from hofft.pipelines import (
  als_hofft,
  als_hofft_sparse_lstsq,
  als_hofft_sparse_smooth
)
from hofft.phase_coeffs import (
  trj_dev_to_phis_alphas,
  compress_phis_alphas,
  rescale_phis_alphas,
  apply_phase_midpoints,
)
from mr_recon.imperfections.field import alpha_segementation
from mr_recon.recons import CG_SENSE_recon
from mr_recon.linops import sense_linop, batching_params
from mr_recon.fourier import sigpy_nufft
from tqdm import tqdm

# params
torch.manual_seed(0)
R = 1
B_compressed = 8
# B_compressed = None
torch_dev = torch.device(0 if torch.cuda.is_available() else 'cpu')
hparams = hofft_params(kern_size=(5,)*2,
                       os=1.25,
                       L=5,
                       reduced_im_size=(120,120),
                       spatial_init='seg',
                       kalpha_method='maxmin',
                       matvec_kwargs={'temporal_batch_size': 2**10},
                       verbose=True)
sparams = sparse_params(Q=200, S=8,
                        spatial_subsample=2**10,
                        temporal_batch_size=2**10, 
                        # spatial_batch_size=2**10,
                        num_validation=30,
                        )

# Load data
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

# ----------------- Process phase coefficients -----------------
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
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack, whiten=True)
spat, temp = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp)
# phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack, norm_dists=True)
# spat, temp = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp)
# if B_compressed is not None:
#     phis_nrm, alphas_nrm = compress_phis_alphas(phis_nrm, alphas_nrm, 
#                                                 B_compressed=B_compressed)

# ----------------- Standard HOFFT Decomposition -----------------
# ALS decomp and apply phase midpoints
rets = als_hofft(phis_nrm, alphas_nrm, hparams=hparams,
                #  rparams=reduce_params(spatial_reduce_size=hparams.reduced_im_size,),
                 num_als_iter=100*0+1)
spatial_factors, kernel_weights = rets
spatial_factors *= spat
kernel_weights *= temp

# Build standard HOFFT linear operator
trj_grd = (trj * hparams.os).round() / hparams.os
A_hofft = hofft_linop(trj_grd, mps, kernel_weights, spatial_factors, dcf, 
                      os_grid=hparams.os, bparams=bparams)
img_hofft = CG_SENSE_recon(A_hofft, ksp, max_iter=10, max_eigen=1.0).cpu()

# ----------------- Compressed HOFFT Decomposition -----------------
# # Indirect Sparse Interpolation
# rets = als_hofft_compressed(phis_nrm, alphas_nrm, hparams=hparams, sparams=sparams,
#                             num_als_iter=100)
# spatial_factors, compressed_kernels, sparse_inds, sparse_coeffs = rets
# bias_kern = None
# hparams.spatial_init = '100_alphas_10'
# rets = mlp_hofft_compressed(phis_nrm, alphas_nrm, hparams=hparams, sparams=sparams,
#                             tparams=training_params(epochs=50 * 100, show_loss=False))
# spatial_factors, compressed_kernels, bias_kern, sparse_inds, sparse_coeffs = rets

# Direct sparse fitting: compare Strategy 2 (lstsq fixed support) against
# Strategy 2.5 (smooth distance-based interpolation, RBF vs inverse-distance
# kernels) -- see math_docs/sparse_fit.md. All three strategies reuse the same
# 'k_alphas' compressed kernels H_comp and only differ in how the sparse
# coefficients C are solved for.
# rets = sparse_fit_hofft_known_coeffs(phis_nrm, alphas_nrm,
#                                      hparams=hparams, sparams=sparams,
#                                      spatial_subsample=2**13, temporal_subsample=2**14)
trj_grd = (trj * hparams.os).round() / hparams.os
sparse_strategies = {
    'Compressed HOFFT (lstsq support)': (als_hofft_sparse_lstsq, 'lstsq'),
    'Compressed HOFFT (smooth RBF, auto-tuned)': (als_hofft_sparse_smooth,
                                                  'rbf'),
    'Compressed HOFFT (smooth inv-dist, auto-tuned)': (als_hofft_sparse_smooth,
                                                       'inv_dist'),
}
comp_imgs = {}
for name, (fit_fn, interp_type) in sparse_strategies.items():
    sparams.interp_type = interp_type
    rets = fit_fn(phis_nrm, alphas_nrm,
                 hparams=hparams, sparams=sparams,
                 num_als_iter=100)
    spatial_factors, compressed_kernels, sparse_inds, sparse_coeffs = rets
    spatial_factors = spatial_factors * spat

    # Build compressed HOFFT linear operator
    # NOTE: the phase-midpoint correction `temp` is passed as temporal_factors (a
    # per-trajectory-point phase on the k-space output) rather than folded into
    # sparse_coeffs -- folding it in would silently drop the bias_kernel's phase.
    A_comp = hofft_compressed_linop(trj_grd, mps,
                                    dcf=dcf,
                                    compressed_kernels=compressed_kernels,
                                    sparse_idxs=sparse_inds, sparse_coeffs=sparse_coeffs,
                                    spatial_factors=spatial_factors, os_grid=hparams.os,
                                    bias_kernel=None,
                                    temporal_factors=temp,
                                    bparams=bparams)
    comp_imgs[name] = CG_SENSE_recon(A_comp, ksp, max_iter=10, max_eigen=1.0).cpu()

# ----------------- Spatio Temporal Splitting Decomposition -----------------
phis_reduced = reduce_spatial(phis, im_size_low=hparams.reduced_im_size, order=3)
tbs = 2**10
cs = torch.zeros((hparams.L, *trj_size), device=torch_dev, dtype=torch.complex64)
for t1 in tqdm(range(0, trj.shape[0], tbs), 'Spatio Temporal Splitting Decomposition'):
  t2 = min(t1 + tbs, trj.shape[0])
  bs, cs[:, t1:t2], _ = alpha_segementation(phis_reduced, alphas[:, t1:t2], 
                                            L=hparams.L, 
                                            L_batch_size=1,
                                            interp_type='lstsq',
                                            method='maxmin',
                                            use_type3=False,
                                            verbose=False)
bs = expand_spatial(bs, im_size_high=im_size, order=3)
nft = sigpy_nufft(im_size, oversamp=hparams.os, width=hparams.kern_size[0])
nft.beta = nft.optimal_beta(torch_dev=torch_dev)
A_split = sense_linop(trj, mps, dcf, 
                      spatial_funcs=bs,
                      temporal_funcs=cs,
                      nufft=nft,
                      bparams=bparams)
img_split = CG_SENSE_recon(A_split, ksp, max_iter=10, max_eigen=1.0).cpu()

# ----------------- Compare ----------------
img_gt = img_gt.cpu()
plt.figure(figsize=(20, 10))
titles = [*comp_imgs.keys(), 'Standard HOFFT', 'Spatio Temporal Splitting']
imgs = [*comp_imgs.values(), img_hofft, img_split]
vmax = img_gt.abs().median() + 3 * img_gt.abs().std()
for i in range(len(imgs)):
    im_nrm = normalize(imgs[i], img_gt)
    nrmse = (im_nrm.abs() - img_gt.abs()).norm() / img_gt.abs().norm()
    plt.subplot(2, len(imgs), i+1)
    plt.imshow(im_nrm.abs().rot90(), cmap='gray', vmin=0, vmax=vmax)
    plt.title(f'{titles[i]}\nNRMSE = {100*nrmse:.2f}%')
    plt.axis('off')
    plt.subplot(2, len(imgs), i+len(imgs)+1)
    plt.imshow((im_nrm.abs() - img_gt.abs()).abs().rot90(), cmap='gray')
    plt.title(f'Error, NRMSE = {100*nrmse:.2f}%')
    plt.axis('off')
plt.tight_layout()
plt.show()