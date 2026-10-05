import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from hofft.utils import gen_grd, maxmin_indices, reduce_spatial, reduce_temporal
from hofft.pipelines import svd_decomp_linop, rescale_phis_alphas
from hofft.decomp import hofft_params
from hofft.phase_coeffs import whiten_phis_alphas, rescale_phis_alphas
from mr_recon.linops import sense_linop, batching_params
from mr_recon.recons import CG_SENSE_recon
from mr_recon.fourier import cufi_nufft
from mr_recon.imperfections.estimation.eddy_focus import (
    build_alpha_trajectory_taylor,
    build_alpha_trajectory,
    normalize_phis,
)

from tqdm import tqdm

dataset = 'coco_spiral'
# dataset = 'tilt_spi_invivo'

# Load data
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trj = torch.load(f'./data/{dataset}/trj.pt', map_location=torch_dev).type(torch.float32)
ksp = torch.load(f'./data/{dataset}/ksp.pt', map_location=torch_dev).type(torch.complex64)
mps = torch.load(f'./data/{dataset}/mps.pt', map_location=torch_dev).type(torch.complex64)
evals = torch.load(f'./data/{dataset}/evals.pt', map_location=torch_dev).type(torch.float32)
dcf = torch.load(f'./data/{dataset}/dcf.pt', map_location=torch_dev).type(torch.float32)
phis = torch.load(f'./data/{dataset}/phis.pt', map_location=torch_dev).type(torch.float32)
alphas = torch.load(f'./data/{dataset}/alphas.pt', map_location=torch_dev).type(torch.float32)
im_size = mps.shape[1:]
trj_size = trj.shape[:-1]
B = phis.shape[0]
C = mps.shape[0]
max_iter = 20

R = 3
alphas = alphas[:, :, ::R].squeeze()
dcf = dcf[:, ::R].squeeze()
trj = trj[:, ::R].squeeze()
ksp = ksp[:, :, ::R].squeeze()

# Remove zero phi alphas terms
idxs = [0,1,2]
phis = phis[idxs]
alphas = alphas[idxs]

# Print shapes
print(f'trj.shape: {trj.shape}')
print(f'mps.shape: {mps.shape}')
print(f'evals.shape: {evals.shape}')
print(f'dcf.shape: {dcf.shape}')
print(f'phis.shape: {phis.shape}')
print(f'alphas.shape: {alphas.shape}')

# Low-res image for phase estimate
ros = slice(None, trj.shape[0]//10)
nft = cufi_nufft(im_size, oversamp=1.25, width=3)
A = sense_linop(trj[ros], mps, dcf[ros], nufft=nft,
                bparams=batching_params(coil_batch_size=C))
img_low_res = CG_SENSE_recon(A, ksp[:, ros], 
                             max_eigen=1.0, max_iter=max_iter)

# Recon image naive 
A = sense_linop(trj, mps, dcf, nufft=nft,
                bparams=batching_params(coil_batch_size=C))
img_recon = CG_SENSE_recon(A, ksp, 
                           max_eigen=1.0, max_iter=max_iter)


# Recon with field terms
hparams = hofft_params(kern_size=(3,)*2, os=1.25, L=30, cur_rank=500)
A_svd = svd_decomp_linop(phis, alphas, mps, trj,
                         hparams=hparams)
img_svd = CG_SENSE_recon(A_svd, ksp, 
                         max_eigen=1.0, max_iter=max_iter)
img_svd *= torch.exp(-1j * img_low_res.angle())

# Try auto focus
phis = normalize_phis(phis)
alphas_auto = build_alpha_trajectory(img_svd, phis, trj, window_size=100).T
A_svd = svd_decomp_linop(phis, alphas + alphas_auto, mps, trj,
                         hparams=hparams)
img_svd_auto = CG_SENSE_recon(A_svd, ksp, 
                         max_eigen=1.0, max_iter=max_iter)


# Show images
imgs = [img_low_res, img_recon, img_svd, img_svd_auto]
plt.figure(figsize=(14, 7))
for i, img in enumerate(imgs):
    plt.subplot(2, len(imgs), i+1)
    plt.imshow(img.abs().cpu().rot90(), cmap='gray')
    plt.axis('off')
    plt.subplot(2, len(imgs), i+1+len(imgs))
    plt.imshow(img.angle().cpu().rot90(), cmap='jet', 
               vmin=-torch.pi, vmax=torch.pi)
    plt.axis('off')
plt.tight_layout()
plt.show()