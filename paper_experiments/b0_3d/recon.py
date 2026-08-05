import gc
import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_recon.fourier import sigpy_nufft
from mr_recon.linops import sense_linop
from mr_recon.recons import CG_SENSE_recon
from mr_recon.imperfections.field import alpha_segementation

from hofft.pipelines import hofft_decomp_linop, time_seg_decomp_linop
from hofft.sparse_fit import sparse_params
from hofft.matvec import matvec_cur
from hofft.decomp import hofft_params
from hofft.utils import expand_spatial, reduce_spatial

# Params
L = 2

# Load data
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fpath = '/local_mount/space/tiger/1/users/abrahamd/mr_data/mrf_b0/data/'
b0 = torch.from_numpy(np.load(fpath + 'b0.npy')).type(torch.float32).to(torch_dev)
dcf = torch.from_numpy(np.load(fpath + 'dcf.npy')).type(torch.float32).to(torch_dev)
trj = torch.from_numpy(np.load(fpath + 'trj.npy')).type(torch.float32).to(torch_dev)
mps = torch.from_numpy(np.load(fpath + 'mps.npy')).type(torch.complex64).to(torch_dev)
ksp = torch.from_numpy(np.load(fpath + 'ksp.npy')).type(torch.complex64).to(torch_dev)
dcf /= dcf.max()
C = mps.shape[0]
im_size = b0.shape

# Subsample
R = 3
G = trj.shape[1]
trj = trj[:, :G//R]
dcf = dcf[:, :G//R]
ksp = ksp[:, :, :G//R]

# B0 phase
ts = torch.arange(trj.shape[0], device=torch_dev) * 2e-6
phis = b0[None,]
alphas = ts[None, :, None, None]

# # SENSE recon
# nft = sigpy_nufft(im_size, width=3)
# nft.beta = nft.optimal_beta(torch_dev=torch_dev)
# phis_red = reduce_spatial(phis, (50,)*3)
# bs, hs, _ = alpha_segementation(phis_red, alphas, L=L, interp_type='lstsq', use_type3=True)
# bs = expand_spatial(bs, im_size)
# A = sense_linop(trj, mps, dcf, 
#                 spatial_funcs=bs,
#                 temporal_funcs=hs,
#                 nufft=nft)

# HOFFT recon
num_als_iter = 100
hparams = hofft_params((3,)*3, 1.25, L,
                        reduced_im_size=(50,)*3,
                        spatial_init='seg',
                        # spatial_init=f'500_alphas_{num_als_iter}',
                        # matvec_kwargs={
                        #     'temporal_batch_size': 2**12,
                        # },
                        # matvec_type=matvec_cur,
                        # matvec_kwargs={
                        #     'rank_phi': 500,
                        #     'rank_alpha': 500,
                        # },
                        verbose=True)
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0]

# A = time_seg_decomp_linop(phis, alphas, mps, trj, dcf=dcf,
#                           normalize_coeffs=True,
#                           use_sigpy=True,
#                           hparams=hparams,)

sparams = sparse_params(Q=500*5, S=16, 
                        interp_type='inv_dist', 
                        temporal_batch_size=2**15,
                        spatial_subsample=2**15,
                        num_validation=300)
fact = 0 if 'alphas' in hparams.spatial_init else 1
alphas = torch.zeros_like(dcf)
alphas[:, ...] = ts[:, None, None]
alphas = alphas[None, ...]
A =  hofft_decomp_linop(phis, alphas, 
                        mps=mps, trj=trj, dcf=dcf, 
                        hparams=hparams, 
                        sparams=sparams,
                        normalize_coeffs=True,
                        # spatial_mask=mask,
                        num_als_iter=num_als_iter*fact,
                        )

# Clear GPU memory
gc.collect()
torch.cuda.empty_cache()

x0 = CG_SENSE_recon(A, ksp, max_iter=2, max_eigen=1.0)

from mr_recon.utils import cvplot
cvplot('./config.yaml', x0)