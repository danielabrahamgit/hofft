import torch
import numpy as np
import matplotlib as mpl
mpl.use('webAgg')
import matplotlib.pyplot as plt

from mr_sim.phantoms import shepp_logan
from mr_sim.trj_lib import trj_lib
from mr_sim.coil_maps import surface_coil_maps

from mr_recon.linops import sense_linop
from mr_recon.fourier import sigpy_nufft, matrix_nufft
from mr_recon.utils import gen_grd, normalize
from mr_recon.algs import density_compensation
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Params
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 256
im_size = (N,)*2

# Image and sensitivity maps
img_gt = shepp_logan(torch_dev).img(im_size)
img_gt /= img_gt.abs().max()
mps = img_gt[None, :] * 0 + 1
# mps, evals = surface_coil_maps(16, im_size, img=img_gt, torch_dev=torch_dev, espirit_crp=-1.0)
# mps *= mask
mps *= (img_gt.abs() > 0.0)

# K-space trajectory and dcf
trj_os_factor = 1.1
im_size_os = (round(N * trj_os_factor),) * 2
trj = trj_lib(im_size_os).spiral_2d(n_shots=1, alpha=1.0)[:, 0, :] / trj_os_factor
param = np.linalg.norm(np.diff(trj, axis=0), axis=-1)
param = np.cumsum(param, axis=0)
param = np.concatenate([param[:1]*0, param], axis=0)
param_new = np.arange(0, param[-1], 1.0)
trj = interp1d(param, trj, axis=0, kind='linear', fill_value='extrapolate')(param_new)
trj = torch.from_numpy(trj).type(torch.float32).to(torch_dev)
trj_size = trj.shape[:-1]
dcf = density_compensation(trj, im_size)

# Mask
mask = 1.0 * (img_gt.abs() > 0.0)
mask = gaussian_filter(mask.cpu(), sigma=4)
mask = torch.from_numpy(mask).to(torch_dev)
mask = 1.0 * (mask > 0)

# GT
nft_sim = matrix_nufft(im_size, spatial_batch_size=2**10)
Asim = sense_linop(trj, mps, nufft=nft_sim)
ksp_gt = Asim(img_gt)

# Save data
torch.save(ksp_gt, './data/sim_spiral/ksp.pt')
torch.save(trj, './data/sim_spiral/trj.pt')
torch.save(dcf, './data/sim_spiral/dcf.pt')
torch.save(mask, './data/sim_spiral/mask.pt')
torch.save(img_gt, './data/sim_spiral/img.pt')
torch.save(mps, './data/sim_spiral/mps.pt')