import torch

import matplotlib as mpl
mpl.use('Webagg')
import matplotlib.pyplot as plt

from mr_sim.phantoms import shepp_logan
from mr_sim.coil_maps import surface_coil_maps
from mr_sim.trj_lib import trj_lib

from mr_recon.linops import sense_linop, batching_params
from mr_recon.algs import density_compensation
from mr_recon.multi_coil.coil_est import csm_from_espirit
from mr_recon.multi_coil.calib import synth_cal
from mr_recon.fourier import fft, matrix_nufft

# Params
R = 3
C = 16
torch_dev = torch.device(5)
im_size = (220, 220)

# Build trajectory 
tl = trj_lib(im_size)
trj = tl.spiral_2d(R, alpha=1.0)
trj = torch.from_numpy(trj)
trj = trj.type(torch.float32).to(torch_dev)
dcf = density_compensation(trj, im_size)

# Phantom
shp = shepp_logan(torch_dev)
img = shp.img(im_size)

# Sensivity maps
mps, evals = surface_coil_maps(C, im_size, 
                               espirit_crp=-1, # no masking
                               img=img, torch_dev=torch_dev)


# Simulate k-space data
nufft = matrix_nufft(im_size, spatial_batch_size=2**10)
A = sense_linop(trj, mps, 
                nufft=nufft,
                bparams=batching_params(C))
ksp = A(img)


# # Simulate single channelk-space data
# mps = torch.ones((1, *im_size), dtype=torch.complex64, device=torch_dev)
# ksp = shp.ksp(trj)[None, :]
# ksp_cal = fft(shp.img(im_size), dim=[-2,-1])[None, :, :]
# ksp_cal = synth_cal(ksp_cal, (32, 32))
# _, evals = csm_from_espirit(ksp_cal, im_size)

    
# Save data
phis = torch.ones((1, *im_size), dtype=torch.float32, device=torch_dev)
alphas = torch.ones((1, *dcf.shape), dtype=torch.float32, device=torch_dev)
fdir = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/sim_spiral'
torch.save(trj.cpu(), f'{fdir}/trj.pt')
torch.save(dcf.cpu(), f'{fdir}/dcf.pt')
torch.save(mps.cpu(), f'{fdir}/mps.pt')
torch.save(ksp.cpu(), f'{fdir}/ksp.pt')
torch.save(evals.cpu(), f'{fdir}/evals.pt')
torch.save(phis.cpu(), f'{fdir}/phis.pt')
torch.save(alphas.cpu(), f'{fdir}/alphas.pt')
