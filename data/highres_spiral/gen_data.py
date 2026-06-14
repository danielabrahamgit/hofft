import torch

import matplotlib as mpl
mpl.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.fourier import fft
from mr_recon.multi_coil.calib import synth_cal
from mr_recon.multi_coil.coil_est import csm_from_espirit

data = torch.load('/local_mount/space/mayday/data/users/zachs/share/ForDaniel/20260610_7t_300um_dataset/data.pt')
recons = torch.load('/local_mount/space/mayday/data/users/zachs/share/ForDaniel/20260610_7t_300um_dataset/recon.pt')
img_gt = recons['matrix'][..., 0]

ksp = data['ksp']
trj = data['trj']
dcf = data['dcf']
mps = data['mps']
b0 = data['b0']
times = data['times']
phis = data['kspha_bases']
alphas = data['kspha'] / (2 * torch.pi)

# Stack b0 and eddy current phase coefficients
phis = torch.cat([b0[None,], phis.moveaxis(-1, 0)], dim=0)
alphas = torch.cat([times[None,], alphas.moveaxis(-1, 0)], dim=0)

# Get mask 
torch_dev = torch.device(5)
ksp_cal = fft(mps, dim=[-2,-1])
ksp_cal = synth_cal(ksp_cal, (64, 64))
_, evals = csm_from_espirit(ksp_cal.to(torch_dev), b0.shape)
evals = evals.cpu()

torch.save(img_gt, 'img_gt.pt')
torch.save(trj, 'trj.pt')
torch.save(dcf, 'dcf.pt')
torch.save(ksp, 'ksp.pt')
torch.save(mps, 'mps.pt')
torch.save(phis, 'phis.pt')
torch.save(evals, 'evals.pt')
torch.save(alphas, 'alphas.pt')