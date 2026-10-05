import torch

import matplotlib as mpl
mpl.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.fourier import fft
from mr_recon.multi_coil.calib import synth_cal
from mr_recon.multi_coil.coil_est import csm_from_espirit
from scipy.ndimage import gaussian_filter

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

# Smooth b0 to 1.5mm, since it was acquired with a 1.5mm multi echo GRE
npix = round(1.5e-3 / 0.3e-3)
b0_smooth = gaussian_filter(b0, sigma=npix)
b0_smooth = torch.from_numpy(b0_smooth)
b0_smooth = b0

# Stack b0 and eddy current phase coefficients
phis = torch.cat([b0_smooth[None,], phis.moveaxis(-1, 0)], dim=0)
alphas = torch.cat([times[None,], alphas.moveaxis(-1, 0)], dim=0)

# Get mask 
torch_dev = torch.device(0)
ksp_cal = fft(mps, dim=[-2,-1])
ksp_cal = synth_cal(ksp_cal, (64, 64))
_, evals = csm_from_espirit(ksp_cal.to(torch_dev), b0.shape)
evals = evals.cpu()

# Save data
torch.save(img_gt, 'img_gt.pt')
torch.save(trj, 'trj.pt')
torch.save(dcf, 'dcf.pt')
torch.save(ksp, 'ksp.pt')
torch.save(mps, 'mps.pt')
torch.save(phis, 'phis.pt')
torch.save(evals, 'evals.pt')
torch.save(alphas, 'alphas.pt')