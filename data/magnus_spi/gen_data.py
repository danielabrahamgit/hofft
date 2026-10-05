import torch

from mr_recon.fourier import fft
from mr_recon.multi_coil.calib import synth_cal
from mr_recon.multi_coil.coil_est import csm_from_espirit

slc = 5
fpath = '/local_mount/space/mayday/data/users/zachs/share/ForDaniel/hoftt_datasets/20260923_magnus_phantom_spiral/'
data = torch.load(fpath + 'data.pt')
img_gt = torch.load(fpath + 'recon.pt')[..., slc]

# Stack 
phis = torch.cat([data['phis_b0'][..., slc][None,], data['phis_kspha'][..., slc]], dim=0)
alphas = torch.cat([data['alphas_b0'][None,], data['alphas_kspha']], dim=0)

# Evals from mps
mps = data['mps'][..., slc]
torch_dev = torch.device(0)
ksp_cal = fft(mps, dim=[-2,-1])
ksp_cal = synth_cal(ksp_cal, (64, 64))
_, evals = csm_from_espirit(ksp_cal.to(torch_dev), mps.shape[1:])
evals = evals.cpu()


# Save
save_path = './data/magnus_spi/'
torch.save(phis, save_path + 'phis.pt')
torch.save(alphas, save_path + 'alphas.pt')
torch.save(mps, save_path + 'mps.pt')
torch.save(data['ksp'][:, slc], save_path + 'ksp.pt')
torch.save(data['trj'], save_path + 'trj.pt')
torch.save(data['dcf'], save_path + 'dcf.pt')
torch.save(evals, save_path + 'evals.pt')
torch.save(img_gt, save_path + 'img_gt.pt')