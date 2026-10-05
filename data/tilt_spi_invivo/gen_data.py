import torch

from scipy.ndimage import gaussian_filter
from mr_recon.fourier import fft
from mr_recon.multi_coil.calib import synth_cal
from mr_recon.multi_coil.coil_est import csm_from_espirit

fpath = '/local_mount/space/mayday/data/users/zachs/share/ForDaniel/hoftt_datasets/20260813_7t_300um_invivo'
data = torch.load(fpath + '/data_corv1.pt', map_location=torch.device('cpu'))
recons = torch.load(fpath + '/recon_corv1.pt', map_location=torch.device('cpu'))
slc = 0
img_gt = recons[..., slc]
ksp = data['ksp'][:, slc]
trj = data['trj']
dcf = data['dcf']
mps = data['mps'][..., slc]
phis_b0 = data['phis_b0'][None, ..., slc]
alphas_b0 = data['alphas_b0'][None,]
phis_kspha = data['phis_kspha'][..., slc]
alphas_kspha = data['alphas_kspha']

# Smooth b0 to 0.9mm, since it was acquired with a 0.9mm multi echo GRE
# npix = round(0.9e-3 / 0.3e-3)
npix = round(3e-3 / 0.3e-3)
b0_smooth = gaussian_filter(phis_b0.cpu().numpy(), sigma=npix)
phis_b0 = torch.from_numpy(b0_smooth)

# Stack b0, coco, and eddy coeffs
try:
    phis_coco = data['phis_coco'][..., slc]
    alphas_coco = data['alphas_coco']
    phis = torch.cat([phis_b0, phis_coco, phis_kspha], dim=0)
    alphas = torch.cat([alphas_b0, alphas_coco, alphas_kspha], dim=0)
except:
    phis = torch.cat([phis_b0, phis_kspha], dim=0)
    alphas = torch.cat([alphas_b0, alphas_kspha], dim=0)

# from hofft.phase_coeffs import visialize_alpha_space
# import matplotlib as mpl
# mpl.use('WebAgg')
# import matplotlib.pyplot as plt
# visialize_alpha_space(phis, alphas, B_compressed=3)
# plt.show()
# quit()


# Get mask 
torch_dev = torch.device(0)
ksp_cal = fft(mps, dim=[-2,-1])
ksp_cal = synth_cal(ksp_cal, (64, 64))
_, evals = csm_from_espirit(ksp_cal.to(torch_dev), mps.shape[1:])
evals = evals.cpu()

# Save data
save_path = './data/tilt_spi_invivo'
torch.save(img_gt, save_path + '/img_gt.pt')
torch.save(trj, save_path + '/trj.pt')
torch.save(dcf, save_path + '/dcf.pt')
torch.save(ksp, save_path + '/ksp.pt')
torch.save(mps, save_path + '/mps.pt')
torch.save(phis, save_path + '/phis.pt')
torch.save(evals, save_path + '/evals.pt')
torch.save(alphas, save_path + '/alphas.pt')