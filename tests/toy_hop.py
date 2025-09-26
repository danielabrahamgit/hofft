import torch
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mr_recon.algs import density_compensation
from mr_recon.linops import batching_params, sense_linop
from mr_recon.recons import CG_SENSE_recon, coil_combine, FISTA_recon
from mr_recon.utils import gen_grd, normalize, np_to_torch, torch_to_np
from mr_recon.fourier import fft, ifft, sigpy_nufft
from mr_sim.physics import GAMMA_BAR
from mr_sim.field_sim import sim_b0
from mr_recon.spatial import spatial_interp, spatial_resize_poly
from mr_recon.imperfections.field import alpha_segementation, b0_to_phis_alphas

from scipy.ndimage import gaussian_filter

from hofft import als_hofft, kb_nufft, multi_apod_kern_linop, funcs_to_phase

# Set seeds
torch.manual_seed(0)
np.random.seed(0)

# load data
fov = 0.22
dt = 2e-6
torch_dev = torch.device(4)
# torch_dev = torch.device('cpu')
data = np.load('/local_mount/space/mayday/data/users/abrahamd/hofft/simulated_data/data.npz')
img = np_to_torch(data['img']).to(torch_dev)
mps = np_to_torch(data['mps']).type(torch.complex64).to(torch_dev)
trj = np_to_torch(data['trj']).type(torch.float32).to(torch_dev)
# ksp = np_to_torch(data['ksp']).type(torch.complex64).to(torch_dev)
# ksp_cal = np_to_torch(data['ksp_cal']).type(torch.complex64).to(torch_dev)
im_size = img.shape
dcf = density_compensation(trj, im_size)

# Simulate data with smooth b0 map
cycles = 3
b0_max = cycles / (dt * trj.shape[0])
rs = gen_grd(im_size).to(torch_dev)
b0 = (rs[...,0] ** 2 + rs[...,1] ** 2) * (img.abs() > 0)
b0 = np_to_torch(gaussian_filter(b0.cpu(), sigma=15)).to(torch_dev)
b0 = b0 / b0.abs().max() * b0_max
b0, ts = b0_to_phis_alphas(b0, dcf.shape, 0, dt)
L_gt = 40
b_gt, h_gt = alpha_segementation(b0, ts,
                                 L=L_gt,
                                 L_batch_size=L_gt,
                                 interp_type='lstsq')
# b_gt = img[None,] * 0 + 1
# h_gt = dcf[None] * 0 + 1
A_gt = sense_linop(trj, mps,
                   spatial_funcs=b_gt,
                   temporal_funcs=h_gt,)
ksp = A_gt(img)

# hofft params
os = round(1.25 * im_size[0]) / im_size[0]
L = 3
kern_size = (5,)*2

# Train hofft over grid 
phis_size = (100,)*2
alpha_size = (50,)*3
rs = gen_grd(im_size).to(torch_dev).moveaxis(-1, 0)
kdevs = trj - (trj * os).round() / os
phis = torch.cat([rs, b0], dim=0)
alphas = torch.cat([kdevs.T, ts], dim=0)
alphas = alphas[2:]
phis = phis[2:]
phis_small = spatial_resize_poly(phis, phis_size, order=3)
# apods_init = torch.ones((L, *phis_small.shape[1:]), device=torch_dev, dtype=torch.float32)
apods_init = 'seg'
weights, apods = als_hofft(phis_small, alphas,
                           im_size, kern_size,
                           apods_init=apods_init,
                           os=os, L=L,
                           num_als_iter=100,
                           use_type3=False)
# weights, apods = kb_nufft(alphas[:2, ..., 0].moveaxis(0, -1), im_size, kern_size, os)
# weights, apods = kb_nufft(trj, im_size, kern_size, os)

for l in range(L):
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.imshow(apods[l].abs().cpu(), cmap='gray')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(apods[l].angle().cpu(), cmap='jet', vmin=-np.pi, vmax=np.pi)
    plt.axis('off')
    plt.tight_layout()
plt.show()
quit()

# Ground truth phase
phz_gt = torch.exp(-2j * np.pi * b0[0].cpu() * ts[0, ::250, None, None].cpu()).angle()
phz_hofft = funcs_to_phase(weights[..., ::250], apods, os).cpu().angle()

# Plot b0 phase
fig, axs = plt.subplots(1, 2)
frames = torch.arange(phz_gt.shape[0])
im_gt = axs[0].imshow(phz_gt[0], cmap='jet', vmin=-np.pi, vmax=np.pi)
img_hofft = axs[1].imshow(phz_hofft[0], cmap='jet', vmin=-np.pi, vmax=np.pi)
axs[0].axis('off')
axs[1].axis('off')
fig.tight_layout()

# Update function for animation
def update(frame):
    im_gt.set_array(phz_gt[frame])
    img_hofft.set_array(phz_hofft[frame])
    return [im_gt, img_hofft]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

# Save the animation as a GIF. 
# Option 1: Using PillowWriter (requires pillow package)
from matplotlib.animation import PillowWriter
writer = PillowWriter(fps=30)
ani.save(f"pw{cycles}_L{L}.gif", writer=writer)#, dpi=200)
