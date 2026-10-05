import torch
import mrinufft as mn

import matplotlib as mpl
mpl.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.algs import density_compensation
from mr_recon.fourier import matrix_nufft, sigpy_nufft
from mr_sim.phantoms import shepp_logan

# Sim params
gamma_bar = 42.5774e6 # Hz / T
N = 220
fov = 0.22
dt = 2e-6
tread = 1e-3
nrot = 0
nshots = round(torch.pi * N * N / 2)
im_size = (N,)*3
nread = round(tread / dt)
torch_dev = torch.device(5)

# Design trajectory
if nrot > 0:
    trj = mn.initialize_3D_cones(
        nshots,
        nread,
        in_out=False,
        spiral=1.0,
        nb_zigzags=nrot,
        width=2.0,
    )
else:
    trj = mn.initialize_3D_phyllotaxis_radial(
        nshots,
        nread,
        in_out=False,
    )
    # trj = mn.initialize_3D_golden_means_radial(
    #     nshots,
    #     nread,
    #     in_out=False,
    # )
trj = torch.from_numpy(trj).type(torch.float32).to(torch_dev)
trj = trj.swapaxes(0, 1) * N
print(f'Trj shape: {trj.shape}')

# Compute gradient and slew 
grad = trj.diff(dim=0) / fov / dt / gamma_bar
slew = grad.diff(dim=0) / dt
print(f'Gmax = {grad.abs().max()*1e3:.2f} mT/m \nSmax = {slew.abs().max():.2f} T/m/s')

# Compute density compensation
dcf = density_compensation(trj, im_size)

# Simulate data
shp = shepp_logan(torch_dev)
img = shp.img(im_size)
# nufft_gt = matrix_nufft(im_size, 
#                         spatial_batch_size=2**6, 
#                         verbose=True)
nufft_gt = sigpy_nufft(im_size, 
                        oversamp=2.0, 
                        width=4)
ksp = nufft_gt(img[None,], trj[None,])[0]

# Save data
fdir = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/shepp_nufft'
torch.save(trj.cpu(), f'{fdir}/trj.pt')
torch.save(dcf.cpu(), f'{fdir}/dcf.pt')
torch.save(ksp.cpu(), f'{fdir}/ksp.pt')
torch.save(img.cpu(), f'{fdir}/img.pt')

# # Try simple recon
# from mr_recon.utils import normalize
# from mr_recon.linops import sense_linop
# from mr_recon.recons import CG_SENSE_recon
# nft = sigpy_nufft(im_size, oversamp=1.25, width=3)
# nft.beta = nft.optimal_beta(torch_dev=torch_dev)
# mps = torch.ones((1, *im_size), dtype=torch.complex64, device=torch_dev)
# ksp = ksp[None,]
# A = sense_linop(trj, mps, dcf, 
#                 nufft=nft)
# img_recon = CG_SENSE_recon(A, ksp, max_eigen=1.0, max_iter=20)

# # Plot
# img = img.cpu()
# img_recon = normalize(img_recon.cpu(), img)
# vmax = img.abs().median() + 3 * img.abs().std()
# slc = (slice(None), slice(None), N//2)
# imgs = [img, img_recon]
# plt.figure(figsize=(14, 7))
# for i in range(len(imgs)):
#     plt.subplot(1, 2, i+1)
#     plt.imshow(imgs[i].abs()[slc].rot90(), cmap='gray', vmin=0, vmax=vmax)
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

# # 3D plot
# trj_plt = trj.cpu()
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# for k in range(trj_plt.shape[1]):
#     ax.plot(*trj_plt[:, k, :].T, color='black', alpha=0.2)
# ax.plot(*trj_plt[:, 0, :].T, color='red', alpha=1.0)

# # Equal axes
# ax.set_aspect('equal')
# ax.set_xlim(-N/2, N/2)
# ax.set_ylim(-N/2, N/2)
# ax.set_zlim(-N/2, N/2)
# ax.axis('off')
# plt.show()
