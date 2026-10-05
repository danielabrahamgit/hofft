import torch
import numpy as np

from hofft.phase_coeffs import b0_to_phis_alphas, coco_to_phis_alphas
from hofft.utils import gen_grd

# Load data
fpath = '/local_mount/space/tiger/1/users/abrahamd/mr_data/mrf_coco/data_11ms'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ksp = torch.from_numpy(np.load(f'{fpath}/ksp.npy')).type(torch.complex64).to(device)
mps = torch.from_numpy(np.load(f'{fpath}/mps.npy')).type(torch.complex64).to(device)
trj = torch.from_numpy(np.load(f'{fpath}/trj.npy')).type(torch.float32).to(device)
dcf = torch.from_numpy(np.load(f'{fpath}/dcf.npy')).type(torch.float32).to(device)
b0  = torch.from_numpy(np.load(f'{fpath}/b0.npy')).type(torch.float32).to(device)
dcf /= dcf.abs().max()
ksp /= ksp.abs().max()
im_size = b0.shape
dt = 5e-6
fov = 0.22

# Show shapes
print(f"ksp shape: {ksp.shape}")
print(f"mps shape: {mps.shape}")
print(f"trj shape: {trj.shape}")
print(f"dcf shape: {dcf.shape}")
print(f"b0 shape: {b0.shape}")

# Get phase coefficients
phis_b0, alphas_b0 = b0_to_phis_alphas(b0, dcf.shape,
                                       ro_dim=0, dt=dt,
                                       repeat_empty_dims=True)
ofs = torch.tensor([0.0, 0.0, 0.0], device=device)
crds = gen_grd(im_size, (fov,)*3).to(device)
crds += ofs
trj_phys = trj / fov
phis_coco, alphas_coco = coco_to_phis_alphas(trj_phys, crds, 
                                             field_strength=0.55, 
                                             ro_dim=0, dt=dt)

phis = torch.cat([phis_b0, phis_coco], dim=0)
alphas = torch.cat([alphas_b0, alphas_coco], dim=0)

# Save data
save_path = './data/mrf_coco'
torch.save(ksp.cpu(), f'{save_path}/ksp.pt')
torch.save(mps.cpu(), f'{save_path}/mps.pt')
torch.save(trj.cpu(), f'{save_path}/trj.pt')
torch.save(dcf.cpu(), f'{save_path}/dcf.pt')
torch.save(b0.cpu(), f'{save_path}/b0.pt')
torch.save(phis.cpu(), f'{save_path}/phis.pt')
torch.save(alphas.cpu(), f'{save_path}/alphas.pt')