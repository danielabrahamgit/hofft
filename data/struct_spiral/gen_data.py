import torch
import numpy as np

import matplotlib as mpl
mpl.use('Webagg')
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from mr_recon.algs import density_compensation


# Load 60 shot data
fpath = '/local_mount/space/tiger/1/users/abrahamd/mr_data/60_shot/data'
trj = torch.from_numpy(np.load(f'{fpath}/trj.npy'))
dcf = torch.from_numpy(np.load(f'{fpath}/dcf.npy'))
ksp = torch.from_numpy(np.load(f'{fpath}/ksp.npy'))
mps = torch.from_numpy(np.load(f'{fpath}/mps.npy'))
evals = torch.from_numpy(np.load(f'{fpath}/evals.npy'))

# Save as tensor 
alphas = torch.ones((1, *dcf.shape), dtype=torch.float32)
phis = torch.ones((1, *evals.shape), dtype=torch.float32)
save_dir = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/struct_spiral'
torch.save(trj, f'{save_dir}/trj.pt')
torch.save(dcf, f'{save_dir}/dcf.pt')
torch.save(ksp, f'{save_dir}/ksp.pt')
torch.save(mps, f'{save_dir}/mps.pt')
torch.save(evals, f'{save_dir}/evals.pt')
torch.save(alphas, f'{save_dir}/alphas.pt')
torch.save(phis, f'{save_dir}/phis.pt')