import torch
import numpy as np

import matplotlib as mpl
mpl.use("webAgg")
import matplotlib.pyplot as plt

from mr_recon.recons import CG_SENSE_recon
from mr_recon.linops import sense_linop
from mr_recon.fourier import sigpy_nufft

from hofft.phase_coeffs import b0_to_phis_alphas, coco_to_phis_alphas, rescale_phis_alphas
from hofft.utils import gen_grd
from hofft.pipelines import alpha_seg_decomp_linop
from hofft.decomp import hofft_params
from hofft.matvec import matvec_cur

# Params
hparams = hofft_params(kern_size=(2,)*3, 
                       os=1.25,
                       L=15,
                       reduced_im_size=(100,)*3,
                       spatial_init='seg',
                       time_reduction_factor=50,
                       matvec_kwargs={'spatial_batch_size': 2 ** 8, 'verbose': True},
                    #    matvec_type=matvec_cur,
                    #    matvec_kwargs={'rank_phi': 500, 'rank_alpha': 500},
                       normalize_coeffs=True,
                       kalpha_method='maxmin',
                       verbose=True,
                       )
cg_kwargs = {'max_iter': 10, 'max_eigen': 1.0, 'verbose': hparams.verbose}

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
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0]
mask = (mps.abs().sum(dim=0) > 0).type(torch.float32)

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

# phis = phis_b0
# alphas = alphas_b0
phis = torch.cat([phis_b0, phis_coco], dim=0)
alphas = torch.cat([alphas_b0, alphas_coco], dim=0)

# Recon
A = alpha_seg_decomp_linop(phis, alphas, mps, trj, 
                           dcf=dcf,
                           hparams=hparams,
                           use_sigpy=True)
# nft = sigpy_nufft(im_size, oversamp=hparams.os, width=hparams.kern_size[0])
# nft.beta = nft.optimal_beta(torch_dev=device)
# A = sense_linop(trj, mps, dcf=dcf, nufft=nft)
img = CG_SENSE_recon(A, ksp, **cg_kwargs)

torch.save(img.cpu(), './data/mrf_coco/img_gt.pt')
quit()

im = img[..., 96].rot90().abs().cpu()
vmax = im.median() + 3 * im.std()
plt.imshow(im, cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.tight_layout()
plt.show()
