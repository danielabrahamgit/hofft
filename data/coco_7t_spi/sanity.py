import torch
import gc

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from time import perf_counter
from tqdm import tqdm
from itertools import product

from mr_recon.fourier import sigpy_nufft, ifft
from mr_recon.recons import CG_SENSE_recon
from mr_recon.utils import gen_grd, normalize, cvplot
from mr_recon.algs import density_compensation
from mr_recon.linops import batching_params, encoding_matrix, sense_linop
from mr_recon.imperfections.field import alpha_segementation, phi_alpha_svd
from mr_recon.multi_coil.calib import synth_cal, calc_coil_subspace
from mr_recon.multi_coil.coil_est import csm_from_espirit
from mr_recon.spatial import fourier_resize

from hofft.pipelines import svd_decomp_linop, sparse_hofft_decomp_linop, qblock_svd_decomp_linop
from hofft.utils import reduce_spatial, expand_spatial
from hofft.decomp import hofft_params
from hofft.sparse_fit import sparse_params
from hofft.matvec import matvec_cur
from hofft.forward_model import hofft_linop, hofft_compressed_linop
from hofft.phase_coeffs import (
  remove_linear_terms,
  visualize_alpha_space,
  coco_to_phis_alphas, 
  b0_to_phis_alphas, 
  rescale_phis_alphas, 
  apply_phase_midpoints,
  trj_dev_to_phis_alphas,
  whiten_phis_alphas,
)

def list_tensors_by_memory(device=None):
    """Lists all active PyTorch tensors sorted by their memory usage.

    Args:
        device (str or torch.device, optional): Filter by 'cuda' or 'cpu'.
    """
    tensor_list = []

    # Iterate through all objects tracked by the garbage collector
    for obj in gc.get_objects():
        try:
            # Check if it's a tensor
            if torch.is_tensor(obj):
                # Calculate true memory size in bytes
                num_elements = obj.nelement()
                element_size = obj.element_size()
                total_bytes = num_elements * element_size

                # Optional filtering by device (torch.device or 'cuda'/'cpu')
                if device is not None:
                    want = torch.device(device) if not isinstance(device, torch.device) else device
                    if obj.device.type != want.type:
                        continue
                    if want.index is not None and obj.device.index != want.index:
                        continue

                tensor_list.append(
                    {
                        "tensor_id": id(obj),
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "device": str(obj.device),
                        "size_mb": total_bytes / (1024**2),
                    }
                )
        except Exception:
            # Handle objects that raise errors during inspection
            continue

    # Sort tensors by size descending
    tensor_list.sort(key=lambda x: x["size_mb"], reverse=True)

    # Print results
    print(
        f"{'Tensor ID':<15} | {'Shape':<25} | {'Dtype':<12} | {'Device':<10} | {'Size (MB)':<10}"
    )
    print("-" * 72)
    for t in tensor_list:
        print(
            f"{t['tensor_id']:<15} | {str(t['shape']):<25} | {t['dtype']:<12} | {t['device']:<10} | {t['size_mb']:10.4f}"
        )


# Params
torch.manual_seed(0)
torch_dev = torch.device('cuda:0')

# Load full data
trj = torch.load('./data/coco_7t_spi/trj.pt', weights_only=True, map_location=torch_dev)
dcf = torch.load('./data/coco_7t_spi/dcf.pt', weights_only=True, map_location=torch_dev)
ksp = torch.load('./data/coco_7t_spi/ksp.pt', weights_only=True, map_location=torch_dev)
evals = torch.load('./data/coco_7t_spi/evals.pt', weights_only=True, map_location=torch_dev)
mps = torch.load('./data/coco_7t_spi/mps.pt', weights_only=True, map_location=torch_dev)
b0 = torch.load('./data/coco_7t_spi/b0.pt', weights_only=True, map_location=torch_dev)
phis = torch.load('./data/coco_7t_spi/phis.pt', weights_only=True, map_location=torch_dev)
alphas = torch.load('./data/coco_7t_spi/alphas.pt', weights_only=True, mmap=True, map_location='cpu')
pred_0 = torch.load('./data/coco_7t_spi/pred_0.pt', weights_only=True, map_location=torch_dev)
coco_term = torch.load('./data/coco_7t_spi/coco_term.pt', weights_only=True, map_location=torch_dev)
ofs = 46
alphas = alphas[:, ofs:ofs+trj.shape[0], :6].to(torch_dev)
im_size = b0.shape
fov = 0.24
C = mps.shape[0]

# Use auto-fitted B0
b0 = torch.load('./data/coco_7t_spi/fit_db0_loop/b0_final.pt', weights_only=True, map_location=torch_dev)['b0']

mask = (evals > 0.95).float()

# Compress like crazy
_, ksp, mps = calc_coil_subspace(ksp[:, :10_000:4, :, ::10], 8, ksp, mps)
C = mps.shape[0]

# Remove zeroth order scanner corrections
ksp *= torch.exp(1j * pred_0)
ksp *= torch.exp(1j * coco_term)

# Crop data down to lower res just for testing
# ros = slice(0, 4_000)
M = 4
# ros = slice(None, 15_000, M)
ros = slice(None, None, M)
ksp = ksp[:, ros].contiguous()
trj = trj[ros].contiguous()
dcf = dcf[ros].contiguous()
alphas = alphas[:, ros].contiguous()
trj_size = dcf.shape

# Resample image domain to same lower res
kmax = trj.abs().max()
N_new = round(kmax.item()) * 2
im_size = (N_new,)*3
mps = reduce_spatial(mps, im_size)
evals = reduce_spatial(evals, im_size)
mask = reduce_spatial(mask, im_size)
b0 = reduce_spatial(b0, im_size)
phis = reduce_spatial(phis, im_size)

# Print shapes
print(f'trj shape: {trj.shape}')
print(f'dcf shape: {dcf.shape}')
print(f'ksp shape: {ksp.shape}')
print(f'evals shape: {evals.shape}')
print(f'mps shape: {mps.shape}')
print(f'b0 shape: {b0.shape}')
print(f'phis shape: {phis.shape}')
print(f'alphas shape: {alphas.shape}')

from hofft.phase_coeffs import sph_bases, coco_bases
shifts = torch.tensor([-0.01, 0.0, 0.0], device=torch_dev)
crds = gen_grd(im_size).to(torch_dev) * fov + shifts * 0
sgns_crds = torch.ones(3, device=torch_dev)
sgns_coco = torch.ones(4, device=torch_dev)
signs_spha = torch.ones(16, device=torch_dev)
signs_spha[0] = -1
signs_spha[4::2] = -1
crds *= sgns_crds
phis_coco = coco_bases(crds[..., 0], crds[..., 1], crds[..., 2])
phis_coco *= sgns_coco[:, None, None, None]
phis_spha = sph_bases(crds[..., 0], crds[..., 1], crds[..., 2])
phis_spha *= signs_spha[:, None, None, None]
phis = torch.cat([phis_coco, phis_spha], dim=0)

# from hofft.phase_coeffs import coco_to_phis_alphas
# phis_coco_check, alphas_coco_check = coco_to_phis_alphas(trj, crds, 6.98, 0, M*1e-6)
# # breakpoint()
# alphas[:4] = alphas_coco_check[:4]
# alphas[:4] /= 2 * torch.pi

# Apply linear terms
# idxs = slice(4, None)
idxs = slice(None)
phis, lin, zero = remove_linear_terms(phis[idxs], alphas[idxs], mask)
trj = lin
ksp *= torch.exp(2j * torch.pi * zero)
# trj = alphas[5:8].moveaxis(0, -1) * fov
# ksp *= torch.exp(2j * torch.pi * alphas[4])

# Stack B0 term
phis, alphas = b0_to_phis_alphas(-b0, trj_size, ro_dim=0, dt=M*1e-6, repeat_empty_dims=False)
# phis_b0, alphas_b0 = b0_to_phis_alphas(-b0, trj_size, ro_dim=0, dt=M*1e-6, repeat_empty_dims=True)
# alphas = torch.cat([alphas_b0, alphas], dim=0)
# phis = torch.cat([phis_b0, phis], dim=0)

# Clear mem
del evals, pred_0, coco_term, b0#, phis_coco, phis_spha
gc.collect()
torch.cuda.empty_cache()

# High Order Correction
L = 20
hparams = hofft_params(kern_size=(3,)*3, os=1.25, L=L,
                       reduced_im_size=(100,)*3,
                       time_reduction_factor=10,
                       cur_rank=500)
bparams = batching_params(coil_batch_size=C, field_batch_size=1)
A = svd_decomp_linop(phis, alphas, mps, trj, hparams, 
                     svd_method='cur',
                     spatial_mask=mask,
                     dcf=dcf,
                     bparams=bparams,
                     use_sigpy=True)
gc.collect()
torch.cuda.empty_cache()

img_recon = CG_SENSE_recon(A, ksp, max_iter=5, max_eigen=1.0).cpu()

if im_size[0] != 320:
    im_size = (320,)*3
    img_recon = fourier_resize(img_recon, im_size, window_after_crop=False)


plt.figure(figsize=(14, 7))
zs = [im_size[2]//2 - 20, im_size[2]//2, im_size[2]//2 + 20]
for i, z in enumerate(zs):
  plt.subplot(1, 3, i+1)
  plt.imshow(img_recon.cpu()[..., z].abs().rot90().cpu(), cmap='gray')
  # plt.imshow(b0.cpu()[..., z].rot90().cpu(), cmap='jet')
  plt.axis('off')
plt.tight_layout()
plt.show()