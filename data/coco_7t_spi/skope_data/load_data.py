import os
import glob
import torch
import numpy as np

import matplotlib as mpl
mpl.use('webAgg')
import matplotlib.pyplot as plt

from mr_sim.phantoms import shepp_logan
from mr_recon.utils import gen_grd, cvplot
from hofft.phase_coeffs import sph_bases, coco_bases, rescale_phis_alphas
from einops import einsum, rearrange

def load_skope(data_folder: str, 
               scan_id: int | str, 
               stat: str) -> torch.Tensor:
    """
    Load flattened 3D Skope data from file and return properly shaped array.

    Parameters
    ----------
    data_folder : str
        Folder containing the file.
    scan_id : int or str
        Scan identifier (file begins with this).
    stat : str
        One of 'raw' (B=16), 'kspha' (B=16), or 'kcoco' (B=4).

    Returns
    -------
    data : np.ndarray
        Array of shape (B, len(dynamics) * nt)
    """
    assert stat in ['raw', 'kspha', 'kcoco'], "stat must be 'raw', 'kspha', or 'kcoco'!"

    # Build the file path pattern
    pattern = os.path.join(data_folder, f"{scan_id}*.{stat}")
    filepath = glob.glob(pattern)[0]  # guaranteed exactly one file

    # Determine dtype
    match stat:
        case 'kspha':
            dtype = '>f8'  # big-endian float64
            B = 16
        case 'kcoco':
            dtype = '>f8'  # big-endian float64
            B = 4
        case 'raw':
            dtype = '>c8'  # big-endian complex64
            B = 16

    # Read binary data
    data = np.fromfile(filepath, dtype=dtype)
    data = data.reshape((-1, B)).T
    data = data.astype(data.dtype.newbyteorder('='))  # torch requires native byte order
    if np.iscomplexobj(data):
        data = torch.from_numpy(data).to(torch.complex64)
    else:
        data = torch.from_numpy(data).to(torch.float32)
    return data

im_size = (320,)*3
fov = 0.24 # meters

# Each file is 18 groups x 250 TRs, stored as (G, T, R) with R fastest.
# Part 1 (scan 6) is even TRs; part 2 (scan 8) is odd TRs.
data_coco_even = load_skope('/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_7t_spi/skope_data/raw', 6, 'kcoco')
data_spha_even = load_skope('/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_7t_spi/skope_data/raw', 6, 'kspha')
data_coco_odd = load_skope('/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_7t_spi/skope_data/raw', 8, 'kcoco')
data_spha_odd = load_skope('/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_7t_spi/skope_data/raw', 8, 'kspha')
data_coco_even = rearrange(data_coco_even, 'B (G T R) -> B R G T', R=23_000, T=250)
data_spha_even = rearrange(data_spha_even, 'B (G T R) -> B R G T', R=23_000, T=250)
data_coco_odd = rearrange(data_coco_odd, 'B (G T R) -> B R G T', R=23_000, T=250)
data_spha_odd = rearrange(data_spha_odd, 'B (G T R) -> B R G T', R=23_000, T=250)

# Interleave TRs: even from part1 (scan 6), odd from part2 (scan 8)
data_coco = torch.zeros((data_coco_even.shape[0], data_coco_even.shape[1], data_coco_even.shape[2], data_coco_even.shape[3] * 2))
data_spha = torch.zeros((data_spha_even.shape[0], data_spha_even.shape[1], data_spha_even.shape[2], data_spha_even.shape[3] * 2))
data_coco[:, :, :, ::2] = data_coco_even
data_spha[:, :, :, ::2] = data_spha_even
data_coco[:, :, :, 1::2] = data_coco_odd
data_spha[:, :, :, 1::2] = data_spha_odd

# trj = torch.load('/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_7t_spi/trj.pt')
# ofs = 46
# trj_skp = data_spha[1:4, ofs:ofs + trj.shape[0], :trj.shape[1], :] * fov / (2 * torch.pi)

data = torch.cat([data_coco, data_spha], dim=0)
crds = gen_grd(im_size)[:, :, :, :] * fov
phis_coco = coco_bases(crds[..., 0], crds[..., 1], crds[..., 2])
phis_spha = sph_bases(crds[..., 0], crds[..., 1], crds[..., 2])
phis = torch.cat([phis_coco, phis_spha], dim=0)
alphas = data / (2 * torch.pi)

# Save data
torch.save(phis, '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_7t_spi/phis.pt')
torch.save(alphas, '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_7t_spi/alphas.pt')
quit()

# Normalize
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)
phis = phis_nrm #+ phis_mp[:, None, None, None]
alphas = alphas_nrm + alphas_mp[:, None,]

plt.figure()
for b in range(12):
    plt.plot(alphas[b+4], label=f'b={b}')
plt.legend()
plt.show()
quit()


phz_coco = torch.exp(-2j * torch.pi * einsum(
    phis[:4], alphas[:4, ::10], 'B ..., B T -> T ...'
))
phz_spha = torch.exp(-2j * torch.pi * einsum(
    phis[4:], alphas[4:, ::10], 'B ..., B T -> T ...'
))

cvplot(phz_coco.angle(), phz_spha.angle())
# plt.plot(data[1])
# plt.show()
