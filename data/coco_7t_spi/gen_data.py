from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import mapvbvd
from scipy.io import loadmat

import matplotlib as mpl
mpl.use('Webagg')
import matplotlib.pyplot as plt

from hofft.phase_coeffs import coco_to_phis_alphas
from mr_recon.algs import density_compensation
from mr_recon.utils import gen_grd, cvplot
from mr_recon.fourier import ifft
from mr_recon.spatial import spatial_resize_poly
from mr_recon.multi_coil.calib import calc_coil_subspace, synth_cal
from mr_recon.multi_coil.coil_est import csm_from_espirit
from einops import rearrange
from tqdm import tqdm


from load_raw_kspace import (
    DEFAULT_RAW,
    TwixLoader,
    assemble_matlab_raw,
    coil_compress_matlab,
    parse_flash_header,
    trim_matlab_raw,
)

# Run from repo root with mr_recon env, e.g.:
#   cd hofft && PYTHONPATH=src python data/coco_7t_spi/gen_data.py

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MAT = (
    '/local_mount/space/mayday/data/users/xc/share/for_Daniel/spiral_7T_flash/'
    'knew_v2_fov240_res075_int6_2213624000_t1t10_g10s400.mat'
)
N_TP_DESIGN = 22_136
N_SPIRAL = 18
N_SPIRAL_CIRCLE = 6
N_ROT = 500
BETA_STEP = 0.131267444 * np.pi
INIT_XITA = 0.0
INIT_BETA = 0.0
K_NORM = 0.5

fov = 0.24
im_size = (320,) * 3

RAW_DAT = "/local_mount/space/mayday/data/users/xc/data/Siemens2026/20260625_stanford_7t/meas_MID00037_FID02420_FLASHSpiral_v01a_tr30_te2.dat"
TR_STRIDE = 1
# SPIRAL_STRIDE = 3
SPIRAL_STRIDE = 18
N_POINT = N_TP_DESIGN
N_COMP_COIL = 16
COIL_COMPRESS = True

# torch_dev = torch.device('cpu')
torch_dev = torch.device('cuda:0')


def load_2d_ktrajectory(
    mat_path: str | Path,
    n_tp: int = N_TP_DESIGN,
) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(mat_path)
    if 'k_adc' not in mat:
        raise KeyError(f'{mat_path} must contain k_adc; found {list(mat.keys())}')

    k_adc = mat['k_adc']
    k_complex = k_adc[:n_tp, 0] + 1j * k_adc[:n_tp, 1]
    k_complex = k_complex / np.abs(k_complex).max() * K_NORM
    return k_complex.real.copy(), k_complex.imag.copy()


def build_k3d(
    kx: np.ndarray,
    ky: np.ndarray,
    n_spiral: int = N_SPIRAL,
    n_spiral_circle: int = N_SPIRAL_CIRCLE,
    n_rot: int = N_ROT,
    init_xita: float = INIT_XITA,
    init_beta: float = INIT_BETA,
    beta_step: float = BETA_STEP,
) -> np.ndarray:
    n_tp = kx.shape[0]
    k_3d = np.zeros((n_tp, 3, n_spiral, n_rot), dtype=np.float32)

    kx0 = kx.astype(np.float64)
    ky0 = ky.astype(np.float64)

    for nn in range(n_rot):
        for ii in range(n_spiral):
            beta = ((ii % n_spiral_circle) + 1 + nn - 1) * beta_step + init_beta
            xita = (ii % n_spiral_circle) * 2 * np.pi / n_spiral_circle + init_xita

            cos_x, sin_x = np.cos(xita), np.sin(xita)
            cos_b, sin_b = np.cos(beta), np.sin(beta)

            k_temp_x = cos_x * kx0 + sin_x * ky0
            k_temp_y = -sin_x * kx0 + cos_x * ky0

            if ii < n_spiral_circle:
                k_3d[:, 0, ii, nn] = k_temp_x
                k_3d[:, 1, ii, nn] = cos_b * k_temp_y
                k_3d[:, 2, ii, nn] = -sin_b * k_temp_y
            elif ii < 2 * n_spiral_circle:
                k_3d[:, 0, ii, nn] = -sin_b * k_temp_y
                k_3d[:, 1, ii, nn] = k_temp_x
                k_3d[:, 2, ii, nn] = cos_b * k_temp_y
            else:
                k_3d[:, 0, ii, nn] = cos_b * k_temp_y
                k_3d[:, 1, ii, nn] = -sin_b * k_temp_y
                k_3d[:, 2, ii, nn] = k_temp_x

    return k_3d


def main() -> None:
    
    # # Save B0 map
    # b0 = loadmat('/local_mount/space/mayday/data/users/xc/data/Siemens2026/20260625_stanford_7t/bss/B0_B1_map.mat')
    # b0 = torch.from_numpy(b0['B0_map']).type(torch.float32)
    # b0 = spatial_resize_poly(b0, im_size, order=3)
    # torch.save(b0.cpu(), SCRIPT_DIR / 'b0.pt')
    
    rep_indices = list(range(0, 6))
    n_tr = N_ROT

    print(f'Subsampling: {len(rep_indices)} Rep groups, {n_tr} TRs, {N_POINT} readout pts')

    # --- trajectory ---
    print(f'Loading 2D spiral design from {DEFAULT_MAT}')
    kx, ky = load_2d_ktrajectory(DEFAULT_MAT, n_tp=N_TP_DESIGN)
    k3d = build_k3d(kx, ky)
    k3d = k3d[:N_POINT, :, rep_indices, :n_tr]

    trj = torch.from_numpy(k3d).float().moveaxis(1, -1) * im_size[0]
    trj = trj.to(torch_dev) 

    # --- k-space ---
    # twix = mapvbvd.mapVBVD(RAW_DAT)
    # img = twix[-1].image
    # img.squeeze = True
    # img.removeOS = False
    # ksp = []
    # for i in tqdm(range(int(img.NRep)), desc='Loading k-space'):
    #     kspi = torch.from_numpy(np.asarray(img[:, :, :, :, i], dtype=np.complex64))
    #     kspi = rearrange(kspi, 'R C T E -> C (E R) T')
    #     ksp.append(kspi)
    # ksp = torch.stack(ksp, dim=-2).to(torch_dev)
    # _, ksp = calc_coil_subspace(ksp[:, :5000, 0, :], N_COMP_COIL, ksp)
    ofs = 3
    ksp = torch.load(SCRIPT_DIR / 'ksp_allread.pt', map_location=torch_dev)[:, ofs:ofs+trj.shape[0]]
    
    # --- dcf + concomitant fields ---
    print('Computing DCF')
    # dcf = density_compensation(trj, im_size)
    dcf = torch.load(SCRIPT_DIR / 'dcf.pt').to(torch_dev)
    
    # Compute coil sensitivity maps
    ksp_cal = synth_cal(ksp, (32,)*3, trj, dcf, num_iter=10, use_toeplitz=True)
    im_size_low = (150,)*3
    mps_low, evals_low = csm_from_espirit(ksp_cal, im_size_low)
    # mps_mag = spatial_resize_poly(mps_low.abs(), im_size, order=3)
    # mps_ang = spatial_resize_poly(mps_low.angle(), im_size, order=3)
    # mps = mps_mag * torch.exp(1j * mps_ang)
    mps = spatial_resize_poly(mps_low, im_size, order=3)
    evals = spatial_resize_poly(evals_low, im_size, order=3)
    
    print(f'Saving to {SCRIPT_DIR}')
    torch.save(trj.cpu(), SCRIPT_DIR / 'trj.pt')
    torch.save(dcf.cpu(), SCRIPT_DIR / 'dcf.pt')
    torch.save(ksp.cpu(), SCRIPT_DIR / 'ksp.pt')
    torch.save(mps.cpu(), SCRIPT_DIR / 'mps.pt')
    torch.save(evals.cpu(), SCRIPT_DIR / 'evals.pt')

    print('Computing concomitant phis / alphas')
    spatial_crds = gen_grd(im_size, (fov,) * 3).to(torch_dev)
    phis, alphas = coco_to_phis_alphas(
        trj / fov, spatial_crds, field_strength=7.0, ro_dim=0, dt=1e-6,
    )
    torch.save(phis.cpu(), SCRIPT_DIR / 'phis_coco.pt')
    torch.save(alphas.cpu(), SCRIPT_DIR / 'alphas_coco.pt')

if __name__ == '__main__':
    main()
