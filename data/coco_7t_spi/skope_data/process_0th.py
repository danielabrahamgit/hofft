import torch
import numpy as np

from scipy.io import loadmat

GAMMA_BAR = 42.5774e6  # Hz / T

def remove_coco_ecc(
    undo_eddy_pth: str,
    trj: torch.Tensor,
    dt: float = 1e-6,
    B0: float = 6.98,
    coco_z_isocenter_x: float = 0.098,
    coco_z_isocenter_y: float = 0.093,
    tshift: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    0th-order COCO and ECC phase predicted from the nominal trajectory.

    Both terms are in radians on the trajectory grid and are meant to be
    subtracted from the Skope B0 coefficient. Gradient is recovered from ``trj``
    as in ``coco_to_phis_alphas`` (``trj`` in 1/m).

    Args
    ----
    undo_eddy_pth : str
        Path to the UndoECC ``*_eddy_phase.mat`` file
    trj : torch.Tensor
        Nominal k-space trajectory with shape (T, ..., d), units of 1/m.
        Time is the leading axis; ``d >= 2`` with (kx, ky, ...)
    dt : float
        Dwell time in seconds
    B0 : float
        Main-field strength in Tesla
    coco_z_isocenter_x, coco_z_isocenter_y : float
        COCO z-isocenter offsets in meters
    flip_z, flip_all_but_z : bool
        RAS sign convention. ``flip_all_but_z`` negates the 0th-order terms;
        ``flip_z`` does not touch them
    tshift : int
        Gradient delay in samples used to window the ECC waveform

    Returns
    -------
    pred_0 : torch.Tensor
        ECC phase, shape (T, ...), radians, relative to t=0
    coco_term : torch.Tensor
        COCO B0 phase, shape (T, ...), radians, zero at t=0
    """

    # Gradient from trajectory: G = dk/dt / gamma_bar  [T/m]
    g = torch.diff(trj, dim=0) / (dt * GAMMA_BAR)
    g = torch.cat((g, g[-1:]), dim=0)
    gx, gy = g[..., 0], g[..., 1]

    # Siemens COCO B0 term at isocenter: dB = (Gx^2 phi_zx^2 + Gy^2 phi_zy^2) / (2 B0)
    coco_hz = (gx**2 * coco_z_isocenter_x**2 + gy**2 * coco_z_isocenter_y**2) / (2 * B0)
    coco_cyc = torch.cumulative_trapezoid(coco_hz, dx=dt, dim=0) * GAMMA_BAR
    coco_term = 2 * torch.pi * torch.cat([torch.zeros_like(coco_cyc[:1]), coco_cyc], dim=0)

    # ECC from UndoECC, windowed to the readout
    Nt = trj.shape[0]
    Nshot = int(np.prod(trj.shape[1:-1])) or 1
    eddy = loadmat(undo_eddy_pth)
    rx = eddy['RXSamples'].squeeze()
    dff = np.diff(rx, axis=0)
    idx_end = np.argwhere(dff == dff.min())[1].item()
    idx_start = np.argwhere(dff == dff.max())[0].item()
    ts, ttot = int(rx[idx_start]), int(rx[idx_end])
    ecc = eddy['ecc_phase'].squeeze().reshape(Nshot, -1).T[ts + tshift:ttot + tshift]
    pred_0 = torch.as_tensor(ecc[:Nt], dtype=trj.dtype, device=trj.device) * -1
    pred_0 = pred_0 - pred_0[:1]
    if trj.ndim > 2:
        pred_0 = pred_0.reshape(coco_term.shape)

    return pred_0, coco_term


fov = 0.24
trj = torch.load('/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_7t_spi/trj.pt')
trj = trj / fov

pred_0, coco_term = remove_coco_ecc(
    undo_eddy_pth='/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_7t_spi/skope_data/proc/ID6_eddy_phase.mat',
    trj=trj,
    dt=1e-6,
    B0=6.98,
    coco_z_isocenter_x=0.098,
    coco_z_isocenter_y=0.093,
    tshift=50,
)

torch.save(pred_0, '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_7t_spi/pred_0.pt')
torch.save(coco_term, '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_7t_spi/coco_term.pt')