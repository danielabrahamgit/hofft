import torch

from typing import Optional

from mr_recon.imperfections.field import b0_to_phis_alphas
from mr_recon.utils import gen_grd
from mr_recon.linops import linop, batching_params
from hofft.model import hofft_params, multi_apod_kern_linop
from hofft.apply import als_hofft

def b0_correction(b0: torch.Tensor,
                  trj: torch.Tensor,
                  mps: torch.Tensor,
                  dcf: Optional[torch.Tensor],
                  ro_dim: int,
                  dt: float,
                  hparams: hofft_params,
                  bparams: Optional[batching_params] = batching_params(),
                  num_als_iter=100) -> linop:
    """
    Creates a HOFFT linop for B0 correction.
    
    Args
    ----
    b0 : torch.Tensor
        B0 map of shape (*solve_size) in Hz
    trj : torch.Tensor
        Trajectory of shape (*trj_size, d), where d = len(im_size)
    mps : torch.Tensor
        Coil sensitivity maps of shape (C, *im_size), where C is the number of coils
    ro_dim : int
        Readout dimension in trj
    dt : float
        Sampling dwell time in seconds
    hparams : hofft_params
        hofft parameters.
    bparams : batching_params
        batching parameters linop
    num_als_iter : int
        number of ALS iterations to run.
    
    Returns
    -------
    linop
        HOFFT linop for B0 correction
    """
    # Consts
    torch_dev = b0.device
    solve_size = b0.shape
    im_size = mps.shape[1:]
    trj_size = trj.shape[:-1]
    
    # Make phis and alphas
    phis_b0, alphas_b0 = b0_to_phis_alphas(b0, trj_size, ro_dim, dt)
    phis_kdev = gen_grd(solve_size).to(torch_dev).moveaxis(-1, 0) # (d, *im_size)
    trj_grd = (trj * hparams.os).round() / hparams.os
    alphas_kdev = (trj - trj_grd).moveaxis(-1, 0) # (d, *trj_size)
    alphas = torch.cat([alphas_b0.expand_as(alphas_kdev)[:1], alphas_kdev], dim=0) # (B+d, *trj_size)
    phis = torch.cat([phis_b0, phis_kdev], dim=0) # (B+d, *im_size)
    
    # # Call ALS hofft
    # weights, apods = als_hofft(phis, alphas, im_size, hparams, 
    #                            num_als_iter=num_als_iter)
    weights = torch.zeros((hparams.L, *hparams.kern_size, *trj_size), device=torch_dev, dtype=torch.complex64)
    apods = torch.zeros((hparams.L, *im_size), device=torch_dev, dtype=torch.complex64)
    
    # Build linop
    linop = multi_apod_kern_linop(trj_grd, mps, 
                                  weights=weights, 
                                  apods=apods,
                                  dcf=dcf,
                                  os_grid=hparams.os,
                                  bparams=bparams)
    
    return linop