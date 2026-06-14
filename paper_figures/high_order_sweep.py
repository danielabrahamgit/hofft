import torch
import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt
import argparse

from time import perf_counter
from tqdm import tqdm
from itertools import product

from mr_recon.fourier import sigpy_nufft
from mr_recon.recons import CG_SENSE_recon
from mr_recon.utils import gen_grd, normalize
from mr_recon.algs import density_compensation
from mr_recon.linops import batching_params, encoding_matrix, sense_linop
from mr_recon.imperfections.field import alpha_segementation, phi_alpha_svd

from hofft.pipelines import als_hofft, kb_nufft, als_hofft_compressed, als_hofft_kbinit
from hofft.decomp import hofft_params
from hofft.matvec import matvec_rnd
from hofft.forward_model import hofft_linop
from hofft.reduce import reduce_params, reduce_spatial, expand_spatial
from hofft.phase_coeffs import (
  coco_to_phis_alphas, 
  b0_to_phis_alphas, 
  rescale_phis_alphas, 
  apply_phase_midpoints,
  trj_dev_to_phis_alphas,
  compress_phis_alphas,
)

# Set random seed
torch.manual_seed(0)

# Parse arguments
args = argparse.ArgumentParser()
args.add_argument('--os', type=float, default=1.25)
args.add_argument('--Wrange', type=str, default='5,6,1')
args.add_argument('--Lrange', type=str, default='5,6,1')
args.add_argument('--gpu', type=int, default=None)
args.add_argument('--R', type=int, default=3)
args.add_argument('--B_compressed', type=int, default=None)
args.add_argument('--num_cg_iter', type=int, default=12)
args.add_argument('--num_als_iter', type=int, default=100*0)
args.add_argument('--spatial_init', type=str, default='100_alphas_10')
args.add_argument('--spatial_reduce_size', type=int, default=120)
args.add_argument('--anderson_order', type=int, default=None)
args.add_argument('--mask_thresh', type=float, default=None)
args.add_argument('--data_path', type=str, default='./data/coco_spiral')
args = args.parse_args()

# Load data
torch_dev = torch.device('cpu') if args.gpu is None else torch.device(args.gpu)
alphas = torch.load(f'{args.data_path}/alphas.pt', map_location=torch_dev)
phis = torch.load(f'{args.data_path}/phis.pt', map_location=torch_dev)
mps = torch.load(f'{args.data_path}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{args.data_path}/trj.pt', map_location=torch_dev)
dcf = torch.load(f'{args.data_path}/dcf.pt', map_location=torch_dev)
ksp = torch.load(f'{args.data_path}/ksp.pt', map_location=torch_dev)
im_size = phis.shape[1:]
trj_size = trj.shape[:-1]
C = mps.shape[0]
print(f'img size: {im_size}')
print(f'trj Size: {trj_size}')
args.os = 2 * round(args.os * im_size[0] / 2) / im_size[0] # makes sure we get an even integer

# Remove small energy alphas or phis
B = phis.shape[0]
phi_energy = phis.reshape((B, -1)).abs().mean(dim=1)
alpha_energy = alphas.reshape((B, -1)).abs().mean(dim=1)
energy = phi_energy * alpha_energy
idxs = torch.argwhere(energy > 1e-6)[:, 0]
phis = phis[idxs]
alphas = alphas[idxs]
B = phis.shape[0]

# # Compress phase coefficients
# phis, alphas = compress_phis_alphas(phis, alphas, 
#                                     B_compressed=8)

# Undersample data and typecast
R = args.R
trj = trj[..., ::R, :].type(torch.float32)
ksp = ksp[..., ::R].type(torch.complex64)
if alphas.shape[-1] > 1:
    alphas = alphas[..., ::R].type(torch.float32)
if R != 1:
    dcf = density_compensation(trj, im_size)
else:
    dcf = dcf[..., ::R].type(torch.float32)

# Set parameters
rparams = reduce_params(spatial_reduce_size=(args.spatial_reduce_size,)*2)
cg_params = {'max_iter': args.num_cg_iter, 'max_eigen': 1.0, 'verbose': False}

# Spatial mask
if args.mask_thresh is not None:
    evals = torch.load(f'{args.data_path}/evals.pt', map_location=torch_dev)
    spatial_mask = 1.0 * (evals > args.mask_thresh)
else:
    spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
mps *= spatial_mask

# ------------------ Expanded Encoding Model ------------------
try: 
    img_gt = torch.load(f'{args.data_path}/img_gt.pt', map_location=torch_dev)
    print('Loaded ground truth from file')
except:
    print('Computing ground truth from via Expanded Encoding Model')
    # Stack total phase
    phis_trj = gen_grd(im_size).to(torch_dev).moveaxis(-1, 0)
    alphas_trj = trj.moveaxis(-1, 0)
    phis_stack = torch.cat([phis, phis_trj], dim=0)
    alphas_stack = torch.cat([alphas, alphas_trj], dim=0)

    # Build encoding model
    bparams = batching_params(C)
    Agt = encoding_matrix(mps, phis_stack, alphas_stack, dcf, 
                        temporal_batch_size=2**10,
                        bparams=bparams,)

    # Recon
    img_gt = CG_SENSE_recon(Agt, ksp, **cg_params)

# Itertate over kernel sizes and number of spatial factors
imgs_hofft = []
imgs_split = []
times_hofft = []
times_split = []
def parse_range(range_str):
    low, high, step = range_str.split(',')
    return torch.arange(int(low), int(high), int(step))
Ws = parse_range(args.Wrange)
Ls = parse_range(args.Lrange)
LWs = [(int(L), int(W)) for L, W in product(Ls, Ws)]
for L, W in tqdm(LWs, 'Sweeping Over L, W'):
    
    # ------------------ Spatio Temporal Splitting ------------------
    # Timeing for decomp
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    # Reduce spatial size
    phis_reduced = reduce_spatial(phis, im_size_low=rparams.spatial_reduce_size, order=3)

    # Spatio-temporal encoding model
    spatial_funcs, temporal_funcs, _ = alpha_segementation(phis_reduced, alphas, 
                                                           L=L, L_batch_size=1,
                                                           interp_type='zero', use_type3=False,
                                                           verbose=False)
    spatial_funcs = expand_spatial(spatial_funcs, im_size_high=im_size, order=3)
    nufft = sigpy_nufft(im_size, oversamp=args.os, width=W)
    beta = nufft.optimal_beta(torch_dev=torch_dev)
    spatial_factor, kern_weights = kb_nufft(trj, im_size, (W,)*2, 
                                            os=args.os, beta=beta)
    
    kern_weights = kern_weights * temporal_funcs[:, None, None, ...]
    spatial_factors = spatial_factor * spatial_funcs
    trj_grd = (args.os * trj).round()/args.os
    Asplit = hofft_linop(trj_grd, mps, kern_weights, spatial_factors, dcf, 
                         os_grid=args.os, bparams=batching_params())
    end.record()
    torch.cuda.synchronize()
    time_split_decomp = start.elapsed_time(end)

    # Recon
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    img_split = CG_SENSE_recon(Asplit, ksp, **cg_params)
    end.record()
    torch.cuda.synchronize()
    time_split_recon = start.elapsed_time(end)
    times_split.append([time_split_decomp, time_split_recon])
    imgs_split.append(img_split.cpu())

    # ------------------ HOFFT ------------------
    hparams = hofft_params(kern_size=(W,)*2,
                           os=args.os,
                           L=L,
                           kalpha_method='maxmin',
                           anderson_order=args.anderson_order,
                           spatial_init=args.spatial_init,
                           matvec_kwargs={'spatial_batch_size': (120**2) // 10},
                           verbose=False)
    
    # Time decomp
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    # Stack phase coefficients, compress, rescale
    phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, hparams.os)
    phis_stack = torch.cat([phis, phis_dev], dim=0)
    alphas_stack = torch.cat([alphas, alphas_dev], dim=0)
    if args.B_compressed is not None:
        phis_stack, alphas_stack = compress_phis_alphas(phis_stack, alphas_stack, 
                                                    B_compressed=args.B_compressed)
    phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack)

    # Perform high order phase decomposition
    spatial_factors, kern_weights = als_hofft(phis_nrm, alphas_nrm, hparams, rparams,
                                              spatial_mask=spatial_mask,
                                              num_als_iter=args.num_als_iter)
    # spatial_factors, kern_weights = als_hofft_compressed(phis_nrm, alphas_nrm, hparams, Q=100, rparams=rparams,
    #                                           spatial_mask=spatial_mask,
    #                                           num_als_iter=args.num_als_iter)
    spatial_factors, kern_weights = apply_phase_midpoints(phis_nrm, alphas_nrm, phis_mp, alphas_mp, spatial_factors, kern_weights)

    # HOFFT model
    trj_grd = (hparams.os * trj).round()/hparams.os
    Ahofft = hofft_linop(trj_grd, mps, kern_weights, spatial_factors, dcf, 
                         os_grid=hparams.os, 
                         bparams=batching_params())
    end.record()
    torch.cuda.synchronize()
    time_hofft_decomp = start.elapsed_time(end)

    # Recon
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    img_hofft = CG_SENSE_recon(Ahofft, ksp, **cg_params)
    end.record()
    torch.cuda.synchronize()
    time_hofft_recon = start.elapsed_time(end)
    times_hofft.append([time_hofft_decomp, time_hofft_recon])
    imgs_hofft.append(img_hofft.cpu())

# ------------------ Save Data ------------------
imgs_hofft = torch.stack(imgs_hofft, dim=0).cpu()
imgs_split = torch.stack(imgs_split, dim=0).cpu()
imgs_hofft = imgs_hofft.reshape((len(Ls), len(Ws), *im_size))
imgs_split = imgs_split.reshape((len(Ls), len(Ws), *im_size))
times_hofft = torch.tensor(times_hofft)
times_split = torch.tensor(times_split)
times_hofft = times_hofft.reshape((len(Ls), len(Ws), 2))
times_split = times_split.reshape((len(Ls), len(Ws), 2))
save_data = {
    'imgs_hofft': imgs_hofft,
    'imgs_split': imgs_split,
    'times_hofft': times_hofft,
    'times_split': times_split,
    'spatial_mask': spatial_mask.cpu(),
    'img_gt': img_gt.cpu(),
    'Ls': Ls,
    'Ws': Ws,
    'os': args.os,
}
data_id = args.data_path.split('/')[-1]
torch.save(save_data, f'./paper_figures/recons/{data_id}.pt')