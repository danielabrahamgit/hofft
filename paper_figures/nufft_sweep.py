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

from hofft.pipelines import kb_nufft, als_nufft
from hofft.decomp import hofft_params
from hofft.forward_model import hofft_linop
from hofft.reduce import reduce_params

# Set random seed
torch.manual_seed(0)

# Parse arguments
args = argparse.ArgumentParser()
args.add_argument('--osrange', type=str, default='1.25,1.5,0.25')
args.add_argument('--Wrange', type=str, default='5,6,1')
# args.add_argument('--Lrange', type=str, default='5,6,1')
args.add_argument('--gpu', type=int, default=None)
args.add_argument('--R', type=int, default=1)
args.add_argument('--num_cg_iter', type=int, default=10)
args.add_argument('--num_als_iter', type=int, default=100)
args.add_argument('--spatial_init', type=str, default='ones')
args.add_argument('--spatial_reduce_size', type=int, default=50)
args.add_argument('--mask_thresh', type=float, default=None)
args = args.parse_args()

# Load data
torch_dev = torch.device('cpu') if args.gpu is None else torch.device(args.gpu)
breakpoint()
fdir = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/shepp_nufft'
trj = torch.load(f'{fdir}/trj.pt', map_location=torch_dev)
dcf = torch.load(f'{fdir}/dcf.pt', map_location=torch_dev)
ksp = torch.load(f'{fdir}/ksp.pt', map_location=torch_dev)
img_gt = torch.load(f'{fdir}/img.pt', map_location=torch_dev)
im_size = img_gt.shape
trj_size = dcf.shape
print(f'img size: {im_size}')
print(f'trj Size: {trj_size}')

# Undersample data and typecast
R = args.R
trj = trj[..., ::R, :].type(torch.float32)
ksp = ksp[..., ::R].type(torch.complex64)
if R != 1:
    dcf = density_compensation(trj, im_size)
else:
    dcf = dcf[..., ::R].type(torch.float32)

# Set parameters
rparams = reduce_params(spatial_reduce_size=(110,)*2)
cg_params = {'max_iter': 20, 'max_eigen': 1.0, 'verbose': False}

# Spatial mask
mps = torch.ones((1, *im_size), dtype=torch.complex64, device=torch_dev)
spatial_mask = torch.ones(im_size, dtype=torch.complex64, device=torch_dev)
mps *= spatial_mask

# Loop over kernel sizes, oversampling factors
imgs_hofft = []
imgs_nufft = []
times_hofft = []
times_nufft = []
def parse_range(range_str):
    low, high, step = range_str.split(',')
    if '.' in low or '.' in high or '.' in step:
        return torch.arange(float(low), float(high), float(step))
    else:
        return torch.arange(int(low), int(high), int(step))
Ws = parse_range(args.Wrange)
Os = parse_range(args.osrange)
OWs = [(int(O), int(W)) for O, W in product(Os, Ws)]
for O, W in tqdm(OWs, 'Sweeping Over O, W'):
    # Make sure oversampling factor is an even integer
    os = 2 * round(O * im_size[0] / 2) / im_size[0]

    # # ------------------ NUFFT Recon ------------------
    # # Setup KB NUFFT operator
    # nufft = sigpy_nufft(im_size, oversamp=os, width=W)
    # beta = nufft.optimal_beta(torch_dev=torch_dev)
    # spatial_factor, kern_weights = kb_nufft(trj, im_size, (W,)*len(im_size), 
    #                                         os=os, beta=beta)
    # trj_grd = (os * trj).round()/os
    # bparams = batching_params()
    # Asplit = hofft_linop(trj_grd, mps, kern_weights, spatial_factor, dcf, 
    #                         os_grid=os, bparams=bparams)

    # # Recon
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    # img_nufft = CG_SENSE_recon(Asplit, ksp, **cg_params)
    # end.record()
    # torch.cuda.synchronize()
    # time_nufft = start.elapsed_time(end)
    # imgs_nufft.append(img_nufft.cpu())
    # times_nufft.append(time_nufft)

    # ------------------ HOFFT Recon ------------------
    L = 1
    hparams = hofft_params(kern_size=(W,)*len(im_size),
                            os=os,
                            L=L,
                            spatial_init=args.spatial_init,
                            verbose=False)

    # Time decomp
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # Perform high order phase decomposition
    spatial_factors, kern_weights = als_nufft(trj, im_size, hparams, 
                                            spatial_mask=spatial_mask,
                                            im_size_low=(args.spatial_reduce_size,)*len(im_size),
                                            num_als_iter=args.num_als_iter)

    # HOFFT linop
    trj_grd = (hparams.os * trj).round()/hparams.os
    bparams = batching_params()
    Ahofft = hofft_linop(trj_grd, mps, kern_weights, spatial_factors, dcf, os_grid=hparams.os, bparams=bparams)
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
    imgs_hofft.append(img_hofft.cpu())
    times_hofft.append([time_hofft_decomp, time_hofft_recon])

# ------------------ Save Data ------------------
imgs_hofft = torch.stack(imgs_hofft, dim=0).cpu()
imgs_nufft = torch.stack(imgs_nufft, dim=0).cpu()
imgs_hofft = imgs_hofft.reshape((len(Os), len(Ws), *im_size))
imgs_nufft = imgs_nufft.reshape((len(Os), len(Ws), *im_size))
times_hofft = torch.tensor(times_hofft)
times_nufft = torch.tensor(times_nufft)
times_hofft = times_hofft.reshape((len(Os), len(Ws), 2))
times_nufft = times_nufft.reshape((len(Os), len(Ws)))
save_data = {
    'imgs_hofft': imgs_hofft,
    'imgs_nufft': imgs_nufft,
    'times_hofft': times_hofft,
    'times_nufft': times_nufft,
    'spatial_mask': spatial_mask.cpu(),
    'img_gt': img_gt.cpu(),
    'Os': Os,
    'Ws': Ws,
    'L': L,
}
torch.save(save_data, f'./paper_figures/recons/shepp_nufft.pt')