import torch

from hofft.forward_model import hofft_linop
from hofft.decomp import hofft_params
from hofft.sparse_decomp import sparse_params
from hofft.pipelines import (
    als_nufft, 
    kb_nufft, 
    als_hofft, 
    sparse_fit_hofft_lstsq_support,
    sparse_fit_hofft_smooth_interp
)

from mr_recon.linops import sense_linop
from mr_recon.fourier import sigpy_nufft, matrix_nufft
from mr_recon.utils import gen_grd
from mr_recon.recons import CG_SENSE_recon
from tqdm import tqdm

# Params
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 256
L = 1
os_list = torch.arange(1.0, 1.5 + 1/16, 1/16).tolist()
W = 2
d = 2
num_als_iter = 100
im_size = (N,)*d
kern_size = (W,)*d

# Load data
img_gt = torch.load('./data/sim_spiral/img.pt')
trj = torch.load('./data/sim_spiral/trj.pt')
dcf = torch.load('./data/sim_spiral/dcf.pt')
mask = torch.load('./data/sim_spiral/mask.pt')

# Dummy coil maps
mps = img_gt[None, :] * 0 + 1
mps *= (img_gt.abs() > 0.0)

# Simulate data
nft_sim = matrix_nufft(im_size, spatial_batch_size=2**10)
Asim = sense_linop(trj, mps, nufft=nft_sim)
ksp_gt = Asim(img_gt)

# -------------- Build High accuracy linop --------------
nft_gt = sigpy_nufft(im_size, oversamp=2.0, width=6)
Agt = sense_linop(trj, mps, dcf, nufft=nft_gt)

# Setup CG SENSE Recon
kwargs = {'max_iter': 10, 'max_eigen': 1.0, 'verbose': False}
img_highacc = CG_SENSE_recon(Agt, ksp_gt, **kwargs)

# Loop over OS and W
imgs_nufft = []
imgs_hofft = []
imgs_hofft_mskd = []
kerns_kb = []
kerns_hft = []
kerns_hft_mskd = []
apods_kb = []
apods_hft = []
apods_hft_mskd = []
for os in tqdm(os_list):

    # -------------- Build NUFFT linop --------------
    # Get optimal beta
    nft = sigpy_nufft(im_size, oversamp=os, width=W)
    nft.beta = nft.optimal_beta()

    # Get HOFFT style tensors using kb nufft funcs
    kb_apod, kb_weights = kb_nufft(trj, im_size, kern_size, os, nft.beta)

    # Build linop
    trj_grd = (trj * os).round() / os
    Anufft = hofft_linop(trj_grd, mps, kb_weights, kb_apod, 
                        dcf=dcf,
                        os_grid=os)

    # -------------- Regular HOFFT --------------
    # HOFFT setup
    hparams = hofft_params(kern_size, os, L=L,
                        reduced_im_size=(100,)*d,
                        solver='pinv',
                        lamda=0.0,
                        spatial_init='ones',
                        verbose=False)
    phis = gen_grd(im_size).moveaxis(-1, 0).to(torch_dev)
    alphas = (trj - trj_grd).moveaxis(-1, 0)

    # HOFFT Decomp
    def hofft_decomp(phis, alphas, num_als_iter, spatial_mask=None, htype='full'):
        # Full HOFFT model
        if htype == 'full':
            spatial_factor, kern_weights = als_hofft(phis, alphas, 
                                                    spatial_mask=spatial_mask,
                                                    hparams=hparams, 
                                                    num_als_iter=num_als_iter)
            return spatial_factor, kern_weights
        
        # Sparse HOFFT model
        sparams = sparse_params(Q=1000, S=4)
        if htype == 'lstsq':
            ret = sparse_fit_hofft_lstsq_support(phis, alphas, 
                                            spatial_mask=spatial_mask,
                                            hparams=hparams, 
                                            sparams=sparams, 
                                            ls_lamda=1e-3,
                                            num_als_iter=num_als_iter*10)
        elif htype == 'smooth':
            ret = sparse_fit_hofft_smooth_interp(phis, alphas, 
                                                spatial_mask=spatial_mask,
                                                hparams=hparams, 
                                                sparams=sparams, 
                                                kernel='rbf',
                                                #  kernel='inv_dist',
                                                auto_tune=True,
                                                num_val=30,
                                                num_als_iter=num_als_iter*10)
            
        # Convert sparse to full
        spatial_factor, compressed_kernels, sparse_inds, sparse_coeffs = ret
        kern_weights = torch.zeros((*compressed_kernels.shape[:-1], *alphas.shape[1:]),
                                device=torch_dev, dtype=torch.complex64)
        for s in range(sparse_inds.shape[0]):
            kern_weights += compressed_kernels[..., sparse_inds[s]] * sparse_coeffs[s]
        
        return spatial_factor, kern_weights
    spatial_factor, kern_weights = hofft_decomp(phis, alphas, num_als_iter)

    # Build HOFFT linop
    Ahofft = hofft_linop(trj_grd, mps, kern_weights, spatial_factor, 
                        dcf=dcf,
                        os_grid=os)

    # -------------- Masked HOFFT --------------
    spatial_factor_mskd, kern_weights_mskd = hofft_decomp(phis, alphas, num_als_iter, mask)
    Ahofft_mskd = hofft_linop(trj_grd, mps, kern_weights_mskd, spatial_factor_mskd, 
                                dcf=dcf,
                                os_grid=os)

    # -------------- Extract Kernel and Apod funcs --------------
    trj_dev = gen_grd((50,)*d).to(torch_dev) / os
    kb_apod, kb_kern = kb_nufft(trj_dev, im_size, kern_size, os, nft.beta)
    hft_apod, hft_kern = hofft_decomp(phis, trj_dev.moveaxis(-1, 0), num_als_iter)
    hft_apod_mskd, hft_kern_mskd = hofft_decomp(phis, trj_dev.moveaxis(-1, 0), num_als_iter, mask)
    apods_kb.append(kb_apod)
    kerns_kb.append(kb_kern)
    apods_hft.append(hft_apod)
    kerns_hft.append(hft_kern)
    apods_hft_mskd.append(hft_apod_mskd)
    kerns_hft_mskd.append(hft_kern_mskd)

    # -------------- Recon --------------
    img_nufft = CG_SENSE_recon(Anufft, ksp_gt, **kwargs)
    img_hofft = CG_SENSE_recon(Ahofft, ksp_gt, **kwargs)
    img_hofft_mskd = CG_SENSE_recon(Ahofft_mskd, ksp_gt, **kwargs)
    imgs_nufft.append(img_nufft)
    imgs_hofft.append(img_hofft)
    imgs_hofft_mskd.append(img_hofft_mskd)

# -------------- Save Recons and kernels --------------
dct = {}
dct['img_gt'] = img_gt.cpu()
dct['os_list'] = torch.tensor(os_list)
dct['mask'] = mask.cpu()
dct['img_highacc'] = img_highacc.cpu()
dct['imgs_nufft'] = torch.stack(imgs_nufft).cpu()
dct['imgs_hofft'] = torch.stack(imgs_hofft).cpu()
dct['imgs_hofft_mskd'] = torch.stack(imgs_hofft_mskd).cpu()
dct['kerns_kb'] = torch.stack(kerns_kb).cpu()
dct['kerns_hft'] = torch.stack(kerns_hft).cpu()
dct['kerns_hft_mskd'] = torch.stack(kerns_hft_mskd).cpu()
dct['apods_kb'] = torch.stack(apods_kb).cpu()
dct['apods_hft'] = torch.stack(apods_hft).cpu()
dct['apods_hft_mskd'] = torch.stack(apods_hft_mskd).cpu()
torch.save(dct, f'./paper_experiments/nufft_2d/W{W}.pt')
