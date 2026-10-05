"""
(L, W) sweep comparing SVD (naive LOBPCG), SVD (CUR), and HOFFT.
"""
import gc
import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from time import perf_counter
from tqdm import tqdm

from mr_recon.recons import CG_SENSE_recon
from mr_recon.linops import sense_linop, batching_params
from mr_recon.fourier import sigpy_nufft
from mr_recon.algs import density_compensation

from hofft.phase_coeffs import remove_linear_terms
from hofft.matvec import matvec_cur, matvec_naive
from hofft.decomp import hofft_params
from hofft.utils import normalize
from hofft.pipelines import (
    alpha_seg_decomp_linop,
    hofft_decomp_linop,
    svd_decomp_linop,
    qblock_svd_decomp_linop,
)

# ---------------------------------------------------------------------------
# Timing / metrics
# ---------------------------------------------------------------------------
class GPUTimer:
    """Time a (possibly GPU-async) block; use CUDA events when available."""

    def __init__(self, torch_dev: torch.device):
        self.is_cuda = torch_dev.type == 'cuda'

    def __enter__(self):
        if self.is_cuda:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
            self._start_evt.record()
        else:
            self._t0 = perf_counter()
        return self

    def __exit__(self, *exc_info):
        if self.is_cuda:
            self._end_evt.record()
            torch.cuda.synchronize()
            self.elapsed = self._start_evt.elapsed_time(self._end_evt) / 1000
        else:
            self.elapsed = perf_counter() - self._t0
        return False

def time_repeated(fn, torch_dev, n_reps=1, reduction='mean'):
    """Run ``fn`` ``n_reps`` times; return ``(elapsed, last_result)``."""
    times, result = [], None
    for _ in range(n_reps):
        with GPUTimer(torch_dev) as t:
            result = fn()
        times.append(t.elapsed)
    times = torch.tensor(times)
    elapsed = {'mean': times.mean, 'median': times.median, 'min': times.min}[reduction]().item()
    return elapsed, result

def nrmse(img, ref, mask=None):
    if mask is not None:
        img, ref = img * mask, ref * mask
    return (torch.linalg.norm(img.abs() - ref.abs()) / torch.linalg.norm(ref.abs())).item()
    # return (torch.linalg.norm(img - ref) / torch.linalg.norm(ref)).item()

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Experiment params
# ---------------------------------------------------------------------------
# dataset = 'tilt_spi_invivo'
# dataset = 'tilt_spi'
# dataset = 'coco_spiral'
# dataset = 'axial_spi_invivo'
# dataset = 'highres_spiral'
# dataset = 'magnus_spi'
dataset = 'magnus_epi'
if dataset == 'coco_spiral':
    Ls_baseline = [5, 10, 15]  # SVD / Alpha Segmentation
    Ls_baseline = [5, 10, 15, 20, 25, 30]  # SVD / Alpha Segmentation
    Ls_hofft = [1, 3, 5, 7, 9]
    Ws = [3]
    R = 3
    reduced_im_size = None
    reduced_im_size = (100, 100)
    time_reduction_factor = None
    num_compressed_bases = 4
    num_representative_alphas = None
    cur_rank = 500
    d = 2
elif dataset == 'tilt_spi':
    Ls_baseline = [10, 15, 20, 25, 30]  # SVD / Alpha Segmentation
    Ls_hofft = [3, 5, 7, 9, 11]
    Ws = [3]
    R = 2
    reduced_im_size = (200, 200)
    reduced_im_size = None
    time_reduction_factor = 10
    num_compressed_bases = 10
    cur_rank = 500
    num_representative_alphas = None
    d = 2
elif dataset == 'tilt_spi_invivo':
    Ls_baseline = [30, 40, 50, 60, 70, 80]  # SVD / Alpha Segmentation
    Ls_baseline = [30, 55, 80]  # SVD / Alpha Segmentation
    Ls_hofft = [10, 15, 20, 25, 30]
    Ws = [3]
    R = 2
    reduced_im_size = (200, 200)
    time_reduction_factor = 10
    num_compressed_bases = 10
    cur_rank = 500
    # cur_rank = None
    num_representative_alphas = None
    d = 2
elif dataset == 'axial_spi_invivo':
    Ls_baseline = [10, 15, 20, 25, 30]  # SVD / Alpha Segmentation
    Ls_hofft = [3, 5, 7, 9, 11]
    Ws = [3]
    R = 2
    reduced_im_size = (400, 400)
    time_reduction_factor = 10
    num_compressed_bases = 10
    cur_rank = 500
    num_representative_alphas = None
    d = 2
elif dataset == 'highres_spiral':
    Ls_baseline = [10, 20, 30, 40, 50]  # SVD / Alpha Segmentation
    Ls_hofft = [5, 10, 15,]
    Ws = [3]
    R = 2
    reduced_im_size = (400, 400)
    time_reduction_factor = 10
    num_compressed_bases = None
    cur_rank = 500
    num_representative_alphas = None
    d = 2
elif dataset == 'mrf_coco':
    Ls_baseline = [1, 3, 5, 10]
    Ls_hofft = [1, 3, 5, 7, 9]
    Ws = [2]
    R = 3
    reduced_im_size = (100,)*3
    time_reduction_factor = 50
    num_compressed_bases = None
    cur_rank = None
    num_representative_alphas = None
    d = 3
elif dataset == 'magnus_spi':
    Ls_baseline = [10, 20, 30, 40, 50]  # SVD / Alpha Segmentation
    Ls_hofft = [5, 7, 9, 11]
    Ws = [3]
    R = 1
    reduced_im_size = None
    time_reduction_factor = 10
    num_compressed_bases = None
    cur_rank = 500
    num_representative_alphas = None
    d = 2
elif dataset == 'magnus_epi':
    Ls_baseline = [20, 30, 40, 50, 60]  # SVD / Alpha Segmentation
    Ls_hofft = [5, 10, 15, 20, 25]
    Ws = [3]
    R = 1
    reduced_im_size = None
    time_reduction_factor = 10
    num_compressed_bases = None
    cur_rank = 500
    num_representative_alphas = None
    d = 2
num_time_reps = 1
time_reduction = 'mean'
mask_thresh = 0.9
os = 1.25

hparams = hofft_params(
    (Ws[0],) * d, os, Ls_hofft[0],
    reduced_im_size=reduced_im_size,
    spatial_init='seg',
    normalize_coeffs=True,
    kalpha_method='maxmin',
    time_reduction_factor = time_reduction_factor,
    num_compressed_bases=num_compressed_bases,
    num_representative_alphas=num_representative_alphas,
    cur_rank=cur_rank,
    solver='solve',
    lamda=1e-3,
    max_als_iter=10,
    verbose=True,
)
hparams.matvec_kwargs = {'spatial_batch_size': 2**10}
cg_kwargs = {'max_iter': 10, 'max_eigen': 1.0, 'verbose': hparams.verbose}
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
fpath = f'./data/{dataset}'
load_kw = {'weights_only': True, 'map_location': torch_dev}
img_gt = torch.load(f'{fpath}/img_gt.pt', **load_kw).type(torch.complex64)
trj = torch.load(f'{fpath}/trj.pt', **load_kw).type(torch.float32)
dcf = torch.load(f'{fpath}/dcf.pt', **load_kw).type(torch.float32)
mps = torch.load(f'{fpath}/mps.pt', **load_kw).type(torch.complex64)
ksp = torch.load(f'{fpath}/ksp.pt', **load_kw).type(torch.complex64)
evals = torch.load(f'{fpath}/evals.pt', **load_kw).type(torch.float32)
phis = torch.load(f'{fpath}/phis.pt', **load_kw).type(torch.float32)
alphas = torch.load(f'{fpath}/alphas.pt', **load_kw).type(torch.float32)
img_gt_cpu = img_gt.cpu()
im_size = mps.shape[1:]
C = mps.shape[0]
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0]
bparams = batching_params(coil_batch_size=C // 2)

# Undersample
trj = trj[:, ::R]
ksp = ksp[:, :, ::R]
alphas = alphas[:, :, ::R]
dcf = density_compensation(trj, im_size)

# Apply mask
mask = (evals > mask_thresh).float()
mps = mps * mask

# ---------------------------------------------------------------------------
# Reference reconstructions (independent of swept L, W)
# ---------------------------------------------------------------------------
W_nominal = 6
nufft_uncorr = sigpy_nufft(im_size, width=W_nominal)
nufft_uncorr.beta = nufft_uncorr.optimal_beta(torch_dev=torch_dev)

# Uncorrected: original trajectory, before folding linear φ terms into k.
A_uncorr_raw = sense_linop(trj, mps, dcf=dcf, nufft=nufft_uncorr, bparams=bparams)
img_uncorr_raw = normalize(CG_SENSE_recon(A_uncorr_raw, ksp, **cg_kwargs), img_gt)

# Remove linear terms
phis_new, trj_term, zeroth_order = remove_linear_terms(phis, alphas, mask=mask)
trj += trj_term
ksp *= torch.exp(2j * torch.pi * zeroth_order)
phis = phis_new * mask

# Drop negligible field terms
B = phis.shape[0]
energy = (phis.reshape((B, -1)).abs().mean(dim=1)
          * alphas.reshape((B, -1)).abs().mean(dim=1))
idxs = torch.argwhere(energy > 1e-6)[:, 0]
phis, alphas = phis[idxs], alphas[idxs]
B = phis.shape[0]

# ---------------------------------------------------------------------------
# Reference reconstructions continued (linear terms now in k)
# ---------------------------------------------------------------------------
A_uncorr = sense_linop(trj, mps, dcf=dcf, nufft=nufft_uncorr, bparams=bparams)
img_uncorr = normalize(CG_SENSE_recon(A_uncorr, ksp, **cg_kwargs), img_gt)


# from hofft.forward_model import expanded_encoding
# from hofft.utils import gen_grd
# rs = gen_grd(im_size).to(torch_dev)
# phis_ee = torch.cat([phis, rs.moveaxis(-1, 0)], dim=0)
# alphas_ee = torch.cat([alphas, trj.moveaxis(-1, 0)], dim=0)
# A_ee = expanded_encoding(mps,  phis_ee, alphas_ee, 
#                          temporal_batch_size=2**10,
#                          dcf=dcf, bparams=bparams)
# img_ee = CG_SENSE_recon(A_ee, ksp, **cg_kwargs)
# torch.save(img_ee.cpu(), f'{fpath}/img_ee_R{R}.pt')
# quit()
img_ee = torch.load(
    f'{fpath}/img_ee_R{R}.pt',
    map_location=torch_dev,
)
# img_ee = img_gt.mT
img_ref = img_ee
print(f'Uncorrected (raw)     NRMSE={nrmse(img_uncorr_raw, img_ref, mask=mask):.4f}')
print(f'Uncorrected (linear)  NRMSE={nrmse(img_uncorr, img_ref, mask=mask):.4f}')

# ---------------------------------------------------------------------------
# Method specs: SVD (naive LOBPCG), SVD (CUR), HOFFT
# ---------------------------------------------------------------------------
def _decomp_kwargs():
    return dict(
        phis=phis, alphas=alphas, mps=mps, trj=trj, dcf=dcf,
        spatial_mask=mask, hparams=hparams, bparams=bparams,
    )


METHODS = [
    {
        'key': 'hofft',
        'label': 'HOFFT',
        'Ls': Ls_hofft,
        'Ws': Ws,
        'decomp': lambda: hofft_decomp_linop(**_decomp_kwargs()),
    },
    {
        'key': 'svd',
        'label': 'SVD',
        'Ls': Ls_baseline,
        'Ws': Ws,
        'decomp': lambda: svd_decomp_linop(
            **_decomp_kwargs(), svd_method='cur',
            use_sigpy=False
        ),
    },
]


# ---------------------------------------------------------------------------
# Sweep (L, W) for each method
# ---------------------------------------------------------------------------
results = {}
imgs = {}
warmed_up = False

for method in METHODS:
    key = method['key']
    Ls, Ws_m = method['Ls'], method['Ws']
    nL, nW = len(Ls), len(Ws_m)

    results[f'nrmse_{key}'] = torch.zeros(nL, nW)
    results[f'decomp_time_{key}'] = torch.zeros(nL, nW)
    results[f'recon_time_{key}'] = torch.zeros(nL, nW)
    imgs[key] = torch.zeros(nL, nW, *im_size, dtype=torch.complex64)

    for j, W in enumerate(tqdm(Ws_m, desc=f'{method["label"]} W', leave=True)):
        for i, L in enumerate(tqdm(Ls, desc=f'{method["label"]} L', leave=False)):
            hparams.kern_size = (W,) * d
            hparams.L = L
            
            # Reset seed
            torch.manual_seed(0)
            results[f'decomp_time_{key}'][i, j], A = time_repeated(
                method['decomp'], torch_dev,
                n_reps=num_time_reps, reduction=time_reduction,
            )
            # A.bparams.field_batch_size = L

            # One-time CUDA warm-up so the first timed recon isn't inflated
            if not warmed_up:
                CG_SENSE_recon(A, ksp, **cg_kwargs)
                if torch_dev.type == 'cuda':
                    torch.cuda.synchronize()
                warmed_up = True

            def recon():
                return normalize(CG_SENSE_recon(A, ksp, **cg_kwargs), img_ref)

            results[f'recon_time_{key}'][i, j], img = time_repeated(
                recon, torch_dev,
                n_reps=num_time_reps, reduction=time_reduction,
            )
            results[f'nrmse_{key}'][i, j] = nrmse(img, img_ref, mask=mask)
            ksp_nrmse = (A(img) - ksp).norm() / ksp.norm()
            imgs[key][i, j] = img.cpu()

            print(
                f'{method["label"]:<20s} L={L:<4d} W={W}: '
                f'NRMSE={results[f"nrmse_{key}"][i, j]:.4f}, '
                f'decomp={results[f"decomp_time_{key}"][i, j]:.2f}s, '
                f'recon={results[f"recon_time_{key}"][i, j]:.2f}s, '
                f'ksp_nrmse={ksp_nrmse:.4f}'
            )
            if hasattr(A, 'clear_plans'):
                A.clear_plans()
            del A, img
            clear_gpu()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
save_path = f'./paper_experiments/{dataset}_sweep_results.pt'
torch.save({
    **results,
    'img_uncorr_raw': img_uncorr_raw.cpu(),
    'img_uncorr': img_uncorr.cpu(),
    'img_ee': img_ee.cpu(),
    'img_gt': img_gt.cpu(),
    **{f'imgs_{m["key"]}': imgs[m['key']] for m in METHODS},
    **{f'Ls_{m["key"]}': m['Ls'] for m in METHODS},
    **{f'Ws_{m["key"]}': m['Ws'] for m in METHODS},
    'os': os,
}, save_path)
print(f'Saved sweep results to {save_path}')
