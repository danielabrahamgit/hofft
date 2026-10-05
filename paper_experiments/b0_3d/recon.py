import gc
import torch
import numpy as np

from time import perf_counter

from mr_recon.recons import CG_SENSE_recon
from mr_recon.utils import cvplot

from hofft.decomp import hofft_params
from hofft.pipelines import svd_decomp_linop

# Params
L = 6

# Load data
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fpath = '/local_mount/space/tiger/1/users/abrahamd/mr_data/mrf_b0/data/'
b0 = torch.from_numpy(np.load(fpath + 'b0.npy')).type(torch.float32).to(torch_dev)
dcf = torch.from_numpy(np.load(fpath + 'dcf.npy')).type(torch.float32).to(torch_dev)
trj = torch.from_numpy(np.load(fpath + 'trj.npy')).type(torch.float32).to(torch_dev)
mps = torch.from_numpy(np.load(fpath + 'mps.npy')).type(torch.complex64).to(torch_dev)
ksp = torch.from_numpy(np.load(fpath + 'ksp.npy')).type(torch.complex64).to(torch_dev)
dcf /= dcf.max()
C = mps.shape[0]
im_size = b0.shape

# Subsample
R = 3
G = trj.shape[1]
trj = trj[:, :G//R]
dcf = dcf[:, :G//R]
ksp = ksp[:, :, :G//R]

# B0 phase on the full trajectory grid
ts = torch.arange(trj.shape[0], device=torch_dev) * 2e-6
phis = b0[None,]
# alphas = torch.zeros_like(dcf)
# alphas[:] = ts[:, None, None]
# alphas = alphas[None]
alphas = ts[None, :, None, None]

hparams = hofft_params(
    (3,) * 3, 1.25, L,
    reduced_im_size=(50,) * 3,
    time_reduction_factor=10,
    normalize_coeffs=True,
    cur_rank=500,
    verbose=True,
)
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0]
decomp_kw = dict(phis=phis, alphas=alphas, mps=mps, trj=trj, dcf=dcf, hparams=hparams, use_sigpy=True)


def _time(fn):
    if torch_dev.type == 'cuda':
        torch.cuda.synchronize()
    t0 = perf_counter()
    out = fn()
    if torch_dev.type == 'cuda':
        torch.cuda.synchronize()
    return perf_counter() - t0, out


# Naive SVD: LOBPCG matvec on the downsampled phase (cur_rank cleared internally)
# vs CUR-SVD of that same reduced phase matrix.
imgs = {}
# for label, svd_method in [('SVD (CUR)', 'cur'), ('SVD (naive)', 'lobpcg')]:
# for label, svd_method in [('SVD (naive)', 'lobpcg'), ('SVD (CUR)', 'cur')]:
for label, svd_method in [('SVD (naive)', 'direct'), ('SVD (CUR)', 'cur')]:
    t_decomp, A = _time(lambda m=svd_method: svd_decomp_linop(**decomp_kw, svd_method=m))
    t_recon, img = _time(lambda: CG_SENSE_recon(A, ksp, max_iter=2, max_eigen=1.0))
    print(f'{label:<16s} decomp={t_decomp:.2f}s  recon={t_recon:.2f}s')
    imgs[label] = img.cpu()
    del A, img
    gc.collect()
    if torch_dev.type == 'cuda':
        torch.cuda.empty_cache()

cvplot(None, **imgs)