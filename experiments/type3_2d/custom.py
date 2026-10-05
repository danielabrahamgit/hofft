import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from hofft.utils import gen_grd, maxmin_indices, reduce_spatial, reduce_temporal
from hofft.phase_coeffs import whiten_phis_alphas, rescale_phis_alphas

from tqdm import tqdm

# dataset = 'coco_spiral'
dataset = 'tilt_spi_invivo'

# Load data
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trj = torch.load(f'./data/{dataset}/trj.pt', map_location=torch_dev).type(torch.float32)
mps = torch.load(f'./data/{dataset}/mps.pt', map_location=torch_dev).type(torch.complex64)
evals = torch.load(f'./data/{dataset}/evals.pt', map_location=torch_dev).type(torch.float32)
dcf = torch.load(f'./data/{dataset}/dcf.pt', map_location=torch_dev).type(torch.float32)
phis = torch.load(f'./data/{dataset}/phis.pt', map_location=torch_dev).type(torch.float32)
alphas = torch.load(f'./data/{dataset}/alphas.pt', map_location=torch_dev).type(torch.float32)
im_size = mps.shape[1:]
trj_size = trj.shape[:-1]
B = phis.shape[0]
C = mps.shape[0]

# Print shapes
print(f'trj.shape: {trj.shape}')
print(f'mps.shape: {mps.shape}')
print(f'evals.shape: {evals.shape}')
print(f'dcf.shape: {dcf.shape}')
print(f'phis.shape: {phis.shape}')
print(f'alphas.shape: {alphas.shape}')

# Downsample
pe = slice(None)
pe = 0
M_low = trj.shape[0] // 4
N_low = (80, 80)
trj = trj[:, pe]
trj = reduce_temporal(trj, M_low, dim=0)
alphas = reduce_temporal(alphas[:, :, pe], M_low, dim=1)
phis = reduce_spatial(phis, N_low)
crds = gen_grd(im_size).to(torch_dev).moveaxis(-1, 0)
crds = reduce_spatial(crds, N_low)
evals = reduce_spatial(evals, N_low)
mps = reduce_spatial(mps, N_low)

# Mask
mask = (evals > 0.95).float()
# mask = mask * 0 
# mask[:N_low[0]//2, :N_low[1]//2] = 1
mask = mask.flatten()

# Flatten
crds = crds.reshape(2, -1) * mask
trj = trj.reshape(-1, 2).T
alphas = alphas.reshape(B, -1)
phis = phis.reshape(B, -1) * mask
mps = mps.reshape((C, -1)) * mask

# Normalize
# phis, alphas = whiten_phis_alphas(phis, alphas, B_compressed=4)
phis, _, alphas, _ = rescale_phis_alphas(phis, alphas,
                                         quantiles=(0.0, 1.0))

def _matching_pursuit(D, Y, k_sparse):
    """Batched MP: each column of the returned codes has ≤ k_sparse nonzeros."""
    residual = Y.clone()
    n_atoms = D.shape[1]
    n_sig = Y.shape[1]
    X = Y.new_zeros(n_atoms, n_sig)
    used = torch.zeros(n_sig, n_atoms, dtype=torch.bool, device=Y.device)
    n_idx = torch.arange(n_sig, device=Y.device)
    for _ in range(k_sparse):
        corr = D.mH @ residual
        corr = corr.masked_fill(used.T, 0)
        idx = corr.abs().argmax(dim=0)
        c = corr[idx, n_idx]
        X[idx, n_idx] = X[idx, n_idx] + c
        residual = residual - D[:, idx] * c
        used[n_idx, idx] = True
    return X


def _codes_ls_support(D, Y, support):
    """Exact LS on a given support: Y ≈ D @ X with X[support[n], n] free."""
    n_atoms, n_sig = D.shape[1], Y.shape[1]
    X = Y.new_zeros(n_atoms, n_sig)
    key, inv = torch.unique(support.sort(dim=1).values, dim=0, return_inverse=True)
    for p in range(key.shape[0]):
        pat = key[p]
        cols = (inv == p).nonzero(as_tuple=False).squeeze(-1)
        X[pat[:, None], cols[None, :]] = torch.linalg.lstsq(
            D[:, pat], Y[:, cols]).solution
    return X


@torch.no_grad()
def ksvd(Y, n_atoms, k_sparse, n_iter=8, D0=None, X0=None, support=None):
    """Approximate K-SVD (Rubinstein et al.).

    Y is (M, N). Returns U, S, V in the same layout as ``torch.svd_lowrank``:
        Y ≈ U @ diag(S) @ V.H
    with each *column of V.H* at most ``k_sparse``-sparse.

    If ``support`` is (N, k) atom indices per column, that geometric support
    replaces matching pursuit. Dictionary updates are unchanged.
    """
    if D0 is None:
        U0, S0, V0 = torch.svd_lowrank(Y, q=n_atoms)
        D = U0 * S0
        X = V0.mH
    else:
        D, X = D0.clone(), X0.clone()
    D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-8)
    n_atoms = D.shape[1]

    for _ in range(n_iter):
        if support is None:
            X = _matching_pursuit(D, Y, k_sparse)
        else:
            X = _codes_ls_support(D, Y, support)
        R = Y - D @ X
        for j in range(n_atoms):
            I = torch.nonzero(X[j].abs() > 0, as_tuple=False).squeeze(-1)
            if I.numel() < 2:
                continue
            R_I = R[:, I] + D[:, j:j + 1] @ X[j:j + 1, I]
            d = R_I @ X[j, I].conj()
            d = d / d.norm().clamp_min(1e-8)
            x_new = d.conj() @ R_I
            R[:, I] = R_I - d[:, None] * x_new
            D[:, j] = d
            X[j, I] = x_new

    S = X.norm(dim=1)
    order = torch.argsort(S, descending=True)
    U = D[:, order]
    S = S[order]
    V = X.mH[:, order] / S.clamp_min(1e-8)
    return U, S, V

phz = torch.exp(-2j * torch.pi * (alphas.T @ phis)) * mask # M N
M, N = phz.shape
aw = alphas.T.contiguous()  # (M, B), already whitened

err_svd = []
err_ksvd = []
err_ksvd_fps = []
err_seg = []
Ls = torch.arange(1, 100, 5)
k_pcnt = 0.5
for L in tqdm(Ls):
    L = int(L)
    k = int(L * k_pcnt)
    # k = 5
    k = max(1, k)

    # K-sparse Alpha-segmentation
    # Spatial atoms: b_l(r) = exp(-j 2π β_l · φ(r)), β_l = l-th FPS anchor.
    # Temporal codes: each α(t) uses the K nearest β's; LS on that support.
    beta_idx = maxmin_indices(aw.float(), L, seed=0)
    betas = alphas[:, beta_idx]  # (B, L)
    spatial = torch.exp(-2j * torch.pi * (betas.T @ phis)) * mask  # (L, N)
    support = torch.topk(
        torch.cdist(aw.double(), betas.T.double()),
        k=k, dim=1, largest=False,
    ).indices
    temporal = _codes_ls_support(spatial.T, phz.T, support).T
    err = (phz - temporal @ spatial).norm() / phz.norm()
    err_seg.append(err.item())

    # Regular SVD
    U, S, V = torch.svd_lowrank(phz, q=L)
    Vh = V.H
    err = ((phz - (U * S) @ Vh) * mask).norm() / phz.norm()
    err_svd.append(err.item())

    # K-SVD on phz.H so sparsity is in U: phz ≈ U diag(S) V.H with each
    # column of U.H at most k_sparse-sparse (each time sample uses ≤ K
    # spatial dictionary atoms).
    Uk_t, Sk, Vk_t = ksvd(phz.mH, n_atoms=L, k_sparse=k,
                        n_iter=8, D0=V * S, X0=U.mH)
    err = ((phz.mH - (Uk_t * Sk) @ Vk_t.mH) * mask[:, None]).norm() / phz.norm()
    err_ksvd.append(err.item())

    # Same K-SVD atom updates, but the support is the FPS k-NN pattern
    # instead of matching pursuit.
    Uf_t, Sf, Vf_t = ksvd(phz.mH, n_atoms=L, k_sparse=k,
                          n_iter=8, D0=V * S, X0=U.mH, support=support)
    err = ((phz.mH - (Uf_t * Sf) @ Vf_t.mH) * mask[:, None]).norm() / phz.norm()
    err_ksvd_fps.append(err.item())


plt.plot(Ls, err_svd, label='SVD')
plt.plot(Ls, err_ksvd, label='K-SVD (MP)')
plt.plot(Ls, err_ksvd_fps, label=f'K-SVD (FPS support, k={k})')
plt.plot(Ls, err_seg, label=f'FPS-NN (k={k})')
plt.legend()
plt.show()
