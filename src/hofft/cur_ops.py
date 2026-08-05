"""
Low-rank approximations of the high-order phase encoding matrix Phi[r,t] =
exp(-2pi j phi(r)·alpha(t)), used for corr = Dw @ Phi in omp_compressed.

Two families:
  - CUR (cluster representatives + pinv coupling)
  - batched SVD-lowrank (optimal rank-k per temporal slice; best rank/accuracy)

Batched drivers avoid storing C (K, T) — only C (K, Tb) per temporal chunk.
"""

import torch

from einops import einsum
from typing import Literal, Optional
from tqdm import tqdm

from mr_recon.utils import pick_K_vectors
from mr_recon.dtypes import complex_dtype

from .phase_coeffs import rescale_phis_alphas

ClusterMethod = Literal['maxmin', 'kmeans']


def setup_cur_phi_clusters(phis: torch.Tensor,
                           alphas: torch.Tensor,
                           rank_phi: int = 256,
                           cluster_method: ClusterMethod = 'maxmin',
                           ) -> tuple:
    """
    One-time CUR spatial setup: rescale phis/alphas and pick fixed phi clusters.

    Returns (phis_full, alphas_nrm, alphas_mp, phi_clusts) for use with
    ``cur_corr_slice`` inside a temporal batch loop.
    """
    phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)
    phis_full = phis_nrm + phis_mp[:, None]
    phi_clusts, _ = pick_K_vectors(
        phis_nrm.T, K=rank_phi, sigma=0, method=cluster_method)
    phi_clusts = phi_clusts + phis_mp
    return phis_full, alphas_nrm, alphas_mp, phi_clusts


def cur_corr_slice(Dw: torch.Tensor,
                   phis_full: torch.Tensor,
                   alphas_nrm: torch.Tensor,
                   alphas_mp: torch.Tensor,
                   phi_clusts: torch.Tensor,
                   t1: int,
                   t2: int,
                   rank_alpha: int = 128,
                   cluster_method: ClusterMethod = 'maxmin',
                   ) -> torch.Tensor:
    """
    Build corr[:, t1:t2] = Dw @ Phi[:, t1:t2] via CUR for one temporal batch.

    Returns corr slice with shape (Q, t2 - t1).
    """
    alphas_full = alphas_nrm[:, t1:t2] + alphas_mp[:, None]
    alpha_clusts, _ = pick_K_vectors(
        alphas_nrm[:, t1:t2].T, K=rank_alpha, sigma=0, method=cluster_method)
    alpha_clusts = alpha_clusts + alphas_mp
    R_raw = torch.exp(-2j * torch.pi * (alpha_clusts @ phis_full))
    C = torch.exp(-2j * torch.pi * (phi_clusts @ alphas_full))
    W = torch.exp(-2j * torch.pi * (alpha_clusts @ phi_clusts.T))
    R = einsum(torch.linalg.pinv(W), R_raw, 'Kp Ka, Ka R -> Kp R')
    return cur_forward(Dw, R, C)


def build_cur_factors(phis: torch.Tensor,
                      alphas: torch.Tensor,
                      rank: int = 128,
                      rank_phi: Optional[int] = None,
                      rank_alpha: Optional[int] = None,
                      cluster_method: ClusterMethod = 'maxmin',
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build CUR factors so Phi[r,t] ≈ sum_k R[k,r] C[k,t].

    Uses rescale_phis_alphas + representative row/column clusters.  Asymmetric
    ranks (rank_phi for spatial reps, rank_alpha for temporal reps) can reduce
    total K while preserving accuracy when spatial and temporal complexity differ.

    Args
    ----
    phis : torch.Tensor
        (B, R) spatial phase coefficients
    alphas : torch.Tensor
        (B, T) temporal phase coefficients
    rank : int
        default rank when rank_phi / rank_alpha are None
    rank_phi : Optional[int]
        number of spatial (phi) representatives
    rank_alpha : Optional[int]
        number of temporal (alpha) representatives
    cluster_method : str
        'maxmin' (default, better coverage) or 'kmeans'

    Returns
    -------
    R : torch.Tensor
        (K, R) with K = rank_phi
    C : torch.Tensor
        (K, T) with K = rank_phi  (same K index couples R and C via U)
    """
    # Consts
    B, R = phis.shape
    T = alphas.shape[1]
    Ka = rank_alpha if rank_alpha is not None else rank
    Kp = rank_phi if rank_phi is not None else rank
    
    # Rescale: maxmin/clustering distance uses the normalized (per-order
    # equalized) coords; phase exponentials use the full phis_nrm+phis_mp
    # affine coords. Using raw coeffs for clustering lets the largest-magnitude
    # phase order dominate farthest-point distance, picking near-duplicate
    # pivots along smaller (but still phase-relevant) orders -- see
    # build_cur_factors_adaptive for why this matters more there.
    phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)
    phis_full = phis_nrm + phis_mp[:, None]
    alphas_full = alphas_nrm + alphas_mp[:, None]

    # Cluster alphas
    if Ka >= T:
        alpha_clusts = alphas_nrm.T
    else:
        alpha_clusts, _ = pick_K_vectors(
            alphas_nrm.T, K=Ka, sigma=0, method=cluster_method)
    alpha_clusts = alpha_clusts + alphas_mp
    
    # Cluster phis
    if Kp >= R:
        phi_clusts = phis_nrm.T
    else:
        phi_clusts, _ = pick_K_vectors(
            phis_nrm.T, K=Kp, sigma=0, method=cluster_method)
    phi_clusts = phi_clusts + phis_mp

    # Compute R and C
    R_raw = torch.exp(-2j * torch.pi * (alpha_clusts @ phis_full))   # (Ka, R)
    C = torch.exp(-2j * torch.pi * (phi_clusts @ alphas_full))       # (Kp, T)

    W = torch.exp(-2j * torch.pi * (alpha_clusts @ phi_clusts.T))    # (Ka, Kp)
    U = torch.linalg.pinv(W)                                           # (Kp, Ka)
    R = einsum(U, R_raw, 'Kp Ka, Ka R -> Kp R')
    return R, C


def maxmin_pivots(vectors: torch.Tensor,
                  K: int,
                  seed: Optional[int] = None) -> torch.Tensor:
    """
    Greedy farthest-point (maxmin) pivot selection, returning indices in the
    order they were picked. Since the picks are incremental, the first k
    indices of a K-pivot call equal a standalone k-pivot call given the same
    seed -- this nesting is what lets build_cur_factors_adaptive grow rank
    one pivot at a time instead of resampling per candidate rank.

    Args
    ----
    vectors : (N, D) candidate points
    K : number of pivots to select
    seed : optional RNG seed for the initial point (fix this for reproducible
        / nested selection across calls)

    Returns
    -------
    idxs : (K,) long tensor, indices into `vectors` in selection order
    """
    N = vectors.shape[0]
    gen = torch.Generator(device=vectors.device)
    if seed is not None:
        gen.manual_seed(seed)
    picked = [int(torch.randint(0, N, (1,), generator=gen, device=vectors.device))]
    dist = torch.linalg.norm(vectors - vectors[picked[-1]], dim=-1)
    for _ in range(1, K):
        nxt = int(torch.argmax(dist))
        picked.append(nxt)
        dist = torch.minimum(dist, torch.linalg.norm(vectors - vectors[nxt], dim=-1))
    return torch.tensor(picked, dtype=torch.long, device=vectors.device)


def _border_inverse(V: torch.Tensor,
                    b: torch.Tensor,
                    c: torch.Tensor,
                    d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extend inv(W_r) = V (r, r) to inv(W_{r+1}) given the new column `b` (r,),
    new row `c` (r,), and corner `d` (scalar), via the Schur-complement
    bordering update -- O(r^2) instead of an O(r^3) pinv from scratch.

    Returns (V_new, s), where s is the Schur complement: small |s| means the
    new pivot is nearly linearly dependent on the existing ones (cheap
    rank-sufficiency signal, analogous to a decaying pivot in rank-revealing
    LU/QR).
    """
    Vb = V @ b
    cV = c @ V
    s = d - c @ Vb
    r = V.shape[0]
    V_new = torch.empty((r + 1, r + 1), dtype=V.dtype, device=V.device)
    V_new[:r, :r] = V + torch.outer(Vb, cV) / s
    V_new[:r, r] = -Vb / s
    V_new[r, :r] = -cV / s
    V_new[r, r] = 1.0 / s
    return V_new, s


def build_cur_factors_adaptive(phis: torch.Tensor,
                               alphas: torch.Tensor,
                               max_rank: int = 1000,
                               min_rank: int = 100,
                               tol: float = 1e-3,
                               check_every: int = 8,
                               n_val: int = 2048,
                               resid_tol: float = 1e-2,
                               seed: int = 0,
                               verbose: bool = False,
                               ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Adaptive-rank CUR: grow a square pivot set (rank_alpha = rank_phi = r) one
    maxmin pivot at a time, maintaining U = inv(W_r) via the Schur-complement
    bordering update in `_border_inverse` instead of an independent
    torch.linalg.pinv per candidate rank. Reconstruction error against a
    held-out sample of Phi entries is checked every `check_every` pivots;
    growth stops as soon as that error is below `tol` (or at `max_rank`, or if
    a new pivot turns out to be numerically dependent on prior ones).

    Cost to reach the final rank r* is O(r*^3) total -- the same as a single
    pinv at r* -- versus O(r*^4) for independently re-inverting W at O(r*)
    candidate ranks.

    Args
    ----
    phis : torch.Tensor
        (B, R) spatial phase coefficients
    alphas : torch.Tensor
        (B, T) temporal phase coefficients
    max_rank : int
        largest rank to grow to before giving up
    min_rank : int
        skip the sequential bordering (and all error checks) below this rank
        by building inv(W_min_rank) directly with one torch.linalg.inv call,
        then bordering up from there. Same FLOP count as growing from rank 1,
        but avoids min_rank-1 sequential Python-level steps -- use this when
        you already know the useful rank is well above 1.
    tol : float
        relative L2 error on held-out Phi entries that stops growth
    check_every : int
        how often (in pivots) to evaluate the held-out error
    n_val : int
        number of held-out (r, t) entries used for the error estimate
    resid_tol : float
        max|inv(W_r) @ W_r - I| tolerance checked at each checkpoint; once
        exceeded, the unregularized bordering recursion has drifted too far
        to trust, so V is re-grounded with a fresh torch.linalg.pinv(W_r)
        (regularized, like build_cur_factors uses at every rank) and growth
        continues from there instead of stopping
    seed : int
        seed for maxmin pivot selection and held-out sampling
    verbose : bool
        print rank/error progress

    Returns
    -------
    R : torch.Tensor
        (K, R) with K = rank_used
    C : torch.Tensor
        (K, T) with K = rank_used
    rank_used : int
        rank at which growth stopped
    """
    # Consts
    B, Rdim = phis.shape
    T = alphas.shape[1]
    torch_dev = phis.device
    max_rank = min(max_rank, Rdim, T)
    min_rank = max(1, min(min_rank, max_rank))

    # Rescale: maxmin pivot order is built from the normalized (per-order
    # equalized) coords -- using raw coeffs here lets the largest-magnitude
    # phase order dominate farthest-point distance, so maxmin repeatedly
    # picks near-duplicate pivots along smaller (but still phase-relevant)
    # orders. That makes W severely ill-conditioned as rank grows, and since
    # this pivot order is nested/seeded (unlike build_cur_factors' fresh
    # per-call draw), one bad early pivot corrupts every higher rank instead
    # of being averaged out.
    phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)
    phis_full = phis_nrm + phis_mp[:, None]
    alphas_full = alphas_nrm + alphas_mp[:, None]

    # Ordered maxmin pivots, grown incrementally
    phi_order = maxmin_pivots(phis_nrm.T, max_rank, seed=seed)
    alpha_order = maxmin_pivots(alphas_nrm.T, max_rank, seed=seed + 1)
    phi_clusts_all = phis_nrm.T[phi_order] + phis_mp        # (max_rank, B)
    alpha_clusts_all = alphas_nrm.T[alpha_order] + alphas_mp  # (max_rank, B)

    # Held-out (r, t) entries of Phi for a cheap error estimate
    gen = torch.Generator(device=torch_dev)
    gen.manual_seed(seed + 2)
    r_val = torch.randint(0, Rdim, (n_val,), generator=gen, device=torch_dev)
    t_val = torch.randint(0, T, (n_val,), generator=gen, device=torch_dev)
    phi_true = torch.exp(-2j * torch.pi * einsum(
        phis_full[:, r_val], alphas_full[:, t_val], 'b n, b n -> n'))

    V = None
    rank_used = max_rank
    errs = []
    resids = []
    for r in tqdm(range(min_rank, max_rank + 1), desc='Sweeping CUR Rank'):
        if V is None:
            # First rank we actually build -- either r=1 (default) or the
            # requested min_rank, computed in one shot rather than bordered
            # up from scratch.
            phi_r_full = phi_clusts_all[:r]
            alpha_r_full = alpha_clusts_all[:r]
            W_r = torch.exp(-2j * torch.pi * (alpha_r_full @ phi_r_full.T))
            V = torch.linalg.inv(W_r)
            resid = (V @ W_r - torch.eye(r, dtype=W_r.dtype, device=W_r.device)).abs().max()
            if resid > 1e-3:
                print(f'Warning: build_cur_factors_adaptive: W at min_rank={r} is '
                     f'poorly conditioned (max|inv(W) @ W - I|={resid.item():.2e}); '
                     f'the true rank may be below min_rank. Growth from here on is '
                     f'unprotected by the near-dependent-pivot check that ranks below '
                     f'min_rank would have gotten -- consider lowering min_rank.')
        else:
            phi_r = phi_clusts_all[r - 1:r]      # (1, B)
            alpha_r = alpha_clusts_all[r - 1:r]  # (1, B)
            phi_prev = phi_clusts_all[:r - 1]
            alpha_prev = alpha_clusts_all[:r - 1]
            b = torch.exp(-2j * torch.pi * (alpha_prev @ phi_r.T)).squeeze(-1)
            c = torch.exp(-2j * torch.pi * (alpha_r @ phi_prev.T)).squeeze(0)
            d = torch.exp(-2j * torch.pi * (alpha_r @ phi_r.T)).reshape(())
            V_new, s = _border_inverse(V, b, c, d)
            if s.abs() < 1e-10:
                rank_used = r - 1
                if verbose:
                    print(f'stopping at rank {rank_used}: pivot {r} '
                         f'nearly dependent (|s|={s.abs().item():.2e})')
                break
            V = V_new

        if r % check_every == 0 or r == max_rank or r == min_rank:
            alpha_r_full = alpha_clusts_all[:r]
            phi_r_full = phi_clusts_all[:r]

            # The Schur-complement recursion has no regularization (unlike
            # pinv's implicit small-singular-value truncation), so floating
            # point error compounds once maxmin runs out of genuinely new,
            # well-separated pivots and W_r starts to become ill-conditioned.
            # Catch that drift via the residual rather than trusting a fixed
            # |s| threshold (by the time |s| is tiny, V is already far gone).
            # Instead of rolling back and stopping growth (which left this
            # method far less accurate at a given rank than build_cur_factors,
            # which uses pinv at every rank), re-ground with a fresh
            # regularized pinv(W_r) and keep growing from there -- this costs
            # one O(r^3) SVD, but only at checkpoints where drift is actually
            # detected, so it stays much cheaper than pinv-per-rank while
            # matching pinv's accuracy exactly where plain inversion breaks
            # down.
            W_r_chk = torch.exp(-2j * torch.pi * (alpha_r_full @ phi_r_full.T))
            resid = (V @ W_r_chk - torch.eye(r, dtype=V.dtype, device=V.device)).abs().max()
            if resid > resid_tol:
                V = torch.linalg.pinv(W_r_chk)
                if verbose:
                    print(f'rank {r}: numerical drift detected '
                         f'(max|inv(W)@W - I|={resid.item():.2e} > {resid_tol}); '
                         f're-grounding with pinv(W)')

            R_raw_val = torch.exp(-2j * torch.pi * (alpha_r_full @ phis_full[:, r_val]))  # (r, n_val)
            C_val = torch.exp(-2j * torch.pi * (phi_r_full @ alphas_full[:, t_val]))      # (r, n_val)
            R_val = V @ R_raw_val
            phi_approx = einsum(R_val, C_val, 'k n, k n -> n')
            err = torch.linalg.norm(phi_approx - phi_true) / torch.linalg.norm(phi_true)
            if verbose:
                print(f'rank {r}: rel err {err.item():.3e} (residual {resid.item():.2e})')
            if err < tol:
                rank_used = r
                break

            resids.append(resid)
            errs.append(err)
    
    # errs = torch.tensor(errs)
    # resids = torch.tensor(resids)
    # ns = torch.arange(min_rank, min_rank + len(errs) * check_every, check_every)
    # import matplotlib.pyplot as plt
    # plt.plot(ns, errs[:len(ns)].cpu())
    # plt.figure()
    # plt.plot(ns, resids[:len(ns)].cpu())
    # plt.show()
    # quit()

    # Final factors at the chosen rank
    phi_clusts = phi_clusts_all[:rank_used]
    alpha_clusts = alpha_clusts_all[:rank_used]
    R_raw = torch.exp(-2j * torch.pi * (alpha_clusts @ phis_full))
    C = torch.exp(-2j * torch.pi * (phi_clusts @ alphas_full))
    R = V @ R_raw
    if verbose:
        print(f'adaptive CUR: rank={rank_used} (max_rank={max_rank})')
    return R, C, rank_used


def cur_forward(Dw: torch.Tensor,
                R: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
    """
    corr[q,t] = sum_r Dw[q,r] Phi[r,t] ≈ (Dw @ R.T) @ C.

    Args
    ----
    Dw : (Q, R)
    R  : (K, R)
    C  : (K, T) or (K, Tb)

    Returns
    -------
    corr : (Q, T) or (Q, Tb)
    """
    coeffs = einsum(Dw, R, 'Q R, K R -> Q K')
    return einsum(coeffs, C, 'Q K, K T -> Q T')


def svd_lowrank_forward(Dw: torch.Tensor,
                        phis: torch.Tensor,
                        alphas: torch.Tensor,
                        rank: int) -> torch.Tensor:
    """
    Optimal rank-`rank` approximation of corr = Dw @ Phi^T for this alpha slice.

    Forms enc = Phi^T (R, T) only for the current T columns; uses torch.svd_lowrank.
    Best rank/accuracy tradeoff for a fixed temporal batch, at the cost of a
    temporary (R, T) encoding matrix.

    Args
    ----
    Dw : (Q, R)
    phis : (B, R)
    alphas : (B, T)
    rank : int

    Returns
    -------
    corr : (Q, T)
    """
    enc = torch.exp(-2j * torch.pi * (alphas.T @ phis))  # (T, R)
    M = enc.T  # (R, T)
    k = min(rank, *M.shape)
    # torch.svd_lowrank does not support complex — use thin SVD (fine for Tb << T)
    U, s, Vh = torch.linalg.svd(M, full_matrices=False)
    Ur = U[:, :k]
    s = s[:k]
    Vh = Vh[:k, :]
    tmp = Dw @ Ur
    return (tmp * s.to(tmp.dtype)) @ Vh


def corr_from_dw_batched(Dw: torch.Tensor,
                         phis: torch.Tensor,
                         alphas: torch.Tensor,
                         temporal_batch_size: int,
                         method: str = 'cur',
                         rank: int = 128,
                         rank_phi: Optional[int] = None,
                         rank_alpha: Optional[int] = None,
                         cluster_method: ClusterMethod = 'maxmin',
                         phi_clusters_once: bool = True,
                         verbose: bool = False) -> torch.Tensor:
    """
    Compute corr (Q, T) = Dw @ Phi in temporal batches without storing C (K, T).

    Methods
    -------
    'cur' : rebuild CUR per batch (alpha clusters + C[:, t1:t2]); spatial phi
            clusters reused if phi_clusters_once=True (R side still rebuilt when
            alpha clusters change).
    'svd_lowrank' : optimal rank-k SVD per batch (usually fewer k for same err).

    Memory per batch: O(K*R + K*Tb) for CUR, O(R*Tb + rank*(R+Tb)) for SVD-lr.
    """
    from tqdm import tqdm

    B, R = phis.shape
    T = alphas.shape[1]
    Q = Dw.shape[0]
    torch_dev = phis.device
    corr_all = torch.empty((Q, T), dtype=complex_dtype, device=torch_dev)

    Kp = rank_phi if rank_phi is not None else rank
    # Rescale once (consistent midpoints across all batches)
    phis_full, alphas_nrm, alphas_mp, phi_clusts = setup_cur_phi_clusters(
        phis, alphas, rank_phi=Kp, cluster_method=cluster_method)

    tbs = temporal_batch_size
    it = range(0, T, tbs)
    if verbose:
        it = tqdm(it, desc=f'corr batched ({method})')

    for t1 in it:
        t2 = min(t1 + tbs, T)

        if method == 'svd_lowrank':
            corr_all[:, t1:t2] = svd_lowrank_forward(
                Dw, phis_full, alphas_nrm[:, t1:t2] + alphas_mp[:, None], rank)
        elif method == 'cur':
            if phi_clusters_once:
                corr_all[:, t1:t2] = cur_corr_slice(
                    Dw, phis_full, alphas_nrm, alphas_mp, phi_clusts,
                    t1, t2, rank_alpha=rank_alpha or rank,
                    cluster_method=cluster_method)
            else:
                R, C = build_cur_factors(
                    phis, alphas[:, t1:t2], rank=rank, rank_phi=rank_phi,
                    rank_alpha=rank_alpha, cluster_method=cluster_method)
                corr_all[:, t1:t2] = cur_forward(Dw, R, C)
        else:
            raise ValueError(f'unknown method {method!r}')

    return corr_all


def estimate_batched_memory_bytes(Q: int, R: int, T: int, Tb: int, K: int,
                                  method: str = 'cur') -> dict:
    """Peak extra memory (bytes) for batched corr vs storing full C (K, T)."""
    full_C = K * T * 8
    if method == 'cur':
        peak = K * R * 8 + K * Tb * 8 + Q * Tb * 8
    else:
        peak = R * Tb * 8 + K * (R + Tb) * 8 + Q * Tb * 8
    return {
        'full_cur_C': full_C,
        'batched_peak': peak,
        'ratio': full_C / peak,
    }
