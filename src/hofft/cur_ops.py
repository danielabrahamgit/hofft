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

from .phase_coeffs import rescale_phis_alphas, whiten_phis_alphas
from .utils import maxmin_indices, maxmin_centroids, kmeans_centroids, fps_multi_center_indices

ClusterMethod = Literal['maxmin', 'kmeans']
NormalizeMethod = Literal['rescale', 'cov', 'mahal', 'svd', '']


def _pivot_indices(coords: torch.Tensor,
                   K: int,
                   cluster_method: ClusterMethod,
                   seed: Optional[int] = None) -> torch.Tensor:
    """
    Return K sample indices into ``coords`` of shape (N, B).

    maxmin uses farthest-point picks (true sample indices). kmeans snaps each
    centroid to its nearest sample so the CUR factors are built from actual
    φ/α columns, not from interpolated centroid coordinates.
    """
    N = coords.shape[0]
    if K >= N:
        return torch.arange(N, device=coords.device)
    if cluster_method == 'maxmin':
        return maxmin_pivots(coords, K, seed=seed)
    cents, _ = pick_K_vectors(coords, K=K, sigma=0, method=cluster_method,
                              return_idxs=False)
    cents_sq = (cents ** 2).sum(dim=-1)
    coords_sq = (coords ** 2).sum(dim=-1)
    d = cents_sq[:, None] + coords_sq[None, :] - 2 * (cents @ coords.T)
    return d.argmin(dim=1)


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


def _prepare_cur_space(
    phis: torch.Tensor,
    alphas: torch.Tensor,
    normalize_method: NormalizeMethod = 'svd',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shared CUR coordinate prep for ``build_cur_factors`` and
    ``build_cur_factors_adaptive``.

    Clustering is done in a metric where Euclidean distance approximates
    phase error; R/C/W are formed from ``phis_full`` / ``alphas_full`` so
    that ``phis_full[:, n] · alphas_full[:, t]`` equals the demeaned
    (or rescaled) total phase. Midpoint / mean phase is factored into
    the unit-modulus ``spat`` and ``temp`` corrections.

    Returns
    -------
    phis_full : (B', N)
        Spatial columns used to form C and W.
    alphas_full : (B', T)
        Temporal columns used to form R and W.
    phis_pivot : (N, B')
        Spatial coordinates for FPS / k-means.
    alphas_pivot : (T, B')
        Temporal coordinates for FPS / k-means.
    spat : (N,)
        Spatial midpoint correction.
    temp : (T,)
        Temporal midpoint correction.
    """
    B = phis.shape[0]

    if normalize_method == 'rescale':
        phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis, alphas)
        phis_full = phis_nrm
        alphas_full = alphas_nrm
        phis_pivot = phis_nrm.T
        alphas_pivot = alphas_nrm.T
        spat = phis_nrm.T @ alphas_mp
        temp = alphas_nrm.T @ phis_mp
        temp = temp + alphas_mp @ phis_mp
        spat = torch.exp(-2j * torch.pi * spat)
        temp = torch.exp(-2j * torch.pi * temp)

    elif normalize_method == 'cov':
        phi0 = phis.mean(dim=1)
        alpha0 = alphas.mean(dim=1)
        phis_demean = phis - phi0[:, None]
        alphas_demean = alphas - alpha0[:, None]
        spat = phis_demean.T @ alpha0
        temp = alphas_demean.T @ phi0
        temp = temp + alpha0 @ phi0
        spat = torch.exp(-2j * torch.pi * spat)
        temp = torch.exp(-2j * torch.pi * temp)
        phis_full = phis_demean
        alphas_full = alphas_demean
        PPT = phis_demean @ phis_demean.T
        AAT = alphas_demean @ alphas_demean.T
        P_eigvals, P_eigvecs = torch.linalg.eigh(PPT)
        A_eigvals, A_eigvecs = torch.linalg.eigh(AAT)
        P_scale = P_eigvals.clamp_min(P_eigvals.max() * 1e-8).rsqrt()
        A_scale = A_eigvals.clamp_min(A_eigvals.max() * 1e-8).rsqrt()
        phis_nrm = (P_scale[:, None] * P_eigvecs.T) @ phis_demean
        alphas_nrm = (A_scale[:, None] * A_eigvecs.T) @ alphas_demean
        phis_pivot = phis_nrm.T
        alphas_pivot = alphas_nrm.T

    elif normalize_method == 'mahal':
        phi0 = phis.mean(dim=1)
        alpha0 = alphas.mean(dim=1)
        phis_demean = phis - phi0[:, None]
        alphas_demean = alphas - alpha0[:, None]
        spat = phis_demean.T @ alpha0
        temp = alphas_demean.T @ phi0
        temp = temp + alpha0 @ phi0
        spat = torch.exp(-2j * torch.pi * spat)
        temp = torch.exp(-2j * torch.pi * temp)

        # sum_n |e^{-j2π Δα · φ(r_n)}|^2 ≈ Δα^T (Φ Φ^T) Δα.
        # W_φ (Φ Φ^T) W_φ^T = I  ⇒  Euclidean FPS on α' = W_φ^{-T} α.
        # Dual: Euclidean FPS on φ' = W_α^{-T} φ. CUR uses φ'' = W_φ φ,
        # α' = W_φ^{-T} α so φ'' · α' = φ · α.
        def _gram_whiten(X):
            evals, evecs = torch.linalg.eigh(X @ X.T)
            lam = evals.clamp_min(evals.max() * 1e-8)
            W = lam.rsqrt()[:, None] * evecs.T
            W_inv_T = lam.sqrt()[:, None] * evecs.T
            return W, W_inv_T

        W_phi, W_phi_inv_T = _gram_whiten(phis_demean)
        _, W_alpha_inv_T = _gram_whiten(alphas_demean)
        phis_full = W_phi @ phis_demean
        alphas_full = W_phi_inv_T @ alphas_demean
        phis_pivot = (W_alpha_inv_T @ phis_demean).T
        alphas_pivot = alphas_full.T

    elif normalize_method == 'svd':
        phi0 = phis.mean(dim=1)
        alpha0 = alphas.mean(dim=1)
        phis_demean = phis - phi0[:, None]
        alphas_demean = alphas - alpha0[:, None]
        spat = phis_demean.T @ alpha0
        temp = alphas_demean.T @ phi0
        temp = temp + alpha0 @ phi0
        spat = torch.exp(-2j * torch.pi * spat)
        temp = torch.exp(-2j * torch.pi * temp)
        phis_nrm, alphas_nrm, S = whiten_phis_alphas(
            phis_demean, alphas_demean,
            B_compressed=B, return_singular_values=True,
        )
        phis_full = phis_nrm
        alphas_full = (alphas_nrm.T * S).T
        phis_pivot = phis_nrm.T * S
        alphas_pivot = alphas_nrm.T * S

    elif normalize_method == '':
        phis_full = phis
        alphas_full = alphas
        phis_pivot = phis_full.T
        alphas_pivot = alphas_full.T
        spat = torch.exp(-2j * torch.pi * phis[0] * 0)
        temp = torch.exp(-2j * torch.pi * alphas[0] * 0)

    else:
        raise ValueError(
            f"Unknown normalize_method {normalize_method!r}. "
            "Expected one of 'rescale', 'cov', 'mahal', 'svd', ''."
        )

    return phis_full, alphas_full, phis_pivot, alphas_pivot, spat, temp


def _cur_factors_from_pivots(phis_full: torch.Tensor,
                             alphas_full: torch.Tensor,
                             phi_idxs: torch.Tensor,
                             alpha_idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CUR R, C (no midpoint correction) from pivot indices. Same pinv as build_cur_factors."""
    phi_clusts = phis_full.T[phi_idxs]
    alpha_clusts = alphas_full.T[alpha_idxs]
    R = torch.exp(-2j * torch.pi * (alpha_clusts @ phis_full))
    C = torch.exp(-2j * torch.pi * (phi_clusts @ alphas_full))
    W = torch.exp(-2j * torch.pi * (alpha_clusts @ phi_clusts.T))
    R = einsum(torch.linalg.pinv(W), R, 'Kp Ka, Ka N -> Kp N')
    return R, C


def _cur_heldout_rel_err(phis_full: torch.Tensor,
                         alphas_full: torch.Tensor,
                         phi_idxs: torch.Tensor,
                         alpha_idxs: torch.Tensor,
                         r_val: torch.Tensor,
                         t_val: torch.Tensor,
                         phi_true: torch.Tensor) -> torch.Tensor:
    """Relative L2 error of pinv-CUR on held-out (r, t) entries of the unit-modulus phase."""
    phi_clusts = phis_full.T[phi_idxs]
    alpha_clusts = alphas_full.T[alpha_idxs]
    R_raw = torch.exp(-2j * torch.pi * (alpha_clusts @ phis_full[:, r_val]))
    C = torch.exp(-2j * torch.pi * (phi_clusts @ alphas_full[:, t_val]))
    W = torch.exp(-2j * torch.pi * (alpha_clusts @ phi_clusts.T))
    approx = einsum(torch.linalg.pinv(W) @ R_raw, C, 'k n, k n -> n')
    return (approx - phi_true).norm() / phi_true.norm()


def build_cur_factors(phis: torch.Tensor,
                      alphas: torch.Tensor,
                      rank: int = 128,
                      rank_phi: Optional[int] = None,
                      rank_alpha: Optional[int] = None,
                      cluster_method: ClusterMethod = 'maxmin',
                      normalize_method: NormalizeMethod = 'svd',
                      seed: int = 0,
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build CUR factors so Phi[r,t] ≈ sum_k R[k,r] C[k,t].

    Args
    ----
    phis : torch.Tensor
        (B, R) spatial phase coefficients (V^T when S is given)
    alphas : torch.Tensor
        (B, T) temporal phase coefficients (U^T when S is given)
    rank : int
        default rank when rank_phi / rank_alpha are None
    rank_phi : Optional[int]
        number of spatial (phi) representatives
    rank_alpha : Optional[int]
        number of temporal (alpha) representatives
    cluster_method : str
        'maxmin' (default, better coverage) or 'kmeans'
    normalize_method : Literal['rescale', 'cov', 'mahal', 'svd', '']
        Method to normalize the coefficients:
        - 'rescale': rescale so phis are in [-1/2, 1/2]
        - 'cov': ZCA-whiten each side with its own Gram (pivot metric only)
        - 'mahal': first-order phase-error metric. W_φ (ΦΦᵀ) W_φᵀ = I,
          α' = W_φ^{-T} α (and the dual for φ); CUR in the W_φ pair
        - 'svd': joint SVD of A^T Φ; cluster in the whitened plane
        - '': no normalization
    seed : int
        RNG seed for the first maxmin pivot (same seed ⇒ same start index)

    Returns
    -------
    R : torch.Tensor
        (K, R) with K = rank_phi
    C : torch.Tensor
        (K, T) with K = rank_phi  (same K index couples R and C via U)
    """
    Ka = rank_alpha if rank_alpha is not None else rank
    Kp = rank_phi if rank_phi is not None else rank

    phis_full, alphas_full, phis_pivot, alphas_pivot, spat, temp = _prepare_cur_space(
        phis, alphas, normalize_method)

    phi_idxs = _pivot_indices(phis_pivot, Kp, cluster_method, seed=seed)
    alpha_idxs = _pivot_indices(alphas_pivot, Ka, cluster_method, seed=seed)
    R, C = _cur_factors_from_pivots(phis_full, alphas_full, phi_idxs, alpha_idxs)
    R = einsum(R, spat, 'Kp N, N -> Kp N')
    C = einsum(C, temp, 'Kp M, M -> Kp M')
    return R, C


def maxmin_pivots(vectors: torch.Tensor,
                  K: int,
                  seed: Optional[int] = None) -> torch.Tensor:
    """
    Greedy farthest-point (maxmin) pivot selection, returning indices in the
    order they were picked. The first k indices of a K-pivot call equal a
    standalone k-pivot call with the same seed, so adaptive rank search can
    reuse one maxmin run and take prefixes.

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
    # Keep indices on device: int(argmax)/int(randint) would sync every pivot.
    picked = torch.empty(K, dtype=torch.long, device=vectors.device)
    picked[0] = torch.randint(0, N, (), generator=gen, device=vectors.device)
    dist = torch.linalg.norm(vectors - vectors[picked[0]], dim=-1)
    for k in range(1, K):
        nxt = dist.argmax()
        picked[k] = nxt
        dist = torch.minimum(dist, torch.linalg.norm(vectors - vectors[nxt], dim=-1))
    return picked


def build_cur_factors_adaptive(phis: torch.Tensor,
                               alphas: torch.Tensor,
                               max_rank: int = 1000,
                               min_rank: int = 16,
                               tol: float = 1e-2,
                               check_every: int = 16,
                               n_val: int = 2048,
                               seed: int = 0,
                               verbose: bool = False,
                               normalize_method: NormalizeMethod = 'svd',
                               cluster_method: ClusterMethod = 'maxmin',
                               normalize_coeffs: Optional[bool] = None,
                               ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Smallest square CUR rank whose held-out phase error is below ``tol``.

    Uses the same ``normalize_method`` / pivot metric / ``pinv(W)`` construction
    as ``build_cur_factors``, so the rank that meets ``tol`` is the rank at
    which ``build_cur_factors(..., rank=r)`` would meet it. Each candidate
    rank is built from scratch (no incremental inverse), which removes the
    Schur-complement drift that used to make the error-vs-rank curve
    unreliable.

    Search: evaluate every ``check_every`` ranks from ``min_rank`` until the
    error drops below ``tol`` (or ``max_rank``), then binary-search that last
    window for the smallest passing rank. maxmin pivots are nested, so one
    FPS run is reused as prefixes. kmeans re-clusters at each evaluated rank.

    Args
    ----
    phis : torch.Tensor
        (B, R) spatial phase coefficients
    alphas : torch.Tensor
        (B, T) temporal phase coefficients
    max_rank : int
        largest rank to try
    min_rank : int
        smallest rank to try
    tol : float
        relative L2 error on held-out Phi entries that stops growth
    check_every : int
        coarse stride before the binary refinement
    n_val : int
        number of held-out (r, t) entries used for the error estimate
    seed : int
        seed for pivot selection and held-out sampling
    verbose : bool
        print rank / error at each evaluation
    normalize_method : {'rescale', 'cov', 'mahal', 'svd', ''}
        Same options as ``build_cur_factors``
    cluster_method : {'maxmin', 'kmeans'}
        Same options as ``build_cur_factors``
    normalize_coeffs : Optional[bool]
        Deprecated. True → 'rescale', False → ''.

    Returns
    -------
    R : torch.Tensor
        (K, R) with K = rank_used
    C : torch.Tensor
        (K, T) with K = rank_used
    rank_used : int
        smallest rank whose held-out error is below ``tol``, or ``max_rank``
    """
    if normalize_coeffs is not None:
        normalize_method = 'rescale' if normalize_coeffs else ''

    N = phis.shape[1]
    T = alphas.shape[1]
    torch_dev = phis.device
    max_rank = min(max_rank, N, T)
    min_rank = max(1, min(min_rank, max_rank))
    check_every = max(1, check_every)

    phis_full, alphas_full, phis_pivot, alphas_pivot, spat, temp = _prepare_cur_space(
        phis, alphas, normalize_method)

    # Nested maxmin prefixes; kmeans has no nesting so it is re-run per rank.
    if cluster_method == 'maxmin':
        phi_idxs_all = _pivot_indices(phis_pivot, max_rank, cluster_method, seed=seed)
        alpha_idxs_all = _pivot_indices(alphas_pivot, max_rank, cluster_method, seed=seed)
    else:
        phi_idxs_all = alpha_idxs_all = None

    def pivots_at(r: int) -> tuple[torch.Tensor, torch.Tensor]:
        if phi_idxs_all is not None:
            return phi_idxs_all[:r], alpha_idxs_all[:r]
        return (
            _pivot_indices(phis_pivot, r, cluster_method, seed=seed),
            _pivot_indices(alphas_pivot, r, cluster_method, seed=seed),
        )

    gen = torch.Generator(device=torch_dev)
    gen.manual_seed(seed + 2)
    r_val = torch.randint(0, N, (n_val,), generator=gen, device=torch_dev)
    t_val = torch.randint(0, T, (n_val,), generator=gen, device=torch_dev)
    phi_true = torch.exp(-2j * torch.pi * einsum(
        phis_full[:, r_val], alphas_full[:, t_val], 'b n, b n -> n'))

    err_cache: dict[int, float] = {}

    def err_at(r: int) -> float:
        if r not in err_cache:
            phi_idxs, alpha_idxs = pivots_at(r)
            err = _cur_heldout_rel_err(
                phis_full, alphas_full, phi_idxs, alpha_idxs,
                r_val, t_val, phi_true,
            )
            err_cache[r] = err.item()
            if verbose:
                print(f'CUR rank {r}: rel err {err_cache[r]:.3e}')
        return err_cache[r]

    def first_at_most(hi: int, lo: int) -> int:
        """Smallest rank in (lo, hi] with error < tol. Assumes err(hi) < tol."""
        left, right, best = lo + 1, hi, hi
        while left <= right:
            mid = (left + right) // 2
            if err_at(mid) < tol:
                best = mid
                right = mid - 1
            else:
                left = mid + 1
        return best

    rank_used = max_rank
    err = err_at(min_rank)
    if err < tol:
        rank_used = min_rank
    else:
        prev, r = min_rank, min_rank
        while r < max_rank:
            prev = r
            r = min(r + check_every, max_rank)
            if err_at(r) < tol:
                rank_used = first_at_most(r, prev)
                break
        else:
            rank_used = max_rank
            if verbose:
                print(f'CUR rank {rank_used}: tol={tol:g} not reached '
                      f'(rel err {err_at(rank_used):.3e})')

    phi_idxs, alpha_idxs = pivots_at(rank_used)
    R, C = _cur_factors_from_pivots(phis_full, alphas_full, phi_idxs, alpha_idxs)
    R = einsum(R, spat, 'Kp N, N -> Kp N')
    C = einsum(C, temp, 'Kp M, M -> Kp M')
    if verbose:
        print(f'adaptive CUR: rank={rank_used}  rel err={err_at(rank_used):.3e}  '
              f'(tol={tol:g}, max_rank={max_rank})')
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
