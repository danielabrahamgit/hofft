"""
Temporally sparse phase factors.

Two K-SVD families live here:

- ``ksvd`` / ``ksvd_temporal``: unconstrained Rubinstein K-SVD (SVD init,
  matching pursuit, per-atom residual update). This is the Q=1 study in
  ``math_docs/ksvd_feas.md``.
- ``ksvd_fit``: OMP on the current dictionary plus block ALS, starting from
  analytic phase atoms. Kept for the interpolative / FPS-NN path.

Supports, fixed-support ALS, and OMP against the current ``B`` are shared
helpers. Rank work in the interpolative path is float64.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from hofft.utils import maxmin_indices


def rel_error(P: torch.Tensor, H: torch.Tensor, Bmat: torch.Tensor) -> float:
    return float((P - H @ Bmat).norm() / P.norm().clamp_min(1e-30))


def phi_gram(phis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    B = phis.shape[0]
    p = phis.reshape(B, -1).double()
    if mask is not None:
        p = p[:, mask.reshape(-1) > 0]
    n = max(p.shape[1], 1)
    return (p @ p.mT) / n


def whitening_matrix(Sig: torch.Tensor) -> torch.Tensor:
    ev, V = torch.linalg.eigh(Sig.double())
    ev = ev.clamp(min=0)
    floor = 1e-8 * float(ev.max().clamp(min=1e-30))
    ev = ev.clamp(min=floor)
    return V * ev.sqrt()


def whitened_alphas(alphas: torch.Tensor, Sigma_sqrt: torch.Tensor) -> torch.Tensor:
    """Return (M, B) whitened alpha cloud: Sigma_sqrt @ alpha."""
    a = alphas.reshape(alphas.shape[0], -1).double()
    return (Sigma_sqrt @ a).T.contiguous()


def arc_length(aw: torch.Tensor) -> torch.Tensor:
    s = aw.new_zeros(aw.shape[0])
    if aw.shape[0] > 1:
        s[1:] = (aw[1:] - aw[:-1]).norm(dim=-1).cumsum(0)
    return s


def effective_rank(aw: torch.Tensor) -> tuple[float, torch.Tensor]:
    X = aw - aw.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(X)
    p = s.square()
    p = p / p.sum().clamp_min(1e-30)
    d_eff = float((p.sum() ** 2) / p.square().sum().clamp_min(1e-30))
    return d_eff, p


def kb_beta(sigma: float, S: int, mode: str = 'beatty') -> float:
    sigma = float(sigma)
    S = int(S)
    if mode == 'ideal':
        return math.pi * S * math.sqrt(max(1.0 - 1.0 / sigma ** 2, 0.0))
    arg = ((S / sigma) * (sigma - 0.5)) ** 2 - 0.8
    return float(math.pi * math.sqrt(arg)) if arg > 0 else 1.0


def _inv_psihat(u: torch.Tensor, beta: float, S: int) -> torch.Tensor:
    """Deapodization: c / sinh(c) with c = sqrt(beta^2 - (pi S u)^2)."""
    u = u.double()
    z2 = beta ** 2 - (math.pi * S * u) ** 2
    c = z2.clamp(min=0).sqrt()
    val = c / torch.sinh(c.clamp(min=1e-12))
    return torch.where(z2 >= 0, val, torch.zeros_like(val))


def _kb_i0_norm(x: torch.Tensor, beta: float) -> torch.Tensor:
    """Kaiser-Bessel on x in [-1, 1], normalized to 1 at 0."""
    from hofft.kb import kb_weights_1d
    xc = x.clamp(-1 + 1e-7, 1 - 1e-7)
    w = kb_weights_1d(xc.float(), float(beta)).double()
    w0 = kb_weights_1d(x.new_zeros(()).float(), float(beta)).double()
    return w / w0.clamp_min(1e-30)


def analytic_gridding_1d(phi: torch.Tensor,
                         alpha: torch.Tensor,
                         sigma: float,
                         S: int,
                         beta_mode: str = 'beatty') -> dict:
    """
    1-D KB interpolation of exp(-2j pi phi * alpha) on a sigma-oversampled
    alpha grid of width S. Returns H (M x L) and Bmat (L x N).

    Spatial maps are deapodized by the discrete DTFT of the interpolator so
    a pure complex exponential is unbiased.
    """
    phi = phi.double().reshape(-1)
    alpha = alpha.double().reshape(-1)
    device = phi.device
    Dphi = float(phi.max() - phi.min())
    Dalpha = float(alpha.max() - alpha.min())
    sigma = float(sigma)
    S = int(S)
    Delta = 1.0 / (sigma * max(Dphi, 1e-12))
    n_inner = max(int(math.ceil(Dalpha / Delta)), 1)
    L = n_inner + S
    alpha0 = float(alpha.min()) - 0.5 * S * Delta
    beta_grid = alpha0 + Delta * torch.arange(L, device=device, dtype=torch.float64)
    beta_kb = kb_beta(sigma, S, beta_mode)

    half = 0.5 * S * Delta
    x = (alpha[:, None] - beta_grid[None, :]) / half
    inside = x.abs() < 1
    kern = torch.where(inside, _kb_i0_norm(x, beta_kb), torch.zeros_like(x))
    H = kern.to(torch.complex128)

    n_taps = torch.arange(S, device=device, dtype=torch.float64) - 0.5 * (S - 1)
    xt = n_taps / (0.5 * S)
    kn = torch.where(xt.abs() < 1 - 1e-12, _kb_i0_norm(xt, beta_kb), torch.zeros_like(xt))
    phi_c = phi - 0.5 * (phi.max() + phi.min())
    nu = phi_c * Delta
    psihat = (kn[None, :] * torch.exp(-2j * math.pi * nu[:, None] * n_taps[None, :])).sum(-1)
    # Match DC to the realized interpolator (row-sum of H).
    dc_h = H.real.sum(dim=1).mean().clamp_min(1e-12)
    dc_psi = psihat.real[torch.argmin(phi_c.abs())].clamp_min(1e-12)
    deap = (dc_psi / dc_h) / psihat.real.clamp(min=1e-12)
    Bmat = torch.exp(-2j * math.pi * beta_grid[:, None] * phi[None, :])
    Bmat = Bmat * deap[None, :].to(Bmat.dtype)

    closed = _inv_psihat(phi_c * Delta, beta_kb, S)
    closed0 = _inv_psihat(phi.new_zeros(1), beta_kb, S)
    closed_n = closed / closed0.clamp_min(1e-30)
    max_deapod = float(closed_n.max().clamp(min=1.0))
    return dict(H=H, Bmat=Bmat, L=L, max_deapod=max_deapod,
                beta=beta_grid, Delta=Delta)


def anchors_fps(aw: torch.Tensor, L: int, seed: int = 0) -> torch.Tensor:
    """Farthest-point anchors. aw is (M, B); returns betas_w (B, L)."""
    L = min(int(L), aw.shape[0])
    idx = maxmin_indices(aw.float(), L, seed=seed)
    return aw[idx].T.contiguous()


def anchors_arclength(aw: torch.Tensor, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Equal-arc-length anchors on the time-ordered curve. Returns (B, L), (L,)."""
    s = arc_length(aw)
    total = float(s[-1].clamp_min(1e-12))
    L = min(int(L), aw.shape[0])
    s_beta = torch.linspace(0.0, total, L, device=aw.device, dtype=aw.dtype)
    idx = torch.searchsorted(s, s_beta).clamp(1, aw.shape[0] - 1)
    s0, s1 = s[idx - 1], s[idx]
    w = ((s_beta - s0) / (s1 - s0).clamp_min(1e-12))[:, None]
    pts = (1.0 - w) * aw[idx - 1] + w * aw[idx]
    return pts.T.contiguous(), s_beta


def support_nearest(aw: torch.Tensor, betas_w: torch.Tensor, S: int) -> torch.Tensor:
    """S nearest FPS/arc anchors in the whitened metric. (M, S) int64."""
    S = min(int(S), betas_w.shape[1])
    d2 = torch.cdist(aw.double(), betas_w.T.double())
    return torch.topk(d2, k=S, dim=1, largest=False).indices.to(torch.int64)


def support_consecutive(aw: torch.Tensor, s_beta: torch.Tensor, S: int) -> torch.Tensor:
    """S consecutive anchors along arc length. (M, S) int64."""
    s = arc_length(aw)
    L = int(s_beta.numel())
    S = min(int(S), L)
    pos = torch.searchsorted(s_beta, s)
    start = (pos - S // 2).clamp(0, max(L - S, 0))
    return start[:, None] + torch.arange(S, device=aw.device, dtype=torch.int64)


def runify_support(support: torch.Tensor, run_len: int) -> torch.Tensor:
    """Copy the middle sample's support onto each contiguous run."""
    if run_len <= 1:
        return support.clone()
    M = support.shape[0]
    out = support.clone()
    for start in range(0, M, int(run_len)):
        end = min(start + int(run_len), M)
        out[start:end] = support[(start + end) // 2]
    return out


def support_stats(support: torch.Tensor, L: int) -> dict:
    taps = int(support.numel())
    counts = torch.bincount(support.reshape(-1), minlength=int(L)).double()
    return dict(
        taps=taps,
        nnz_mean=float(counts.mean()),
        nnz_max=float(counts.max()),
        dead_factors=int((counts == 0).sum()),
        n_patterns=int(torch.unique(support.sort(dim=1).values, dim=0).shape[0]),
    )


def _ls_on_support(Y: torch.Tensor, atoms: torch.Tensor, lamda: float) -> torch.Tensor:
    """Y (b, N), atoms (b, s, N) -> coeffs (b, s) for Y ≈ h @ atoms."""
    G = torch.einsum('bsn,btn->bst', atoms, atoms.conj())
    rhs = torch.einsum('bn,bsn->bs', Y, atoms.conj())
    s = atoms.shape[1]
    eye = torch.eye(s, dtype=G.dtype, device=G.device)
    ridge = lamda * G.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1).clamp_min(1e-30)
    G = G + ridge[:, None, None] * eye
    return torch.linalg.solve(G, rhs.unsqueeze(-1)).squeeze(-1)


def support_omp(P: torch.Tensor,
                Bmat: torch.Tensor,
                S: int,
                batch: int = 64) -> torch.Tensor:
    """
    Batched OMP on the *current* dictionary Bmat (L x N).

    Model: each row a_m ≈ h @ B[Omega]. Dictionary atoms are rows of Bmat.
    """
    M, N = P.shape
    L = Bmat.shape[0]
    S = min(int(S), L)
    out = torch.empty((M, S), dtype=torch.int64, device=P.device)
    eye_cache = {}
    for m0 in range(0, M, batch):
        m1 = min(m0 + batch, M)
        Y = P[m0:m1]
        b = Y.shape[0]
        residual = Y.clone()
        used = torch.zeros(b, L, dtype=torch.bool, device=P.device)
        chosen = torch.empty(b, S, dtype=torch.int64, device=P.device)
        n_idx = torch.arange(b, device=P.device)
        for s in range(S):
            corr = residual @ Bmat.mH
            corr = corr.masked_fill(used, 0)
            idx = corr.abs().argmax(dim=1)
            chosen[:, s] = idx
            used[n_idx, idx] = True
            atoms = Bmat[chosen[:, :s + 1]]
            h = _ls_on_support(Y, atoms, lamda=1e-12)
            residual = Y - torch.einsum('bs,bsn->bn', h, atoms)
        out[m0:m1] = chosen
        del residual, used, corr, atoms, h
    return out


def support_omp_simultaneous(Yc: torch.Tensor,
                             Bmat: torch.Tensor,
                             S: int,
                             batch: int = 32) -> torch.Tensor:
    """
    Simultaneous OMP: Yc is (C, M, N), one shared k-sparse support per time m.
    Correlation is the Frobenius inner product summed over coils.
    """
    C, M, N = Yc.shape
    L = Bmat.shape[0]
    S = min(int(S), L)
    out = torch.empty((M, S), dtype=torch.int64, device=Yc.device)
    Y = Yc.permute(1, 0, 2).contiguous()  # M C N
    for m0 in range(0, M, batch):
        m1 = min(m0 + batch, M)
        block = Y[m0:m1]  # b C N
        b = block.shape[0]
        residual = block.clone()
        used = torch.zeros(b, L, dtype=torch.bool, device=Yc.device)
        chosen = torch.empty(b, S, dtype=torch.int64, device=Yc.device)
        n_idx = torch.arange(b, device=Yc.device)
        for s in range(S):
            # corr[b,l] = || residual[b] @ B[l].conj() || over coils
            corr = torch.einsum('bcn,ln->blc', residual, Bmat.conj()).norm(dim=-1)
            corr = corr.masked_fill(used, 0)
            idx = corr.argmax(dim=1)
            chosen[:, s] = idx
            used[n_idx, idx] = True
            atoms = Bmat[chosen[:, :s + 1]]  # b s N
            # per-coil LS, then reconstruct
            recon = []
            for c in range(C):
                h = _ls_on_support(block[:, c], atoms, lamda=1e-12)
                recon.append(torch.einsum('bs,bsn->bn', h, atoms))
            residual = block - torch.stack(recon, dim=1)
        out[m0:m1] = chosen
    return out


def h_update(P: torch.Tensor,
             Bmat: torch.Tensor,
             support: torch.Tensor,
             lamda: float = 0.0) -> tuple[torch.Tensor, int]:
    """Exact LS for H with frozen supports. One solve per unique pattern."""
    M, L = P.shape[0], Bmat.shape[0]
    S = support.shape[1]
    H = P.new_zeros((M, L))
    key, inv = torch.unique(support.sort(dim=1).values, dim=0, return_inverse=True)
    n_pat = int(key.shape[0])
    for p in range(n_pat):
        pat = key[p]
        rows = torch.nonzero(inv == p, as_tuple=False).squeeze(-1)
        atoms = Bmat[pat]  # S x N
        # P[rows] ≈ coef @ atoms  ->  atoms.T @ coef.T = P[rows].T
        coef = torch.linalg.lstsq(atoms.T, P[rows].T).solution.T
        H[rows[:, None], pat[None, :]] = coef
    return H, n_pat


def h_update_dense(P: torch.Tensor,
                   Bmat: torch.Tensor,
                   support: torch.Tensor,
                   lamda: float = 0.0) -> torch.Tensor:
    """Dense masked LS, one solve per row. Reference for test 5."""
    M, L = P.shape[0], Bmat.shape[0]
    H = P.new_zeros((M, L))
    for m in range(M):
        om = support[m]
        H[m, om] = torch.linalg.lstsq(Bmat[om].T, P[m]).solution
    return H


def b_update(P: torch.Tensor, H: torch.Tensor, lamda: float = 0.0) -> torch.Tensor:
    """B = argmin ||H B - P||_F. lstsq, not normal equations (dead atoms)."""
    return torch.linalg.lstsq(H, P).solution


def init_factors(phis: torch.Tensor,
                 alphas: torch.Tensor,
                 betas: torch.Tensor,
                 support: torch.Tensor,
                 Sigma_sqrt: Optional[torch.Tensor] = None,
                 P: Optional[torch.Tensor] = None,
                 lamda: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic atoms b_l(r) = exp(-2j pi phi(r).beta_l), then LS H on the support."""
    phis_flt = phis.reshape(phis.shape[0], -1).double()
    Bmat = torch.exp(-2j * math.pi * (betas.double().T @ phis_flt))
    if P is None:
        P = torch.exp(-2j * math.pi * (alphas.double().T @ phis_flt))
    H, _ = h_update(P, Bmat, support, lamda=lamda)
    return H, Bmat


def fit_fixed_support(P: torch.Tensor,
                      support: torch.Tensor,
                      H0: torch.Tensor,
                      n_iter: int = 12,
                      lamda: float = 0.0) -> dict:
    """ALS with frozen supports. Both half-steps are exact LS; must be monotone."""
    H = H0.clone()
    errs = []
    n_patterns = 0
    Bmat = None
    for _ in range(n_iter):
        Bmat = b_update(P, H, lamda=lamda)
        H, n_patterns = h_update(P, Bmat, support, lamda=lamda)
        errs.append(rel_error(P, H, Bmat))
    rise = 0.0
    for a, b in zip(errs, errs[1:]):
        rise = max(rise, b - a)
    return dict(H=H, Bmat=Bmat, B=Bmat, err=errs[-1], errs=errs,
                monotone=rise, n_patterns=n_patterns)


def ksvd_atom_update(P: torch.Tensor,
                     H: torch.Tensor,
                     Bmat: torch.Tensor,
                     support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One K-SVD pass: SVD of the residual on each atom's temporal support."""
    R = P - H @ Bmat
    L = Bmat.shape[0]
    for j in range(L):
        I = torch.nonzero(H[:, j].abs() > 0, as_tuple=False).squeeze(-1)
        if I.numel() < 2:
            continue
        R_I = R[I] + H[I, j:j + 1] @ Bmat[j:j + 1]
        try:
            U, s, Vh = torch.linalg.svd(R_I, full_matrices=False)
        except RuntimeError:
            continue
        Bmat[j] = Vh[0]
        H[I, j] = U[:, 0] * s[0]
        R[I] = R_I - H[I, j:j + 1] @ Bmat[j:j + 1]
    return H, Bmat


def _matching_pursuit(D: torch.Tensor, Y: torch.Tensor, k_sparse: int) -> torch.Tensor:
    """Batched MP: each column of the returned codes has ≤ k_sparse nonzeros."""
    residual = Y.clone()
    n_atoms = D.shape[1]
    n_sig = Y.shape[1]
    X = Y.new_zeros(n_atoms, n_sig)
    used = torch.zeros(n_sig, n_atoms, dtype=torch.bool, device=Y.device)
    n_idx = torch.arange(n_sig, device=Y.device)
    k_sparse = min(int(k_sparse), n_atoms)
    for _ in range(k_sparse):
        corr = D.mH @ residual
        corr = corr.masked_fill(used.T, 0)
        idx = corr.abs().argmax(dim=0)
        c = corr[idx, n_idx]
        X[idx, n_idx] = X[idx, n_idx] + c
        residual = residual - D[:, idx] * c
        used[n_idx, idx] = True
    return X


def _codes_ls(D: torch.Tensor, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Exact LS for column-sparse ``X``: ``Y ≈ D @ X``. Groups identical supports."""
    used = X.abs() > 0
    nnz = used.sum(dim=0)
    S = int(nnz.min().clamp(min=1))
    if int(nnz.max()) != S:
        # Mixed sparsity (dead-atom replacement); fall back per unique row-count.
        S = int(nnz.max().clamp(min=1))
    coords = used.T.nonzero(as_tuple=False)
    if coords.shape[0] != Y.shape[1] * S:
        X_new = X.new_zeros(X.shape)
        for n in range(Y.shape[1]):
            pat = used[:, n].nonzero(as_tuple=False).squeeze(-1)
            if pat.numel() == 0:
                continue
            X_new[pat, n] = torch.linalg.lstsq(D[:, pat], Y[:, n]).solution
        return X_new
    support = coords[:, 1].reshape(Y.shape[1], S)
    key, inv = torch.unique(support.sort(dim=1).values, dim=0, return_inverse=True)
    X_new = X.new_zeros(X.shape)
    for p in range(key.shape[0]):
        pat = key[p]
        cols = (inv == p).nonzero(as_tuple=False).squeeze(-1)
        coef = torch.linalg.lstsq(D[:, pat], Y[:, cols]).solution
        X_new[pat[:, None], cols[None, :]] = coef
    return X_new


def _replace_dead_atoms(D: torch.Tensor, Y: torch.Tensor, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Aharon–Elad: unused atoms become residuals of the worst-fit columns.

    Without this, SVD-init K-SVD never updates atoms that matching pursuit
    did not pick on the first pass, so NRMSE is independent of extra ``L``.
    """
    usage = (X.abs() > 0).sum(dim=1)
    dead = (usage < 2).nonzero(as_tuple=False).squeeze(-1)
    if dead.numel() == 0:
        return D, X
    residual = Y - D @ X
    worst = torch.argsort(residual.norm(dim=0), descending=True)
    n_take = min(int(dead.numel()), int(Y.shape[1]))
    for i in range(n_take):
        j = int(dead[i])
        n = int(worst[i])
        r = residual[:, n]
        nrm = r.norm().clamp_min(1e-8)
        D[:, j] = r / nrm
        X[j] = 0
        X[:, n] = 0
        X[j, n] = nrm
    D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return D, X


@torch.no_grad()
def ksvd(Y: torch.Tensor,
         n_atoms: int,
         k_sparse: int,
         n_iter: int = 8,
         D0: Optional[torch.Tensor] = None,
         X0: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unconstrained K-SVD (Rubinstein et al.).

    ``Y`` is ``(m, n)``. Returns ``U, S, V`` in the same layout as
    ``torch.svd_lowrank``:

        Y ≈ U @ diag(S) @ V.H

    with each *column of* ``V.H`` at most ``k_sparse``-sparse.
    """
    if D0 is None:
        q = min(int(n_atoms), *Y.shape)
        U0, S0, V0 = torch.svd_lowrank(Y, q=q)
        D = U0 * S0
        X = V0.mH
    else:
        D, X = D0.clone(), X0.clone()
    D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-8)
    n_atoms = D.shape[1]
    k_sparse = min(int(k_sparse), n_atoms)

    for _ in range(n_iter):
        X = _matching_pursuit(D, Y, k_sparse)
        X = _codes_ls(D, Y, X)
        D, X = _replace_dead_atoms(D, Y, X)
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
        D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-8)

    S = X.norm(dim=1)
    order = torch.argsort(S, descending=True)
    U = D[:, order]
    S = S[order]
    V = X.mH[:, order] / S.clamp_min(1e-8)
    return U, S, V


@torch.no_grad()
def ksvd_temporal(P: torch.Tensor,
                  n_atoms: int,
                  k_sparse: int,
                  n_iter: int = 8) -> dict:
    """K-SVD of a phase matrix ``P`` (M, N) with k-sparse *rows of U*.

    ``P ≈ U diag(S) V.H``; each time sample uses at most ``k_sparse`` spatial
    atoms. Warm-starts from the rank-``n_atoms`` SVD of ``P``.
    """
    q = min(int(n_atoms), *P.shape)
    U0, S0, V0 = torch.svd_lowrank(P, q=q)
    Ut, S, Vt = ksvd(P.mH, q, k_sparse, n_iter=n_iter, D0=V0 * S0, X0=U0.mH)
    U, V = Vt, Ut
    recon = (U * S) @ V.mH
    err = float((P - recon).norm() / P.norm().clamp_min(1e-30))
    n_live = int((U.abs().sum(dim=0) > 0).sum().item())
    del recon
    return dict(U=U, S=S, V=V, err=err, n_live=n_live)


def ksvd_fit(P: torch.Tensor,
             B0: torch.Tensor,
             k: int,
             n_outer: int = 20,
             lamda: float = 1e-12,
             per_atom: bool = False,
             run_len: int = 1,
             omp_batch: int = 64,
             coding: str = 'omp') -> dict:
    """
    Alternate OMP on the current B, exact LS H, then dictionary update.

    Tracks the best iterate: support re-selection is allowed to break monotonicity.
    """
    Bmat = B0.clone()
    nrm = Bmat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Bmat = Bmat / nrm
    best = None
    history = []
    H = None
    support = None
    for it in range(n_outer):
        if coding == 'topk':
            support = (P @ Bmat.mH).abs().topk(k, dim=1).indices
        else:
            support = support_omp(P, Bmat, k, batch=omp_batch)
        if run_len > 1:
            support = runify_support(support, run_len)
        H, n_pat = h_update(P, Bmat, support, lamda=lamda)
        err_h = rel_error(P, H, Bmat)
        Bmat = b_update(P, H, lamda=lamda)
        if per_atom:
            H, Bmat = ksvd_atom_update(P, H, Bmat, support)
            nrm = Bmat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            Bmat = Bmat / nrm
            H = H * nrm.T
        err = rel_error(P, H, Bmat)
        rec = dict(it=it, err_after_h=err_h, err=err, n_patterns=n_pat)
        history.append(rec)
        if best is None or err < best['err']:
            best = dict(H=H.clone(), B=Bmat.clone(), Bmat=Bmat.clone(),
                        support=support.clone(), err=err, n_patterns=n_pat, it=it)
    best['history'] = history
    best['errs'] = [h['err'] for h in history]
    return best


def theoretical_sigma(k: int, eps: float = 1e-2) -> float:
    """Solve eps = exp(-pi k sqrt(1-1/sigma)) for sigma (ksvd_feas.md Sec. 0)."""
    t = -math.log(eps) / (math.pi * k)
    if t >= 1.0:
        return float('inf')
    return 1.0 / max(1.0 - t ** 2, 1e-12)


def theoretical_nu(k: int, eps: float = 1e-2) -> float:
    """SBP-dominated nu ≈ required sigma."""
    return theoretical_sigma(k, eps)
