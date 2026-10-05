"""
Dyadic spatial tree and Gram-based rank machinery (feas.md Sec. 10, 11, 25).

Why a private partitioner instead of ``qblock_feas.blocks.make_blocks``
----------------------------------------------------------------------
``make_blocks`` tiles the *mask bounding box* with ``stride = ceil(bbox / Q)``. Cells at
successive ``Q`` are therefore **not nested**: with ``bbox = 100``, level 3 has stride 13
and level 2 has stride 25, so the child cell ``[13, 26)`` straddles parents ``[0, 25)``
and ``[25, 50)``. A tree needs exact containment, so the tree partition is built here on
the padded grid, where ``n_i`` is divisible by ``2**L`` and every level is an exact
subdivision. ``make_blocks`` is still the right tool for the non-nested rank atlas and
for the Stage 3/4 block geometry, and is used there.

Why Grams instead of SVDs
-------------------------
Every rank question reduces to the ``C x C`` Gram ``G_q = S_q W_q S_q^H``: singular
values are ``sqrt(eig(G_q))`` and the eigenvectors *are* the basis ``U_q``. This costs
``C^2 N`` per scale instead of an SVD per block, and -- the reason it matters --
**Grams are additive up the tree**, ``G_p = sum_c G_c``. One pass over the finest leaves
therefore yields the exact rank of every node at every level for free, which is the whole
Stage-2 curve. Verified against a direct SVD in ``test_tree.py``.

Block ordering is row-major over the per-axis block index, so level ``l`` is naturally
shaped ``(2**l,) * d`` and a parent is the sum of its ``2**d`` even/odd siblings.
"""
import numpy as np
import torch

from typing import Optional, Sequence

# feas.md Sec. 10.3 asks for both conventions; they are related by energy = 1 - tol^2.
FROB_TOLS = (1e-2, 1e-3, 1e-4)
ENERGY_FRACS = (0.99, 0.999, 0.9999)


# ---------------------------------------------------------------------------
# Dyadic partition
# ---------------------------------------------------------------------------
def check_dyadic(im_size: Sequence[int], level: int):
    step = 1 << int(level)
    bad = [n for n in im_size if n % step]
    if bad:
        raise ValueError(f'grid {tuple(im_size)} is not divisible by 2**{level}; '
                         f'use common.pad_grid to build the analysis grid')


def block_shape(im_size: Sequence[int], level: int) -> tuple:
    """Voxels per block at ``level``."""
    check_dyadic(im_size, level)
    return tuple(n >> level for n in im_size)


def to_blocks(x: torch.Tensor, level: int) -> torch.Tensor:
    """
    ``(C, *im_size) -> (Q, C, N_q)`` with ``Q = 2**(d*level)``, row-major block order.

    The returned tensor is contiguous, so callers pay one copy of ``x``.
    """
    C = x.shape[0]
    im_size = tuple(x.shape[1:])
    d = len(im_size)
    b = block_shape(im_size, level)
    s = 1 << level
    view = x.reshape((C,) + tuple(v for i in range(d) for v in (s, b[i])))
    # (C, s,b0, s,b1, ...) -> (s,s,..., C, b0,b1,...)
    perm = [1 + 2 * i for i in range(d)] + [0] + [2 + 2 * i for i in range(d)]
    return view.permute(perm).reshape(s ** d, C, -1).contiguous()


def from_blocks(xb: torch.Tensor, im_size: Sequence[int], level: int) -> torch.Tensor:
    """Inverse of :func:`to_blocks`."""
    im_size = tuple(im_size)
    d = len(im_size)
    b = block_shape(im_size, level)
    s = 1 << level
    C = xb.shape[1]
    view = xb.reshape((s,) * d + (C,) + b)
    # (s,s,..., C, b0,b1,...) -> (C, s,b0, s,b1, ...)
    perm = [d] + [v for i in range(d) for v in (i, d + 1 + i)]
    return view.permute(perm).reshape((C,) + im_size)


def gen_grd64(im_size: Sequence[int], device=None) -> torch.Tensor:
    """
    ``(*im_size, d)`` spatial grid in **float64**, matching ``hofft.utils.gen_grd``.

    ``gen_grd`` returns ``real_dtype`` (float32), whose ~1e-8 roundoff is fine for image
    work but not for the block-centre phase, which feas.md Sec. 12 requires to be exact.
    Same convention: voxel ``j`` on axis ``i`` sits at ``(j - N_i//2) / N_i``.
    """
    lins = [torch.arange(-(n // 2), n // 2 + n % 2, dtype=torch.float64,
                         device=device) / n for n in im_size]
    return torch.stack(torch.meshgrid(*lins, indexing='ij'), dim=-1)


def block_centers(im_size: Sequence[int], level: int) -> torch.Tensor:
    """
    ``(Q, d)`` block-window centres in ``gen_grd`` coordinates, float64.

    Matches ``qblock_feas/blocks.py``: voxel ``j`` on axis ``i`` sits at
    ``(j - N_i//2)/N_i``, so a window ``[lo, lo+b)`` is centred at ``(lo + b//2 - N//2)/N``.
    The block-centre phase ``exp(-2j*pi*k.c)`` must be built from exactly this.
    """
    im_size = tuple(im_size)
    d = len(im_size)
    b = block_shape(im_size, level)
    s = 1 << level
    grids = torch.meshgrid(*[torch.arange(s) for _ in range(d)], indexing='ij')
    lo = torch.stack([g.reshape(-1) * b[i] for i, g in enumerate(grids)], dim=-1)
    N = torch.tensor(im_size)
    bt = torch.tensor(b)
    return (lo + bt // 2 - N // 2).double() / N


# ---------------------------------------------------------------------------
# Grams
# ---------------------------------------------------------------------------
def block_grams(S: torch.Tensor,
                level: int,
                weights: Optional[torch.Tensor] = None,
                chunk: int = 4096) -> torch.Tensor:
    """
    Per-block Gram matrices ``G_q = S_q W_q S_q^H``.

    Args
    ----
    S : torch.Tensor
        ``(C, *im_size)`` field, on the padded analysis grid
    level : int
        Tree level; ``Q = 2**(d*level)`` blocks
    weights : Optional[torch.Tensor]
        ``(*im_size)`` non-negative voxel weights (object mask, or ``|img_ref|``)
    chunk : int
        Blocks per batch, to bound peak memory

    Returns
    -------
    torch.Tensor
        ``(Q, C, C)`` complex Hermitian PSD, in ``complex128``
    """
    if weights is not None:
        S = S * weights.sqrt().to(S.dtype)
    Xb = to_blocks(S, level)                     # (Q, C, N_q)
    Q = Xb.shape[0]
    out = torch.empty(Q, Xb.shape[1], Xb.shape[1],
                      dtype=torch.complex128, device=S.device)
    for i in range(0, Q, chunk):
        x = Xb[i:i + chunk].to(torch.complex128)
        out[i:i + chunk] = x @ x.conj().transpose(-1, -2)
    return out


def merge_grams(G: torch.Tensor, d: int) -> torch.Tensor:
    """
    One level up: ``(2**l,)*d + (C,C) -> (2**(l-1),)*d + (C,C)``.

    Exact, because ``G_p = sum_{c in children(p)} G_c`` for disjoint children. Summing
    even and odd slices along each of the first ``d`` axes in turn performs the ``2**d``
    sibling sum without materializing the groups.
    """
    out = G
    for ax in range(d):
        pre = (slice(None),) * ax
        out = out[pre + (slice(0, None, 2),)] + out[pre + (slice(1, None, 2),)]
    return out


def gram_pyramid(S: torch.Tensor,
                 level: int,
                 weights: Optional[torch.Tensor] = None) -> list:
    """
    Grams at every level from ``level`` down to 0, computed once at the leaves.

    Returns a list indexed by level: ``out[l]`` has shape ``(2**(d*l), C, C)``.
    """
    d = S.ndim - 1
    C = S.shape[0]
    leaf = block_grams(S, level, weights)
    out = [None] * (level + 1)
    out[level] = leaf
    g = leaf.reshape((1 << level,) * d + (C, C))
    for l in range(level - 1, -1, -1):
        g = merge_grams(g, d)
        out[l] = g.reshape(-1, C, C)
    return out


# ---------------------------------------------------------------------------
# Ranks and bases
# ---------------------------------------------------------------------------
def gram_eigs(G: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of a batch of Hermitian PSD Grams, descending, clamped at 0."""
    return torch.linalg.eigvalsh(G).flip(-1).clamp(min=0)


def occupied(G: torch.Tensor, rtol: float = 1e-12) -> torch.Tensor:
    """Boolean mask of blocks carrying non-negligible energy."""
    tr = G.diagonal(dim1=-2, dim2=-1).real.sum(-1)
    return tr > rtol * tr.max().clamp(min=1e-300)


def rank_from_eigs(eigs: torch.Tensor,
                   tol: Optional[float] = None,
                   energy: Optional[float] = None) -> torch.Tensor:
    """
    Rank per block at a relative-Frobenius residual ``tol`` or a retained ``energy``.

    ``tol`` is the residual on the *matrix*: ``||S - S_k||_F / ||S||_F <= tol``, i.e.
    ``sum_{j>k} lambda_j / sum_j lambda_j <= tol**2``. An all-zero block gets rank 0.
    """
    if (tol is None) == (energy is None):
        raise ValueError('pass exactly one of tol, energy')
    cs = eigs.cumsum(-1)
    tot = cs[..., -1:]
    dead = tot.squeeze(-1) <= 0
    frac = torch.where(tot > 0, (tot - cs) / tot.clamp(min=1e-300),
                       torch.zeros_like(cs))
    thresh = tol ** 2 if tol is not None else (1.0 - energy)
    r = (frac > thresh).sum(-1) + 1
    return torch.where(dead, torch.zeros_like(r), r)


def bases_from_grams(G: torch.Tensor, ranks: torch.Tensor) -> list:
    """
    Per-block orthonormal bases ``U_q`` (``(C, k_q)``) and singular values.

    Returns ``[(U_q, s_q), ...]``; ``s_q`` are the retained singular values
    ``sqrt(lambda)``, which the energy-weighted parent merge needs.
    """
    vals, vecs = torch.linalg.eigh(G)
    vals = vals.flip(-1).clamp(min=0)
    vecs = vecs.flip(-1)
    out = []
    for q in range(G.shape[0]):
        k = int(ranks[q])
        out.append((vecs[q, :, :k], vals[q, :k].sqrt()))
    return out


def parent_basis(children: list,
                 tol: float,
                 weighted: bool = True) -> tuple:
    """
    Parent basis from the union of child subspaces (feas.md Sec. 5.2, Sec. 25).

    Args
    ----
    children : list
        ``[(U_c, s_c), ...]`` from :func:`bases_from_grams`
    tol : float
        Relative-Frobenius truncation tolerance
    weighted : bool
        If True, concatenate ``U_c diag(s_c)`` so directions are weighted by the energy
        they actually carry. If False, concatenate the bare ``U_c``, which is the
        feas.md Sec. 25 pseudocode and treats every child direction as equally important.

    Returns
    -------
    (U_p, s_p, transfers)
        ``transfers[i] = U_p^H U_c_i``, the child -> parent mixing matrix.
    """
    cols = [(U * s if weighted else U) for U, s in children if U.shape[1] > 0]
    if not cols:
        C = children[0][0].shape[0]
        z = torch.zeros(C, 0, dtype=children[0][0].dtype, device=children[0][0].device)
        return z, z.new_zeros(0).real, [z.conj().T @ U for U, _ in children]
    Ucat = torch.cat(cols, dim=1)
    U, s, _ = torch.linalg.svd(Ucat, full_matrices=False)
    k = int(rank_from_eigs(s.square()[None], tol=tol)[0])
    Up = U[:, :k]
    return Up, s[:k], [Up.conj().T @ Uc for Uc, _ in children]


def principal_angles(Ua: torch.Tensor, Ub: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians, ascending) between two orthonormal column spans."""
    if Ua.shape[1] == 0 or Ub.shape[1] == 0:
        return torch.zeros(0, device=Ua.device)
    s = torch.linalg.svdvals(Ua.conj().T @ Ub).clamp(-1, 1)
    return torch.arccos(s)


def residual_grams(G: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    Method-D residual Grams (feas.md Sec. 7): ``(I - P) G (I - P)`` with ``P`` the
    projector onto what the ancestors already explain.
    """
    C = G.shape[-1]
    I = torch.eye(C, dtype=G.dtype, device=G.device)
    Q = I - P
    return Q @ G @ Q


# ---------------------------------------------------------------------------
# Field truncation -- the Stage 1b operator-error probe
# ---------------------------------------------------------------------------
def truncate_field(S: torch.Tensor,
                   level: int,
                   ranks: torch.Tensor,
                   G: torch.Tensor) -> torch.Tensor:
    """
    Block-truncated field ``s_hat = sum_q 1_{R_q} U_q U_q^H s`` on the same grid.

    This is what makes Gate 1 interpretable: pushing ``s_hat`` through the *global*
    NUFFT gives the exact forward-operator error of local rank truncation, with no
    hierarchical operator needed (feas.md Failure 9).
    """
    vals, vecs = torch.linalg.eigh(G)
    vecs = vecs.flip(-1)                                  # (Q, C, C) descending
    C = S.shape[0]
    keep = (torch.arange(C, device=S.device)[None, :] < ranks[:, None].to(S.device))
    V = vecs.to(S.dtype) * keep[:, None, :].to(S.dtype)   # zero the dropped columns
    P = V @ V.conj().transpose(-1, -2)                    # (Q, C, C) projector
    Xb = to_blocks(S, level)                              # (Q, C, N_q)
    return from_blocks(P @ Xb, S.shape[1:], level)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def rank_stats(ranks: torch.Tensor, occ: torch.Tensor) -> dict:
    """Median / mean / max / sum of the per-block rank over occupied blocks only."""
    r = ranks[occ].double()
    if r.numel() == 0:
        return dict(median=0.0, mean=0.0, max=0.0, sum=0.0, n_occ=0)
    return dict(median=float(r.median()), mean=float(r.mean()), max=float(r.max()),
                sum=float(r.sum()), n_occ=int(r.numel()))


def rank_map(ranks: torch.Tensor, level: int, d: int) -> np.ndarray:
    """Per-block ranks reshaped to ``(2**level,)*d`` for the spatial rank map figure."""
    s = 1 << level
    return ranks.reshape((s,) * d).cpu().numpy()
