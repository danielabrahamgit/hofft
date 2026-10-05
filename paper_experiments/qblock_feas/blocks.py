"""
Block partitioning, per-block affine absorption, and per-block rank accounting for the
block-partitioned SVD feasibility study (``math_docs/Qblock_svd_feas.md``).

Geometry conventions
--------------------
Voxel index ``j`` along axis ``i`` sits at ``gen_grd`` coordinate ``(j - N_i//2) / N_i``.
A block's FFT window spans indices ``[win_lo, win_lo + V)``, so with

    D_i      = V_i / N_i                          # block extent, fraction of FOV
    c_i      = (win_lo_i + V_i//2 - N_i//2) / N_i # window center in global coords
    r'_i     = (j - win_lo_i - V_i//2) / V_i      # local coords, a gen_grd(V) grid

we get ``r = c + D * r'`` exactly, hence ``k . r = k . c + (D * k) . r'``. In local
coordinates a block is therefore an ordinary ``V``-voxel image sampled at
``kappa = D * k``, which lands in ``[-V/2, V/2)`` whenever ``k`` is in ``[-N/2, N/2)``.
That is Sec. 1.2: the block problem is the global problem at a smaller size, so ``beta``,
``os`` and the ``W``-tap stencil all carry over unchanged.

Two block extents are tracked separately and they are not the same thing:

* the **mask cell**, a ``stride``-sized tile. Cells tile the mask bounding box exactly and
  disjointly, which is what makes ``sum_q m_q = 1`` hold.
* the **FFT window**, a uniform ``V``-sized box with ``V >= stride`` rounded up to an
  FFT-friendly size. Windows may overlap. This is what sets FFT cost (Sec. 7).
"""
import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Optional, Sequence

from einops import einsum

from hofft.kb import kb_apod_1d, sample_kb_kernel
from hofft.utils import gen_grd


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------
def next_smooth(n: int, radices: Sequence[int] = (2, 3, 5)) -> int:
    """Smallest integer ``>= n`` whose prime factors all lie in ``radices``."""
    n = int(max(n, 1))
    while True:
        m = n
        for p in radices:
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(n, 1))))


def mask_bbox(mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Tight inclusive bounding box of a boolean/float support mask."""
    d = mask.ndim
    m = mask > 0
    lo, hi = np.zeros(d, dtype=int), np.zeros(d, dtype=int)
    for i in range(d):
        other = tuple(j for j in range(d) if j != i)
        occ = torch.argwhere(m.any(dim=other) if other else m)[:, 0]
        lo[i], hi[i] = int(occ.min()), int(occ.max())
    return lo, hi


@dataclass
class BlockSet:
    """A support-aware, FFT-friendly partition of the imaging region."""
    style: str
    im_size: tuple                  # grid the masks live on
    Q_per_axis: tuple               # requested blocks per axis
    V: tuple                        # uniform FFT window size (voxels)
    stride: tuple                   # mask cell size (voxels)
    labels: torch.Tensor            # (*im_size) int64, kept-block id or -1
    idx: list                       # per kept block, flat indices into prod(im_size)
    win_lo: torch.Tensor            # (Q_kept, d) int64 FFT window lower corner
    centers: torch.Tensor           # (Q_kept, d) float64 window center, gen_grd coords
    D: torch.Tensor                 # (d,) float64 block extent as a fraction of FOV
    bbox: tuple = field(default=())  # (lo, hi) of the support mask

    @property
    def d(self) -> int:
        return len(self.im_size)

    @property
    def Q_tot(self) -> int:
        return int(np.prod(self.Q_per_axis))

    @property
    def Q_kept(self) -> int:
        return len(self.idx)

    @property
    def occupancy(self) -> torch.Tensor:
        """Masked voxels per kept block."""
        return torch.tensor([len(i) for i in self.idx], dtype=torch.float64)

    @property
    def window_volume(self) -> int:
        """Voxels in one FFT window. FFT cost scales with this, not with occupancy."""
        return int(np.prod(self.V))

    def local_coords(self, q: int) -> torch.Tensor:
        """``r'`` of block ``q``'s masked voxels, shape (N_q, d), in [-1/2, 1/2)."""
        sub = torch.stack(torch.unravel_index(self.idx[q], self.im_size), dim=-1)
        V = torch.tensor(self.V, device=sub.device)
        return (sub - self.win_lo[q].to(sub.device) - V // 2).double() / V

    def summary(self) -> dict:
        occ = self.occupancy
        return dict(
            style=self.style, Q_per_axis=tuple(self.Q_per_axis), Q_tot=self.Q_tot,
            Q_kept=self.Q_kept, kept_frac=self.Q_kept / self.Q_tot, V=tuple(self.V),
            stride=tuple(self.stride), window_volume=self.window_volume,
            occ_min=float(occ.min()), occ_max=float(occ.max()),
            occ_mean=float(occ.mean()),
            occ_frac_mean=float((occ / self.window_volume).mean()),
        )


def make_blocks(mask: torch.Tensor,
                Q_per_axis: Sequence[int],
                fft_friendly: bool = True,
                pow2: bool = False,
                style: Optional[str] = None) -> BlockSet:
    """
    Partition the support of ``mask`` into ``prod(Q_per_axis)`` uniform blocks.

    Mask cells tile the support bounding box by ``stride = ceil(bbox / Q)`` and are
    disjoint; FFT windows are the uniform, FFT-friendly ``V >= stride`` boxes centered on
    each cell. Blocks with no masked voxels are dropped.

    Args
    ----
    mask : torch.Tensor
        Support mask with shape ``im_size``
    Q_per_axis : Sequence[int]
        Requested block count per axis
    fft_friendly : bool
        Round ``V`` up to a 5-smooth size
    pow2 : bool
        Round ``V`` up to a power of two instead
    style : Optional[str]
        Label carried through to the results table

    Returns
    -------
    BlockSet
    """
    im_size = tuple(mask.shape)
    d = len(im_size)
    assert len(Q_per_axis) == d
    lo, hi = mask_bbox(mask)
    bbox = hi - lo + 1

    stride = np.array([int(np.ceil(bbox[i] / Q_per_axis[i])) for i in range(d)])
    V = stride.copy()
    if pow2:
        V = np.array([next_pow2(v) for v in V])
    elif fft_friendly:
        V = np.array([next_smooth(v) for v in V])
    V = np.minimum(V, np.array(im_size))          # a window cannot exceed the grid

    # Assign every voxel to a cell, then to a kept-block id
    sub = torch.stack(torch.meshgrid(
        *[torch.arange(n, device=mask.device) for n in im_size], indexing='ij'), dim=-1)
    lo_t = torch.tensor(lo, device=mask.device)
    stride_t = torch.tensor(stride, device=mask.device)
    Q_t = torch.tensor(np.asarray(Q_per_axis), device=mask.device)
    cell = ((sub - lo_t) // stride_t).clamp(min=torch.zeros_like(Q_t), max=Q_t - 1)
    cell_id = torch.zeros(im_size, dtype=torch.long, device=mask.device)
    for i in range(d):
        cell_id = cell_id * int(Q_per_axis[i]) + cell[..., i]
    cell_id = torch.where(mask > 0, cell_id, torch.full_like(cell_id, -1))

    labels = torch.full(im_size, -1, dtype=torch.long, device=mask.device)
    idx, win_lo_rows = [], []
    for c in range(int(np.prod(Q_per_axis))):
        sel = torch.argwhere(cell_id.reshape(-1) == c)[:, 0]
        if sel.numel() == 0:
            continue                                # Sec. 3.1 step 4
        q = len(idx)
        labels.reshape(-1)[sel] = q
        idx.append(sel)

        # Center the window on the cell, then clamp so it stays inside the grid
        cq = np.unravel_index(c, tuple(Q_per_axis))
        cell_start = np.array([lo[i] + cq[i] * stride[i] for i in range(d)])
        pad = (V - stride) // 2
        w = np.clip(cell_start - pad, 0, np.array(im_size) - V)
        cell_end = np.minimum(cell_start + stride, np.array(im_size))
        assert np.all(w <= cell_start) and np.all(w + V >= cell_end), \
            f'window {w}+{V} does not contain cell {cell_start}+{stride}'
        win_lo_rows.append(w)

    win_lo = torch.tensor(np.stack(win_lo_rows), dtype=torch.long)
    N_t = torch.tensor(im_size, dtype=torch.long)
    V_t = torch.tensor(V, dtype=torch.long)
    centers = (win_lo + V_t // 2 - N_t // 2).double() / N_t
    return BlockSet(
        style=style or f'uniform{tuple(Q_per_axis)}', im_size=im_size,
        Q_per_axis=tuple(int(q) for q in Q_per_axis), V=tuple(int(v) for v in V),
        stride=tuple(int(s) for s in stride), labels=labels, idx=idx,
        win_lo=win_lo, centers=centers, D=V_t.double() / N_t, bbox=(lo, hi),
    )


def make_slabs(mask: torch.Tensor, Q: int, axis: int, **kw) -> BlockSet:
    """``Q`` slabs perpendicular to ``axis`` (the ``d_eff = 1``-aware partition)."""
    Q_per_axis = [1] * mask.ndim
    Q_per_axis[axis] = Q
    return make_blocks(mask, Q_per_axis, style=f'slab_ax{axis}(Q={Q})', **kw)


def erode_mask(mask: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """Erode a mask by ``iters`` voxels with a cross structuring element."""
    m = mask > 0
    for _ in range(iters):
        e = m.clone()
        for i in range(mask.ndim):
            e = e & torch.roll(m, 1, dims=i) & torch.roll(m, -1, dims=i)
        m = e
    return m


def jacobian_principal_axes(phis: torch.Tensor,
                            mask: torch.Tensor,
                            sigma_sqrt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Principal directions of the whitened phase Jacobian pooled over the support.

    Rows of the pooled matrix are the whitened gradient vectors ``Sigma^{1/2} J[:, :, n]``,
    so the leading eigenvector is the direction along which the phase gradient carries the
    most (whitened) energy. Slabs perpendicular to it are the ``d_eff = 1`` partition of
    Sec. 1.4, and the eigenvalue spread is a direct read-out of ``d_eff``.

    ``phis`` arrives multiplied by a hard mask, so the mask boundary carries a huge
    spurious gradient in *every* direction. Pooling over the eroded mask keeps only voxels
    whose central-difference stencil is entirely inside the support; without that, a purely
    ``d_eff = 1`` field reads out as isotropic.

    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    mask : torch.Tensor
        Support mask with shape ``im_size``
    sigma_sqrt : torch.Tensor
        Temporal whitening matrix with shape (B, B)

    Returns
    -------
    evals : torch.Tensor
        Normalized eigenvalues of the pooled (d, d) Gram matrix, descending
    evecs : torch.Tensor
        Corresponding directions with shape (d, d), columns matching ``evals``
    """
    im_size = phis.shape[1:]
    d = len(im_size)
    B = phis.shape[0]
    grads = torch.gradient(phis.double(), spacing=[1.0 / n for n in im_size],
                           dim=tuple(range(1, d + 1)))
    jac = torch.stack(grads, dim=1).reshape((B, d, -1))          # B d N
    jac = einsum(sigma_sqrt.double(), jac, 'A B, B d N -> A d N')
    sel = torch.argwhere(erode_mask(mask, 1).reshape(-1))[:, 0]
    A = jac[..., sel].permute(0, 2, 1).reshape((-1, d))          # (B*N_interior, d)
    G = A.T @ A
    evals, evecs = torch.linalg.eigh(G)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    return evals / evals.sum().clamp(min=1e-30), evecs[:, order]


# ---------------------------------------------------------------------------
# Per-block affine absorption (Sec. 1.1)
# ---------------------------------------------------------------------------
def fit_block_affine(phis: torch.Tensor,
                     bs: BlockSet,
                     weights: Optional[torch.Tensor] = None
                     ) -> tuple[torch.Tensor, torch.Tensor, list]:
    """
    Mask-weighted per-block affine fit ``m_q phi_b ~= m_q (C_q[b,:] . r + chat_q[b])``.

    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    bs : BlockSet
        Partition to fit over
    weights : Optional[torch.Tensor]
        Per-voxel weights with shape ``im_size``

    Returns
    -------
    C : torch.Tensor
        Linear coefficients with shape (Q_kept, B, d); shears the trajectory
    chat : torch.Tensor
        Constants with shape (Q_kept, B); becomes the scalar ``zeroth_q``
    phi_res : list
        Per block, the residual ``phi_res_q`` with shape (B, N_q)
    """
    im_size = bs.im_size
    d = bs.d
    B = phis.shape[0]
    rs = gen_grd(im_size).to(phis.device).reshape((-1, d)).double()
    phis_flt = phis.reshape((B, -1)).double()
    w_flt = None if weights is None else weights.reshape(-1).double()

    C = torch.zeros((bs.Q_kept, B, d), dtype=torch.float64, device=phis.device)
    chat = torch.zeros((bs.Q_kept, B), dtype=torch.float64, device=phis.device)
    phi_res = []
    for q, sel in enumerate(bs.idx):
        A = torch.cat([torch.ones((sel.numel(), 1), dtype=torch.float64,
                                  device=phis.device), rs[sel]], dim=1)
        Y = phis_flt[:, sel].T                                    # N_q B
        if w_flt is not None:
            sw = w_flt[sel].clamp(min=0).sqrt()[:, None]
            coef = torch.linalg.lstsq(A * sw, Y * sw).solution
        else:
            coef = torch.linalg.lstsq(A, Y).solution              # (d+1) B
        chat[q] = coef[0]
        C[q] = coef[1:].T
        phi_res.append((Y - A @ coef).T.contiguous())             # B N_q
    return C, chat, phi_res


def block_trajectory(trj: torch.Tensor,
                     alphas: torch.Tensor,
                     C: torch.Tensor) -> torch.Tensor:
    """
    Per-block sheared trajectory ``k_q(t) = k(t) + C_q^T alpha(t)``.

    Args
    ----
    trj : torch.Tensor
        Trajectory with shape (M, d)
    alphas : torch.Tensor
        Temporal coefficients with shape (B, M)
    C : torch.Tensor
        Per-block linear coefficients with shape (Q, B, d)

    Returns
    -------
    trj_q : torch.Tensor
        Sheared trajectories with shape (Q, M, d)
    """
    return trj.double()[None] + einsum(C, alphas.double(), 'Q B d, B M -> Q M d')


# ---------------------------------------------------------------------------
# Per-block SVD and pooled rank allocation (Sec. 3.2)
# ---------------------------------------------------------------------------
def phase_matrix(phis_flt: torch.Tensor, alphas_anchor: torch.Tensor) -> torch.Tensor:
    """``P(m, n) = exp(-2j pi phi(r_n) . alpha(t_m))`` in complex128."""
    return torch.exp(-2j * np.pi * (alphas_anchor.double().T @ phis_flt.double()))


def block_singular_values(phi_res: list, alphas_anchor: torch.Tensor) -> list:
    """
    Singular values of each block's residual phase matrix, in float64.

    Returned on the CPU: the rank-allocation bookkeeping downstream is tiny and index-heavy,
    and keeping it host-side avoids a device round-trip per block.
    """
    return [torch.linalg.svdvals(phase_matrix(pr, alphas_anchor)).cpu()
            for pr in phi_res]


def allocate_ranks(svals: list, S: int, floor: int = 1) -> tuple[np.ndarray, float]:
    """
    Pooled-threshold rank allocation: the Frobenius-optimal split of a budget ``S``.

    Because blocks are disjoint and ``sum_q m_q = 1``, global squared error is
    ``sum_q ||E_q||_F^2``, so taking the ``S`` largest singular values anywhere is optimal.
    A floor of one factor per kept block is imposed because a block with ``L_q = 0`` has no
    forward model at all (Sec. 1.3, ``S >= Q_tot``).

    Args
    ----
    svals : list
        Per-block singular values, descending
    S : int
        Total rank budget
    floor : int
        Minimum rank per block

    Returns
    -------
    L_q : np.ndarray
        Allocated rank per block with shape (Q,)
    sq_err : float
        Resulting ``sum_q sum_{j >= L_q} s_{q,j}^2``
    """
    svals = [s.cpu() for s in svals]
    Q = len(svals)
    L_q = np.full(Q, floor, dtype=int)
    L_q = np.minimum(L_q, [len(s) for s in svals])
    budget = int(S) - int(L_q.sum())
    if budget > 0:
        pool = torch.cat([s[L_q[q]:] for q, s in enumerate(svals)])
        owner = torch.cat([torch.full((len(s) - L_q[q],), q) for q, s in enumerate(svals)])
        take = min(budget, pool.numel())
        top = torch.argsort(pool, descending=True)[:take]
        for q, cnt in zip(*np.unique(owner[top].numpy(), return_counts=True)):
            L_q[q] += int(cnt)
    sq = sum(float(s[L_q[q]:].square().sum()) for q, s in enumerate(svals))
    return L_q, sq


def pooled_error_curve(svals: list,
                       norm_sq: float,
                       floor: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Relative error as a function of total rank ``S`` under pooled-threshold allocation.

    Args
    ----
    svals : list
        Per-block singular values, descending
    norm_sq : float
        ``||P||_F^2`` of the full (all-block) target, i.e. ``M * N_mask``
    floor : int
        Minimum rank per block

    Returns
    -------
    S : np.ndarray
        Total rank, from ``floor * Q`` upward
    err : np.ndarray
        ``sqrt(sum_q ||E_q||_F^2 / norm_sq)``
    """
    svals = [s.cpu() for s in svals]
    Q = len(svals)
    base = np.minimum(np.full(Q, floor, dtype=int), [len(s) for s in svals])
    sq_base = sum(float(s[base[q]:].square().sum()) for q, s in enumerate(svals))
    pool = torch.cat([s[base[q]:].square() for q, s in enumerate(svals)])
    pool = torch.sort(pool, descending=True).values
    rem = sq_base - torch.cumsum(pool, dim=0)
    sq = np.concatenate([[sq_base], rem.clamp(min=0).cpu().numpy()])
    S = int(base.sum()) + np.arange(len(sq))
    return S, np.sqrt(np.maximum(sq, 0) / norm_sq)


def global_error_curve(svals: torch.Tensor, norm_sq: float) -> tuple[np.ndarray, np.ndarray]:
    """Relative error of the optimal rank-``L`` global approximation."""
    tot = float(svals.square().sum())
    sq = tot - torch.cumsum(svals.square(), dim=0)
    sq = np.concatenate([[tot], sq.clamp(min=0).cpu().numpy()])
    return np.arange(len(sq)), np.sqrt(np.maximum(sq, 0) / norm_sq)


def rank_at_error(S: np.ndarray, err: np.ndarray, target: float) -> Optional[float]:
    """Smallest rank on a monotone (S, err) curve reaching ``target``; None if never."""
    ok = np.argwhere(err <= target)[:, 0]
    return float(S[ok[0]]) if ok.size else None


# ---------------------------------------------------------------------------
# Cost model (Sec. 1.3)
# ---------------------------------------------------------------------------
def cost_ratios(r: float,
                bs: BlockSet,
                os: float,
                rho: float,
                im_size_cost: Optional[tuple] = None) -> dict:
    """
    Predicted cost of the block model relative to a global rank-``L`` splitting model.

    ``lam`` and the FFT ratio use the *measured* padded window volume rather than the
    idealized ``N / Q_tot``, so bounding-box padding and dropped blocks are charged
    honestly (Sec. 7). For a perfectly tiling partition the two agree.

    Args
    ----
    r : float
        Rank penalty ``S / L``
    bs : BlockSet
        Partition
    os : float
        Grid oversampling
    rho : float
        Measured gather/FFT wall-clock ratio of the global linop
    im_size_cost : Optional[tuple]
        Grid the cost is evaluated on. The partition is built on a reduced grid, but cost
        is incurred at native resolution; the block extent ``D`` is grid-independent, so
        the window is rescaled to ``D * im_size_cost``. Defaults to the partition's grid.

    Returns
    -------
    dict
        ``lam``, ``Q_eff``, ``fft_ratio``, ``gather_ratio``, ``speedup``
    """
    im_size_cost = tuple(bs.im_size) if im_size_cost is None else tuple(im_size_cost)
    V_cost = [max(round(float(bs.D[i]) * im_size_cost[i]), 1) for i in range(bs.d)]
    n_g = float(np.prod(block_os(os, im_size_cost)[0]))
    n_b = float(np.prod(block_os(os, V_cost)[0]))
    lam = np.log(n_b) / np.log(n_g)
    Q_eff = n_g / n_b
    fft_ratio = r * lam / Q_eff
    gather_ratio = r
    return dict(lam=float(lam), Q_eff=float(Q_eff), fft_ratio=float(fft_ratio),
                gather_ratio=float(gather_ratio),
                speedup=float((1.0 + rho) / (fft_ratio + gather_ratio * rho)))


def storage_terms(S: int,
                  bs: BlockSet,
                  M: int,
                  W: int,
                  im_size_cost: Optional[tuple] = None) -> dict:
    """
    Parameter counts. The block model has no ``L*W*M`` kernel term (Sec. 7).

    Args
    ----
    S : int
        Total block rank ``sum_q L_q``
    bs : BlockSet
        Partition
    M : int
        Number of k-space samples
    W : int
        Stencil width
    im_size_cost : Optional[tuple]
        Grid to evaluate storage on; defaults to the partition's grid

    Returns
    -------
    dict
        ``block_per_factor`` (``V + M``), ``hofft_per_factor`` (``N + W^d M``), and the
        block total ``sum_q L_q (V + M)``
    """
    im_size_cost = tuple(bs.im_size) if im_size_cost is None else tuple(im_size_cost)
    N = int(np.prod(im_size_cost))
    V = float(np.prod([max(round(float(bs.D[i]) * im_size_cost[i]), 1)
                       for i in range(bs.d)]))
    return dict(block=float(S * (V + M)), block_per_factor=float(V + M),
                hofft_per_factor=float(N + (W ** bs.d) * M), S=int(S))


# ---------------------------------------------------------------------------
# Block KB gather (Sec. 1.2, validated by test 4)
# ---------------------------------------------------------------------------
def kb_beta(os: float, W: int) -> float:
    """``beta`` depends only on ``(os, W)``, not on grid size (Sec. 1.2)."""
    arg = ((W / os) * (os - 0.5)) ** 2 - 0.8
    return float(np.pi * arg ** 0.5) if arg > 0 else 1.0


def block_os(os: float, V: Sequence[int]) -> tuple[list, list]:
    """
    Per-axis padded block size and the exact oversampling it realizes.

    ``run_sweep`` snaps the global ``os`` so that ``os*N`` is an even integer, which is
    what lets ``hofft_linop`` index its grid with ``round(os*k)``. A block of size
    ``V = N/Q`` generally has non-integer ``os*V``, so the block grid is padded up to an
    even ``V_os >= os*V`` and the realized oversampling ``os_q = V_os/V >= os`` is used for
    the geometry. ``beta`` still comes from the nominal ``(os, W)`` (Sec. 1.2), so a block
    is never *less* accurate than the global grid on this account.

    Args
    ----
    os : float
        Nominal oversampling
    V : Sequence[int]
        Block size in voxels

    Returns
    -------
    V_os : list
        Padded grid size per axis
    os_q : list
        Realized oversampling per axis
    """
    V_os = [2 * int(np.ceil(os * v / 2)) for v in V]
    return V_os, [V_os[i] / V[i] for i in range(len(V))]


def block_kb_forward(x_local: torch.Tensor,
                     kappa: torch.Tensor,
                     V: tuple,
                     os: float,
                     W: int,
                     beta: Optional[float] = None) -> torch.Tensor:
    """
    KB-NUFFT of one block, evaluated in the block's own local coordinates.

    Identical in structure to the global KB-NUFFT: apodize, zero-pad, FFT, gather a
    ``W``-tap stencil with wrap-around indexing. Gather indices are taken modulo the padded
    grid, which is exact here because the local grid's period in ``kappa`` units is ``V``
    (Sec. 1.2, tested in test 5). Passing ``V = im_size`` recovers the global KB-NUFFT
    exactly, which is how test 4 compares the two.

    Args
    ----
    x_local : torch.Tensor
        Block image in local coordinates with shape ``V``
    kappa : torch.Tensor
        Local trajectory ``D * k`` with shape (M, d), in cycles per block FOV
    V : tuple
        Block size in voxels
    os : float
        Nominal oversampling factor
    W : int
        Stencil width
    beta : Optional[float]
        KB shape parameter; defaults to ``kb_beta(os, W)``

    Returns
    -------
    y : torch.Tensor
        Samples with shape (M,)
    """
    d = len(V)
    beta = kb_beta(os, W) if beta is None else beta
    kern_size = (W,) * d
    dev = x_local.device
    V_os, os_q = block_os(os, V)
    os_q_t = torch.tensor(os_q, device=dev, dtype=torch.float32)

    # Apodize (mirrors hofft.pipelines.kb_nufft, per-axis at the realized os_q)
    rs = gen_grd(V).to(dev)
    apod = kb_apod_1d(rs / os_q_t, beta, W).prod(dim=-1) / W ** d
    k_corr = (W % 2 == 0) * torch.ones(d, device=dev) / os_q_t / 2
    apod = apod * torch.exp(-2j * np.pi * einsum(rs, k_corr, '... d, d -> ...'))

    # Zero-pad and centered FFT
    xp = torch.zeros(V_os, dtype=torch.complex64, device=dev)
    lo = [(V_os[i] // 2) - (V[i] // 2) for i in range(d)]
    xp[tuple(slice(lo[i], lo[i] + V[i]) for i in range(d))] = (apod * x_local).type(
        torch.complex64)
    X = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(xp)))

    # Gather a W-tap stencil around each sample, wrapping the index
    kern_vecs = gen_grd(kern_size, kern_size).reshape((-1, d)).to(dev)
    V_os_t = torch.tensor(V_os, device=dev)
    base = (kappa.float() * os_q_t).round() + V_os_t // 2
    idxs = (base[:, None, :] + kern_vecs).long() % V_os_t                  # M K d
    flat = torch.zeros(idxs.shape[:-1], dtype=torch.long, device=dev)
    for i in range(d):
        flat = flat * V_os[i] + idxs[..., i]
    taps = X.reshape(-1)[flat]                                             # M K

    kdevs = kappa.float() - (os_q_t * kappa.float()).round() / os_q_t
    wts = sample_kb_kernel(kdevs, kern_size, float(np.mean(os_q)), beta)
    wts = wts.reshape((-1, kappa.shape[0]))
    return (taps * wts.T.type(taps.dtype)).sum(dim=-1)


def dense_block_gather(x_local: torch.Tensor,
                       kappa: torch.Tensor,
                       V: tuple) -> torch.Tensor:
    """Exact ``sum_n x(r'_n) exp(-2j pi kappa . r'_n)`` for a block, as a reference."""
    d = len(V)
    rs = gen_grd(V).to(x_local.device).reshape((-1, d))
    E = torch.exp(-2j * np.pi * (kappa.float() @ rs.T))
    return E @ x_local.reshape(-1).type(E.dtype)
