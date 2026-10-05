"""
Per-factor trajectory shear for HOFFT (feasibility study).

Each HOFFT spatial factor ``l`` gets its own affine model of the phase bases over its
support::

    phi_b(r) ~= c_{b,l} + g_{b,l} . r + phi^res_{b,l}(r)

The constant is absorbed into the kernel weights and the linear term becomes a k-space
shear ``s_l(t) = G_l^T alpha(t)``, so factor ``l`` reads its FFT grid at an integer
offset ``Delta_l(t) = round(os * (k + s_l)) - round(os * k)``. Setting ``G_l = 0``
recovers standard HOFFT exactly.

The shear pushes gather points outside the nominal k-space extent, so each factor needs
its own padded grid. The feasibility metric is therefore the effective factor count
``L_eff = sum_l rho_l``, where ``rho_l`` is the per-factor grid growth.
"""
import torch
import numpy as np

from typing import Optional
from einops import einsum

from .utils import gen_grd

__all__ = [
    'alpha_second_moment',
    'spatial_jacobian',
    'shear_features',
    'weighted_kmeans',
    'kaffine_clustering',
    'init_label_candidates',
    'fit_affine_per_cluster',
    'shear_trajectory',
    'grid_growth',
    'residual_diagnostics',
    'soft_membership',
]


def alpha_second_moment(alphas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Temporal second moment of the phase coefficients and its matrix square root.

    ``Sigma_alpha = (1/M) sum_t alpha(t) alpha(t)^T`` is the metric that makes distances
    between Jacobians measure actual phase error rather than raw coefficient difference,
    which is what lets bases with different physical units be compared.

    Args
    ----
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)

    Returns
    -------
    sigma : torch.Tensor
        Second moment with shape (B, B)
    sigma_sqrt : torch.Tensor
        Symmetric PSD square root with shape (B, B)
    """
    B = alphas.shape[0]
    a = alphas.reshape((B, -1)).double()
    sigma = (a @ a.T) / a.shape[1]
    evals, evecs = torch.linalg.eigh(sigma)
    evals = evals.clamp(min=0.0)
    sigma_sqrt = (evecs * evals.sqrt()) @ evecs.T
    return sigma.to(alphas.dtype), sigma_sqrt.to(alphas.dtype)


def spatial_jacobian(phis: torch.Tensor) -> torch.Tensor:
    """
    Central-difference Jacobian of the phase bases on the normalized grid.

    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)

    Returns
    -------
    jac : torch.Tensor
        d phi_b / d r_i with shape (B, d, *im_size), where r is the ``gen_grd``
        coordinate spanning [-1/2, 1/2)
    """
    im_size = phis.shape[1:]
    d = len(im_size)
    # gen_grd spacing along axis i is 1 / im_size[i]
    grads = torch.gradient(phis, spacing=[1.0 / n for n in im_size],
                           dim=tuple(range(1, d + 1)))
    return torch.stack(grads, dim=1)


def shear_features(phis: torch.Tensor,
                   jac: torch.Tensor,
                   sigma_sqrt: torch.Tensor,
                   ell: float) -> torch.Tensor:
    """
    Whitened value-and-gradient clustering features.

    ``F(r) = [phi(r), ell * J(r)]`` whitened by ``Sigma_alpha^{1/2}`` so that Euclidean
    distance in feature space equals the RMS-over-time phase discrepancy. ``ell = 0``
    reduces to pure value clustering.

    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    jac : torch.Tensor
        Jacobian with shape (B, d, *im_size)
    sigma_sqrt : torch.Tensor
        Whitening matrix with shape (B, B)
    ell : float
        Gradient length scale in units of normalized r

    Returns
    -------
    feats : torch.Tensor
        Feature matrix with shape (N, B * (1 + d))
    """
    B = phis.shape[0]
    d = jac.shape[1]
    stack = torch.cat([phis[:, None], ell * jac], dim=1)  # B (1+d) *im_size
    stack = stack.reshape((B, 1 + d, -1))
    feats = einsum(sigma_sqrt, stack, 'B1 B2, B2 P N -> B1 P N')
    return feats.reshape((B * (1 + d), -1)).T.contiguous()


def weighted_kmeans(feats: torch.Tensor,
                    weights: torch.Tensor,
                    L: int,
                    n_iter: int = 25,
                    seed: int = 0) -> torch.Tensor:
    """
    Weighted Lloyd's algorithm with farthest-point initialization.

    Weighting by signal magnitude keeps voxels in signal voids from dragging centroids.

    Args
    ----
    feats : torch.Tensor
        Feature matrix with shape (N, F)
    weights : torch.Tensor
        Non-negative voxel weights with shape (N,)
    L : int
        Number of clusters
    n_iter : int
        Lloyd iterations
    seed : int
        Seed for the initial pivot

    Returns
    -------
    labels : torch.Tensor
        Cluster assignment with shape (N,) in [0, L-1]
    """
    N = feats.shape[0]
    if L == 1:
        return torch.zeros(N, dtype=torch.long, device=feats.device)

    # Farthest-point init restricted to voxels that carry weight
    valid = torch.argwhere(weights > 0)[:, 0]
    gen = torch.Generator(device=feats.device)
    gen.manual_seed(seed)
    start = valid[torch.randint(0, valid.numel(), (), generator=gen, device=feats.device)]
    centroids = torch.empty((L, feats.shape[1]), dtype=feats.dtype, device=feats.device)
    centroids[0] = feats[start]
    dist = torch.linalg.norm(feats - centroids[0], dim=-1)
    dist[weights <= 0] = -1.0
    for l in range(1, L):
        centroids[l] = feats[dist.argmax()]
        dist = torch.minimum(dist, torch.linalg.norm(feats - centroids[l], dim=-1))
        dist[weights <= 0] = -1.0

    labels = torch.zeros(N, dtype=torch.long, device=feats.device)
    nbs = 1 << 14
    for _ in range(n_iter):
        for n1 in range(0, N, nbs):
            n2 = min(n1 + nbs, N)
            labels[n1:n2] = torch.cdist(feats[n1:n2], centroids).argmin(dim=-1)
        # Weighted centroid update; empty clusters keep their previous centroid
        num = torch.zeros_like(centroids)
        den = torch.zeros(L, dtype=feats.dtype, device=feats.device)
        num.index_add_(0, labels, feats * weights[:, None])
        den.index_add_(0, labels, weights)
        alive = den > 0
        centroids[alive] = num[alive] / den[alive, None]
    return labels


def kaffine_clustering(phis: torch.Tensor,
                       weights: torch.Tensor,
                       sigma_sqrt: torch.Tensor,
                       L: int,
                       init_labels,
                       n_iter: int = 20,
                       include_linear: bool = True
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Lloyd's algorithm on per-cluster affine models rather than on feature vectors.

    Assign each voxel to the factor whose affine model predicts it best, refit, repeat.
    This optimizes the quantity the shear actually cares about -- the post-fit residual
    -- instead of the hand-tuned ``[phi, ell * J]`` proxy, so it removes ``ell`` as a
    confound. With ``include_linear=False`` it reduces to whitened k-means on values,
    which is the no-shear baseline.

    The affine variant is badly local-minimum prone: started from a value clustering it
    sits on level sets, where every affine fit is nearly constant and no voxel wants to
    move. Pass several inits (see ``init_label_candidates``) and the best is returned.

    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    weights : torch.Tensor
        Non-negative voxel weights with shape (N,)
    sigma_sqrt : torch.Tensor
        Whitening matrix with shape (B, B)
    L : int
        Number of clusters
    init_labels : torch.Tensor | list[torch.Tensor]
        Starting assignment(s) with shape (N,)
    n_iter : int
        Lloyd iterations
    include_linear : bool
        Fit constant plus linear (True) or constant only (False)

    Returns
    -------
    labels : torch.Tensor
        Cluster assignment with shape (N,)
    consts : torch.Tensor
        Affine offsets with shape (L, B)
    grads : torch.Tensor
        Linear coefficients with shape (L, B, d)
    obj : float
        Weighted whitened residual energy of the returned solution
    """
    if isinstance(init_labels, torch.Tensor):
        init_labels = [init_labels]

    B = phis.shape[0]
    im_size = phis.shape[1:]
    d = len(im_size)
    N = int(np.prod(im_size))
    torch_dev = phis.device

    rs = gen_grd(im_size).to(torch_dev).reshape((N, d))
    phis_w = (sigma_sqrt @ phis.reshape((B, N))).T          # N B
    nbs = 1 << 14

    def _assign(consts, grads):
        consts_w = (sigma_sqrt @ consts.T).T                # L B
        grads_w = einsum(sigma_sqrt, grads, 'B1 B2, L B2 d -> L B1 d')
        out = torch.empty(N, dtype=torch.long, device=torch_dev)
        obj = torch.zeros((), device=torch_dev)
        for n1 in range(0, N, nbs):
            n2 = min(n1 + nbs, N)
            pred = consts_w[:, None] + einsum(
                grads_w, rs[n1:n2], 'L B d, N d -> L N B')
            cost = (phis_w[None, n1:n2] - pred).square().sum(dim=-1)  # L N
            best, idx = cost.min(dim=0)
            out[n1:n2] = idx
            obj = obj + (weights[n1:n2] * best).sum()
        return out, float(obj)

    best = None
    for init in init_labels:
        labels = init.clone()
        for _ in range(n_iter):
            consts, grads = fit_affine_per_cluster(
                phis, labels, weights, L, include_linear=include_linear)
            new_labels, _ = _assign(consts, grads)
            if bool(torch.all(new_labels == labels)):
                break
            labels = new_labels
        consts, grads = fit_affine_per_cluster(
            phis, labels, weights, L, include_linear=include_linear)
        _, obj = _assign(consts, grads)
        if best is None or obj < best[3]:
            best = (labels, consts, grads, obj)
    return best


def init_label_candidates(phis: torch.Tensor,
                          jac: torch.Tensor,
                          weights: torch.Tensor,
                          sigma_sqrt: torch.Tensor,
                          L: int,
                          ells=(0.0, 0.25, 1.0),
                          n_spatial: int = 2,
                          seed: int = 0) -> list:
    """
    Assorted starting partitions for :func:`kaffine_clustering`.

    Mixes value/gradient feature clusterings with purely spatial Voronoi tilings. The
    spatial ones matter: affine models only mean something on spatially local supports,
    and no feature clustering reliably produces those.

    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    jac : torch.Tensor
        Jacobian with shape (B, d, *im_size)
    weights : torch.Tensor
        Non-negative voxel weights with shape (N,)
    sigma_sqrt : torch.Tensor
        Whitening matrix with shape (B, B)
    L : int
        Number of clusters
    ells : tuple
        Gradient length scales to seed feature clusterings with
    n_spatial : int
        Number of spatial Voronoi tilings (different seeds)
    seed : int
        Base RNG seed

    Returns
    -------
    inits : list[torch.Tensor]
        Candidate label vectors, each with shape (N,)
    """
    im_size = phis.shape[1:]
    d = len(im_size)
    N = int(np.prod(im_size))
    rs = gen_grd(im_size).to(phis.device).reshape((N, d))

    inits = [weighted_kmeans(shear_features(phis, jac, sigma_sqrt, e), weights, L)
             for e in ells]
    inits += [weighted_kmeans(rs, weights, L, seed=seed + s) for s in range(n_spatial)]
    return inits


def soft_membership(labels: torch.Tensor,
                    L: int,
                    im_size: tuple,
                    smooth_iters: int = 0) -> torch.Tensor:
    """
    Partition of unity from hard cluster labels, optionally smoothed.

    Args
    ----
    labels : torch.Tensor
        Cluster assignment with shape (N,)
    L : int
        Number of clusters
    im_size : tuple
        Spatial shape
    smooth_iters : int
        Box-blur passes applied to each membership map

    Returns
    -------
    memb : torch.Tensor
        Memberships with shape (L, *im_size) summing to one over L
    """
    memb = torch.zeros((L, labels.numel()), dtype=torch.float32, device=labels.device)
    memb[labels, torch.arange(labels.numel(), device=labels.device)] = 1.0
    memb = memb.reshape((L, *im_size))
    if smooth_iters > 0:
        d = len(im_size)
        conv = {1: torch.nn.functional.conv1d,
                2: torch.nn.functional.conv2d,
                3: torch.nn.functional.conv3d}[d]
        kern = torch.ones((1, 1) + (3,) * d, device=labels.device) / (3 ** d)
        for _ in range(smooth_iters):
            memb = conv(memb[:, None], kern, padding=1)[:, 0]
        memb = memb / memb.sum(dim=0, keepdim=True).clamp(min=1e-8)
    return memb


def fit_affine_per_cluster(phis: torch.Tensor,
                           labels: torch.Tensor,
                           weights: torch.Tensor,
                           L: int,
                           fit: str = 'ls',
                           include_linear: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted affine fit of every phase basis over every cluster support.

    The whitening metric drops out of this fit: the design matrix ``[1, r]`` is shared
    across bases, so the minimizer is the ordinary weighted least-squares solution.

    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    labels : torch.Tensor
        Cluster assignment with shape (N,)
    weights : torch.Tensor
        Non-negative voxel weights with shape (N,)
    L : int
        Number of clusters
    fit : str
        ``'ls'`` for least squares or ``'minimax'`` for a few IRLS passes toward the
        Chebyshev fit (residual range, not L2 norm, is what sets the required L)
    include_linear : bool
        When False, fit only the constant. This is the no-shear baseline and must be
        fit on its own rather than by zeroing the linear part of the affine fit.

    Returns
    -------
    consts : torch.Tensor
        Affine offsets with shape (L, B)
    grads : torch.Tensor
        Linear coefficients with shape (L, B, d), zero when ``include_linear`` is False
    """
    B = phis.shape[0]
    im_size = phis.shape[1:]
    d = len(im_size)
    torch_dev = phis.device
    N = int(np.prod(im_size))

    rs = gen_grd(im_size).to(torch_dev).reshape((N, d))
    ones = torch.ones((N, 1), device=torch_dev)
    design = torch.cat([ones, rs], dim=1) if include_linear else ones
    targets = phis.reshape((B, N)).T  # N B

    consts = torch.zeros((L, B), device=torch_dev)
    grads = torch.zeros((L, B, d), device=torch_dev)
    for l in range(L):
        w = weights * (labels == l)
        if (w > 0).sum() < d + 1:
            continue
        coef = _weighted_lstsq(design, targets, w)
        if fit == 'minimax':
            coef = _irls_minimax(design, targets, w, coef)
        consts[l] = coef[0]
        if include_linear:
            grads[l] = coef[1:].T
    return consts, grads


def _weighted_lstsq(design: torch.Tensor,
                    targets: torch.Tensor,
                    w: torch.Tensor) -> torch.Tensor:
    """Solve min_x sum_n w_n ||targets_n - design_n x||^2, returning (1+d, B)."""
    dw = design * w[:, None]
    gram = design.T @ dw
    rhs = dw.T @ targets
    gram = gram + 1e-9 * gram.diagonal().abs().mean() * torch.eye(
        gram.shape[0], device=gram.device)
    return torch.linalg.solve(gram, rhs)


def _irls_minimax(design: torch.Tensor,
                  targets: torch.Tensor,
                  w: torch.Tensor,
                  coef: torch.Tensor,
                  n_iter: int = 12,
                  p_final: float = 16.0) -> torch.Tensor:
    """Iteratively reweighted least squares driving the L2 fit toward Chebyshev."""
    active = w > 0
    for it in range(n_iter):
        p = 2.0 + (p_final - 2.0) * (it + 1) / n_iter
        resid = (targets - design @ coef).norm(dim=1)
        scale = resid[active].max().clamp(min=1e-12)
        wt = w * (resid / scale).clamp(min=1e-6) ** (p - 2)
        coef = _weighted_lstsq(design, targets, wt)
    return coef


def shear_trajectory(trj: torch.Tensor,
                     alphas: torch.Tensor,
                     grads: torch.Tensor,
                     os: float,
                     cap: Optional[float] = None,
                     im_size: Optional[tuple] = None
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-factor sheared trajectory and its integer grid offset.

    Args
    ----
    trj : torch.Tensor
        Nominal trajectory with shape (*trj_size, d)
    alphas : torch.Tensor
        Temporal phase coefficients with shape (B, *trj_size)
    grads : torch.Tensor
        Per-factor linear coefficients with shape (L, B, d)
    os : float
        Grid oversampling factor
    cap : Optional[float]
        Shear cap as a fraction of ``N/2``; ``None`` leaves the shear uncapped
    im_size : Optional[tuple]
        Image shape, required when ``cap`` is given

    Returns
    -------
    kappa : torch.Tensor
        Sheared trajectory with shape (L, *trj_size, d)
    delta : torch.Tensor
        Integer grid offset with shape (L, *trj_size, d)
    tau : torch.Tensor
        Applied shear shrinkage per factor with shape (L,)
    """
    L = grads.shape[0]
    shear = einsum(grads, alphas, 'L B d, B ... -> L ... d')

    tau = torch.ones(L, device=trj.device)
    if cap is not None:
        assert im_size is not None, 'cap requires im_size'
        cap_abs = cap * min(im_size) / 2
        peak = shear.reshape((L, -1)).abs().amax(dim=1).clamp(min=1e-12)
        tau = (cap_abs / peak).clamp(max=1.0)
        shear = shear * tau.reshape((L,) + (1,) * (shear.ndim - 1))

    kappa = trj[None] + shear
    delta = ((os * kappa).round() - (os * trj[None]).round())
    return kappa, delta, tau


def grid_growth(trj: torch.Tensor,
                kappa: torch.Tensor,
                os: float,
                kern_size: tuple,
                im_size: tuple) -> tuple[torch.Tensor, float, float]:
    """
    Per-factor FFT grid growth and the effective factor count.

    This is the padding cost model of feas_test.md Sec 1.6, which assumes the sheared
    gather index cannot wrap. That assumption does not hold here: for a discrete voxel
    image on an even grid with integer ``os * N``, the term ``exp(-2i pi r.z/os)`` is
    exactly periodic in ``z`` with period ``os * N``, so the index wraps losslessly and
    the true cost is ``rho_l = 1``. See ``stage1_checks.check_wrap_exactness``. The
    span ratio is kept because it still measures how far the shear pushes samples.

    Args
    ----
    trj : torch.Tensor
        Nominal trajectory with shape (*trj_size, d)
    kappa : torch.Tensor
        Sheared trajectory with shape (L, *trj_size, d)
    os : float
        Grid oversampling factor
    kern_size : tuple
        Interpolation stencil size
    im_size : tuple
        Image shape

    Returns
    -------
    rho : torch.Tensor
        Per-factor grid growth with shape (L,), relative to the unsheared grid
    L_eff : float
        ``sum_l rho_l``
    rho_nominal : float
        Growth of the unsheared grid over ``os * im_size``, for reference
    """
    d = trj.shape[-1]
    L = kappa.shape[0]
    kappa_flt = kappa.reshape((L, -1, d)) * os
    trj_flt = trj.reshape((-1, d)) * os

    span_base = (trj_flt.amax(dim=0) - trj_flt.amin(dim=0)
                 + torch.tensor(kern_size, device=trj.device, dtype=trj.dtype))
    span = (kappa_flt.amax(dim=1) - kappa_flt.amin(dim=1)
            + torch.tensor(kern_size, device=trj.device, dtype=trj.dtype))

    rho = (span / span_base).prod(dim=-1)
    nominal = float((span_base / (os * torch.tensor(
        im_size, device=trj.device, dtype=trj.dtype))).prod())
    return rho, float(rho.sum()), nominal


def residual_diagnostics(phis: torch.Tensor,
                         jac: torch.Tensor,
                         alphas_sub: torch.Tensor,
                         labels: torch.Tensor,
                         weights: torch.Tensor,
                         consts: torch.Tensor,
                         grads: torch.Tensor,
                         chunk: int = 1 << 12
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Residual phase range and residual gradient after the per-factor affine fit.

    The residual range is what sets how many factors HOFFT needs; the residual gradient
    is what determines whether the ``W`` stencil is doing any work.

    Args
    ----
    phis : torch.Tensor
        Spatial phase maps with shape (B, *im_size)
    jac : torch.Tensor
        Jacobian with shape (B, d, *im_size)
    alphas_sub : torch.Tensor
        Subsampled temporal coefficients with shape (B, T_sub)
    labels : torch.Tensor
        Cluster assignment with shape (N,)
    weights : torch.Tensor
        Non-negative voxel weights with shape (N,)
    consts : torch.Tensor
        Affine offsets with shape (L, B)
    grads : torch.Tensor
        Linear coefficients with shape (L, B, d); pass zeros for the no-shear baseline

    Returns
    -------
    range_per_factor : torch.Tensor
        ``max_t [max_r phi_res . alpha - min_r phi_res . alpha]`` with shape (L,). Peak
        to peak rather than max-abs, because a spatially constant phase is absorbed by
        the kernel weights for free and so must not be charged to either variant.
    range_weighted : torch.Tensor
        Signal-weighted RMS of the mean-removed residual with shape (L,)
    grad_max_per_factor : torch.Tensor
        ``max_t max_r |(J - G)^T alpha|_inf`` with shape (L,)
    grad_samples : torch.Tensor
        Per-voxel ``max_t |(J - G)^T alpha|_inf`` with shape (N,), for histograms
    """
    B = phis.shape[0]
    d = jac.shape[1]
    L = consts.shape[0]
    N = labels.numel()
    T = alphas_sub.shape[1]
    torch_dev = phis.device

    im_size = phis.shape[1:]
    rs = gen_grd(im_size).to(torch_dev).reshape((N, d))
    phis_flt = phis.reshape((B, N))
    jac_flt = jac.reshape((B, d, N))

    range_per_factor = torch.zeros(L, device=torch_dev)
    range_weighted = torch.zeros(L, device=torch_dev)
    grad_max_per_factor = torch.zeros(L, device=torch_dev)
    grad_samples = torch.zeros(N, device=torch_dev)

    for l in range(L):
        idx = torch.argwhere((labels == l) & (weights > 0))[:, 0]
        if idx.numel() == 0:
            continue
        rmax = torch.full((T,), -torch.inf, device=torch_dev)
        rmin = torch.full((T,), torch.inf, device=torch_dev)
        acc_s = torch.zeros(T, device=torch_dev)
        acc_sq = torch.zeros(T, device=torch_dev)
        acc_w = torch.zeros((), device=torch_dev)
        gmax = torch.zeros((), device=torch_dev)
        for i1 in range(0, idx.numel(), chunk):
            sub = idx[i1:i1 + chunk]
            # Residual phase, (n_sub, T_sub)
            res_b = phis_flt[:, sub] - consts[l][:, None] - einsum(
                grads[l], rs[sub], 'B d, N d -> B N')
            res = res_b.T @ alphas_sub
            rmax = torch.maximum(rmax, res.amax(dim=0))
            rmin = torch.minimum(rmin, res.amin(dim=0))
            wsub = weights[sub]
            acc_s = acc_s + (wsub[:, None] * res).sum(dim=0)
            acc_sq = acc_sq + (wsub[:, None] * res.square()).sum(dim=0)
            acc_w = acc_w + wsub.sum()
            # Residual gradient, (n_sub, d, T_sub)
            jres = jac_flt[:, :, sub] - grads[l][:, :, None]
            gres = einsum(jres, alphas_sub, 'B d N, B T -> N d T').abs().amax(dim=1)
            gmax = torch.maximum(gmax, gres.amax())
            grad_samples[sub] = gres.amax(dim=-1)
        acc_w = acc_w.clamp(min=1e-12)
        var = (acc_sq / acc_w - (acc_s / acc_w).square()).clamp(min=0)
        range_per_factor[l] = (rmax - rmin).amax()
        range_weighted[l] = var.mean().sqrt()
        grad_max_per_factor[l] = gmax
    return range_per_factor, range_weighted, grad_max_per_factor, grad_samples
