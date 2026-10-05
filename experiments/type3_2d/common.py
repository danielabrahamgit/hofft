"""
Shared linear algebra for the 2D type-3 + residual-HOFFT feasibility study
(math_docs/t3n_synergy.md, Stages A–C).

Never materializes the full N×M phase or exponential. Phase is in cycles:
    Ψ_nm = r_n·k_m + Σ_a φ_a(r_n) α_a(t_m)
    (Ax)_m = Σ_n x_n exp(-i 2π Ψ_nm)

Dataset preprocessing matches paper_experiments/hybrid_feas/common.py and
paper_experiments/run_sweep.py (same R, ESPIRiT 0.9 mask, remove_linear_terms).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from hofft.phase_coeffs import remove_linear_terms
from hofft.utils import gen_grd

MASK_THRESH = 0.9
OS_NOMINAL = 1.25
REL_QR_TOL = 1e-10
COND_T_MAX = 100.0
C0 = 1.0
PRIMARY_EPS = 1e-3
EPSILONS = (1e-2, 1e-3, 1e-4)
# η = Smax / Sgeo. Values < 1 are optional compression, not the main experiment.
ETA_SWEEP = (1.0, 1.5, 2.0, 4.0, 8.0)
SEEDS = (0, 1, 2)
BLOCK_NS, BLOCK_NT = 1024, 2048
HELD_NS, HELD_NT = 512, 1024
SIGMA_FLOOR = 1e-14

# Same reconstruction undersampling as run_sweep / hybrid_feas.
DATASETS = {
    'coco_spiral': dict(R=3),
    'tilt_spi_invivo': dict(R=2),
}

# cuFINUFFT 2D type-1/2 on these volumes uses upsampfac=2 (hybrid_feas REPORT).
UPSAMPFAC_BASE = 2.0
# Stencil work: W=3 taps/axis → 9 gathers; conservative uses W=6 → 36.
W_OPT, W_CONS = 9.0, 36.0
# N'_j ≈ smoothceil(γ s_j + β). Optimistic / conservative type-3 grid models.
GAMMA_OPT, BETA_OPT = 1.25, 4.0
GAMMA_CONS, BETA_CONS = 2.5, 16.0


def _snap_os(os_nominal: float, n: int) -> float:
    return 2 * round(os_nominal * n / 2) / n


def next_smooth(n: int, radices=(2, 3, 5)) -> int:
    n = int(max(n, 1))
    while True:
        m = n
        for p in radices:
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


@dataclass
class PhaseData:
    """Masked 2D cycle-phase factors after run_sweep preprocessing."""
    name: str
    B: torch.Tensor          # (Nsrc, 2+K) float64  [R | Φ]
    D: torch.Tensor          # (M, 2+K) float64      [K | C]
    R: torch.Tensor          # (Nsrc, 2)
    Ktraj: torch.Tensor      # (M, 2)
    Phi: torch.Tensor        # (Nsrc, K)
    C: torch.Tensor          # (M, K)
    mask: torch.Tensor       # (*im_size) float, native grid
    im_size: tuple
    n_coils: int
    n_high_order: int
    R_undersample: int
    os: float
    n_fft_base: int
    n_src: int
    n_tgt: int
    energy_src: torch.Tensor  # (Nsrc,) HO spatial energy
    energy_tgt: torch.Tensor  # (M,) HO temporal energy


@dataclass
class CoreSVD:
    QB: torch.Tensor
    RB: torch.Tensor
    QD: torch.Tensor
    RD: torch.Tensor
    U: torch.Tensor
    sigma: torch.Tensor
    V: torch.Tensor
    core: torch.Tensor
    rank: int
    centered: bool
    bbar: Optional[torch.Tensor] = None
    dbar: Optional[torch.Tensor] = None
    a: Optional[torch.Tensor] = None
    c: Optional[torch.Tensor] = None


@dataclass
class Candidate:
    cid: str
    family: str
    p: int
    F: torch.Tensor
    Q: torch.Tensor
    a: torch.Tensor
    c: torch.Tensor
    T: Optional[np.ndarray] = None
    notes: str = ''
    metrics: dict = field(default_factory=dict)


def load_phase_data(name: str, device: torch.device,
                    repo: Path | None = None) -> PhaseData:
    if name not in DATASETS:
        raise KeyError(f'unknown dataset {name}; expected {list(DATASETS)}')
    cfg = DATASETS[name]
    repo = Path(repo or Path.cwd())
    fpath = repo / 'data' / name
    kw = {'weights_only': True, 'map_location': 'cpu'}

    trj = torch.load(fpath / 'trj.pt', **kw).float()
    evals = torch.load(fpath / 'evals.pt', **kw).float()
    phis = torch.load(fpath / 'phis.pt', **kw).float()
    alphas = torch.load(fpath / 'alphas.pt', **kw).float()
    mps = torch.load(fpath / 'mps.pt', **kw)
    n_coils = int(mps.shape[0])
    del mps

    R = cfg['R']
    if trj.ndim >= 3 and R > 1:
        trj = trj[:, ::R].contiguous()
        alphas = alphas[:, :, ::R].contiguous()

    im_size = tuple(evals.shape)
    if len(im_size) != 2:
        raise ValueError(f'{name} is {len(im_size)}D; this study is 2D-only')
    mask = (evals > MASK_THRESH).float()
    phis, trj_term, _zeroth = remove_linear_terms(phis, alphas, mask=mask)
    trj = trj + trj_term
    phis = phis * mask

    Bho = phis.shape[0]
    energy = (phis.reshape((Bho, -1)).abs().mean(dim=1)
              * alphas.reshape((Bho, -1)).abs().mean(dim=1))
    keep = torch.argwhere(energy > 1e-6)[:, 0]
    phis, alphas = phis[keep].contiguous(), alphas[keep].contiguous()

    rs = gen_grd(im_size).reshape(-1, 2)
    mflat = mask.reshape(-1) > 0
    Rcrd = rs[mflat].to(device=device, dtype=torch.float64)
    Phi = phis.reshape(phis.shape[0], -1)[:, mflat].T.to(device=device, dtype=torch.float64)
    Ktraj = trj.reshape(-1, 2).to(device=device, dtype=torch.float64)
    Ctmp = alphas.reshape(alphas.shape[0], -1).T.to(device=device, dtype=torch.float64)

    B = torch.cat([Rcrd, Phi], dim=1)
    D = torch.cat([Ktraj, Ctmp], dim=1)
    os = _snap_os(OS_NOMINAL, im_size[0])
    n_fft = int(round(UPSAMPFAC_BASE * im_size[0])) * int(round(UPSAMPFAC_BASE * im_size[1]))
    return PhaseData(
        name=name, B=B, D=D, R=Rcrd, Ktraj=Ktraj, Phi=Phi, C=Ctmp,
        mask=mask, im_size=im_size, n_coils=n_coils,
        n_high_order=int(Phi.shape[1]), R_undersample=R, os=os,
        n_fft_base=n_fft, n_src=int(Rcrd.shape[0]), n_tgt=int(Ktraj.shape[0]),
        energy_src=(Phi.square().sum(dim=1)),
        energy_tgt=(Ctmp.square().sum(dim=1)),
    )


def qr_core_svd(B: torch.Tensor, D: torch.Tensor, center: bool) -> CoreSVD:
    """Economy QR + core SVD of Ψ = B D^T. Never forms Ψ."""
    if center:
        bbar = B.mean(dim=0)
        dbar = D.mean(dim=0)
        Bc = B - bbar
        Dc = D - dbar
        a = B @ dbar
        c = Dc @ bbar
    else:
        bbar = dbar = a = c = None
        Bc, Dc = B, D
    QB, RB = torch.linalg.qr(Bc, mode='reduced')
    QD, RD = torch.linalg.qr(Dc, mode='reduced')
    core = RB @ RD.mT
    U, sigma, Vh = torch.linalg.svd(core, full_matrices=False)
    V = Vh.mT
    rank = int((sigma > SIGMA_FLOOR * sigma[0].clamp(min=SIGMA_FLOOR)).sum())
    return CoreSVD(
        QB=QB, RB=RB, QD=QD, RD=RD, U=U, sigma=sigma, V=V, core=core,
        rank=rank, centered=center, bbar=bbar, dbar=dbar, a=a, c=c,
    )


def balanced_factors(core: CoreSVD, p: int) -> tuple[torch.Tensor, torch.Tensor]:
    p = min(p, int(core.sigma.numel()), core.rank)
    s = core.sigma[:p].clamp(min=0).sqrt()
    F = core.QB @ (core.U[:, :p] * s)
    Q = core.QD @ (core.V[:, :p] * s)
    return F, Q


def phase_tail(sigma: torch.Tensor, p: int) -> float:
    tot = float(sigma.square().sum().clamp(min=SIGMA_FLOOR))
    return float(sigma[p:].square().sum() / tot) ** 0.5 if p < sigma.numel() else 0.0


def rms_from_tail(sigma: torch.Tensor, p: int, n: int, m: int) -> float:
    tail = float(sigma[p:].square().sum()) if p < sigma.numel() else 0.0
    return math.sqrt(tail / max(n * m, 1))


def reconstruct_sampled_phase(core: CoreSVD, Bi: torch.Tensor, Dj: torch.Tensor,
                              Ii: torch.Tensor, Jj: torch.Tensor) -> torch.Tensor:
    """Ψ_IJ = (Q_B U Σ) (Q_D V)^T on sampled rows/cols, plus offsets if centered."""
    Ffull = core.QB @ (core.U * core.sigma)
    Qfull = core.QD @ core.V
    ph = Ffull[Ii] @ Qfull[Jj].mT
    if core.centered:
        ph = ph + core.a[Ii][:, None] + core.c[Jj][None, :]
    return ph


def dense_phase(B: torch.Tensor, D: torch.Tensor, Ii, Jj) -> torch.Tensor:
    return B[Ii] @ D[Jj].mT


def bandwidth(F: torch.Tensor, Q: torch.Tensor) -> dict:
    """Full-data extents. Never clip outliers."""
    widths_f = (F.max(dim=0).values - F.min(dim=0).values).cpu()
    widths_q = (Q.max(dim=0).values - Q.min(dim=0).values).cpu()
    s = (widths_f * widths_q).numpy()
    S = float(np.prod(C0 + s))
    q = torch.tensor([0.01, 0.99], device=F.device, dtype=F.dtype)
    pct_f = torch.quantile(F, q, dim=0)
    pct_q = torch.quantile(Q, q, dim=0)
    s_pct = ((pct_f[1] - pct_f[0]) * (pct_q[1] - pct_q[0])).cpu().numpy()
    return dict(
        widths_F=widths_f.numpy().tolist(),
        widths_Q=widths_q.numpy().tolist(),
        s=s.tolist(),
        S=S,
        s_p01_p99=s_pct.tolist(),
        S_p01_p99=float(np.prod(C0 + s_pct)),
        aspect=float(s[0] / max(s[1], 1e-18)),
    )


def fold_midpoint_offsets(F, Q, a, c):
    """Center latent coords; fold separable terms into a, c. Widths unchanged."""
    f0 = F.mean(dim=0)
    q0 = Q.mean(dim=0)
    Fc, Qc = F - f0, Q - q0
    a_new = a + Fc @ q0 + 0.5 * (f0 * q0).sum()
    c_new = c + Qc @ f0 + 0.5 * (f0 * q0).sum()
    return Fc, Qc, a_new, c_new


def apply_T(F, Q, T: torch.Tensor):
    """F → F T, Q → Q T^{-T}. Phase F Q^T is invariant."""
    TinvT = torch.linalg.inv(T).mT
    return F @ T, Q @ TinvT


def cond2(T: torch.Tensor) -> float:
    s = torch.linalg.svdvals(T)
    return float((s[0] / s[-1].clamp(min=1e-30)).cpu())


def rotation_T(theta: float, device, dtype) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], device=device, dtype=dtype)


def shear_T(axis: int, sigma: float, device, dtype) -> torch.Tensor:
    T = torch.eye(2, device=device, dtype=dtype)
    if axis == 0:
        T[0, 1] = sigma
    else:
        T[1, 0] = sigma
    return T


def bandwidth_S(F: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Differentiable proxy volume S = Π_j (1 + ΔF_j ΔQ_j)."""
    S = F.new_ones(())
    for j in range(F.shape[1]):
        S = S * (C0 + (F[:, j].max() - F[:, j].min()) * (Q[:, j].max() - Q[:, j].min()))
    return S


def separable_from_FQ(B, D, F, Q):
    """Exact row/column means of Ψ - FQ^T, without forming the matrix."""
    dbar, qbar = D.mean(0), Q.mean(0)
    a = B @ dbar - F @ qbar
    c = D @ B.mean(0) - Q @ F.mean(0) - a.mean()
    return a, c


def mix_FQ(R, Phi, Ktraj, C, U, V):
    """F = R + Φ U, Q = K + C V. U,V are (n_high_order, 2)."""
    F = R if U is None else R + Phi @ U
    Q = Ktraj if V is None else Ktraj + C @ V
    return F, Q


def ho_mix_directions(Phi: torch.Tensor, C: torch.Tensor, seed: int = 0) -> list:
    """Unit (K, 2) mixing directions: PCA, leading bases, and one random draw."""
    K = Phi.shape[1]
    device, dtype = Phi.device, Phi.dtype
    dirs = []
    Pc = Phi - Phi.mean(0)
    Cc = C - C.mean(0)
    if K >= 1:
        _, _, Vhp = torch.linalg.svd(Pc, full_matrices=False)
        _, _, Vhc = torch.linalg.svd(Cc, full_matrices=False)
        Up = torch.zeros(K, 2, device=device, dtype=dtype)
        Vc = torch.zeros(K, 2, device=device, dtype=dtype)
        r = min(2, K)
        Up[:, :r] = Vhp[:r, :].mT
        Vc[:, :r] = Vhc[:r, :].mT
        dirs.append(('pca', Up, Vc))
        Ue = torch.zeros(K, 2, device=device, dtype=dtype)
        Ve = torch.zeros(K, 2, device=device, dtype=dtype)
        for j in range(r):
            Ue[j, j] = 1.0 / Phi[:, j].std().clamp(min=1e-8)
            Ve[j, j] = 1.0 / C[:, j].std().clamp(min=1e-8)
        dirs.append(('lead', Ue, Ve))
    g = torch.Generator(device=device).manual_seed(seed)
    if K >= 2:
        Ur = torch.linalg.qr(torch.randn(K, 2, generator=g, device=device, dtype=dtype))[0]
        g = torch.Generator(device=device).manual_seed(seed + 1)
        Vr = torch.linalg.qr(torch.randn(K, 2, generator=g, device=device, dtype=dtype))[0]
    else:
        Ur = torch.zeros(K, 2, device=device, dtype=dtype)
        Vr = torch.zeros(K, 2, device=device, dtype=dtype)
        if K == 1:
            Ur[0, 0] = 1.0
            Vr[0, 0] = 1.0
    dirs.append(('rand', Ur, Vr))
    return dirs


def scale_mix_to_eta(R, Phi, Ktraj, C, U0, V0, eta, S_geo, mode='joint'):
    """Binary-search a scalar so S(F,Q) is just above η S_geo (full-data extents)."""
    target = float(eta) * float(S_geo)
    zU = torch.zeros_like(U0)
    zV = torch.zeros_like(V0)

    def S_of(a):
        U = None if mode == 'temp' else a * U0
        V = None if mode == 'spat' else a * V0
        if mode == 'temp':
            U = zU * 0
        if mode == 'spat':
            V = zV * 0
        F, Q = mix_FQ(R, Phi, Ktraj, C, U, V)
        return float(bandwidth(F, Q)['S']), F, Q, U, V

    s0, F, Q, U, V = S_of(0.0)
    if target <= s0 * 1.01:
        return 0.0, F, Q, U, V, s0
    lo, hi = 0.0, 1.0
    shi, F, Q, U, V = S_of(hi)
    while shi < target and hi < 1e5:
        hi *= 2.0
        shi, F, Q, U, V = S_of(hi)
        if not math.isfinite(shi):
            hi /= 2.0
            break
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        sm, Fm, Qm, Um, Vm = S_of(mid)
        if sm < target:
            lo, F, Q, U, V, s0 = mid, Fm, Qm, Um, Vm, sm
        else:
            hi, F, Q, U, V, s0 = mid, Fm, Qm, Um, Vm, sm
    return hi, F, Q, U, V, s0


def best_rotation_cleanup(F, Q):
    """Phase-preserving rotation that minimizes S. Shears skipped if they raise S."""
    best_T = rotation_T(0.0, F.device, F.dtype)
    best_S, best_F, best_Q = bandwidth(F, Q)['S'], F, Q
    for deg in range(0, 180, 5):
        T = rotation_T(math.radians(deg), F.device, F.dtype)
        Ft, Qt = apply_T(F, Q, T)
        S = bandwidth(Ft, Qt)['S']
        if S < best_S:
            best_S, best_T, best_F, best_Q = S, T, Ft, Qt
    return best_F, best_Q, best_T, best_S


def sv_tail_rel(svals: torch.Tensor, l: int) -> float:
    s2 = svals.square()
    tot = float(s2.sum().clamp(min=SIGMA_FLOOR))
    tail = float(s2[l:].sum()) if l < svals.numel() else 0.0
    return math.sqrt(tail / tot)


def rank_report(svals: torch.Tensor, eps: float) -> dict:
    cap = int(svals.numel())
    L = rank_epsilon(svals, eps)
    tail = sv_tail_rel(svals, L)
    met = tail <= eps + 1e-15
    return dict(L=L, cap=cap, tail=tail, target_met=bool(met),
                L_label=str(L) if met else f'>{cap}')


def phase_fro_from_core(core: CoreSVD, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """||C_Ψ - X Y^T||_F^2 exact, X = Q_B^T F, Y = Q_D^T Q."""
    return torch.linalg.norm(core.core - X @ Y.mT) ** 2


def candidate_metrics(cand: Candidate, core_ref: CoreSVD, n: int, m: int,
                      S_geo: float) -> dict:
    bw = bandwidth(cand.F, cand.Q)
    X = core_ref.QB.mT @ cand.F
    Y = core_ref.QD.mT @ cand.Q
    loss = float(phase_fro_from_core(core_ref, X, Y).cpu())
    rms = math.sqrt(loss / max(n * m, 1))
    rel = math.sqrt(loss / float(core_ref.sigma.square().sum().clamp(min=SIGMA_FLOOR)))
    T = cand.T
    return dict(
        **bw,
        phase_rms_cycles=rms,
        phase_rel_fro=rel,
        S_over_Sgeo=bw['S'] / max(S_geo, 1e-18),
        cond_T=None if T is None else float(np.linalg.cond(T)),
        p=cand.p,
        family=cand.family,
        cid=cand.cid,
        notes=cand.notes,
    )


def residual_phase_block(B, D, cand: Candidate, I, J) -> torch.Tensor:
    ph = B[I] @ D[J].mT
    ph = ph - cand.a[I][:, None] - cand.c[J][None, :]
    ph = ph - cand.F[I] @ cand.Q[J].mT
    return ph


def residual_exp_block(B, D, cand: Candidate, I, J) -> torch.Tensor:
    # complex64 is enough for ε down to 1e-4; SVD is much faster than complex128
    return torch.exp(-2j * math.pi * residual_phase_block(B, D, cand, I, J)).to(torch.complex64)


def residual_exp_from_psi(Psi: torch.Tensor, cand: Candidate, I, J) -> torch.Tensor:
    ph = Psi - cand.a[I][:, None] - cand.c[J][None, :] - cand.F[I] @ cand.Q[J].mT
    return torch.exp(-2j * math.pi * ph).to(torch.complex64)


def rank_epsilon(svals: torch.Tensor, eps: float) -> int:
    s2 = svals.square()
    tot = s2.sum().clamp(min=SIGMA_FLOOR)
    tail = torch.flip(torch.cumsum(torch.flip(s2, [0]), 0), [0])
    # tail[l] = sum_{j>=l} σ_j^2 ; want min l s.t. sqrt(sum_{j>l} / tot) <= eps
    # sum_{j>l} = tail[l+1] if l+1 < n else 0
    n = svals.numel()
    rel = torch.zeros(n + 1, device=svals.device, dtype=svals.dtype)
    rel[0] = 1.0
    if n > 1:
        rel[1:n] = (tail[1:] / tot).clamp(min=0).sqrt()
    rel[n] = 0.0
    hit = torch.nonzero(rel <= eps, as_tuple=False)
    return int(hit[0, 0].item())


def stratified_indices(n: int, n_pick: int, energy: torch.Tensor,
                       coords: torch.Tensor, seed: int) -> torch.Tensor:
    """Mix of random, high-energy, and coordinate-extrema source/target indices."""
    n_pick = min(n_pick, n)
    g = torch.Generator(device=energy.device).manual_seed(seed)
    n_rand = max(n_pick // 2, 1)
    n_hi = max(n_pick // 4, 0)
    n_ext = n_pick - n_rand - n_hi
    rand = torch.randperm(n, generator=g, device=energy.device)[:n_rand]
    k_hi = min(max(n_hi * 8, n_hi), n)
    hi_pool = torch.topk(energy, k_hi).indices
    hi = hi_pool[torch.randperm(hi_pool.numel(), generator=g, device=energy.device)[:n_hi]]
    # 4 bbox corners of the 2D coords, plus axis extrema
    c2 = coords[:, :2]
    lo, hi_c = c2.min(0).values, c2.max(0).values
    corners = torch.stack([
        torch.stack([lo[0], lo[1]]), torch.stack([lo[0], hi_c[1]]),
        torch.stack([hi_c[0], lo[1]]), torch.stack([hi_c[0], hi_c[1]]),
    ])
    d2 = ((c2[:, None, :] - corners[None, :, :]) ** 2).sum(-1)
    corner_idx = d2.argmin(dim=0)
    axis_idx = torch.stack([c2[:, 0].argmin(), c2[:, 0].argmax(),
                            c2[:, 1].argmin(), c2[:, 1].argmax()])
    ext = torch.unique(torch.cat([corner_idx, axis_idx]))
    if ext.numel() < n_ext:
        extra = torch.randperm(n, generator=g, device=energy.device)
        ext = torch.unique(torch.cat([ext, extra]))
    ext = ext[:n_ext]
    idx = torch.unique(torch.cat([rand, hi, ext]))
    if idx.numel() < n_pick:
        extra = torch.randperm(n, generator=g, device=energy.device)
        idx = torch.unique(torch.cat([idx, extra]))
    return idx[:n_pick]


def _nystrom_from_svd(U, s, Vh, E_Iho_Jfit, E_Ifit_Jho, E_ho, L: int) -> float:
    """
    Held-out error of extensible SVD factors: the worse of
    column-space generalization (new times, same sources) and
    row-space generalization (new sources, same times).

    Two-sided Nyström of E_ho is not used for the gate — a locally
    low-rank block of exp(-i2π φ·α) is not globally that rank.
    """
    L = max(1, min(L, int(s.numel())))
    UL = U[:, :L]
    VhL = Vh[:L, :]
    den_c = torch.linalg.norm(E_Ifit_Jho).clamp(min=1e-30)
    den_r = torch.linalg.norm(E_Iho_Jfit).clamp(min=1e-30)
    col = torch.linalg.norm(E_Ifit_Jho - UL @ (UL.mH @ E_Ifit_Jho)) / den_c
    row = torch.linalg.norm(E_Iho_Jfit - (E_Iho_Jfit @ VhL.mH) @ VhL) / den_r
    return float(torch.maximum(col, row).cpu())


def heldout_nystrom_error(E_fit, E_Iho_Jfit, E_Ifit_Jho, E_ho, L: int) -> float:
    """
    Fit rank-L factors on E_fit by SVD, extend by least squares, score E_ho.
    E ≈ B_R H_R^T with B_R = U √S, H_R = conj(V) √S.
    """
    L = max(1, min(L, min(E_fit.shape)))
    U, s, Vh = torch.linalg.svd(E_fit, full_matrices=False)
    return _nystrom_from_svd(U, s, Vh, E_Iho_Jfit, E_Ifit_Jho, E_ho, L)


def min_rank_heldout(E_fit, E_Iho_Jfit, E_Ifit_Jho, E_ho, eps: float,
                     Lmax: int = 128, factors=None) -> tuple[int, float]:
    """Smallest rank whose Nyström extension meets `eps` on the held-out block."""
    if factors is None:
        U, s, Vh = torch.linalg.svd(E_fit, full_matrices=False)
    else:
        U, s, Vh = factors
    lo, hi = 1, min(int(Lmax), int(s.numel()),
                    int((s > 1e-8 * s[0]).sum().item() or 1))
    # Prefer the smallest L that meets eps; if none, the L with the best error
    # (do not default to Lmax — the tail is ill-conditioned).
    best_L, best_err = 1, _nystrom_from_svd(U, s, Vh, E_Iho_Jfit, E_Ifit_Jho, E_ho, 1)
    # coarse scan then refine around the first hit / the minimum
    grid = sorted(set(
        [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, hi] + list(range(lo, min(hi, 32) + 1))
    ))
    grid = [k for k in grid if 1 <= k <= hi]
    for mid in grid:
        err = _nystrom_from_svd(U, s, Vh, E_Iho_Jfit, E_Ifit_Jho, E_ho, mid)
        if err < best_err - 1e-12 or (abs(err - best_err) <= 1e-12 and mid < best_L):
            best_L, best_err = mid, err
        if err <= eps:
            # shrink to the smallest feasible
            hi2 = mid
            lo2 = 1 if mid == 1 else max(1, mid // 2)
            while lo2 <= hi2:
                m2 = (lo2 + hi2) // 2
                e2 = _nystrom_from_svd(U, s, Vh, E_Iho_Jfit, E_Ifit_Jho, E_ho, m2)
                if e2 <= eps:
                    best_L, best_err = m2, e2
                    hi2 = m2 - 1
                else:
                    lo2 = m2 + 1
            break
    return best_L, best_err


def nprime_axis(s_j: float, gamma: float, beta: float) -> int:
    return next_smooth(int(math.ceil(gamma * float(s_j) + beta)))


def predict_cost(L: int, Lp: int, s, n_src: int, n_tgt: int, n_fft_base: int,
                 gamma: float, beta: float, W: float) -> dict:
    n1 = nprime_axis(s[0], gamma, beta)
    n2 = nprime_axis(s[1], gamma, beta)
    Np = n1 * n2
    N = float(n_fft_base)
    M = float(n_tgt)
    Ns = float(n_src)
    t_base = L * (N * math.log(max(N, 2.0)) + W * M)
    t_new = Lp * (Np * math.log(max(Np, 2.0)) + W * Ns + W * M)
    ratio = t_new / max(t_base, 1e-30)
    be = (N * math.log(max(N, 2.0)) + W * M) / max(
        Np * math.log(max(Np, 2.0)) + W * Ns + W * M, 1e-30)
    return dict(
        nprime=(n1, n2), Nprime=Np, rho=Np / max(N, 1),
        T_new_over_T_base=ratio, break_even_Lratio=be,
        W=W, gamma=gamma, beta=beta,
    )


def gpu_mem_GiB(device) -> float:
    if device.type != 'cuda':
        return 0.0
    return torch.cuda.get_device_properties(device).total_memory / 2 ** 30


def memory_ok(Lp: int, n_src: int, n_tgt: int, Nprime: int,
              gpu_gib: float, coils: int, headroom=0.20) -> dict:
    """Complex64 factors + one rank-batch FFT workspace. Conservative."""
    bytes_c64 = 8
    factors = Lp * (n_src + n_tgt) * bytes_c64
    fft_ws = Nprime * 16  # complex128 workspace bound
    batch = coils * Lp * n_src * bytes_c64
    peak = factors + fft_ws + batch
    budget = gpu_gib * (1 - headroom) * 2 ** 30
    return dict(
        factors_MiB=factors / 2 ** 20,
        fft_ws_MiB=fft_ws / 2 ** 20,
        batch_MiB=batch / 2 ** 20,
        peak_GiB=peak / 2 ** 30,
        gpu_GiB=gpu_gib,
        ok=peak <= budget,
    )


def pareto_mask(rms: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Nondominated: no other point has both smaller RMS and smaller S."""
    keep = np.ones(len(rms), dtype=bool)
    for i in range(len(rms)):
        better = (rms <= rms[i] + 1e-18) & (S <= S[i] + 1e-18)
        strictly = (rms < rms[i] - 1e-18) | (S < S[i] - 1e-18)
        if np.any(better & strictly):
            keep[i] = False
    return keep
