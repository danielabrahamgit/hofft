"""
Iteration 1 of residual-B0 from sharpness autofocus.

  R(ksp * exp(-j 2π f t)) for f in {-100, -90, ..., +100} Hz
  per-patch argmax_f of z-scored Crete blur-effect (skimage)
  weighted quadratic fit → db0(r)

Encoding matches sanity readout (M=4, first 15k) at full kmax grid
(~246³, all interleaves). ofs=46, origin=0, B0 time-seg (no HO SVD).

Run from repo root:
  srun --jobid=7996 --overlap env PYTHONUNBUFFERED=1 PYTHONPATH=src \\
    python data/coco_7t_spi/fit_db0.py --recompute
  python data/coco_7t_spi/fit_db0.py          # rescore cached imgs
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.filters import sobel

from mr_recon.imperfections.field import alpha_segementation
from mr_recon.multi_coil.calib import calc_coil_subspace
from mr_recon.recons import CG_SENSE_recon
from mr_recon.utils import gen_grd

from hofft.phase_coeffs import b0_to_phis_alphas, remove_linear_terms, sph_bases
from hofft.utils import reduce_spatial

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_params import (
    C_COMP, FOV, L_B0, M, N_GROUP, ROS,
    build_phis, default_signs, make_A,
)

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'fit_db0_iter1.pt'
FREQS = torch.arange(-100, 101, 10)  # Hz
PATCH = 32
STRIDE = 16
CG_ITERS = 5
MASK_FRAC = 0.3
OFS = 46
# Re-blur width. Must be << patch so the metric can still see extra blur.
MIN_CACHE_N = 200  # ignore the old 100³ sweep cache
H_SIZE = 7  # leftover Crete path only
FOCUS_SIGMA = 1.2
FOCUS_WIN = 9
FOCUS_STRUCT_Q = 0.40
FOCUS_PROM_MIN = 0.008
FOCUS_RATIO_Q = 0.80  # keep in-mask voxels at or above this max/min quantile
FOCUS_MAX_FIT = 80_000


def patch_starts(n: int, patch: int, stride: int) -> list[int]:
    starts = list(range(0, max(n - patch, 0) + 1, stride))
    if starts[-1] + patch < n:
        starts.append(n - patch)
    return starts


def poly_design(xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = xyz.unbind(-1)
    return torch.stack([
        torch.ones_like(x),
        x, y, z,
        x * x, y * y, z * z,
        x * y, x * z, y * z,
    ], dim=-1)


def recon_stack(device: torch.device):
    print(f'device={device}  freqs={FREQS.tolist()} Hz', flush=True)
    trj_full = torch.load(ROOT / 'trj.pt', weights_only=True, map_location=device)
    dcf_full = torch.load(ROOT / 'dcf.pt', weights_only=True, map_location=device)
    ksp = torch.load(ROOT / 'ksp.pt', weights_only=True, map_location=device)
    evals = torch.load(ROOT / 'evals.pt', weights_only=True, map_location=device)
    mps = torch.load(ROOT / 'mps.pt', weights_only=True, map_location=device)
    b0 = torch.load(ROOT / 'b0.pt', weights_only=True, map_location=device)
    pred_0 = torch.load(ROOT / 'pred_0.pt', weights_only=True, map_location=device)
    coco_term = torch.load(ROOT / 'coco_term.pt', weights_only=True, map_location=device)
    alphas_mm = torch.load(ROOT / 'alphas.pt', weights_only=True, mmap=True, map_location='cpu')
    T_full = trj_full.shape[0]

    mask = (evals > 0.95).float()
    _, ksp, mps = calc_coil_subspace(ksp[:, :10_000:4, :, ::10], C_COMP, ksp, mps)
    ksp = ksp * torch.exp(1j * pred_0.to(ksp.device))
    ksp = ksp * torch.exp(1j * coco_term.to(ksp.device))
    ksp = ksp[:, ROS, :N_GROUP].contiguous()
    dcf = dcf_full[ROS, :N_GROUP].contiguous()
    kmax = float(trj_full[ROS, :N_GROUP].abs().max())
    N_new = int(round(kmax) * 2)
    im_size = (N_new,) * 3
    mps = reduce_spatial(mps, im_size)
    mask = (reduce_spatial(mask, im_size) > 0.5).float()
    b0 = reduce_spatial(b0, im_size)
    dt = M * 1e-6
    n_ro = ksp.shape[1]
    t = torch.arange(n_ro, device=device, dtype=torch.float32) * dt
    t = t[:, None, None]

    sl = alphas_mm[:, OFS:OFS + T_full, :N_GROUP][:, ROS]
    al = sl.to(device=device, dtype=torch.float32).contiguous()
    origin = torch.zeros(3, device=device)
    sc, sco, ss = default_signs(device)
    phis = build_phis(im_size, origin, sc, sco, ss, device)
    _, trj_lin, zero = remove_linear_terms(phis, al, mask)
    ksp = ksp * torch.exp(2j * torch.pi * zero)
    phis_b0, alphas_b0 = b0_to_phis_alphas(
        -b0, tuple(dcf.shape), ro_dim=0, dt=dt, repeat_empty_dims=False)
    spat, temp, _ = alpha_segementation(
        phis_b0, alphas_b0, L=L_B0, interp_type='zero',
        method='maxmin', verbose=False)
    A, _nft = make_A(trj_lin, mps, dcf, spat, temp)
    print(f'R ready  ksp {tuple(ksp.shape)}  im {im_size}', flush=True)

    freqs = FREQS.to(device=device, dtype=torch.float32)
    imgs = []
    for f in freqs:
        ksp_f = ksp * torch.exp(-2j * torch.pi * f * t)
        img = CG_SENSE_recon(A, ksp_f, max_iter=CG_ITERS, max_eigen=1.0, verbose=False)
        imgs.append(img.detach())
        print(f'  f={float(f):+6.1f} Hz  ||img||={img.abs().mean():.3e}', flush=True)
    imgs = torch.stack(imgs, dim=0).abs()
    if hasattr(A, 'clear_plans'):
        A.clear_plans()
    return imgs.cpu(), mask.cpu(), b0.cpu()


def load_or_recon(device: torch.device):
    if CACHE.is_file() and '--recompute' not in sys.argv:
        cached = torch.load(CACHE, weights_only=True, map_location='cpu')
        n = int(cached['imgs'].shape[-1])
        if n >= MIN_CACHE_N:
            print(f'rescore from {CACHE}  im={tuple(cached["imgs"].shape[-3:])}', flush=True)
            b0 = cached.get('b0')
            if b0 is None:
                b0 = torch.load(ROOT / 'b0.pt', weights_only=True, map_location='cpu')
                b0 = reduce_spatial(b0, tuple(cached['imgs'].shape[-3:]))
            return cached['imgs'], cached['mask'], b0
        print(f'ignoring low-res cache im={n}³', flush=True)
    return recon_stack(device)


def save_freq_images(imgs: torch.Tensor, out_dir: Path):
    """Mid-z magnitude for every demod frequency: montage + one PNG each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    F = imgs.shape[0]
    mid = imgs.shape[-1] // 2
    sl = imgs[..., mid]
    vmax = float(torch.quantile(sl[F // 2], 0.995).clamp(min=1e-12))
    for i, f in enumerate(FREQS.tolist()):
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.imshow(sl[i].rot90(), cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(f'f = {f:+.0f} Hz')
        ax.axis('off')
        fig.tight_layout()
        path = out_dir / f'{i:02d}_f{f:+.0f}Hz.png'
        fig.savefig(path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'  wrote {path}', flush=True)

    ncols = 6
    nrows = (F + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows))
    axes = np.asarray(axes).ravel()
    for i, f in enumerate(FREQS.tolist()):
        axes[i].imshow(sl[i].rot90(), cmap='gray', vmin=0, vmax=vmax)
        axes[i].set_title(f'{f:+.0f} Hz', fontsize=10)
        axes[i].axis('off')
    for ax in axes[F:]:
        ax.axis('off')
    fig.suptitle('R(ksp exp(-j 2π f t)) mid-z', fontsize=12)
    fig.tight_layout()
    montage = out_dir / 'all_freqs_midz.png'
    fig.savefig(montage, dpi=140)
    plt.close(fig)
    print(f'  wrote {montage}', flush=True)


def sph_bases_upto(x, y, z, order: int = 5) -> torch.Tensor:
    """Real solid harmonics through ``order``. Shape ``((order+1)**2, *)``."""
    if order < 2 or order > 5:
        raise ValueError(f'order must be 2..5, got {order}')
    assert x.shape == y.shape == z.shape
    tup = (None,) + (slice(None),) * x.ndim
    x, y, z = x[tup], y[tup], z[tup]
    x2, y2, z2 = x * x, y * y, z * z
    r2 = x2 + y2 + z2
    x3, y3, z3 = x2 * x, y2 * y, z2 * z
    terms = [
        torch.ones_like(x), x, y, z,
        x * y, z * y, 3 * z2 - r2, x * z, x2 - y2,
        3 * y * x2 - y3, x * y * z, (5 * z2 - r2) * y,
        5 * z3 - 3 * z * r2, (5 * z2 - r2) * x, z * (x2 - y2), x3 - 3 * x * y2,
    ]
    if order >= 4:
        r4 = r2 * r2
        z4 = z2 * z2
        terms += [
            35 * z4 - 30 * z2 * r2 + 3 * r4,
            x * z * (7 * z2 - 3 * r2),
            y * z * (7 * z2 - 3 * r2),
            (x2 - y2) * (7 * z2 - r2),
            (2 * x * y) * (7 * z2 - r2),
            z * (x3 - 3 * x * y2),
            z * (3 * x2 * y - y3),
            x2 * x2 - 6 * x2 * y2 + y2 * y2,
            4 * x * y * (x2 - y2),
        ]
    if order >= 5:
        r4 = r2 * r2
        z4 = z2 * z2
        terms += [
            63 * z2 * z3 - 70 * z3 * r2 + 15 * z * r4,
            x * (21 * z4 - 14 * z2 * r2 + r4),
            y * (21 * z4 - 14 * z2 * r2 + r4),
            (x2 - y2) * z * (9 * z2 - r2),
            (2 * x * y) * z * (9 * z2 - r2),
            (x3 - 3 * x * y2) * (9 * z2 - r2),
            (3 * x2 * y - y3) * (9 * z2 - r2),
            z * (x2 * x2 - 6 * x2 * y2 + y2 * y2),
            z * (4 * x * y * (x2 - y2)),
            x2 * x3 - 10 * x3 * y2 + 5 * x * y2 * y2,
            5 * x2 * x2 * y - 10 * x2 * y3 + y2 * y3,
        ]
    return torch.cat(terms[: (order + 1) ** 2], dim=0)


def fit_sph_db0(cents, f_opt, keep, crds, weights, order: int = 5):
    """Weighted spherical-harmonic fit of patch f* → db0 [Hz]."""
    xyz = cents[keep]
    y = f_opt[keep]
    w = weights[keep].clamp(min=0)
    w = w / w.mean().clamp(min=1e-30)
    B = sph_bases_upto(xyz[..., 0], xyz[..., 1], xyz[..., 2], order=order)
    Afit = B.T * w[:, None]
    coef, *_ = torch.linalg.lstsq(Afit, y * w)
    Bvol = sph_bases_upto(crds[..., 0], crds[..., 1], crds[..., 2], order=order)
    db0 = (Bvol * coef.reshape((-1,) + (1,) * (Bvol.ndim - 1))).sum(0)
    return db0, coef


def hp_energy_ratio(vol: np.ndarray, sigma: float = FOCUS_SIGMA, win: int = FOCUS_WIN) -> np.ndarray:
    """Local high-pass energy / local intensity energy. Scale-invariant focus measure."""
    lp = gaussian_filter(vol, sigma=sigma, mode='nearest')
    hp = vol - lp
    num = uniform_filter(hp * hp, size=win, mode='nearest')
    den = uniform_filter(vol * vol, size=win, mode='nearest')
    return (num / np.maximum(den, 1e-20)).astype(np.float32)


def dense_focus_maps(imgs: torch.Tensor, mask: torch.Tensor):
    """Focus-stack: per-voxel argmax of high-pass energy ratio.

    Only keep voxels that are in the object, have structure, a clear peak
    (winner beats runner-up), and are not on the frequency wall.
    """
    vols = np.asarray(imgs, dtype=np.float32)
    F = vols.shape[0]
    f0 = int((FREQS == 0).nonzero(as_tuple=False).reshape(-1)[0]) if (FREQS == 0).any() else F // 2
    ref = vols[f0]
    edge2 = sobel(ref) ** 2
    e_den = uniform_filter(ref * ref, size=FOCUS_WIN, mode='nearest')
    struct = (uniform_filter(edge2, size=FOCUS_WIN, mode='nearest')
              / np.maximum(e_den, 1e-20)).astype(np.float32)

    best = np.full(ref.shape, -np.inf, dtype=np.float32)
    second = np.full(ref.shape, -np.inf, dtype=np.float32)
    worst = np.full(ref.shape, np.inf, dtype=np.float32)
    f_idx = np.zeros(ref.shape, dtype=np.int16)
    global_s = []
    m = np.asarray(mask) > 0.5
    for fi in range(F):
        s = hp_energy_ratio(vols[fi])
        global_s.append(float(s[m].mean()) if m.any() else float(s.mean()))
        better = s > best
        second = np.where(better, best, np.maximum(second, s))
        best = np.where(better, s, best)
        np.minimum(worst, s, out=worst)
        f_idx = np.where(better, fi, f_idx)
        print(f'  focus f={FREQS[fi]:+.0f} Hz  mean_in_mask={global_s[-1]:.5f}', flush=True)

    ratio = (best / np.maximum(worst, 1e-20)).astype(np.float32)
    freqs = FREQS.to(dtype=torch.float32).numpy()
    f_opt = torch.from_numpy(freqs[f_idx].astype(np.float32))
    prom = torch.from_numpy(((best - second) / np.maximum(best, 1e-12)).astype(np.float32))
    struct_t = torch.from_numpy(struct)
    f_idx_t = torch.from_numpy(f_idx.astype(np.int64))
    interior = (f_idx_t > 0) & (f_idx_t < F - 1)
    if m.any():
        rthr = float(np.quantile(ratio[m], FOCUS_RATIO_Q))
    else:
        rthr = 1.0
    keep = torch.from_numpy(m) & interior & (torch.from_numpy(ratio) >= rthr)
    aif = torch.from_numpy(np.take_along_axis(vols, f_idx[None], 0)[0])
    print(f'focus keep {int(keep.sum())}/{keep.numel()}  '
          f'max/min ratio >= {rthr:.3f} (q={FOCUS_RATIO_Q})', flush=True)
    print('kept f* counts:', flush=True)
    vals = f_opt[keep]
    for f in FREQS.tolist():
        print(f'  {f:+4.0f} Hz  {int((vals == f).sum())}', flush=True)
    return dict(f_opt=f_opt, keep=keep, prom=prom, struct=struct_t,
                aif=aif, f_idx=f_idx_t, global_s=global_s,
                ratio=torch.from_numpy(ratio), ratio_thr=rthr)


def fit_sph_from_dense(crds, f_opt, keep, weights, order: int = 5):
    """SH fit on a dense in-focus mask, subsampled if needed."""
    idx = keep.reshape(-1).nonzero(as_tuple=False).reshape(-1)
    w = weights.reshape(-1)[idx].clamp(min=0)
    if idx.numel() > FOCUS_MAX_FIT:
        p = (w / w.sum().clamp(min=1e-30)).double()
        sel = torch.multinomial(p, FOCUS_MAX_FIT, replacement=False)
        idx, w = idx[sel], w[sel]
        print(f'  SH fit on {int(idx.numel())} / {int(keep.sum())} voxels', flush=True)
    xyz = crds.reshape(-1, 3)[idx]
    y = f_opt.reshape(-1)[idx]
    w = w / w.mean().clamp(min=1e-30)
    B = sph_bases_upto(xyz[..., 0], xyz[..., 1], xyz[..., 2], order=order)
    Afit = B.T * w[:, None]
    coef, *_ = torch.linalg.lstsq(Afit, y * w)
    Bvol = sph_bases_upto(crds[..., 0], crds[..., 1], crds[..., 2], order=order)
    db0 = (Bvol * coef.reshape((-1,) + (1,) * (Bvol.ndim - 1))).sum(0)
    return db0, coef


def prep_b0_recon(device: torch.device, b0_map: torch.Tensor):
    """Same encoding as the f-sweep, with a caller-supplied B0 [Hz]."""
    trj_full = torch.load(ROOT / 'trj.pt', weights_only=True, map_location=device)
    dcf_full = torch.load(ROOT / 'dcf.pt', weights_only=True, map_location=device)
    ksp = torch.load(ROOT / 'ksp.pt', weights_only=True, map_location=device)
    evals = torch.load(ROOT / 'evals.pt', weights_only=True, map_location=device)
    mps = torch.load(ROOT / 'mps.pt', weights_only=True, map_location=device)
    pred_0 = torch.load(ROOT / 'pred_0.pt', weights_only=True, map_location=device)
    coco_term = torch.load(ROOT / 'coco_term.pt', weights_only=True, map_location=device)
    alphas_mm = torch.load(ROOT / 'alphas.pt', weights_only=True, mmap=True, map_location='cpu')
    T_full = trj_full.shape[0]

    mask = (evals > 0.95).float()
    _, ksp, mps = calc_coil_subspace(ksp[:, :10_000:4, :, ::10], C_COMP, ksp, mps)
    ksp = ksp * torch.exp(1j * pred_0.to(ksp.device))
    ksp = ksp * torch.exp(1j * coco_term.to(ksp.device))
    ksp = ksp[:, ROS, :N_GROUP].contiguous()
    dcf = dcf_full[ROS, :N_GROUP].contiguous()
    kmax = float(trj_full[ROS, :N_GROUP].abs().max())
    im_size = (int(round(kmax) * 2),) * 3
    mps = reduce_spatial(mps, im_size)
    mask = (reduce_spatial(mask, im_size) > 0.5).float()
    b0_map = b0_map.to(device)
    if tuple(b0_map.shape) != im_size:
        b0_map = reduce_spatial(b0_map, im_size)
    dt = M * 1e-6

    sl = alphas_mm[:, OFS:OFS + T_full, :N_GROUP][:, ROS]
    al = sl.to(device=device, dtype=torch.float32).contiguous()
    origin = torch.zeros(3, device=device)
    sc, sco, ss = default_signs(device)
    phis = build_phis(im_size, origin, sc, sco, ss, device)
    _, trj_lin, zero = remove_linear_terms(phis, al, mask)
    ksp = ksp * torch.exp(2j * torch.pi * zero)
    phis_b0, alphas_b0 = b0_to_phis_alphas(
        -b0_map, tuple(dcf.shape), ro_dim=0, dt=dt, repeat_empty_dims=False)
    spat, temp, _ = alpha_segementation(
        phis_b0, alphas_b0, L=L_B0, interp_type='zero',
        method='maxmin', verbose=False)
    A, nft = make_A(trj_lin, mps, dcf, spat, temp)
    print(f'A ready  ksp {tuple(ksp.shape)}  im {im_size}  '
          f'b0[{float(b0_map.min()):+.1f},{float(b0_map.max()):+.1f}] Hz', flush=True)
    return A, nft, ksp, mask, b0_map


def apply_sh_and_recon():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cached = torch.load(CACHE, weights_only=True, map_location='cpu')
    imgs = cached['imgs']
    mask = cached['mask']
    b0 = cached['b0']
    im_size = tuple(imgs.shape[-3:])
    crds = gen_grd(im_size) * FOV
    print('dense focus (HP energy ratio; fit on high max/min voxels)', flush=True)
    foc = dense_focus_maps(imgs, mask)
    f_opt, keep = foc['f_opt'], foc['keep']
    w = (foc['ratio'] - 1.0).clamp(min=0)
    order = 5
    db0, coef = fit_sph_from_dense(crds, f_opt, keep, w, order=order)
    print(f'sph{order} n_terms={coef.numel()}  '
          f'coef[0:4]={" ".join(f"{float(c):+.3g}" for c in coef[:4])}', flush=True)
    print('db0 Hz in mask: '
          f'mean={float(db0[mask > 0.5].mean()):+.2f}  '
          f'|mean|/max = {float(db0[mask > 0.5].abs().mean()):.2f}/'
          f'{float(db0[mask > 0.5].abs().max()):.2f}', flush=True)

    # Same support as the original B0: do not paint SH into the background.
    support = mask > 0.5
    b0_new = torch.where(support, b0 + db0, b0)
    db0 = torch.where(support, db0, torch.zeros_like(db0))
    torch.save(dict(db0=db0, b0=b0, b0_new=b0_new, coef=coef, order=order,
                    keep=keep.cpu(), ratio=foc['ratio'].cpu(), mask=mask),
               ROOT / f'fit_db0_sh{order}.pt')
    if device.type == 'cuda':
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    A, nft, ksp, mask_d, _ = prep_b0_recon(device, b0_new)
    img_new = CG_SENSE_recon(A, ksp, max_iter=CG_ITERS, max_eigen=1.0, verbose=False)
    img_new = img_new.detach().abs().cpu()
    if hasattr(A, 'clear_plans'):
        A.clear_plans()
    del A, nft, ksp
    print(f'new recon ||img||={float(img_new.mean()):.3e}', flush=True)

    f0 = int((FREQS == 0).nonzero(as_tuple=False).reshape(-1)[0])
    img_old = imgs[f0]
    n2 = im_size[2]
    zs = [n2 // 2 - 20, n2 // 2, n2 // 2 + 20]
    flim = float(FREQS.abs().max())
    mid = n2 // 2
    f_show = f_opt.clone()
    f_show[~keep] = float('nan')
    fig_f, ax_f = plt.subplots(1, 4, figsize=(14, 3.6))
    ax_f[0].imshow(img_old[..., mid].rot90(), cmap='gray')
    ax_f[0].set_title('R(f=0)')
    rshow = foc['ratio'].clone()
    rshow[mask <= 0.5] = float('nan')
    im_r = ax_f[1].imshow(rshow[..., mid].rot90(), cmap='magma', vmin=1.0,
                          vmax=float(torch.quantile(foc['ratio'][mask > 0.5], 0.99)))
    ax_f[1].set_title('max/min metric')
    plt.colorbar(im_r, ax=ax_f[1], fraction=0.046)
    im_fs = ax_f[2].imshow(f_show[..., mid].rot90(), cmap='RdBu_r', vmin=-flim, vmax=flim)
    ax_f[2].set_title(f'f*  (ratio≥{foc["ratio_thr"]:.2f})')
    plt.colorbar(im_fs, ax=ax_f[2], fraction=0.046)
    ax_f[3].imshow(keep[..., mid].rot90(), cmap='gray')
    ax_f[3].set_title('fit mask')
    for ax in ax_f:
        ax.axis('off')
    fig_f.tight_layout()
    focus_path = ROOT / 'fit_db0_focus.png'
    fig_f.savefig(focus_path, dpi=140)
    plt.close(fig_f)
    print(f'wrote {focus_path}', flush=True)
    vmax = float(torch.quantile(img_old[..., n2 // 2], 0.995).clamp(min=1e-12))
    flim = float(FREQS.abs().max())
    b0lim = float(torch.quantile(b0[mask > 0.5].abs(), 0.99).clamp(min=1))

    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    for row, z in enumerate(zs):
        axes[row, 0].imshow(img_old[..., z].rot90(), cmap='gray', vmin=0, vmax=vmax)
        axes[row, 1].imshow(img_new[..., z].rot90(), cmap='gray', vmin=0, vmax=vmax)
        axes[row, 2].imshow((img_new[..., z] - img_old[..., z]).rot90(),
                            cmap='RdBu_r', vmin=-0.15 * vmax, vmax=0.15 * vmax)
        im_d = axes[row, 3].imshow(db0[..., z].rot90(), cmap='RdBu_r', vmin=-flim, vmax=flim)
        im_b = axes[row, 4].imshow(b0_new[..., z].rot90(), cmap='RdBu_r',
                                   vmin=-b0lim, vmax=b0lim)
        for ax in axes[row]:
            ax.axis('off')
        axes[row, 0].set_ylabel(f'z={z}', fontsize=10)
    axes[0, 0].set_title('old B0  (f=0)')
    axes[0, 1].set_title(f'new B0 = old + sph{order} db0')
    axes[0, 2].set_title('new − old')
    axes[0, 3].set_title(f'sph{order} db0 [Hz]')
    axes[0, 4].set_title('updated B0 [Hz]')
    plt.colorbar(im_d, ax=axes[:, 3], fraction=0.05, shrink=0.8)
    plt.colorbar(im_b, ax=axes[:, 4], fraction=0.05, shrink=0.8)
    fig.tight_layout()
    fig_path = ROOT / f'fit_db0_sh{order}.png'
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    mid = n2 // 2
    for name, vol in [('old_b0', img_old), ('new_b0', img_new)]:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.imshow(vol[..., mid].rot90(), cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(name.replace('_', ' '))
        ax.axis('off')
        fig.tight_layout()
        p = ROOT / 'fit_db0_freqs' / f'{name}_midz.png'
        fig.savefig(p, dpi=140, bbox_inches='tight')
        plt.close(fig)

    torch.save(dict(
        db0=db0, b0=b0, b0_new=b0_new, coef=coef, order=order,
        img_old=img_old, img_new=img_new, mask=mask,
    ), ROOT / f'fit_db0_sh{order}.pt')
    print(f'wrote {fig_path}', flush=True)


def recon_freq_stack(device: torch.device, b0_map: torch.Tensor):
    """21-frequency CG stack with the given B0 [Hz]."""
    A, nft, ksp, mask, b0_d = prep_b0_recon(device, b0_map)
    dt = M * 1e-6
    t = torch.arange(ksp.shape[1], device=ksp.device, dtype=torch.float32) * dt
    t = t[:, None, None]
    freqs = FREQS.to(device=ksp.device, dtype=torch.float32)
    imgs = []
    for f in freqs:
        img = CG_SENSE_recon(
            A, ksp * torch.exp(-2j * torch.pi * f * t),
            max_iter=CG_ITERS, max_eigen=1.0, verbose=False)
        imgs.append(img.detach().abs().cpu())
        print(f'  f={float(f):+6.1f} Hz  ||img||={img.abs().mean():.3e}', flush=True)
    if hasattr(A, 'clear_plans'):
        A.clear_plans()
    del A, nft, ksp
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return torch.stack(imgs, dim=0), mask.cpu(), b0_d.cpu()


def save_loop_figures(out_dir: Path, it: int, imgs, mask, b0, b0_new, db0,
                      foc, img_f0):
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = imgs.shape[-1] // 2
    flim = float(FREQS.abs().max())
    f_show = foc['f_opt'].clone()
    f_show[~foc['keep']] = float('nan')
    rshow = foc['ratio'].clone()
    rshow[mask <= 0.5] = float('nan')
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    ax[0].imshow(img_f0[..., mid].rot90(), cmap='gray')
    ax[0].set_title(f'iter {it}  R(f=0)')
    im_r = ax[1].imshow(rshow[..., mid].rot90(), cmap='magma', vmin=1.0,
                        vmax=float(torch.quantile(foc['ratio'][mask > 0.5], 0.99)))
    ax[1].set_title('max/min metric')
    plt.colorbar(im_r, ax=ax[1], fraction=0.046)
    im_f = ax[2].imshow(f_show[..., mid].rot90(), cmap='RdBu_r', vmin=-flim, vmax=flim)
    ax[2].set_title(f'f*  (ratio≥{foc["ratio_thr"]:.2f})')
    plt.colorbar(im_f, ax=ax[2], fraction=0.046)
    ax[3].imshow(foc['keep'][..., mid].rot90(), cmap='gray')
    ax[3].set_title('fit mask')
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(out_dir / f'{it:02d}_focus.png', dpi=140)
    plt.close(fig)

    zs = [mid - 20, mid, mid + 20]
    vmax = float(torch.quantile(img_f0[..., mid], 0.995).clamp(min=1e-12))
    b0lim = float(torch.quantile(b0[mask > 0.5].abs(), 0.99).clamp(min=1))
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for row, z in enumerate(zs):
        axes[row, 0].imshow(img_f0[..., z].rot90(), cmap='gray', vmin=0, vmax=vmax)
        axes[row, 1].imshow(db0[..., z].rot90(), cmap='RdBu_r', vmin=-flim, vmax=flim)
        axes[row, 2].imshow(b0[..., z].rot90(), cmap='RdBu_r', vmin=-b0lim, vmax=b0lim)
        axes[row, 3].imshow(b0_new[..., z].rot90(), cmap='RdBu_r', vmin=-b0lim, vmax=b0lim)
        for a in axes[row]:
            a.axis('off')
        axes[row, 0].set_ylabel(f'z={z}')
    axes[0, 0].set_title(f'iter {it}  f=0')
    axes[0, 1].set_title('sph5 db0 [Hz]')
    axes[0, 2].set_title('B0 in')
    axes[0, 3].set_title('B0 out')
    fig.tight_layout()
    fig.savefig(out_dir / f'{it:02d}_db0.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(img_f0[..., mid].rot90(), cmap='gray', vmin=0, vmax=vmax)
    ax.set_title(f'iter {it}  f=0')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_dir / f'{it:02d}_f0_midz.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def iterate_autofocus(n_iter: int = 5):
    """Feed updated B0 back in; sweep / high-ratio sph5 fit / update, ``n_iter`` times."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = ROOT / 'fit_db0_sh5.pt'
    if seed.is_file():
        prev = torch.load(seed, weights_only=True, map_location='cpu')
        b0 = prev['b0_new'].cpu()
        print(f'seed B0 from {seed}  '
              f'[{float(b0.min()):+.1f},{float(b0.max()):+.1f}] Hz', flush=True)
    else:
        b0 = torch.load(ROOT / 'b0.pt', weights_only=True, map_location='cpu')
        print('seed B0 from b0.pt', flush=True)

    out_dir = ROOT / 'fit_db0_loop'
    out_dir.mkdir(parents=True, exist_ok=True)
    f0 = int((FREQS == 0).nonzero(as_tuple=False).reshape(-1)[0])
    history = []
    for it in range(1, n_iter + 1):
        print(f'\n======== autofocus iter {it}/{n_iter} ========', flush=True)
        if device.type == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()
        imgs, mask, b0_used = recon_freq_stack(device, b0)
        if tuple(b0.shape) != tuple(mask.shape):
            b0 = reduce_spatial(b0, tuple(mask.shape))
        crds = gen_grd(tuple(mask.shape)) * FOV
        foc = dense_focus_maps(imgs, mask)
        w = (foc['ratio'] - 1.0).clamp(min=0)
        db0, coef = fit_sph_from_dense(crds, foc['f_opt'], foc['keep'], w, order=5)
        support = mask > 0.5
        db0 = torch.where(support, db0, torch.zeros_like(db0))
        b0_new = torch.where(support, b0 + db0, b0)
        stats = dict(
            iter=it,
            n_keep=int(foc['keep'].sum()),
            ratio_thr=float(foc['ratio_thr']),
            db0_mean=float(db0[support].mean()),
            db0_mean_abs=float(db0[support].abs().mean()),
            db0_max_abs=float(db0[support].abs().max()),
            global_s=foc['global_s'],
        )
        print(f'iter {it}  db0 mean={stats["db0_mean"]:+.2f}  '
              f'|mean|/max={stats["db0_mean_abs"]:.2f}/{stats["db0_max_abs"]:.2f}  '
              f'keep={stats["n_keep"]}  ratio≥{stats["ratio_thr"]:.2f}', flush=True)
        save_loop_figures(out_dir, it, imgs, mask, b0, b0_new, db0, foc, imgs[f0])
        torch.save(dict(
            b0=b0, db0=db0, b0_new=b0_new, coef=coef,
            img_f0=imgs[f0], mask=mask, keep=foc['keep'],
            ratio=foc['ratio'], f_opt=foc['f_opt'],
        ), out_dir / f'{it:02d}.pt')
        history.append(stats)
        (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
        b0 = b0_new
        del imgs, foc
        gc.collect()

    mid_imgs = []
    for it in range(1, n_iter + 1):
        d = torch.load(out_dir / f'{it:02d}.pt', weights_only=True, map_location='cpu')
        mid_imgs.append(d['img_f0'][..., d['img_f0'].shape[-1] // 2])
    vmax = float(torch.quantile(mid_imgs[0], 0.995).clamp(min=1e-12))
    fig, axes = plt.subplots(1, n_iter, figsize=(3.2 * n_iter, 3.4))
    if n_iter == 1:
        axes = [axes]
    for it, ax, sl in zip(range(1, n_iter + 1), axes, mid_imgs):
        ax.imshow(sl.rot90(), cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(f'iter {it}')
        ax.axis('off')
    fig.suptitle('f=0 after each autofocus update')
    fig.tight_layout()
    fig.savefig(out_dir / 'summary_f0.png', dpi=140)
    plt.close(fig)
    torch.save(dict(b0=b0, history=history), out_dir / 'b0_final.pt')
    print(f'wrote {out_dir / "summary_f0.png"}', flush=True)


def main():
    if '--iterate' in sys.argv:
        n = 5
        for a in sys.argv:
            if a.startswith('--n='):
                n = int(a.split('=', 1)[1])
        iterate_autofocus(n)
        return
    if '--apply-sh' in sys.argv:
        apply_sh_and_recon()
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgs, mask, b0 = load_or_recon(device)
    freqs = FREQS.to(dtype=torch.float32)
    im_size = tuple(imgs.shape[-3:])
    print(f'freqs={FREQS.tolist()} Hz  metric=z-scored Crete blur_effect  '
          f'h_size={H_SIZE}  im={im_size}', flush=True)

    n0, n1, n2 = im_size
    s0 = patch_starts(n0, PATCH, STRIDE)
    s1 = patch_starts(n1, PATCH, STRIDE)
    s2 = patch_starts(n2, PATCH, STRIDE)
    P = (len(s0), len(s1), len(s2))
    F = imgs.shape[0]
    imgs_np = imgs.numpy()
    sharp = torch.zeros((F, *P), dtype=torch.float32)
    mfrac = torch.zeros(P, dtype=torch.float32)
    for i, i0 in enumerate(s0):
        for j, j0 in enumerate(s1):
            for k, k0 in enumerate(s2):
                mfrac[i, j, k] = mask[i0:i0 + PATCH, j0:j0 + PATCH, k0:k0 + PATCH].mean()
                if float(mfrac[i, j, k]) < MASK_FRAC:
                    continue
                for fi in range(F):
                    sharp[fi, i, j, k] = patch_sharpness(
                        imgs_np[fi, i0:i0 + PATCH, j0:j0 + PATCH, k0:k0 + PATCH])
        print(f'  scored patch plane {i + 1}/{len(s0)}', flush=True)

    f_idx = sharp.argmax(dim=0)
    f_opt = freqs[f_idx]
    prom = sharp.max(dim=0).values - sharp.mean(dim=0)
    interior = (f_idx > 0) & (f_idx < F - 1)
    keep = (mfrac >= MASK_FRAC) & (prom > 0) & interior
    print(f'patches {P}  kept {int(keep.sum())}/{keep.numel()} '
          f'(dropped walls / empty)', flush=True)
    vals = f_opt[keep]
    print('kept f* counts:', flush=True)
    for f in FREQS.tolist():
        print(f'  {f:+4.0f} Hz  {int((vals == f).sum())}', flush=True)

    # Global curve on the masked mid-volume (same metric, one volume per f).
    mnp = mask.numpy() > 0.5
    print('global 1-blur_effect (z-scored, masked bbox):', flush=True)
    # tight bbox around mask to keep this cheap
    inz = np.where(mnp)
    slb = tuple(slice(int(a.min()), int(a.max()) + 1) for a in inz)
    for fi, f in enumerate(FREQS.tolist()):
        vol = imgs_np[fi][slb]
        print(f'  {f:+4.0f} Hz  {patch_sharpness(vol, h_size=11):.5f}', flush=True)

    crds = gen_grd(im_size) * FOV
    cents = []
    for i, i0 in enumerate(s0):
        for j, j0 in enumerate(s1):
            for k, k0 in enumerate(s2):
                cents.append(crds[i0 + PATCH // 2, j0 + PATCH // 2, k0 + PATCH // 2])
    cents = torch.stack(cents, dim=0).reshape(*P, 3)
    xyz = cents[keep]
    y = f_opt[keep]
    w = (prom[keep] * mfrac[keep]).clamp(min=0)
    w = w / w.mean().clamp(min=1e-30)
    Afit = poly_design(xyz) * w[:, None]
    rhs = y * w
    coef, *_ = torch.linalg.lstsq(Afit, rhs)
    db0 = (poly_design(crds) * coef).sum(dim=-1)
    print('quadratic db0 Hz: '
          f'c={float(coef[0]):+.2f}  '
          f'lin=({float(coef[1]):+.2f},{float(coef[2]):+.2f},{float(coef[3]):+.2f})  '
          f'|db0| mean/max in mask = '
          f'{float(db0[mask > 0.5].abs().mean()):.2f}/'
          f'{float(db0[mask > 0.5].abs().max()):.2f}',
          flush=True)

    f_vol = torch.full(im_size, float('nan'))
    for i, i0 in enumerate(s0):
        for j, j0 in enumerate(s1):
            for k, k0 in enumerate(s2):
                if not bool(keep[i, j, k]):
                    continue
                f_vol[i0:i0 + PATCH, j0:j0 + PATCH, k0:k0 + PATCH] = f_opt[i, j, k]

    mid = n2 // 2
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    f0 = int((FREQS == 0).nonzero(as_tuple=False).reshape(-1)[0]) if (FREQS == 0).any() else F // 2
    mag0 = imgs[f0, ..., mid]
    axes[0].imshow(mag0.rot90(), cmap='gray')
    axes[0].set_title('R(f=0) mid-z')
    flim = float(FREQS.abs().max())
    im1 = axes[1].imshow(f_vol[..., mid].rot90(), cmap='RdBu_r', vmin=-flim, vmax=flim)
    axes[1].set_title('per-patch f* [Hz]')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(db0[..., mid].rot90(), cmap='RdBu_r', vmin=-flim, vmax=flim)
    axes[2].set_title('quadratic db0 [Hz]')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[3].imshow((b0[..., mid] / (2 * b0.abs().max().clamp(min=1e-6))).rot90(),
                   cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[3].set_title('current b0 (scaled)')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig_path = ROOT / 'fit_db0_iter1.png'
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    payload = dict(
        freqs=[int(f) for f in FREQS],
        metric='skimage.measure.blur_effect (Crete 2007), z-scored, 1-blur',
        h_size=H_SIZE,
        im_size=list(im_size),
        patch=PATCH, stride=STRIDE,
        n_kept=int(keep.sum()),
        coef=[float(c) for c in coef],
        db0_mean_abs=float(db0[mask > 0.5].abs().mean()),
        db0_max_abs=float(db0[mask > 0.5].abs().max()),
    )
    (ROOT / 'fit_db0_iter1.json').write_text(json.dumps(payload, indent=2))
    torch.save(dict(
        imgs=imgs, f_opt=f_opt, keep=keep, sharp=sharp,
        db0=db0, mask=mask, b0=b0, coef=coef,
        f_vol=f_vol, cents=cents,
    ), CACHE)
    print(f'wrote {fig_path}', flush=True)
    save_freq_images(imgs, ROOT / 'fit_db0_freqs')


if __name__ == '__main__':
    main()
