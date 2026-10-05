"""
Demo figure: linear k-space vs high-order α-space on tilt_spi_invivo
(one shot of the multi-shot spiral).

One cohesive two-panel figure (k-space | α-space), plus two phase
snapshots at a single readout time (``t_show`` samples, ``dt_ms`` ms each).
Set ``save = True`` to write PNG/PDF under paper_experiments/figure_hop/.

  python paper_experiments/figure_hop.py
"""
import numpy as np
import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from einops import einsum

from hofft.phase_coeffs import rescale_phis_alphas, whiten_phis_alphas, remove_linear_terms
from hofft.utils import gen_grd

fpath = './data/tilt_spi_invivo'
shot = 0
t_window = 23_000
t_show = 14_630
dt_ms = 1 / 1000  # ms per readout sample
save = False
out_dir = Path('./paper_experiments/figure_hop')
if save:
    out_dir.mkdir(parents=True, exist_ok=True)

# Stacking from data/tilt_spi/gen_data.py: [b0, coco (4), kspha]
I_B0 = slice(0, 1)
I_COCO = slice(1, 5)
I_KSPHA = slice(5, None)

kwargs = {'weights_only': True, 'map_location': 'cpu'}
phis = torch.load(f'{fpath}/phis.pt', **kwargs).float()
alphas = torch.load(f'{fpath}/alphas.pt', **kwargs).float()[:, :, shot]
trj = torch.load(f'{fpath}/trj.pt', **kwargs).float()[:, shot]
evals = torch.load(f'{fpath}/evals.pt', **kwargs).float()
mask = (evals > 0.998).float()
im_size = phis.shape[1:]
crds = gen_grd(im_size).rot90(dims=[0,1])

# Strip spatially linear (and DC) terms from every basis before any SVD.
phis, trj_term, zeroth_order = remove_linear_terms(phis, alphas, mask=mask)


def _live(phis_g, alphas_g, thresh=1e-8):
    energy = (phis_g.reshape(phis_g.shape[0], -1).abs().mean(1)
              * alphas_g.reshape(alphas_g.shape[0], -1).abs().mean(1))
    keep = energy > thresh
    if not bool(keep.any()):
        return phis_g[:1], alphas_g[:1]
    return phis_g[keep], alphas_g[keep]


def compress_group(phis_g: torch.Tensor, alphas_g: torch.Tensor, rank: int = 1):
    """Rank-`rank` SVD of a coefficient group (no-op if already that size)."""
    phis_g, alphas_g = _live(phis_g, alphas_g)
    if phis_g.shape[0] <= rank:
        return phis_g, alphas_g
    return whiten_phis_alphas(phis_g, alphas_g, B_compressed=rank)


# B0 stays 1 basis; coco and eddy each collapse to their leading SVD mode.
phis_b0, alphas_b0 = compress_group(phis[I_B0], alphas[I_B0], 1)
phis_coco, alphas_coco = compress_group(phis[I_COCO], alphas[I_COCO], 1)
phis_eddy, alphas_eddy = compress_group(phis[I_KSPHA], alphas[I_KSPHA], 1)
alphas_b0 = alphas_b0 - alphas_b0[:, :1].clone()
phis = torch.cat([phis_b0, phis_coco, phis_eddy], dim=0)
alphas = torch.cat([alphas_b0, alphas_coco, alphas_eddy], dim=0)
print(f'compressed HO bases {tuple(phis.shape)}  (B0, coco, eddy)')

# Spatial maps in [-1/2, 1/2]; α traces in wraps.
phis_nrm, phis0, alphas_nrm, alphas0 = rescale_phis_alphas(phis, alphas, mask=mask,
                                                 quantiles=(0.0, 1.0))
for b in range(phis_nrm.shape[0]):
    alphas_nrm[b] = alphas_nrm[b] + alphas0[b]

groups = {
    'b0': {'idx': [0], 'ls': '--', 'label': r'$\alpha_{B_0}(t)$'},
    'coco': {'idx': [1], 'ls': ':', 'label': r'$\alpha_{\mathrm{coco}}(t)$'},
    'eddy': {'idx': [2], 'ls': '-', 'label': r'$\alpha_{\mathrm{eddy}}(t)$'},
}

# One readout sample so k·r has few wraps.
t_idx = [min(t_show, t_window - 1)]
t_ms = [i * dt_ms for i in t_idx]
print(f'time sample {t_idx[0]}  ({t_ms[0]:.3f} ms)')

# Crop to the object support.
slc = (slice(0, -50), slice(12, -22))


def cycles_group(idx, t):
    if len(idx) == 0:
        return torch.zeros(im_size)
    return einsum(phis[idx], alphas[idx, t], 'B ..., B -> ...')


def cycles_kx(t):
    return trj[t, 0] * crds[..., 0]


def cycles_ky(t):
    return trj[t, 1] * crds[..., 1]


def cycles_ktrj(t):
    return cycles_kx(t) + cycles_ky(t)


def cycles_ho(t):
    return einsum(phis, alphas[:, t], 'B ..., B -> ...')


def wrapped(cycles: torch.Tensor) -> torch.Tensor:
    ang = torch.angle(torch.exp(-2j * torch.pi * cycles))
    return ang.masked_fill(mask < 0.5, float('nan'))


def masked_basis(img: torch.Tensor) -> torch.Tensor:
    return img.masked_fill(mask < 0.5, float('nan'))[slc].cpu()


def colored_line(ax, x, y, t, cmap, norm, lw=1.1):
    pts = torch.stack([x, y], dim=-1).cpu().numpy()
    segments = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=lw)
    lc.set_array(t[:-1].cpu().numpy())
    ax.add_collection(lc)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    return lc


def colored_line_3d(ax, x, y, z, t, cmap, norm, lw=1.1):
    pts = torch.stack([x, y, z], dim=-1).cpu().numpy()
    segments = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=lw)
    lc.set_array(t[:-1].cpu().numpy())
    ax.add_collection(lc)
    mid = torch.stack([x.mean(), y.mean(), z.mean()])
    half = 0.5 * torch.stack([
        x.max() - x.min(), y.max() - y.min(), z.max() - z.min(),
    ]).max().item()
    half = half * 1.05 if half > 0 else 1.0
    ax.set_xlim(float(mid[0] - half), float(mid[0] + half))
    ax.set_ylim(float(mid[1] - half), float(mid[1] + half))
    ax.set_zlim(float(mid[2] - half), float(mid[2] + half))
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    return lc


def mark_times(ax, t_ms, colors, ys_at_t):
    for t, c, ys in zip(t_ms, colors, ys_at_t):
        ys = torch.as_tensor(ys).reshape(-1).cpu()
        ax.scatter(torch.full((ys.numel(),), float(t)), ys,
                   marker='x', color=c, s=90, linewidths=2.2, zorder=5)


def style_trace_ax(ax):
    ax.tick_params(colors='k', width=2.0, length=5, labelsize=14)
    for lab in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        lab.set_fontweight('heavy')
        lab.set_color('k')
    ax.xaxis.label.set(fontweight='heavy', color='k', fontsize=16)
    ax.yaxis.label.set(fontweight='heavy', color='k', fontsize=16)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # ax.legend(loc='upper left', frameon=False, fontsize=15, ncol=1,
    #           handlelength=2.8, labelcolor='k',
    #           prop={'weight': 'heavy', 'size': 15})

phase_cmap = plt.cm.jet.copy()
phase_cmap.set_bad('white')
phi_cmap = plt.cm.RdBu_r.copy()
phi_cmap.set_bad('white')
time_cmap = plt.cm.plasma
# time_cmap = plt.cm.turbo
time_norm = mpl.colors.Normalize(vmin=0, vmax=(t_window - 1) * dt_ms)
t_colors = [time_cmap(time_norm(t)) for t in t_ms]

rows = [
    ('kx', lambda t: wrapped(cycles_kx(t))),
    ('ky', lambda t: wrapped(cycles_ky(t))),
    ('kspace', lambda t: wrapped(cycles_ktrj(t))),
    ('b0', lambda t: wrapped(cycles_group(groups['b0']['idx'], t))),
    ('coco', lambda t: wrapped(cycles_group(groups['coco']['idx'], t))),
    ('eddy', lambda t: wrapped(cycles_group(groups['eddy']['idx'], t))),
    ('ho', lambda t: wrapped(cycles_ho(t))),
]

mpl.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 1.2,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


def save_fig(fig, name, pad_inches=0.02):
    if not save:
        return
    png = out_dir / f'{name}.png'
    pdf = out_dir / f'{name}.pdf'
    fig.savefig(png, dpi=180, bbox_inches='tight', pad_inches=pad_inches)
    fig.savefig(pdf, bbox_inches='tight', pad_inches=pad_inches)
    print(f'Saved {png}')


def bare_imshow(ax, img, cmap, vmin, vmax):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


# =====================================================================
# k-space | α-space
# =====================================================================
fig = plt.figure(figsize=(16.2, 11.8))
gs = fig.add_gridspec(
    3, 2,
    height_ratios=[1.05, 1.25, 1.35],
    width_ratios=[2, 3],
    hspace=0.32, wspace=0.22,
    left=0.07, right=0.90, top=0.93, bottom=0.06,
)

fig.text(0.26, 0.97, r'$k$-space', ha='center', va='top',
         fontsize=16, fontweight='heavy')
fig.text(0.64, 0.97, r'high-order  $\alpha$-space', ha='center', va='top',
         fontsize=16, fontweight='heavy')

# ----- spatial bases -----
k_maps = [(r'$r_x$', crds[..., 0]), (r'$r_y$', crds[..., 1])]
a_maps = [
    (r'$\varphi_1$', phis_nrm[0]),
    (r'$\varphi_2$', phis_nrm[1]),
    (r'$\varphi_3$', phis_nrm[2]),
]
gs_km = gs[0, 0].subgridspec(1, 2, wspace=0.08)
gs_am = gs[0, 1].subgridspec(1, 3, wspace=0.08)
im_phi = None
for j, (label, img) in enumerate(k_maps):
    ax = fig.add_subplot(gs_km[0, j])
    im_phi = bare_imshow(ax, masked_basis(img), phi_cmap, -0.5, 0.5)
    ax.set_title(label, fontweight='heavy', fontsize=13, pad=4)
for j, (label, img) in enumerate(a_maps):
    ax = fig.add_subplot(gs_am[0, j])
    im_phi = bare_imshow(ax, masked_basis(img), phi_cmap, -0.5, 0.5)
    ax.set_title(label, fontweight='heavy', fontsize=13, pad=4)

cax_phi = fig.add_axes([0.915, 0.70, 0.012, 0.20])
cb_phi = fig.colorbar(im_phi, cax=cax_phi)
cb_phi.set_ticks([-0.5, 0.0, 0.5])
cb_phi.set_label(r'$\varphi(\mathbf{r})$', fontsize=11, fontweight='heavy')

# ----- coefficient traces -----
ts = torch.arange(t_window) * dt_ms
ax_k = fig.add_subplot(gs[1, 0])
ax_a = fig.add_subplot(gs[1, 1])

ax_k.plot(ts, trj[:t_window, 0].cpu(), color='k', lw=3.5, ls='-', label=r'$k_x(t)$')
ax_k.plot(ts, trj[:t_window, 1].cpu(), color='k', lw=3.5, ls='--', label=r'$k_y(t)$')
ax_k.axhline(0.0, color='k', lw=0.8, zorder=0)
ax_k.set_xlim(0, (t_window - 1) * dt_ms)
ax_k.set_xlabel('time [ms]')
ax_k.set_ylabel(r'$k(t)$  [cycles]')
mark_times(ax_k, t_ms, t_colors, [trj[t, :2] for t in t_idx])
style_trace_ax(ax_k)

for spec in groups.values():
    b = spec['idx'][0]
    ax_a.plot(ts, alphas_nrm[b, :t_window].cpu(),
              color='k', lw=3.5, ls=spec['ls'], label=spec['label'])
ax_a.axhline(0.0, color='k', lw=0.8, zorder=0)
ax_a.set_xlim(0, (t_window - 1) * dt_ms)
ax_a.set_xlabel('time [ms]')
ax_a.set_ylabel(r'$\alpha(t)$  [wraps]')
mark_times(ax_a, t_ms, t_colors, [alphas_nrm[:, t] for t in t_idx])
style_trace_ax(ax_a)

# ----- trajectories colored by time -----
ax_ktr = fig.add_subplot(gs[2, 0])
ax_al = fig.add_subplot(gs[2, 1], projection='3d')

kx = trj[:t_window, 0]
ky = trj[:t_window, 1]
colored_line(ax_ktr, kx, ky, ts, time_cmap, time_norm, lw=1.15)
for t, c in zip(t_idx, t_colors):
    ax_ktr.scatter(trj[t, 0], trj[t, 1], marker='x', color=c, s=90,
                   linewidths=2.2, zorder=4)
ax_ktr.axhline(0.0, color='k', lw=0.6, zorder=0)
ax_ktr.axvline(0.0, color='k', lw=0.6, zorder=0)
ax_ktr.set_aspect('equal')
ax_ktr.set_xlabel(r'$k_x$', fontweight='heavy')
ax_ktr.set_ylabel(r'$k_y$', fontweight='heavy')
ax_ktr.tick_params(labelsize=8)
for sp in ax_ktr.spines.values():
    sp.set_visible(False)

a_pc = alphas_nrm[:, :t_window]
colored_line_3d(ax_al, a_pc[0], a_pc[1], a_pc[2], ts, time_cmap, time_norm, lw=1.15)
for t, c in zip(t_idx, t_colors):
    ax_al.scatter(a_pc[0, t], a_pc[1, t], a_pc[2, t],
                  marker='x', color=c, s=90, linewidths=2.2, zorder=4)
ax_al.tick_params(labelsize=7, pad=-2)

sm = mpl.cm.ScalarMappable(norm=time_norm, cmap=time_cmap)
sm.set_array([])
cax_t = fig.add_axes([0.915, 0.06, 0.012, 0.28])
cb_t = fig.colorbar(sm, cax=cax_t)
cb_t.set_label('time [ms]', fontsize=11, fontweight='heavy')
for t, c in zip(t_ms, t_colors):
    cb_t.ax.scatter(0.5, t, marker='x', color=c, s=70, zorder=5,
                    linewidths=2.0, clip_on=False,
                    transform=cb_t.ax.get_yaxis_transform())

save_fig(fig, 'kspace_alphaspace')

# =====================================================================
# Phase grids: linear k-space  |  high-order
# =====================================================================
phase_groups = [
    ('phase_kspace', [
        (r'$k_x r_x$', rows[0][1]),
        (r'$k_y r_y$', rows[1][1]),
        (r'$k\cdot r$', rows[2][1]),
    ]),
    ('phase_alpha', [
        (r'$B_0$', rows[3][1]),
        ('coco', rows[4][1]),
        ('eddy', rows[5][1]),
        (r'$\varphi\cdot\alpha$', rows[6][1]),
    ]),
]

im = None
for name, specs in phase_groups:
    n_row = len(specs)
    fig_p, axes = plt.subplots(1, n_row, figsize=(2.4 * n_row + 0.6, 2.6))
    if n_row == 1:
        axes = [axes]
    t = t_idx[0]
    for ax, (ylabel, phase_fn) in zip(axes, specs):
        im = bare_imshow(ax, phase_fn(t)[slc].cpu(), phase_cmap,
                         -float(torch.pi), float(torch.pi))
        ax.set_title(ylabel, fontsize=12, fontweight='heavy', pad=4)
    fig_p.suptitle(rf'$t={t_ms[0]:.2f}\,\mathrm{{ms}}$', fontsize=13, fontweight='heavy', y=1.02)
    fig_p.tight_layout(w_pad=0.2)
    save_fig(fig_p, name)

fig_c, ax_c = plt.subplots(figsize=(3.6, 0.55))
cb = fig_c.colorbar(im, cax=ax_c, orientation='horizontal')
cb.set_label('phase [rad]', fontsize=11, fontweight='heavy')
cb.set_ticks([-torch.pi, 0, torch.pi])
cb.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
save_fig(fig_c, 'cbar_phase', pad_inches=0.08)

print(f'Saved figures to {out_dir}/' if save else 'Display only (save=False)')
plt.show()
