"""
Demo GIF: two sources of high-order phase in MRI.

Trajectory in alpha-space (alphas = number of phase wraps):
  1. Out-and-back along +alpha_0  -> B0-only phase
  2. Out-and-back along +alpha_1  -> eddy-only phase
  3. Smooth random path           -> combinations

Total phase factor:
  exp(-2j pi (phi_0 alpha_0 + phi_1 alpha_1))
"""
import numpy as np
import torch

import matplotlib
matplotlib.use('webAgg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

from mr_recon.utils import gen_grd
from mr_recon.spatial import spatial_resize_poly


# ---------------------------------------------------------------------------
# Spatial bases (already normalized so |alpha| = wraps at the peak of phi)
# ---------------------------------------------------------------------------
slc = 0
fpath = '/local_mount/space/tiger/1/users/abrahamd/mr_data/congyu_spiral_2d/data'
b0 = np.load(f'{fpath}/b0.npy')[..., slc]
mps = np.load(f'{fpath}/mps.npy')[..., slc]
mask = 1.0 * (np.mean(np.abs(mps), axis=0) > 1e-6)

b0 = gaussian_filter(b0, 0.5)
b0 = torch.from_numpy(b0).type(torch.float32)
b0 = spatial_resize_poly(b0, mps.shape[1:], order=3).numpy()
phi0 = b0 * mask
phi0 /= np.abs(phi0).max()
crds = gen_grd(mps.shape[1:]).numpy()
phi1 = (3 * crds[..., 1] * crds[..., 0]**2 - crds[..., 1]**3) * mask
phi1 = (crds[..., 0] ** 2 + crds[..., 1] ** 2) * mask
phi1 /= np.abs(phi1).max()

plt.imshow(phi1 / mask, cmap='RdBu_r', vmin=-0.5*2, vmax=0.5*2)
plt.axis('off')
plt.show()
quit()


# ---------------------------------------------------------------------------
# Smooth alpha-space trajectory
# ---------------------------------------------------------------------------
DT_MOVE = 0.035       # seconds per moving frame
DT_PEAK = 0.70        # pause at pure-axis peaks
DT_ORIGIN = 0.45      # pause at origin between chapters

A0_PEAK = 2.0         # wraps
A1_PEAK = 2.0
A_COMB = 2.0
N_LEG = 40            # samples for each one-way axis leg
N_COMBINED = 120      # samples for the combined tour


def ease_inout(u):
    u = np.asarray(u, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))


def eased_line(n, start, end):
    return start + (end - start) * ease_inout(np.linspace(0.0, 1.0, n))


def append_segment(a0_list, a1_list, dt_list, lab_list, a0, a1, label, dt):
    """Append moving samples; drop a duplicate junction with the previous tip."""
    a0 = np.atleast_1d(np.asarray(a0, dtype=np.float64))
    a1 = np.atleast_1d(np.asarray(a1, dtype=np.float64))
    if a0_list and np.isclose(a0[0], a0_list[-1]) and np.isclose(a1[0], a1_list[-1]):
        a0, a1 = a0[1:], a1[1:]
    if a0.size == 0:
        return
    a0_list.extend(a0.tolist())
    a1_list.extend(a1.tolist())
    dt_list.extend([dt] * len(a0))
    lab_list.extend([label] * len(a0))


def hold_last(dt_list, extra_dt):
    """Linger on the current tip by extending its display duration."""
    dt_list[-1] += extra_dt


def smooth_random_path(n, a_max, seed=3, n_knots=8):
    """Smooth random path visiting mixed quadrants; start/end at origin."""
    rng = np.random.default_rng(seed)
    knots = np.zeros((n_knots, 2))
    angles = np.linspace(0.35, 0.35 + 2.15 * np.pi, n_knots - 2, endpoint=False)
    radii = rng.uniform(0.6, 1.0, size=n_knots - 2) * a_max
    knots[1:-1, 0] = radii * np.cos(angles) * rng.uniform(0.85, 1.15, n_knots - 2)
    knots[1:-1, 1] = radii * np.sin(angles) * rng.uniform(0.85, 1.15, n_knots - 2)

    s = np.linspace(0.0, 1.0, n_knots)
    cs0 = CubicSpline(s, knots[:, 0], bc_type=((1, 0.0), (1, 0.0)))
    cs1 = CubicSpline(s, knots[:, 1], bc_type=((1, 0.0), (1, 0.0)))
    u = np.linspace(0.0, 1.0, n)
    a0, a1 = cs0(u), cs1(u)
    peak = max(np.abs(a0).max(), np.abs(a1).max(), 1e-12)
    return a0 * (a_max / peak), a1 * (a_max / peak)


a0_s, a1_s, dts, labels = [], [], [], []
lab0 = r'$\alpha_0$ only (B0)'
lab1 = r'$\alpha_1$ only (eddy)'
labc = 'combined'

# 1) +alpha_0 out, pause, back, pause
append_segment(a0_s, a1_s, dts, labels, eased_line(N_LEG, 0.0, A0_PEAK), np.zeros(N_LEG), lab0, DT_MOVE)
hold_last(dts, DT_PEAK - DT_MOVE)
append_segment(a0_s, a1_s, dts, labels, eased_line(N_LEG, A0_PEAK, 0.0), np.zeros(N_LEG), lab0, DT_MOVE)
hold_last(dts, DT_ORIGIN - DT_MOVE)

# 2) +alpha_1 out, pause, back, pause
append_segment(a0_s, a1_s, dts, labels, np.zeros(N_LEG), eased_line(N_LEG, 0.0, A1_PEAK), lab1, DT_MOVE)
hold_last(dts, DT_PEAK - DT_MOVE)
append_segment(a0_s, a1_s, dts, labels, np.zeros(N_LEG), eased_line(N_LEG, A1_PEAK, 0.0), lab1, DT_MOVE)
hold_last(dts, DT_ORIGIN - DT_MOVE)

# 3) smooth combined tour, brief end pause
a0_comb, a1_comb = smooth_random_path(N_COMBINED, A_COMB, seed=3, n_knots=8)
append_segment(a0_s, a1_s, dts, labels, a0_comb, a1_comb, labc, DT_MOVE)
hold_last(dts, DT_ORIGIN - DT_MOVE)

alpha0 = np.asarray(a0_s)
alpha1 = np.asarray(a1_s)
durations = np.asarray(dts)          # seconds
labels = np.asarray(labels)
n_frames = len(alpha0)

# Chapter boundaries for colored path
n0 = int(np.sum(labels == lab0))
n1 = int(np.sum(labels == lab1))

c_a0, c_a1, c_comb = '#2a6f97', '#2a9d8f', '#6d597a'


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_trj, ax_phase) = plt.subplots(1, 2, figsize=(11, 5))

# Full path, color-coded by chapter
ax_trj.plot(alpha0[:n0], alpha1[:n0], color=c_a0, lw=2.0, alpha=0.35, label=r'$\alpha_0$ only')
ax_trj.plot(alpha0[n0:n0 + n1], alpha1[n0:n0 + n1], color=c_a1, lw=2.0, alpha=0.35, label=r'$\alpha_1$ only')
ax_trj.plot(alpha0[n0 + n1:], alpha1[n0 + n1:], color=c_comb, lw=2.0, alpha=0.35, label='combined')
ax_trj.axhline(0.0, color='0.75', lw=0.8, zorder=0)
ax_trj.axvline(0.0, color='0.75', lw=0.8, zorder=0)
ax_trj.plot(0, 0, 'o', color='0.4', ms=5, zorder=3)
ax_trj.axis('off')

point, = ax_trj.plot([], [], 'o', color='#e76f51', ms=10, zorder=5, animated=True)
trail, = ax_trj.plot([], [], color='#e76f51', lw=2.2, alpha=0.9, animated=True)

# ax_trj.set_xlabel(r'$\alpha_0$  (B0 wraps)', fontsize=12)
# ax_trj.set_ylabel(r'$\alpha_1$  (eddy wraps)', fontsize=12)
# ax_trj.set_title(r'$\alpha$-space trajectory', fontsize=13)
lim = A_COMB * 1.25
ax_trj.set_xlim(-lim, lim)
ax_trj.set_ylim(-lim, lim)
ax_trj.set_aspect('equal')
# ax_trj.grid(True, alpha=0.3)
# ax_trj.legend(loc='lower right', fontsize=9, framealpha=0.9)
hud = ax_trj.text(
    0.02, 0.98, '', transform=ax_trj.transAxes,
    va='top', ha='left', fontsize=11, animated=True,
    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85),
)


def phase_map(i):
    phase = phi0 * alpha0[i] + phi1 * alpha1[i]
    ang = np.angle(np.exp(-2j * np.pi * phase))
    ang[mask <= 1e-6] = np.nan
    return ang


im = ax_phase.imshow(
    phase_map(0), cmap='jet', vmin=-np.pi, vmax=np.pi, animated=True,
)
# ax_phase.set_title(r'$\angle\,\exp(-2\pi j\,(\phi_0\alpha_0+\phi_1\alpha_1))$', fontsize=13)
# ax_phase.set_xlabel(r'$r_x$', fontsize=12)
# ax_phase.set_ylabel(r'$r_y$', fontsize=12)
ax_phase.set_aspect('equal')
ax_phase.axis('off')
# cbar = fig.colorbar(im, ax=ax_phase, fraction=0.046, pad=0.04)
# cbar.set_label('phase [rad]', fontsize=10)

fig.suptitle(
    r'Two high-order phase sources: B0 ($\phi_0=b_0$) + eddy '
    r'($\phi_1=3r_y r_x^2-r_y^3$)',
    fontsize=12,
)
fig.tight_layout()

fig.canvas.draw()
background = fig.canvas.copy_from_bbox(fig.bbox)

frames = []
for i in range(n_frames):
    fig.canvas.restore_region(background)
    point.set_data([alpha0[i]], [alpha1[i]])
    trail.set_data(alpha0[: i + 1], alpha1[: i + 1])
    hud.set_text(
        f'{labels[i]}\n'
        f'$\\alpha_0$={alpha0[i]:+.2f}, $\\alpha_1$={alpha1[i]:+.2f}'
    )
    im.set_data(phase_map(i))
    ax_trj.draw_artist(trail)
    ax_trj.draw_artist(point)
    # ax_trj.draw_artist(hud)
    ax_phase.draw_artist(im)
    fig.canvas.blit(fig.bbox)
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    frames.append(buf)

out_path = 'highorder_phase_demo.gif'
# imageio/Pillow GIF durations are in milliseconds
duration_ms = [max(1, int(round(1000 * d))) for d in durations]
imageio.mimsave(out_path, frames, duration=duration_ms, loop=0)
print(f'Saved {out_path}  ({len(frames)} frames, {durations.sum():.1f}s)')
