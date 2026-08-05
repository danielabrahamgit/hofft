import numpy as np
import torch

import matplotlib
matplotlib.use('webAgg')
import matplotlib.pyplot as plt
from PIL import Image

# --- k-space trajectory: 2D sinusoid (Lissajous-style) ---
n_frames = 150
kmax = 2.0          # cycles / FOV
fx, fy = 3.0, 3.0    # frequencies for the two axes

t = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
kx = kmax * (t / (2 * np.pi) - 0.5)
ky = kmax * np.sin(fy * t)

# --- spatial grid ---
N = 220
r = np.linspace(-0.5, 0.5, N)  # normalized FOV coords
X, Y = np.meshgrid(r, r)

# # Load real field map 
# fm = torch.load('b0_coil.pt').type(torch.float32) / 10e3
# mask = 1.0 * (fm.abs() > 0.0).flatten()
# X_flt = torch.from_numpy(X).flatten().type(torch.float32)
# Y_flt = torch.from_numpy(Y).flatten().type(torch.float32)
# A = torch.stack([X_flt, Y_flt, X_flt * 0 + 1], dim=1)
# b = fm.flatten()
# x = torch.linalg.lstsq(A * mask[:, None], b * mask).solution
# dff = ((A @ x - b) * mask).reshape(X.shape)
# plt.imshow(fm, cmap='jet', vmin=fm.min(), vmax=fm.max())
# plt.figure()
# plt.imshow(dff, cmap='jet', vmin=fm.min(), vmax=fm.max())
# plt.show()
# quit()


# high-order spatial basis function phi(r)
bases = np.stack([X**2, Y**2, X**3, Y**3, X**5, Y**5], axis=-1)
B = bases.shape[-1]
for i in range(B):
    bases[..., i] -= bases[..., i].mean()
    bases[..., i] /= bases[..., i].std()
coeffs = np.random.randn(B)
phi = bases @ coeffs
amp = np.sin(fy * t)  # same temporal envelope driving ky

# --- figure/axes setup ---
fig, (ax_traj, ax_phasor, ax_highorder) = plt.subplots(1, 3, figsize=(15, 5))

ax_traj.plot(kx, ky, color='green', lw=1, alpha=0.5)
point, = ax_traj.plot([], [], 'o', color='green', ms=8, animated=True)
ax_traj.axis('off')

phasor0 = np.angle(np.exp(2j * np.pi * (kx[0] * X + ky[0] * Y)))
im = ax_phasor.imshow(
    phasor0, cmap='jet', vmin=-np.pi, vmax=np.pi,
    extent=[r[0], r[-1], r[0], r[-1]], origin='lower', animated=True,
)
ax_phasor.axis('off')

highorder0 = np.angle(np.exp(-2j * np.pi * amp[0] * phi))
im_ho = ax_highorder.imshow(
    highorder0, cmap='jet', vmin=-np.pi, vmax=np.pi,
    extent=[r[0], r[-1], r[0], r[-1]], origin='lower', animated=True,
)
ax_highorder.axis('off')

fig.tight_layout()

# Render the static background once, then blit only the point + images each
# frame -- avoids a full canvas redraw per frame.
fig.canvas.draw()
background = fig.canvas.copy_from_bbox(fig.bbox)

frames = []
for i in range(n_frames):
    fig.canvas.restore_region(background)
    point.set_data([kx[i]], [ky[i]])
    im.set_data(np.angle(np.exp(2j * np.pi * (kx[i] * X + ky[i] * Y))))
    im_ho.set_data(np.angle(np.exp(-2j * np.pi * amp[i] * phi)))
    ax_traj.draw_artist(point)
    ax_phasor.draw_artist(im)
    ax_highorder.draw_artist(im_ho)
    fig.canvas.blit(fig.bbox)
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames.append(Image.fromarray(buf).convert('RGB'))

# GIF-saving cost is dominated by per-frame adaptive palette quantization on
# the 'jet' colormap. Quantize once against a shared palette (no dithering)
# and reuse it for every frame instead of re-running median-cut per frame.
palette_frame = frames[0].convert('P', palette=Image.ADAPTIVE, colors=256)
quantized = [f.quantize(palette=palette_frame, dither=Image.NONE) for f in frames]

out_path = 'kspace_phasor.gif'
quantized[0].save(
    out_path, save_all=True, append_images=quantized[1:],
    duration=40, loop=0, disposal=2,
)
print(f'Saved {out_path}')
