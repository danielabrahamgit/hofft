import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from hofft.utils import normalize

# Load data
save_path = './paper_experiments/highres_spiral/sweep_results.pt'
results = torch.load(save_path)
Ls_nufft = results['Ls_nufft']
Ws_nufft = results['Ws_nufft']
Ls_hofft = results['Ls_hofft']
Ws_hofft = results['Ws_hofft']
img_uncorr = results['img_uncorr']
img_ee = results['img_ee']
img_gt = results['img_gt']
imgs_ts = results['imgs_ts']
imgs_hofft = results['imgs_hofft']
img_ref = img_ee
im_size = img_gt.shape

# ------------ Plot: metrics vs L, one line per (method, W) ------------
# Color encodes method (TS NUFFT vs HOFFT); linestyle encodes kernel width W.
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
panels = [
    ('nrmse', 'NRMSE'), ('decomp_time', 'Decomp time [s]'), ('recon_time', 'Recon time [s]'),
]
methods = [
    ('ts', 'TS NUFFT', 'red', 'o', Ls_nufft, Ws_nufft),
    ('hofft', 'HOFFT', 'green', 's', Ls_hofft, Ws_hofft),
]
linestyles = ['-', '--', ':', '-.']
all_Ws = sorted(set(Ws_nufft) | set(Ws_hofft))
W_to_ls = {W: linestyles[i % len(linestyles)] for i, W in enumerate(all_Ws)}

nrmse_max = 0.33
for ax, (key, title) in zip(axes[:3], panels):
    for suffix, method_label, color, marker, Ls, Ws in methods:
        for j, W in enumerate(Ws):
            ax.plot(Ls, results[f'{key}_{suffix}'][:, j].numpy(),
                    color=color, linestyle=W_to_ls[W], marker=marker,
                    label=f'{method_label}, W={W}')
    ax.set_xlabel('L')
    ax.set_ylabel(title)
    ax.set_title(title)
    if key == 'nrmse':
        ax.set_ylim(0, nrmse_max)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both' if key == 'nrmse' else 'major')

# NRMSE vs recon time -- the accuracy/speed tradeoff, points connected in
# increasing-L order per (method, W) line.
ax = axes[3]
for suffix, method_label, color, marker, Ls, Ws in methods:
    for j, W in enumerate(Ws):
        total_time = results[f'decomp_time_{suffix}'][:, j].numpy() + \
                     results[f'recon_time_{suffix}'][:, j].numpy()
        ax.plot(total_time,
                results[f'nrmse_{suffix}'][:, j].numpy(),
                color=color, linestyle=W_to_ls[W], marker=marker,
                label=f'{method_label}, W={W}')
# ax.set_yscale('log')
ax.set_xlabel('Total time [s]')
ax.set_ylabel('NRMSE')
ax.set_title('NRMSE vs Total Time')
ax.set_xlim(0, 30)
ax.set_ylim(0, nrmse_max)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(f'./paper_experiments/highres_spiral/sweep_lineplots.png', dpi=150)

# ------------ Plot: qualitative comparison at the largest (L, W) ------------
# from mr_recon.utils import cvplot
# cvplot('./config.yaml', img_uncorr.cpu(), img_ee.cpu(), imgs_ts[-1, -1], imgs_hofft[-1, -1])
# quit()
imgs = [img_uncorr.cpu(), img_ee.cpu(), imgs_ts[-1, -1], imgs_hofft[-1, -1]]
vmin = 0
img_ref = img_ref.cpu()
vmax = img_gt.abs().median() + 4 * img_gt.abs().std()
# pl, pr = 0.6, 0.85
# pu, pd = 0.35, 0.6
pl, pr = 0.0, 1.0
pu, pd = 0.0, 1.0
img_slc = (slice(round(pu*im_size[1]), round(pd*im_size[1])),
           slice(round(pl*im_size[0]), round(pr*im_size[0]))) 
titles = ['Uncorrected', 'Expanded Encoding (precomputed)', f'TS NUFFT (L={Ls_nufft[-1]}, W={Ws_nufft[-1]})', f'HOFFT (L={Ls_hofft[-1]}, W={Ws_hofft[-1]})']
plt.figure(figsize=(14, 7))
M = 5
for i, img in enumerate(imgs):
    img = normalize(img, img_ref)
    plt.subplot(2, len(imgs), i+1)
    plt.imshow(img.abs().cpu().rot90()[img_slc], cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title(titles[i])

    err = img.abs() - img_ref.abs()
    plt.subplot(2, len(imgs), i+1+len(imgs))
    plt.imshow(err.abs().cpu().rot90()[img_slc], cmap='gray', vmin=vmin/M, vmax=vmax/M)
    plt.axis('off')
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.show()
