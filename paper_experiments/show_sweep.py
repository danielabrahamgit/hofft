"""
Reload highres_spiral sweep results and make paper-ready plots.
"""
import torch
import os

import matplotlib as mpl
mpl.use('WebAgg')
import matplotlib.pyplot as plt

from hofft.utils import normalize

# dataset = 'tilt_spi_invivo'
# dataset = 'tilt_spi'
# dataset = 'coco_spiral'
# dataset = 'axial_spi_invivo'
# dataset = 'highres_spiral'
# dataset = 'magnus_spi'
dataset = 'magnus_epi'
save_path = f'./paper_experiments/{dataset}_sweep_results.pt'
results = torch.load(save_path, weights_only=False, map_location='cpu')

# Params
# nrmse_max = 0.33 * 0 + 0.05
# time_max = 24 
nrmse_max = None
time_max = None
nrmse_tol = 0.025       # target NRMSE
total_time_tol = 50.0   # target total time [s]
err_scale = 5

img_uncorr_raw = results.get('img_uncorr_raw')
img_uncorr = results['img_uncorr']
img_ee = results['img_ee']
img_gt = results['img_gt']
img_ref = img_ee
im_size = img_gt.shape

METHODS = []
if 'Ls_svd' in results:
    METHODS.append({
        'key': 'svd',
        'label': 'SVD',
        'color': 'orange',
        'marker': 'o',
        'Ls': results['Ls_svd'],
        'Ws': results['Ws_svd'],
        'imgs': results['imgs_svd'],
    })
if 'Ls_hofft' in results:
    METHODS.append({
        'key': 'hofft',
        'label': 'HOFFT',
        'color': '#2ca02c',
        'marker': 's',
        'Ls': results['Ls_hofft'],
        'Ws': results['Ws_hofft'],
        'imgs': results['imgs_hofft'],
    })

# ---------------------------------------------------------------------------
# Paper-ready line plots
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14,
    'axes.linewidth': 2.0,
    'lines.linewidth': 4.0,
    'lines.markersize': 14,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

plot_kw = dict(lw=4.5, ms=16, mew=2.0, markeredgecolor='white', zorder=3)
linestyles = ['-', '--', ':', '-.']
all_Ws = sorted({W for m in METHODS for W in m['Ws']})
W_to_ls = {W: linestyles[i % len(linestyles)] for i, W in enumerate(all_Ws)}


fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
metric_panels = [
    ('nrmse', 'NRMSE', 'NRMSE vs L'),
    ('decomp_time', 'Decomp time [s]', 'Decomp time vs L'),
    ('recon_time', 'Recon time [s]', 'Recon time vs L'),
]

for ax, (key, ylabel, title) in zip(axes[:3], metric_panels):
    for m in METHODS:
        for j, W in enumerate(m['Ws']):
            label = m['label'] if len(all_Ws) == 1 else f'{m["label"]}, W={W}'
            ax.plot(
                m['Ls'], results[f'{key}_{m["key"]}'][:, j].numpy(),
                color=m['color'], linestyle=W_to_ls[W], marker=m['marker'],
                label=label, markerfacecolor=m['color'], **plot_kw,
            )
    ax.set_xlabel('L')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if key == 'nrmse' and nrmse_max is not None:
        ax.set_ylim(0, nrmse_max)
    else:
        ax.set_ylim(bottom=0)

ax = axes[3]
for m in METHODS:
    for j, W in enumerate(m['Ws']):
        total = (results[f'decomp_time_{m["key"]}'][:, j]
                 + results[f'recon_time_{m["key"]}'][:, j]).numpy()
        label = m['label'] if len(all_Ws) == 1 else f'{m["label"]}, W={W}'
        ax.plot(
            total, results[f'nrmse_{m["key"]}'][:, j].numpy(),
            color=m['color'], linestyle=W_to_ls[W], marker=m['marker'],
            label=label, markerfacecolor=m['color'], **plot_kw,
        )
ax.set_xlabel('Total time [s]')
ax.set_ylabel('NRMSE')
ax.set_title('NRMSE vs Total Time')
if time_max is not None:
    ax.set_xlim(0, time_max)
if nrmse_max is not None:
    ax.set_ylim(0, nrmse_max)

for ax in axes:
    ax.grid(True, which='major', alpha=0.35, lw=1.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    leg = ax.legend(
        loc='best', frameon=True, fancybox=False,
        edgecolor='0.4', framealpha=0.95, borderpad=0.5,
        handlelength=2.0, markerscale=1.05,
    )
    leg.get_frame().set_linewidth(1.5)

# fig.suptitle('High-res spiral: SVD / Alpha Segmentation / HOFFT',
#              fontsize=28, fontweight='bold', y=1.02)
fig.tight_layout()
os.makedirs(f'./paper_experiments/{dataset}', exist_ok=True)
fig.savefig(f'./paper_experiments/{dataset}/sweep_lineplots.png', dpi=200)

# ---------------------------------------------------------------------------
# Qualitative comparison: match NRMSE or total time across methods
# ---------------------------------------------------------------------------

# Display order: Uncorrected, EE, Alpha Seg, SVD, HOFFT
methods_by_key = {m['key']: m for m in METHODS}


def closest_idx(metric, target):
    return (metric - target).abs().argmin().item()


def pick_method_image(key, metric, target):
    m = methods_by_key[key]
    nW = len(m['Ws'])
    flat = closest_idx(metric, target)
    i, j = divmod(flat, nW)
    L, W = m['Ls'][i], m['Ws'][j]
    nrmse_ij = results[f'nrmse_{key}'][i, j].item()
    total_ij = (results[f'decomp_time_{key}'][i, j]
                + results[f'recon_time_{key}'][i, j]).item()
    title = f'{m["label"]}\nL={L}, W={W}\nNRMSE={nrmse_ij:.1%}, {total_ij:.1f}s'
    return m['imgs'][i, j], title


def plot_image_error_figure(qual_imgs, qual_titles, fig_title, out_path):
    # pcnt_left = 0.05
    # pcnt_right = 0.35
    # pcnt_top = 0.35
    # pcnt_bottom = 0.65
    pcnt_left = 0.35
    pcnt_right = 0.65
    pcnt_top = 0.55
    pcnt_bottom = 0.85
    im_size = qual_imgs[0].shape
    slc = (slice(round(pcnt_top * im_size[0]), round(pcnt_bottom * im_size[0])), 
           slice(round(pcnt_left * im_size[1]), round(pcnt_right * im_size[1])))
    slc = slice(None)
    
    n = len(qual_imgs)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7))
    for col, (img, title) in enumerate(zip(qual_imgs, qual_titles)):
        img = normalize(img, img_ref)
        axes[0, col].imshow(img.abs().rot90()[slc], cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(title, fontsize=12)
        axes[0, col].axis('off')

        err = (img.abs() - img_ref.abs()).abs()
        # err = (img - img_ref).abs()
        axes[1, col].imshow(err.rot90()[slc], cmap='gray',
                            vmin=vmin / err_scale, vmax=vmax / err_scale)
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Image', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel(f'|Error| (×{err_scale})', fontsize=14, fontweight='bold')
    for row in (0, 1):
        axes[row, 0].axis('on')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

    fig.suptitle(fig_title, fontsize=18, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved {out_path}')
    return fig


vmin = 0.0
vmax = img_gt.abs().median() + 4 * img_gt.abs().std()
refs = []
if img_uncorr_raw is not None:
    refs.append((img_uncorr_raw, 'Uncorrected'))
    refs.append((img_uncorr, 'Linear-corrected'))
else:
    refs.append((img_uncorr, 'Uncorrected'))
refs.append((img_ee, 'Expanded Encoding'))

fig_specs = [
    (f'Matched NRMSE ≈ {nrmse_tol * 100}%', 'nrmse', nrmse_tol, 'sweep_images_matched_nrmse.png'),
    (f'Matched total time ≈ {total_time_tol} s', 'total_time', total_time_tol, 'sweep_images_matched_time.png'),
]
for fig_title, metric_name, target, fname in fig_specs:
    qual_imgs = [img for img, _ in refs]
    qual_titles = [title for _, title in refs]
    for key in methods_by_key.keys():
        if metric_name == 'nrmse':
            metric = results[f'nrmse_{key}']
        else:
            metric = results[f'decomp_time_{key}'] + results[f'recon_time_{key}']
        img, title = pick_method_image(key, metric, target)
        qual_imgs.append(img)
        qual_titles.append(title)

    out_path = f'./paper_experiments/{dataset}/{fname}'
    plot_image_error_figure(qual_imgs, qual_titles, fig_title, out_path)

plt.show()
