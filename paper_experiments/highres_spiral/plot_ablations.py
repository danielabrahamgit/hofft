"""
Plot NRMSE vs total time for HOFFT ablation sweeps.

Styled for PowerPoint (large fonts / markers / linewidths).
"""
import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# ---- slide-friendly defaults ----
mpl.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
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

root = './paper_experiments/highres_spiral'
ablations = [
    ('cur_ka_nonorm', 'CUR + No Whitening', '#1f77b4', 'o'),
    # ('cur_ka',        'CUR + $k_\\alpha$',  '#2ca02c', 's'),
    ('cur_noka',      'CUR',                '#ff7f0e', '^'),
    ('nocur_noka',    'no CUR',             '#d62728', 'D'),
]

fig, ax = plt.subplots(figsize=(10, 7))

for tag, label, color, marker in ablations:
    path = f'{root}/sweep_results_{tag}.pt'
    d = torch.load(path, weights_only=False, map_location='cpu')
    nrmse = d['nrmse_hofft'][:, 0].numpy()
    total = (d['decomp_time_hofft'][:, 0] + d['recon_time_hofft'][:, 0]).numpy()
    ax.plot(
        total, nrmse,
        color=color, marker=marker, label=label,
        lw=4.5, ms=16, mew=2.0, markerfacecolor=color,
        markeredgecolor='white', zorder=3,
    )

ax.set_xlabel('Total time [s]')
ax.set_ylabel('NRMSE')
ax.set_title(r'HOFFT ablations  ($L=5\rightarrow 20$)')
ax.set_ylim(bottom=0)
ax.grid(True, which='major', alpha=0.35, lw=1.2)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(2.0)

leg = ax.legend(
    loc='upper right', frameon=True, fancybox=False,
    edgecolor='0.4', framealpha=0.95, borderpad=0.6,
    handlelength=2.2, markerscale=1.1,
)
leg.get_frame().set_linewidth(1.5)

fig.tight_layout()
out = f'{root}/ablation_nrmse_vs_time.png'
fig.savefig(out, dpi=200)
print(f'Saved {out}')
