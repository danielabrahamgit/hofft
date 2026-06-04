import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from mr_recon.utils import normalize

# Load data
data = torch.load('./paper_figures/recons/coco_spiral.pt',
                  map_location=torch.device('cpu'),
                  weights_only=True)
imgs_hofft = data['imgs_hofft']
imgs_split = data['imgs_split']
times_hofft = data['times_hofft']
times_split = data['times_split']
img_gt = data['img_gt']
Ls = data['Ls']
Ws = data['Ws']
os = data['os']

# Compute NRMSEs
nrmses_hofft = torch.zeros(len(Ls), len(Ws))
nrmses_split = torch.zeros(len(Ls), len(Ws))
for l in range(len(Ls)):
    for w in range(len(Ws)):
        img_hofft = normalize(imgs_hofft[l, w], img_gt)
        img_split = normalize(imgs_split[l, w], img_gt)
        nrmses_hofft[l, w] = (img_hofft.abs() - img_gt.abs()).norm() / img_gt.norm()
        nrmses_split[l, w] = (img_split.abs() - img_gt.abs()).norm() / img_gt.norm()

# Plot figure
def plot_lw_recons(imgs: torch.Tensor, 
                   img_gt: torch.Tensor, 
                   S: int = 2,
                   title: str = ''):
    plt.figure(figsize=(len(Ls)*S, len(Ws)*S))
    plt.suptitle(f'{title}')
    vmax = img_gt.abs().median() + 3 * img_gt.abs().std()
    for l in range(len(Ls)):
        for w in range(len(Ws)):
            img = normalize(imgs[l, w], img_gt)
            plt.subplot(len(Ws), len(Ls), w*len(Ls) + l + 1)
            plt.imshow(img.abs().rot90(), cmap='gray',
                       vmin=0, vmax=vmax)
            
            if l == 0:
                plt.ylabel(f'W={Ws[w]}')
            if w == len(Ws) - 1:
                plt.xlabel(f'L={Ls[l]}')

            # Turn axis off but keep labels
            plt.xticks([])
            plt.yticks([])
            
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

def plot_errors(nrmses1: torch.Tensor, 
                nrmses2: torch.Tensor,
                label1: str = '',
                label2: str = ''):
    plt.figure(figsize=(14, 7))
    plt.title(f'Error Figure')
    for w in range(len(Ws)):
        line, = plt.plot(Ls, nrmses1[:, w], label=f'{label1} W={Ws[w]}')
        plt.plot(Ls, nrmses2[:, w], linestyle='--', color=line.get_color(), label=f'{label2} W={Ws[w]}')
    plt.xlabel('L')
    plt.ylabel('NRMSE (%)')
    plt.legend()
    plt.tight_layout()

def plot_times(times1: torch.Tensor,
               times2: torch.Tensor,
               label1: str = '',
               label2: str = ''):
    plt.figure(figsize=(14,7))
    plt.title(f'Timing Figure')
    for k,w in enumerate(Ws):
        line, = plt.plot(Ls, times1[:, k], label=f'{label1} W={w}')
        plt.plot(Ls, times2[:, k], linestyle='--', color=line.get_color(), label=f'{label2} W={w}')
    plt.xlabel('L')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
        
def plot_time_vs_error(times1: torch.Tensor,
                       nrmses1: torch.Tensor,
                       times2: torch.Tensor,
                       nrmses2: torch.Tensor,
                       label1: str = '',
                       label2: str = ''):
    plt.figure(figsize=(14,7))
    plt.title(f'Time vs Error')
    # for k,w in enumerate(Ws):
    #     scat = plt.scatter(times1[:, k], nrmses1[:, k].log10(), label=f'{label1} W={w}')
    #     plt.scatter(times2[:, k], nrmses2[:, k].log10(), marker='x', color=scat.get_facecolor(), label=f'{label2} W={w}')
    for k,w in enumerate(Ws):
        line, = plt.semilogy(times1[:, k], nrmses1[:, k], label=f'{label1} W={w}')
        plt.semilogy(times2[:, k], nrmses2[:, k], linestyle='--', color=line.get_color(), label=f'{label2} W={w}')
    plt.xlabel('Time (s)')
    plt.ylabel('NRMSE')
    plt.legend()

plot_lw_recons(imgs_hofft, img_gt, title='HOFFT Reconstructions')
plot_lw_recons(imgs_split, img_gt, title='Splitting Reconstructions')
plot_errors(nrmses_hofft, nrmses_split, 
            label1='HOFFT', 
            label2='Splitting')
plot_times(times_hofft.sum(dim=-1), times_hofft[..., -1], 
           label1='HOFFT', 
           label2='Splitting')
plot_time_vs_error(times_hofft[..., -1], nrmses_hofft, 
                   times_hofft[..., -1], nrmses_split,
                   label1='HOFFT',
                   label2='Splitting')
plt.show()
