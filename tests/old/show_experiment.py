import torch

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

from mr_recon.utils import normalize, torch_to_np
from skimage.metrics import structural_similarity

# fpath = 'tiny_nufft_experiment_10000_old.pt'
# fpath = 'tiny_nufft_experiment_kb.pt'
# fpath = 'high_order_experiments/100_iter.pt'
fpath = 'high_order_experiments/svd.pt'

dct = torch.load(fpath, weights_only=True)
Ls = dct['Ls']
ks = dct['ks']
idxs = dct['idxs']
imgs = dct['imgs'].abs()
img_gt = torch.load('high_order_experiments/img_gt.pt', weights_only=True).abs()
im_size = img_gt.shape

errs = torch.zeros((len(Ls), len(ks), len(ks), *im_size))
ssims = torch.zeros((len(Ls), len(ks), len(ks)))
nrmses = torch.zeros((len(Ls), len(ks), len(ks)))
for n in range(len(idxs)):
    lidx = idxs[n, 0]
    kidx1 = idxs[n, 1]
    kidx2 = idxs[n, 2]
    
    err = (normalize(imgs[n], img_gt, mag=True, ofs=True) - img_gt).abs()
    nrmse = err.norm() / img_gt.norm()
    errs[lidx, kidx1, kidx2] = err
    nrmses[lidx, kidx1, kidx2] = nrmse
    
    ssim = structural_similarity(torch_to_np(imgs[n] / imgs[n].max()), 
                                 torch_to_np(img_gt / img_gt.max()), 
                                 data_range=1.0)
    ssims[lidx, kidx1, kidx2] = ssim

diag = torch.arange(len(ks))
w = 1
# w = 0.5
plt.figure()
plt.title(f'Reconstruction Error NRMSE Log Scale')
plt.imshow(nrmses[:, diag, diag].log10(), vmin=-1.6, vmax=0,
           extent=[ks.min()-w, ks.max()+w, Ls.min()-w, Ls.max()+w],
           origin='lower', aspect='auto', cmap='inferno')
plt.colorbar()
plt.xlabel('Kernel Width')
plt.xticks(ks)
plt.yticks(Ls)
plt.ylabel('Number of Apodization Functions')

plt.figure()
plt.title(f'Reconstruction SSIM')
plt.imshow(ssims[:, diag, diag], vmin=0.9, vmax=.98,
           extent=[ks.min()-w, ks.max()+w, Ls.min()-w, Ls.max()+w],
           origin='lower', aspect='auto', cmap='inferno')
plt.colorbar()
plt.xlabel('Kernel Width')
plt.xticks(ks)
plt.yticks(Ls)
plt.ylabel('Number of Apodization Functions')
plt.show()

# imgs = imgs.reshape((len(Ls), len(ks), len(ks), *im_size))
# from pyeyes import ComparativeViewer
# cv = ComparativeViewer({'errs': errs},
#                        ['L', 'Kx', 'Ky', 'X', 'Y'],
#                        ['Y', 'X'])
# cv.launch()
