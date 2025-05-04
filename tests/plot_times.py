import torch

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

from mr_recon.utils import normalize

fpath = 'tiny_nufft_experiment_test5.pt'
fpath_times = 'times_gpu_rep20.pt'

times = torch.load(fpath_times, weights_only=True)
dct = torch.load(fpath, weights_only=True)
Ls = dct['Ls']
ks = dct['ks']
idxs = dct['idxs']
imgs = dct['imgs']
img_gt = torch.load('img_gt.pt', weights_only=True)
im_size = img_gt.shape

times = times.reshape((len(Ls), len(ks), len(ks)))


diag = torch.arange(len(ks))
plt.title(f'Log Recon Time')
plt.imshow(times[:, diag, diag].log10(),
           extent=[ks.min()-0.5, ks.max()+0.5, Ls.min()-0.5, Ls.max()+0.5],
           origin='lower', aspect='auto', cmap='grey')
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
