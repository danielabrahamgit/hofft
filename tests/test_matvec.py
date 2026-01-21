import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from hofft.matvec import (
    matvec_naive, 
    matvec_type3, 
    matvec_svd, 
    matvec_cur,
    matvec_histogram
)
from mr_recon.utils import gen_grd

torch.manual_seed(0)

# consts
im_size = (220,)
trj_size = (100,)
torch_dev = torch.device(5)

# Generate phis alphas
rs = gen_grd(im_size).to(torch_dev).moveaxis(-1,0)
phis = rs ** 3
phis /= phis.abs().max() * 2
alphas = 25 * (torch.rand(trj_size, device=torch_dev) ** 2).sort().values[None,]

# Compute naive
op = lambda x, mv : mv.normal(x[None,])[0]
matvec = matvec_naive(phis, alphas)
x = torch.randn(im_size, device=torch_dev, dtype=torch.complex64)
# x = torch.randn(trj_size, device=torch_dev, dtype=torch.complex64)
y = op(x, matvec)

# Compute type3
matvec = matvec_type3(phis, alphas, width=3)
y_type3 = op(x, matvec)

# # Compute SVD
# matvec = matvec_svd(phis, alphas, svd_rank=10, svd_method='torch', num_iter=15)
# y_svd = op(x, matvec)

# # Compute CUR
# matvec = matvec_cur(phis, alphas, cur_rank=20)
# y_cur = op(x, matvec)

# # Compute histogram
# matvec = matvec_histogram(phis, alphas, 
#                         #   dphi=0.005, 
#                           dalpha=0.1, 
#                         #   Kphi=220, 
#                         #   Kalpha=100,
#                           )
# y_hist = op(x, matvec)

# Compare
plt.figure(figsize=(14, 7))
plt.plot(y.real.cpu().numpy(), label='Naive', alpha=0.5)
plt.plot(y_type3.real.cpu().numpy(), label='Type3', alpha=0.5)
# plt.plot(y_svd.real.cpu().numpy(), label='SVD', alpha=0.5)
# plt.plot(y_cur.real.cpu().numpy(), label='CUR', alpha=0.5)
# plt.plot(y_hist.real.cpu().numpy(), label='Histogram', alpha=0.5)
plt.legend()
plt.show()