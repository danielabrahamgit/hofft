"""
Measures the compressed HOFFT model phase error at FULL resolution (after
spatial factor expansion) for 'lstsq' vs 'grid' sparse decompositions.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import einsum

from hofft.decomp import hofft_params, build_kern_bases
from hofft.sparse_decomp import sparse_params
from hofft.pipelines import als_hofft_compressed
from hofft.phase_coeffs import trj_dev_to_phis_alphas, compress_phis_alphas, rescale_phis_alphas

torch.manual_seed(0)
R_under = 3
B_compressed = 4
torch_dev = torch.device(1)
hparams = hofft_params(kern_size=(5,)*2, os=1.25, L=5, reduced_im_size=(120,120),
                       spatial_init='seg', verbose=False)

fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_spiral'
mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev)
trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev)
alphas = torch.load(f'{fpath}/alphas.pt', map_location=torch_dev)
phis = torch.load(f'{fpath}/phis.pt', map_location=torch_dev)
im_size = mps.shape[1:]
hparams.os = 2 * round(hparams.os * im_size[0] / 2) / im_size[0]

trj = trj[..., ::R_under, :].type(torch.float32)
trj_size = trj.shape[:-1]

B = phis.shape[0]
phi_energy = phis.reshape((B, -1)).abs().mean(dim=1)
alpha_energy = alphas.reshape((B, -1)).abs().mean(dim=1)
idxs = torch.argwhere(phi_energy * alpha_energy > 1e-6)[:, 0]
phis, alphas = phis[idxs], alphas[idxs]

phis_dev, alphas_dev = trj_dev_to_phis_alphas(trj, im_size, hparams.os)
phis_stack = torch.cat([phis, phis_dev], dim=0)
alphas_stack = torch.cat([alphas, alphas_dev], dim=0)
phis_stack, alphas_stack = compress_phis_alphas(phis_stack, alphas_stack, B_compressed=B_compressed)
phis_nrm, phis_mp, alphas_nrm, alphas_mp = rescale_phis_alphas(phis_stack, alphas_stack)

B = phis_nrm.shape[0]
R = int(np.prod(im_size))
T = int(np.prod(trj_size))
phis_flt = phis_nrm.reshape((B, R))
alphas_flt = alphas_nrm.reshape((B, T))

t_eval = torch.randperm(T, device=torch_dev)[:500]
truth = torch.exp(-2j * np.pi * einsum(phis_flt, alphas_flt[:, t_eval], 'B R, B T -> T R'))
kern_full = build_kern_bases(hparams.kern_size, im_size, os=hparams.os).to(torch_dev)
kern_flt = kern_full.reshape((-1, R))

common = dict(temporal_batch_size=2**10, lamda=0.0, normalize_coeffs=True)
configs = [
    ('lstsq',        sparse_params(Q=200, K=5, beta_method='maxmin', interp_method='lstsq', **common)),
    ('grid', sparse_params(interp_method='grid', grid_spacing=0.25, grid_width=2, **common)),
]

err_maps = {}
for label, sparams in configs:
    sf, ck, si, sc = als_hofft_compressed(phis_nrm, alphas_nrm, hparams=hparams,
                                          sparams=sparams, num_als_iter=100)
    L = sf.shape[0]
    sf_flt = sf.reshape((L, R))
    Kw = ck.shape[1] * ck.shape[2]
    ck_flt = ck.reshape((L, Kw, -1))
    si_flt = si.reshape((si.shape[0], T))[:, t_eval]
    sc_flt = sc.reshape((sc.shape[0], T))[:, t_eval]

    # kernel weights per eval time: w[l, kw, t] = sum_s ck[l, kw, si[s,t]] * sc[s,t]
    w = einsum(ck_flt[:, :, si_flt], sc_flt, 'L Kw S T, S T -> L Kw T')
    model = einsum(sf_flt, kern_flt, w, 'L R, Kw R, L Kw T -> T R')

    err_t = (model - truth).norm(dim=-1) / truth.norm(dim=-1)
    err_map = (model - truth).abs().mean(dim=0).reshape(im_size)
    err_maps[label] = err_map.cpu()
    print(f'{label:14s} full-res model err: mean {err_t.mean():.3e} max {err_t.max():.3e}')

fig, axs = plt.subplots(1, len(err_maps), figsize=(6*len(err_maps), 6))
vmax = max(m.max().item() for m in err_maps.values())
for ax, (k, m) in zip(axs, err_maps.items()):
    im = ax.imshow(m.rot90(), cmap='inferno', vmin=0, vmax=vmax)
    ax.set_title(f'{k} (mean |err| per voxel)'); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig('experiments/_tmp_diag_expand.png', dpi=120)
print('saved experiments/_tmp_diag_expand.png')
