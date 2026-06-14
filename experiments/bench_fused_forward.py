import torch
import numpy as np
from einops import einsum

from mr_recon.linops import batching_params
from mr_recon._func.indexing import multi_index
from mr_recon.fourier import fft

from hofft.forward_model import hofft_compressed_linop


def reference_forward(A, img):
    """Original PyTorch forward (pre-fusion) using the linop's stored tensors."""
    D = A.idx_kerns.shape[-1]
    C = A.mps.shape[0]
    L = A.compressed_kernels.shape[0]
    cbs = A.bparams.coil_batch_size
    fbs = A.bparams.field_batch_size
    ksp = torch.zeros(A.oshape, device=img.device, dtype=torch.complex64)
    for c1 in range(0, C, cbs):
        c2 = min(c1 + cbs, C)
        Sx = A.mps[c1:c2] * img
        for l1 in range(0, L, fbs):
            l2 = min(l1 + fbs, L)
            MSx = einsum(Sx, A.spatial_factors[l1:l2], 'C ..., L ... -> C L ...')
            MSx = A.padder(MSx)
            FMSx = fft(MSx, dim=tuple(range(-D, 0)))
            FMSx *= np.prod(FMSx.shape[-D:]) ** 0.5 / np.prod(A.im_size) ** 0.5
            blocks = multi_index(FMSx, D, A.idx_kerns).moveaxis(-1, 2)
            kw = A.compressed_kernels[l1:l2, :, A.sparse_idxs]
            kw = einsum(kw, A.sparse_coeffs, 'L K S ..., S ... -> L K ...')
            ksp[c1:c2] += einsum(blocks, kw, 'C L K ..., L K ... -> C ...')
    return ksp


def build(im_size, trj_size, C, L, kern, S, Q, os_grid, bparams, dev):
    D = len(im_size)
    trj = torch.rand((*trj_size, D), device=dev) * 0  # zeros -> on-grid trivially
    mps = torch.randn((C, *im_size), dtype=torch.complex64, device=dev)
    comp = torch.randn((L, *kern, Q), dtype=torch.complex64, device=dev)
    sidx = torch.randint(0, Q, (S, *trj_size), device=dev, dtype=torch.long)
    scoef = torch.randn((S, *trj_size), dtype=torch.complex64, device=dev)
    spat = torch.randn((L, *im_size), dtype=torch.complex64, device=dev)
    A = hofft_compressed_linop(trj, mps, compressed_kernels=comp, sparse_idxs=sidx,
                               sparse_coeffs=scoef, spatial_factors=spat,
                               os_grid=os_grid, bparams=bparams)
    return A


def main():
    dev = torch.device('cuda')
    torch.manual_seed(0)

    # --- Correctness on a moderate shape that the old path can also run ---
    im_size = (48, 48, 48)
    trj_size = (200, 8, 30)
    bparams = batching_params(coil_batch_size=1, field_batch_size=1)
    A = build(im_size, trj_size, C=4, L=4, kern=(3, 3, 3), S=8, Q=64,
              os_grid=1.0, bparams=bparams, dev=dev)
    img = torch.randn(im_size, dtype=torch.complex64, device=dev)

    out = A.forward(img)
    ref = reference_forward(A, img)
    rel = (out - ref).abs().max() / ref.abs().max()
    print(f"[correctness] max rel err: {rel.item():.3e}  -> {'PASS' if rel < 1e-4 else 'FAIL'}")

    # --- Peak memory: fused vs reference on a bigger shape ---
    im_size = (96, 96, 96)
    trj_size = (800, 8, 60)
    A = build(im_size, trj_size, C=6, L=8, kern=(3, 3, 3), S=8, Q=300,
              os_grid=1.0, bparams=bparams, dev=dev)
    img = torch.randn(im_size, dtype=torch.complex64, device=dev)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    out = A.forward(img)
    torch.cuda.synchronize()
    fused_peak = torch.cuda.max_memory_allocated() / 1e9

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    ref = reference_forward(A, img)
    torch.cuda.synchronize()
    ref_peak = torch.cuda.max_memory_allocated() / 1e9

    rel = (out - ref).abs().max() / ref.abs().max()
    print(f"[big] T={int(np.prod(trj_size))}  max rel err: {rel.item():.3e}")
    print(f"[big] fused peak mem     : {fused_peak:.3f} GB")
    print(f"[big] reference peak mem : {ref_peak:.3f} GB")
    print(f"[big] reduction          : {ref_peak / fused_peak:.1f}x")


if __name__ == '__main__':
    main()
