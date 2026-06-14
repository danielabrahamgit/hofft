import torch
import numpy as np

from mr_recon.linops import batching_params, sense_linop
from mr_recon.fourier import sigpy_nufft
from hofft.forward_model import hofft_compressed_linop


def main():
    dev = torch.device(1)
    torch.cuda.set_device(dev)
    torch.manual_seed(1)


    # Realistic b0_3d shapes
    im_size = (220, 220, 220)
    trj_size = (1678, 16, 500)
    C, L, S, Q = 6, 8, 8, 1000
    kern = (5,) * 3
    os_grid = 1.25
    os_grid = 2 * round(os_grid * im_size[0] / 2) / im_size[0]

    D = len(im_size)
    trj = torch.zeros((*trj_size, D), device=dev)
    mps = torch.randn((C, *im_size), dtype=torch.complex64, device=dev)
    comp = torch.randn((L, *kern, Q), dtype=torch.complex64, device=dev)
    sidx = torch.randint(0, Q, (S, *trj_size), device=dev, dtype=torch.long)
    scoef = torch.randn((S, *trj_size), dtype=torch.complex64, device=dev)
    spat = torch.randn((L, *im_size), dtype=torch.complex64, device=dev)
    
    # HOFFT compressed
    bparams = batching_params(coil_batch_size=1, field_batch_size=1)
    A = hofft_compressed_linop(trj, mps, compressed_kernels=comp, sparse_idxs=sidx,
                               sparse_coeffs=scoef, spatial_factors=spat,
                               os_grid=os_grid, bparams=bparams)
    
    # NUFFT
    nft = sigpy_nufft(im_size, oversamp=os_grid, width=kern[0])
    Anufft = sense_linop(trj, mps, 
                         nufft=nft, 
                         bparams=bparams)
    
    # TS-NUFFT
    cs = torch.randn((L, *trj_size), dtype=torch.complex64, device=dev)
    Anufft_ts = sense_linop(trj, mps, 
                            spatial_funcs=spat,
                            temporal_funcs=cs,
                            nufft=nft, 
                            bparams=bparams)

    import time
    img = torch.randn(im_size, dtype=torch.complex64, device=dev)
    base = torch.cuda.memory_allocated() / 1e9
    ksp = torch.randn(A.oshape, dtype=torch.complex64, device=dev)

    def timed(fn, label, n=3):
        fn(); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(n):
            out = fn()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{label:9s}: warm {dt*1e3:7.1f} ms   peak {peak:5.2f} GB")
        return out

    print(f"resident before ops : {base:.2f} GB")
    timed(lambda: Anufft.normal(img), "NUFFT normal")
    timed(lambda: Anufft_ts.normal(img), "TS-NUFFT normal")
    timed(lambda: A.normal(img), "HOFFT normal")
    print(f"(old path needed ~23 GB for a single (L,K,S,*trj) gather and OOM'd)")


if __name__ == '__main__':
    main()
