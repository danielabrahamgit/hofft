import torch, time, numpy as np
from einops import einsum
from mr_recon.fourier import fft
from hofft.triton_forward import hofft_forward_fused

dev = torch.device('cuda')
torch.manual_seed(0)

im_size = (220, 220, 220)
trj_size = (1678, 16, 500)
C, L, S, Q = 6, 8, 8, 1000
K = 27
os_grid = 2 * round(1.25 * im_size[0] / 2) / im_size[0]
im_os = [round(s * os_grid) for s in im_size]
NPIX = int(np.prod(im_os))
T = int(np.prod(trj_size))

FMSx = torch.randn((1, 1, NPIX), dtype=torch.complex64, device=dev)
comp = torch.randn((1, K, Q), dtype=torch.complex64, device=dev)
idx_lin = torch.randint(0, NPIX, (T, K), device=dev, dtype=torch.int32)
sidx = torch.randint(0, Q, (S, T), device=dev, dtype=torch.long)
scoef = torch.randn((S, T), dtype=torch.complex64, device=dev)

def timeit(fn, n=5):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3

# Triton kernel only (one (c,l) batch)
t_kern = timeit(lambda: hofft_forward_fused(FMSx, idx_lin, comp, sidx, scoef))
print(f"triton kernel (1 c,l batch): {t_kern:.1f} ms  -> x48 = {t_kern*48:.0f} ms")

# FFT only (one 275^3 batch)
x = torch.randn((1, 1, *im_os), dtype=torch.complex64, device=dev)
t_fft = timeit(lambda: fft(x, dim=(-3, -2, -1)))
print(f"3D FFT ({im_os}) 1 batch     : {t_fft:.1f} ms  -> x48 = {t_fft*48:.0f} ms")

for bt in [128, 256, 512, 1024]:
    t = timeit(lambda: hofft_forward_fused(FMSx, idx_lin, comp, sidx, scoef, BLOCK_T=bt))
    print(f"  BLOCK_T={bt:5d}: {t:.1f} ms")
