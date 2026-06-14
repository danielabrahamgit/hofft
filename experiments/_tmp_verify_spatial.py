"""Temp script: verify new spatial normal equations vs old einsum chain at realistic sizes."""
import time
import torch
from einops import einsum

torch.manual_seed(0)
dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
dtype = torch.complex64

L, K, Q = 5, 125, 200   # K = prod(kern_size) = 5^3
N = 48**3               # moderate so the OLD code fits unbatched
B = 2**16               # batch size used by new code

ck = torch.randn(L, K, Q, dtype=dtype, device=dev) / K**0.5
kern = torch.randn(K, N, dtype=dtype, device=dev)
sb = torch.randn(Q, N, dtype=dtype, device=dev)
mask = torch.ones(N, dtype=dtype, device=dev)  # ones so old (mask^1 in f) == new (mask^2)


def old_spatial(ck, sb, kern, mask, batch=120):  # batch=120 as in pipelines.py before
    M = torch.zeros((N, L, L), dtype=dtype, device=dev)
    f = torch.zeros((N, L), dtype=dtype, device=dev)
    for n1 in range(0, N, batch):
        n2 = min(n1 + batch, N)
        left = einsum(kern[:, n1:n2].conj() * mask[n1:n2], ck.conj(), 'K N, L K Q -> N L Q')
        right = einsum(kern[:, n1:n2] * mask[n1:n2], ck, 'K N, L K Q -> N L Q')
        M[n1:n2] = einsum(left, right, 'N L1 Q, N L2 Q -> N L1 L2')
        inner = einsum(ck.conj(), kern[:, n1:n2].conj() * mask[n1:n2], 'L K Q, K N -> N Q L')
        f[n1:n2] = einsum(inner, sb[:, n1:n2], 'N Q L, Q N -> N L')
    return M, f


def new_spatial(ck, sb, kern, mask, batch=B):
    mask2 = mask * mask
    M = torch.empty((N, L, L), dtype=dtype, device=dev)
    f = torch.empty((N, L), dtype=dtype, device=dev)
    for n1 in range(0, N, batch):
        n2 = min(n1 + batch, N)
        A = einsum(ck, kern[:, n1:n2], 'L K Q, K N -> N L Q')
        Ah = A.conj() * mask2[n1:n2, None, None]
        M[n1:n2] = einsum(Ah, A, 'N L1 Q, N L2 Q -> N L1 L2')
        f[n1:n2] = einsum(Ah, sb[:, n1:n2], 'N L Q, Q N -> N L')
    return M, f


def timeit(fn, *args, reps=3, **kw):
    fn(*args, **kw)  # warmup
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args, **kw)
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)
    return out, (time.perf_counter() - t0) / reps


(M_new, f_new), t_new = timeit(new_spatial, ck, sb, kern, mask)
(M_old, f_old), t_old = timeit(old_spatial, ck, sb, kern, mask, reps=1, batch=2**14)
_, t_old120 = timeit(old_spatial, ck, sb, kern, mask, reps=1)  # original batch=120

err_M = (M_old - M_new).abs().max() / M_old.abs().max()
err_f = (f_old - f_new).abs().max() / f_old.abs().max()
print(f'device={dev}  L={L} K={K} Q={Q} N={N}')
print(f'M rel err: {err_M:.2e}   f rel err: {err_f:.2e}')
print(f'old (batch=120):   {t_old120*1e3:9.1f} ms')
print(f'old (batch=2^14):  {t_old*1e3:9.1f} ms')
print(f'new (batch=2^16):  {t_new*1e3:9.1f} ms   speedup vs orig: {t_old120/t_new:.1f}x')

# Solver comparison (jitter like in decomp.py)
Mj = M_new.clone()
diag_scale = Mj.diagonal(dim1=-2, dim2=-1).real.mean()
Mj.diagonal(dim1=-2, dim2=-1).add_(1e-6 * diag_scale)
_, t_pinv = timeit(lambda: torch.linalg.pinv(M_new) @ f_new[..., None])
_, t_solve = timeit(lambda: torch.linalg.solve(Mj, f_new[..., None]))
x_pinv = torch.linalg.pinv(M_new) @ f_new[..., None]
x_solve = torch.linalg.solve(Mj, f_new[..., None])
err_x = (x_pinv - x_solve).abs().max() / x_pinv.abs().max()
print(f'pinv: {t_pinv*1e3:9.1f} ms   solve: {t_solve*1e3:9.1f} ms   '
      f'speedup: {t_pinv/t_solve:.1f}x   sol rel err: {err_x:.2e}')
