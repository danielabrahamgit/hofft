import torch
from einops import einsum

from mr_recon._func.indexing import multi_index, multi_grid, ravel
from hofft.triton_forward import hofft_forward_fused, hofft_adjoint_fused


def reference(FMSx, idx_kerns, comp_kernels, sparse_idxs, sparse_coeffs, D):
    """Current PyTorch path (forward_model.py lines 357-366) for one (C,L) batch."""
    blocks = multi_index(FMSx, D, idx_kerns)          # (C, L, *trj_size, K)
    blocks = blocks.moveaxis(-1, 2)                    # (C, L, K, *trj_size)
    kw = comp_kernels[:, :, sparse_idxs]               # (L, K, S, *trj_size)
    kw = einsum(kw, sparse_coeffs, 'L K S ..., S ... -> L K ...')
    return einsum(blocks, kw, 'C L K ..., L K ... -> C ...')


def reference_adjoint(y, idx_kerns, comp_kernels, sparse_idxs, sparse_coeffs, im_size_os):
    """Current PyTorch adjoint path (forward_model.py lines 413-421)."""
    kw = comp_kernels[:, :, sparse_idxs]               # (L, K, S, *trj_size)
    kw = einsum(kw, sparse_coeffs, 'L K S ..., S ... -> L K ...')
    Ky = einsum(y, kw.conj(), 'C ..., L K ... -> C L ... K')
    Ky = multi_grid(Ky, idx_kerns, im_size_os)         # (C, L, *im_size_os)
    return Ky


def main():
    torch.manual_seed(0)
    dev = torch.device('cuda')

    # Small but non-trivial shapes
    C, L, K, S, Q, D = 3, 4, 27, 8, 50, 3
    im_size_os = (16, 18, 20)
    trj_size = (40, 16, 9)
    NPIX = int(torch.tensor(im_size_os).prod())
    T = int(torch.tensor(trj_size).prod())

    FMSx = torch.randn((C, L, *im_size_os), dtype=torch.complex64, device=dev)
    comp_kernels = torch.randn((L, K, Q), dtype=torch.complex64, device=dev)
    sparse_idxs = torch.randint(0, Q, (S, *trj_size), device=dev, dtype=torch.long)
    sparse_coeffs = torch.randn((S, *trj_size), dtype=torch.complex64, device=dev)

    im_size_os_t = torch.tensor(im_size_os, device=dev)
    idx_kerns = torch.randint(0, 10_000, (*trj_size, K, D), device=dev, dtype=torch.int32)
    idx_kerns = idx_kerns % im_size_os_t.to(torch.int32)

    # Reference
    ref = reference(FMSx, idx_kerns, comp_kernels, sparse_idxs, sparse_coeffs, D)
    ref = ref.reshape(C, T)

    # Fused
    idx_lin = ravel(idx_kerns, im_size_os, dim=-1).reshape(T, K)   # (T, K)
    FMSx_flat = FMSx.reshape(C, L, NPIX)
    out = hofft_forward_fused(FMSx_flat, idx_lin,
                              comp_kernels,
                              sparse_idxs.reshape(S, T),
                              sparse_coeffs.reshape(S, T))

    err = (out - ref).abs()
    rel = err.max() / ref.abs().max()
    print(f"NPIX={NPIX} T={T}")
    print("--- forward ---")
    print(f"max abs err : {err.max().item():.3e}")
    print(f"max rel err : {rel.item():.3e}")
    print(f"ref scale   : {ref.abs().max().item():.3e}")
    assert rel < 1e-4, "Fused forward does not match reference!"
    print("forward PASS")

    # --- Adjoint vs reference ---
    y = torch.randn((C, *trj_size), dtype=torch.complex64, device=dev)
    ref_adj = reference_adjoint(y, idx_kerns, comp_kernels, sparse_idxs, sparse_coeffs,
                                im_size_os).reshape(C, L, NPIX)
    out_adj = hofft_adjoint_fused(y.reshape(C, T), idx_lin, comp_kernels,
                                  sparse_idxs.reshape(S, T), sparse_coeffs.reshape(S, T),
                                  NPIX=NPIX)
    rel_adj = (out_adj - ref_adj).abs().max() / ref_adj.abs().max()
    print("--- adjoint ---")
    print(f"max rel err : {rel_adj.item():.3e}")
    assert rel_adj < 1e-4, "Fused adjoint does not match reference!"
    print("adjoint PASS")

    # --- Dot-product (transpose) test: <A x, y> == <x, A^H y> ---
    # A x = forward over flattened FMSx input; here treat FMSx as the input vector.
    Ax = out                              # (C, T)
    AHy = out_adj                         # (C, L, NPIX)
    lhs = (Ax.conj() * y.reshape(C, T)).sum()
    rhs = (FMSx.reshape(C, L, NPIX).conj() * AHy).sum()
    rel_dot = (lhs - rhs).abs() / lhs.abs()
    print("--- dot-product test  <Ax,y> vs <x,A^H y> ---")
    print(f"<Ax, y>   = {lhs.item():.6e}")
    print(f"<x, A^H y>= {rhs.item():.6e}")
    print(f"rel diff  : {rel_dot.item():.3e}")
    assert rel_dot < 1e-4, "Forward/adjoint are not transposes!"
    print("transpose PASS")


if __name__ == '__main__':
    main()
