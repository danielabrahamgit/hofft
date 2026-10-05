from typing import Optional

import torch
import triton
import triton.language as tl

__all__ = [
    'hofft_forward_fused',
    'hofft_adjoint_fused',
]

# Used when auto_tune=False (the default): no search, no per-shape recompile
# cost, just launch directly with these. Manually found to work well for the
# highres_spiral (2D) case -- but per the module-level note below, the true
# optimum shifts with problem shape and GPU, so this is a starting point, not
# a universal answer. Pass auto_tune=True to search instead when you can
# afford the one-time cost (see hofft_forward_fused/hofft_adjoint_fused).
BLOCK_T_DEFAULT = 128
NUM_WARPS_DEFAULT = 8

# Candidate (BLOCK_T, num_warps) combinations to search over when auto_tune=True.
# The optimum shifts with T, Cb, and K (larger K in 3D changes register
# pressure per unrolled iteration, which changes the BLOCK_T/num_warps
# trade-off) -- so this is autotuned per-shape rather than hardcoded.
_AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_T': 64}, num_warps=2),
    triton.Config({'BLOCK_T': 64}, num_warps=4),
    triton.Config({'BLOCK_T': 128}, num_warps=4),
    triton.Config({'BLOCK_T': 128}, num_warps=8),
    triton.Config({'BLOCK_T': 256}, num_warps=4),
    triton.Config({'BLOCK_T': 256}, num_warps=8),
    triton.Config({'BLOCK_T': 512}, num_warps=8),
    triton.Config({'BLOCK_T': 512}, num_warps=16),
    triton.Config({'BLOCK_T': 1024}, num_warps=8),
    triton.Config({'BLOCK_T': 1024}, num_warps=16),
]
# Re-search whenever any of these (runtime, non-constexpr) args change --
# they're what plausibly shifts the optimal config. Lb/K/S/HAS_BIAS are
# tl.constexpr, so a change in any of those already compiles (and tunes) an
# entirely separate kernel specialization without needing to be listed here.
_AUTOTUNE_KEY = ['T', 'Cb', 'NPIX', 'Q']


@triton.jit
def _hofft_forward_kernel(
    fmsx_re_ptr, fmsx_im_ptr,        # (Cb * Lb * NPIX,) float32
    comp_re_ptr, comp_im_ptr,        # (Lb * K * Q,) float32
    sidx_ptr,                        # (S * T,) int (linear into Q, per trj point)
    sc_re_ptr, sc_im_ptr,            # (S * T,) float32  (sparse coeffs)
    idx_lin_ptr,                     # (T * K,) int (linear into NPIX, per trj point)
    bias_re_ptr, bias_im_ptr,        # (Lb * K,) float32  (per-(l,k) bias kernel)
    out_re_ptr, out_im_ptr,          # (Cb * T,) float32
    Cb, Q, NPIX, T,
    Lb: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    Fused forward: for each trajectory point t and coil c,

        out[c, t] = sum_l sum_k FMSx[c, l, idx_lin[t, k]]
                              * ( bias[l, k] + sum_s comp[l, k, sidx[s, t]] * sc[s, t] )

    One program handles a tile of BLOCK_T trajectory points across all coils.
    Neither the (C,L,K,*trj) blocks tensor nor the (L,K,S,*trj) gather is
    materialized -- everything is accumulated in registers per point.

    `Lb`/`K`/`S` are tl.constexpr so the (l, k, s) loops fully unroll at
    compile time: the compiler can then interleave/pipeline the otherwise
    latency-bound dependent loads across iterations instead of executing a
    genuine serial runtime loop with a branch every iteration.
    """
    pid = tl.program_id(0)
    offs_t = (pid * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    mask_t = offs_t < T

    offs_c = tl.arange(0, BLOCK_C).to(tl.int64)
    mask_c = offs_c < Cb

    fmask = mask_c[:, None] & mask_t[None, :]

    acc_re = tl.zeros((BLOCK_C, BLOCK_T), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_C, BLOCK_T), dtype=tl.float32)

    for l in range(Lb):
        for k in range(K):
            # kern_weights[l, k, t] = sum_s comp[l, k, sidx[s, t]] * sc[s, t]
            w_re = tl.zeros((BLOCK_T,), dtype=tl.float32)
            w_im = tl.zeros((BLOCK_T,), dtype=tl.float32)
            ck_base = (l * K + k) * Q
            for s in range(S):
                s_off = s * T + offs_t
                sidx = tl.load(sidx_ptr + s_off, mask=mask_t, other=0).to(tl.int64)
                ck_re = tl.load(comp_re_ptr + ck_base + sidx, mask=mask_t, other=0.0)
                ck_im = tl.load(comp_im_ptr + ck_base + sidx, mask=mask_t, other=0.0)
                sc_re = tl.load(sc_re_ptr + s_off, mask=mask_t, other=0.0)
                sc_im = tl.load(sc_im_ptr + s_off, mask=mask_t, other=0.0)
                w_re += ck_re * sc_re - ck_im * sc_im
                w_im += ck_re * sc_im + ck_im * sc_re

            if HAS_BIAS:
                # bias[l, k] is constant across t: one scalar load, broadcast
                bk_off = l * K + k
                w_re += tl.load(bias_re_ptr + bk_off)
                w_im += tl.load(bias_im_ptr + bk_off)

            # gather FMSx[c, l, idx_lin[t, k]] for all coils -> (BLOCK_C, BLOCK_T)
            gidx = tl.load(idx_lin_ptr + offs_t * K + k, mask=mask_t, other=0).to(tl.int64)
            f_off = (offs_c[:, None] * Lb + l) * NPIX + gidx[None, :]
            f_re = tl.load(fmsx_re_ptr + f_off, mask=fmask, other=0.0)
            f_im = tl.load(fmsx_im_ptr + f_off, mask=fmask, other=0.0)

            acc_re += f_re * w_re[None, :] - f_im * w_im[None, :]
            acc_im += f_re * w_im[None, :] + f_im * w_re[None, :]

    out_off = offs_c[:, None] * T + offs_t[None, :]
    tl.store(out_re_ptr + out_off, acc_re, mask=fmask)
    tl.store(out_im_ptr + out_off, acc_im, mask=fmask)


# Autotuned kernels are built lazily. triton.autotune() touches the active CUDA
# driver at construction time, so doing it at import fails on CPU-only hosts
# (RuntimeError: 0 active drivers). See _get_forward_kernel_autotuned /
# _get_adjoint_kernel_autotuned below.
_hofft_forward_kernel_autotuned = None
_hofft_adjoint_kernel_autotuned = None


def _get_forward_kernel_autotuned():
    """Lazily wrap _hofft_forward_kernel with @triton.autotune (CUDA only)."""
    global _hofft_forward_kernel_autotuned
    if _hofft_forward_kernel_autotuned is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "triton.autotune requires a CUDA device; "
                "call with auto_tune=False on CPU, or run on GPU.")
        _hofft_forward_kernel_autotuned = triton.autotune(
            configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY,
        )(_hofft_forward_kernel)
    return _hofft_forward_kernel_autotuned


def hofft_forward_fused(FMSx: torch.Tensor,
                        idx_lin: torch.Tensor,
                        comp_kernels: torch.Tensor,
                        sparse_idxs: torch.Tensor,
                        sparse_coeffs: torch.Tensor,
                        bias_kernel: Optional[torch.Tensor] = None,
                        auto_tune: bool = False) -> torch.Tensor:
    """
    Fused HOFFT forward block-extraction + sparse-kernel apply.

    Computes, for each coil c and trajectory point t:
        out[c, t] = sum_l sum_k FMSx[c, l, idx_lin[t, k]]
                              * ( bias_kernel[l, k]
                                  + sum_s comp_kernels[l, k, sparse_idxs[s, t]] * sparse_coeffs[s, t] )

    Args
    ----
    FMSx : (Cb, Lb, NPIX) complex64, contiguous
        Oversampled FFT image, spatial dims flattened to NPIX = prod(im_size_os).
    idx_lin : (T, K) int (int32/int64)
        Raveled linear indices into NPIX for each (trajectory point, kernel offset).
    comp_kernels : (Lb, K, Q) complex64
        Compressed kernels (this field batch).
    sparse_idxs : (S, T) int
        Sparse indices into Q for each (sparsity term, trajectory point).
    sparse_coeffs : (S, T) complex64
        Sparse coefficients.
    bias_kernel : (Lb, K) complex64, optional
        Per-(field, kernel offset) bias added to every trajectory point's
        reconstructed kernel weight. If None, no bias is applied.
    auto_tune : bool
        If False (default), launches directly with BLOCK_T_DEFAULT/
        NUM_WARPS_DEFAULT -- no search, no one-time per-shape tuning cost.
        If True, uses @triton.autotune to search _AUTOTUNE_CONFIGS the first
        time this (T, Cb, NPIX, Q) shape is seen, then caches the winner.

    Returns
    -------
    out : (Cb, T) complex64
        k-space contribution for this (coil, field) batch.
    """
    assert FMSx.is_cuda, "fused kernel requires CUDA tensors"
    Cb, Lb, NPIX = FMSx.shape
    Lk, K, Q = comp_kernels.shape
    S, T = sparse_idxs.shape
    assert Lk == Lb, f"FMSx field dim {Lb} != comp_kernels field dim {Lk}"
    assert idx_lin.shape == (T, K), f"idx_lin shape {tuple(idx_lin.shape)} != {(T, K)}"
    assert sparse_coeffs.shape == (S, T)

    dev = FMSx.device

    # Split complex into contiguous real/imag float32 buffers
    fmsx = FMSx.reshape(-1).contiguous()
    fmsx_re = fmsx.real.contiguous()
    fmsx_im = fmsx.imag.contiguous()

    comp = comp_kernels.reshape(-1).contiguous()
    comp_re = comp.real.contiguous()
    comp_im = comp.imag.contiguous()

    sc = sparse_coeffs.reshape(-1).contiguous()
    sc_re = sc.real.contiguous()
    sc_im = sc.imag.contiguous()

    sidx = sparse_idxs.reshape(-1).contiguous().to(torch.int32)
    idx_lin_flat = idx_lin.reshape(-1).contiguous().to(torch.int32)

    HAS_BIAS = bias_kernel is not None
    if HAS_BIAS:
        assert bias_kernel.shape == (Lb, K), \
            f"bias_kernel shape {tuple(bias_kernel.shape)} != {(Lb, K)}"
        bk = bias_kernel.reshape(-1).contiguous()
        bias_re = bk.real.contiguous()
        bias_im = bk.imag.contiguous()
    else:
        bias_re = torch.empty((1,), device=dev, dtype=torch.float32)
        bias_im = bias_re

    return _launch_forward_kernel(fmsx_re, fmsx_im, comp_re, comp_im, sidx, sc_re, sc_im,
                                  idx_lin_flat, bias_re, bias_im, HAS_BIAS,
                                  Cb, Lb, K, Q, NPIX, T, S, auto_tune)


def _launch_forward_kernel(fmsx_re: torch.Tensor, fmsx_im: torch.Tensor,
                           comp_re: torch.Tensor, comp_im: torch.Tensor,
                           sidx: torch.Tensor, sc_re: torch.Tensor, sc_im: torch.Tensor,
                           idx_lin_flat: torch.Tensor, bias_re: torch.Tensor, bias_im: torch.Tensor,
                           HAS_BIAS: bool, Cb: int, Lb: int, K: int, Q: int, NPIX: int,
                           T: int, S: int, auto_tune: bool = False) -> torch.Tensor:
    """Launches _hofft_forward_kernel given already-flattened, contiguous,
    correctly-typed (float32/int32) buffers. No marshalling is performed here --
    callers own splitting/caching those buffers."""
    dev = fmsx_re.device
    out_re = torch.empty((Cb * T,), device=dev, dtype=torch.float32)
    out_im = torch.empty((Cb * T,), device=dev, dtype=torch.float32)

    BLOCK_C = max(triton.next_power_of_2(Cb), 1)
    common_args = (fmsx_re, fmsx_im, comp_re, comp_im, sidx, sc_re, sc_im,
                   idx_lin_flat, bias_re, bias_im, out_re, out_im, Cb, Q, NPIX, T)
    common_kwargs = dict(Lb=Lb, K=K, S=S, HAS_BIAS=HAS_BIAS, BLOCK_C=BLOCK_C)

    if auto_tune:
        grid = lambda META: (triton.cdiv(T, META['BLOCK_T']),)
        _get_forward_kernel_autotuned()[grid](*common_args, **common_kwargs)
    else:
        grid = (triton.cdiv(T, BLOCK_T_DEFAULT),)
        _hofft_forward_kernel[grid](*common_args, **common_kwargs,
                                    BLOCK_T=BLOCK_T_DEFAULT, num_warps=NUM_WARPS_DEFAULT)

    out = torch.complex(out_re, out_im).reshape(Cb, T)
    return out

@triton.jit
def _hofft_adjoint_kernel(
    y_re_ptr, y_im_ptr,              # (Cb * T,) float32
    comp_re_ptr, comp_im_ptr,        # (Lb * K * Q,) float32
    sidx_ptr,                        # (S * T,) int
    sc_re_ptr, sc_im_ptr,            # (S * T,) float32
    idx_lin_ptr,                     # (T * K,) int
    bias_re_ptr, bias_im_ptr,        # (Lb * K,) float32  (per-(l,k) bias kernel)
    grid_re_ptr, grid_im_ptr,        # (Cb * Lb * NPIX,) float32, zero-init, atomic_add
    Cb, Q, NPIX, T,
    Lb: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    Fused adjoint (transpose of _hofft_forward_kernel). For each trajectory
    point t, coil c, field l, kernel offset k:

        grid[c, l, idx_lin[t, k]] += y[c, t]
                                     * conj( bias[l, k] + sum_s comp[l, k, sidx[s, t]] * sc[s, t] )

    Scatter-adds (via atomic_add) into the oversampled grid. This is the exact
    transpose of the forward gather+contract, so A^H matches A for CG.

    `Lb`/`K`/`S` are tl.constexpr for the same reason as in
    `_hofft_forward_kernel` -- fully unrolled loops let the compiler pipeline
    the dependent loads instead of serializing them.
    """
    pid = tl.program_id(0)
    offs_t = (pid * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    mask_t = offs_t < T

    offs_c = tl.arange(0, BLOCK_C).to(tl.int64)
    mask_c = offs_c < Cb
    cmask = mask_c[:, None] & mask_t[None, :]

    # Load y[c, t] for all coils -> (BLOCK_C, BLOCK_T)
    y_off = offs_c[:, None] * T + offs_t[None, :]
    y_re = tl.load(y_re_ptr + y_off, mask=cmask, other=0.0)
    y_im = tl.load(y_im_ptr + y_off, mask=cmask, other=0.0)

    for l in range(Lb):
        for k in range(K):
            # kern_weights[l, k, t] = sum_s comp[l, k, sidx[s, t]] * sc[s, t]
            w_re = tl.zeros((BLOCK_T,), dtype=tl.float32)
            w_im = tl.zeros((BLOCK_T,), dtype=tl.float32)
            ck_base = (l * K + k) * Q
            for s in range(S):
                s_off = s * T + offs_t
                sidx = tl.load(sidx_ptr + s_off, mask=mask_t, other=0).to(tl.int64)
                ck_re = tl.load(comp_re_ptr + ck_base + sidx, mask=mask_t, other=0.0)
                ck_im = tl.load(comp_im_ptr + ck_base + sidx, mask=mask_t, other=0.0)
                sc_re = tl.load(sc_re_ptr + s_off, mask=mask_t, other=0.0)
                sc_im = tl.load(sc_im_ptr + s_off, mask=mask_t, other=0.0)
                w_re += ck_re * sc_re - ck_im * sc_im
                w_im += ck_re * sc_im + ck_im * sc_re

            if HAS_BIAS:
                # bias[l, k] is constant across t: one scalar load, broadcast
                bk_off = l * K + k
                w_re += tl.load(bias_re_ptr + bk_off)
                w_im += tl.load(bias_im_ptr + bk_off)

            # val[c, t] = y[c, t] * conj(w[t]);  conj(w) = w_re - i w_im
            # (a + bi)(w_re - i w_im) = (a*w_re + b*w_im) + i(b*w_re - a*w_im)
            vr = y_re * w_re[None, :] + y_im * w_im[None, :]
            vi = y_im * w_re[None, :] - y_re * w_im[None, :]

            gidx = tl.load(idx_lin_ptr + offs_t * K + k, mask=mask_t, other=0).to(tl.int64)
            g_off = (offs_c[:, None] * Lb + l) * NPIX + gidx[None, :]
            tl.atomic_add(grid_re_ptr + g_off, vr, mask=cmask)
            tl.atomic_add(grid_im_ptr + g_off, vi, mask=cmask)


def _get_adjoint_kernel_autotuned():
    """Lazily wrap _hofft_adjoint_kernel with @triton.autotune (CUDA only).

    reset_to_zero is required here (unlike the forward kernel): this kernel
    accumulates into grid_re_ptr/grid_im_ptr via atomic_add rather than
    overwriting via tl.store, so without it, repeated benchmark trials during
    the search would add on top of each other and corrupt the result of the
    call that triggers the search.
    """
    global _hofft_adjoint_kernel_autotuned
    if _hofft_adjoint_kernel_autotuned is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "triton.autotune requires a CUDA device; "
                "call with auto_tune=False on CPU, or run on GPU.")
        _hofft_adjoint_kernel_autotuned = triton.autotune(
            configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY,
            reset_to_zero=['grid_re_ptr', 'grid_im_ptr'],
        )(_hofft_adjoint_kernel)
    return _hofft_adjoint_kernel_autotuned


def hofft_adjoint_fused(y: torch.Tensor,
                        idx_lin: torch.Tensor,
                        comp_kernels: torch.Tensor,
                        sparse_idxs: torch.Tensor,
                        sparse_coeffs: torch.Tensor,
                        NPIX: int,
                        bias_kernel: Optional[torch.Tensor] = None,
                        auto_tune: bool = False) -> torch.Tensor:
    """
    Fused HOFFT adjoint: gridding of k-space onto the oversampled grid.

    Exact transpose of `hofft_forward_fused`. Computes, for each coil c and
    field l:
        grid[c, l, p] = sum_{t, k : idx_lin[t,k]==p} y[c, t]
                        * conj( bias_kernel[l, k]
                                + sum_s comp_kernels[l, k, sparse_idxs[s, t]] * sparse_coeffs[s, t] )

    Args
    ----
    y : (Cb, T) complex64
        DCF-weighted k-space for this coil batch (trajectory flattened to T).
    idx_lin : (T, K) int
        Raveled linear indices into NPIX.
    comp_kernels : (Lb, K, Q) complex64
    sparse_idxs : (S, T) int
    sparse_coeffs : (S, T) complex64
    NPIX : int
        prod(im_size_os).
    bias_kernel : (Lb, K) complex64, optional
        Per-(field, kernel offset) bias added to every trajectory point's
        reconstructed kernel weight (matching the forward). If None, no bias.
    auto_tune : bool
        If False (default), launches directly with BLOCK_T_DEFAULT/
        NUM_WARPS_DEFAULT -- no search, no one-time per-shape tuning cost.
        If True, uses @triton.autotune to search _AUTOTUNE_CONFIGS the first
        time this (T, Cb, NPIX, Q) shape is seen, then caches the winner.

    Returns
    -------
    grid : (Cb, Lb, NPIX) complex64
        Gridded oversampled k-space (ready for ifft).
    """
    assert y.is_cuda, "fused kernel requires CUDA tensors"
    Cb, T = y.shape
    Lb, K, Q = comp_kernels.shape
    S, Ts = sparse_idxs.shape
    assert Ts == T
    assert idx_lin.shape == (T, K), f"idx_lin shape {tuple(idx_lin.shape)} != {(T, K)}"
    assert sparse_coeffs.shape == (S, T)

    dev = y.device

    yf = y.reshape(-1).contiguous()
    y_re = yf.real.contiguous()
    y_im = yf.imag.contiguous()

    comp = comp_kernels.reshape(-1).contiguous()
    comp_re = comp.real.contiguous()
    comp_im = comp.imag.contiguous()

    sc = sparse_coeffs.reshape(-1).contiguous()
    sc_re = sc.real.contiguous()
    sc_im = sc.imag.contiguous()

    sidx = sparse_idxs.reshape(-1).contiguous().to(torch.int32)
    idx_lin_flat = idx_lin.reshape(-1).contiguous().to(torch.int32)

    HAS_BIAS = bias_kernel is not None
    if HAS_BIAS:
        assert bias_kernel.shape == (Lb, K), \
            f"bias_kernel shape {tuple(bias_kernel.shape)} != {(Lb, K)}"
        bk = bias_kernel.reshape(-1).contiguous()
        bias_re = bk.real.contiguous()
        bias_im = bk.imag.contiguous()
    else:
        bias_re = torch.empty((1,), device=dev, dtype=torch.float32)
        bias_im = bias_re

    return _launch_adjoint_kernel(y_re, y_im, comp_re, comp_im, sidx, sc_re, sc_im,
                                  idx_lin_flat, bias_re, bias_im, HAS_BIAS,
                                  Cb, Lb, K, Q, NPIX, T, S, auto_tune)


def _launch_adjoint_kernel(y_re: torch.Tensor, y_im: torch.Tensor,
                           comp_re: torch.Tensor, comp_im: torch.Tensor,
                           sidx: torch.Tensor, sc_re: torch.Tensor, sc_im: torch.Tensor,
                           idx_lin_flat: torch.Tensor, bias_re: torch.Tensor, bias_im: torch.Tensor,
                           HAS_BIAS: bool, Cb: int, Lb: int, K: int, Q: int, NPIX: int,
                           T: int, S: int, auto_tune: bool = False) -> torch.Tensor:
    """Launches _hofft_adjoint_kernel given already-flattened, contiguous,
    correctly-typed (float32/int32) buffers. No marshalling is performed here --
    callers own splitting/caching those buffers."""
    dev = y_re.device
    grid_re = torch.zeros((Cb * Lb * NPIX,), device=dev, dtype=torch.float32)
    grid_im = torch.zeros((Cb * Lb * NPIX,), device=dev, dtype=torch.float32)

    BLOCK_C = max(triton.next_power_of_2(Cb), 1)
    common_args = (y_re, y_im, comp_re, comp_im, sidx, sc_re, sc_im,
                   idx_lin_flat, bias_re, bias_im, grid_re, grid_im, Cb, Q, NPIX, T)
    common_kwargs = dict(Lb=Lb, K=K, S=S, HAS_BIAS=HAS_BIAS, BLOCK_C=BLOCK_C)

    if auto_tune:
        grid = lambda META: (triton.cdiv(T, META['BLOCK_T']),)
        _get_adjoint_kernel_autotuned()[grid](*common_args, **common_kwargs)
    else:
        grid = (triton.cdiv(T, BLOCK_T_DEFAULT),)
        _hofft_adjoint_kernel[grid](*common_args, **common_kwargs,
                                    BLOCK_T=BLOCK_T_DEFAULT, num_warps=NUM_WARPS_DEFAULT)

    out = torch.complex(grid_re, grid_im).reshape(Cb, Lb, NPIX)
    return out
