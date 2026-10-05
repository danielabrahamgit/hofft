"""
Validation for the block machinery (tests 1-6 of math_docs/Qblock_svd_feas.md).

Test 4 runs first, as instructed: it is the one that validates the Sec. 1.2 claim that a
block's coarser k-space grid costs no accuracy, and every cost number in the study is
void if it fails.

Run with:
    PYTHONPATH=src python paper_experiments/qblock_feas/test_blocks.py
"""
import sys

import numpy as np
import torch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks import (  # noqa: E402
    allocate_ranks,
    block_kb_forward,
    block_os,
    block_singular_values,
    dense_block_gather,
    fit_block_affine,
    global_error_curve,
    make_blocks,
    make_slabs,
    phase_matrix,
    pooled_error_curve,
)
from hofft.phase_coeffs import remove_linear_terms  # noqa: E402
from hofft.utils import gen_grd  # noqa: E402

FAILED = []


def check(name: str, cond: bool, msg: str = '') -> None:
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f'  ({msg})' if msg else ''))
    if not cond:
        FAILED.append(name)


def _disc_mask(im_size, torch_dev, radius=0.45):
    rs = gen_grd(im_size).to(torch_dev)
    return rs.square().sum(-1).sqrt() < radius


# ---------------------------------------------------------------------------
# Test 4 -- block FFT accuracy (validates Sec. 1.2; run before anything else)
# ---------------------------------------------------------------------------
def _kb_rel_err(size, os, W, torch_dev, M=4096, seed=0):
    """
    Relative KB-NUFFT error for white noise filling the FOV of a ``size`` grid.

    White noise is the roughest possible content, so this is the worst case, and it fills
    the FOV in both the global and the block case -- which is the only fair comparison.
    """
    gen = torch.Generator(device=torch_dev)
    gen.manual_seed(seed)
    d = len(size)
    x = torch.randn(size, dtype=torch.complex64, device=torch_dev, generator=gen)
    kap = (torch.rand((M, d), device=torch_dev, generator=gen) - 0.5) * size[0]
    y = block_kb_forward(x, kap, size, os, W).type(torch.complex128)
    ref = dense_block_gather(x, kap, size).type(torch.complex128)
    return float((y - ref).norm() / ref.norm())


def test_block_fft_accuracy(torch_dev):
    """
    A block's padded FFT is coarser in k by 1/D, but its content is compactly supported
    over extent D, so a W-tap KB stencil should reach the same accuracy as in the global
    case. Two separate claims are checked:

    (a) accuracy parity -- the KB error at grid size V equals the KB error at grid size N,
        each measured on content filling its own FOV. Feeding the global NUFFT a
        block-supported image instead would flatter it by the oversampling factor 1/D and
        prove nothing.
    (b) geometry -- the local-coordinate path (kappa = D k, plus the exp(-2j pi k . r_c)
        center phase, plus wrapped indexing) reproduces a direct dense evaluation on the
        global grid, to that same KB precision.
    """
    print('\ntest 4: block FFT accuracy at matched (os, W)')
    N, os = 220, 1.25
    torch.manual_seed(0)

    for W in (3, 6):
        e_glob = _kb_rel_err((N, N), os, W, torch_dev)
        for V in (55, 44, 20):
            e_blk = _kb_rel_err((V, V), os, W, torch_dev)
            V_os, os_q = block_os(os, (V, V))
            check(f'(a) V={V}, W={W}: block KB error matches global',
                  e_blk < 1.5 * e_glob,
                  f'block {e_blk:.3e} vs global {e_glob:.3e} ({e_blk / e_glob:.2f}x), '
                  f'{V_os[0]}-pt FFT, delta_k={N / V_os[0]:.2f}, 1/D={N / V:.1f}, '
                  f'os_q={os_q[0]:.3f}')

    for V, W in ((55, 3), (20, 3), (55, 6), (20, 6)):
        im_size, V_size = (N, N), (V, V)
        # A block placed off-center, so the exp(-2j pi k . r_c) factor is exercised
        win_lo = np.array([37, 148])
        x_local = torch.randn(V_size, dtype=torch.complex64, device=torch_dev)

        M = 4096
        k = (torch.rand((M, 2), device=torch_dev) - 0.5) * N

        # Exact, in global coordinates over the block's voxels
        sub = torch.stack(torch.meshgrid(
            *[torch.arange(win_lo[i], win_lo[i] + V, device=torch_dev)
              for i in range(2)], indexing='ij'), dim=-1).reshape((-1, 2))
        rs = (sub - torch.tensor(im_size, device=torch_dev) // 2).double() / \
            torch.tensor(im_size, device=torch_dev)
        y_exact = torch.exp(-2j * np.pi * (k.double() @ rs.T)) @ \
            x_local.reshape(-1).type(torch.complex128)

        # Block path: local coords, kappa = D k, then the center phase
        D = V / N
        c = torch.tensor((win_lo + V // 2 - np.array(im_size) // 2) / np.array(im_size),
                         device=torch_dev, dtype=torch.float32)
        y_blk = block_kb_forward(x_local, D * k, V_size, os, W)
        y_blk = y_blk * torch.exp(-2j * np.pi * (k @ c))

        e_b = float((y_blk.type(torch.complex128) - y_exact).norm() / y_exact.norm())
        e_ref = _kb_rel_err(V_size, os, W, torch_dev)
        check(f'(b) V={V}, W={W}: local-coord block path matches global dense evaluation',
              e_b < 1.5 * e_ref, f'{e_b:.3e} vs the block KB floor {e_ref:.3e}')


# ---------------------------------------------------------------------------
# Test 5 -- wrap-freeness of the sheared per-block gather
# ---------------------------------------------------------------------------
def test_wrap_freeness(torch_dev):
    """
    Local coordinates are multiples of 1/V and the block grid's period in kappa units is
    exactly V, so gather indices may leave the nominal range and still be exact. This is
    what makes the per-block trajectory shear cost no grid growth (Sec. 1.2).
    """
    print('\ntest 5: wrap-freeness of out-of-range block gathers')
    N, V, os, W = 220, 55, 1.25, 3
    torch.manual_seed(0)
    x_local = torch.randn((V, V), dtype=torch.complex64, device=torch_dev)

    M = 2048
    kappa_in = (torch.rand((M, 2), device=torch_dev) - 0.5) * V
    # Push samples several periods out, as a per-block shear would
    shift = torch.tensor([[1.0, -2.0]], device=torch_dev) * V
    kappa_out = kappa_in + shift

    y_in = block_kb_forward(x_local, kappa_in, (V, V), os, W)
    y_out = block_kb_forward(x_local, kappa_out, (V, V), os, W)
    check('shifted-by-a-period gathers agree with in-range gathers',
          float((y_in - y_out).abs().max() / y_in.abs().mean()) < 1e-4,
          f'max rel dev {float((y_in - y_out).abs().max() / y_in.abs().mean()):.2e}')

    # The bar is parity with the in-range gather, not absolute accuracy: at W=3 the KB
    # stencil itself is only good to ~3e-2, and wrapping must not add to that.
    def _err(y, kappa):
        ref = dense_block_gather(x_local, kappa, (V, V)).type(torch.complex128)
        return float((y.type(torch.complex128) - ref).norm() / ref.abs().norm())

    e_in, e_out = _err(y_in, kappa_in), _err(y_out, kappa_out)
    check('out-of-range gather is as accurate as the in-range gather',
          e_out < 1.05 * e_in, f'out {e_out:.3e} vs in {e_in:.3e} '
                               f'({e_out / e_in:.3f}x)')

    idx_span = float((os * kappa_out).abs().amax())
    check('gather indices genuinely exceed the nominal range',
          idx_span > os * V / 2, f'|os*kappa|max={idx_span:.0f} vs nominal {os * V / 2:.0f}')


# ---------------------------------------------------------------------------
# Test 2 -- partition validity
# ---------------------------------------------------------------------------
def test_partition_validity(torch_dev):
    print('\ntest 2: partition validity')
    im_size = (100, 100)
    mask = _disc_mask(im_size, torch_dev)
    n_mask = int(mask.sum())

    for bs in [make_blocks(mask, (4, 4)), make_blocks(mask, (8, 8)),
               make_blocks(mask, (2, 8)), make_slabs(mask, 8, 0),
               make_slabs(mask, 16, 1)]:
        counts = torch.zeros(im_size, device=torch_dev)
        for sel in bs.idx:
            counts.reshape(-1)[sel] += 1
        ok_unit = bool((counts[mask] == 1).all())
        ok_outside = bool((counts[~mask] == 0).all())
        ok_total = sum(len(s) for s in bs.idx) == n_mask
        ok_labels = bool((bs.labels[mask] >= 0).all() and (bs.labels[~mask] == -1).all())
        check(f'{bs.style}: sum_q m_q = 1 on the mask, 0 off it, labels consistent',
              ok_unit and ok_outside and ok_total and ok_labels,
              f'Q_kept={bs.Q_kept}/{bs.Q_tot}, V={bs.V}, stride={bs.stride}')

        # Windows must contain their cells and stay inside the grid
        in_grid = bool(((bs.win_lo >= 0).all() and
                        (bs.win_lo + torch.tensor(bs.V) <= torch.tensor(im_size)).all()))
        contained = True
        for q, sel in enumerate(bs.idx):
            sub = torch.stack(torch.unravel_index(sel, im_size), dim=-1).cpu()
            contained &= bool((sub >= bs.win_lo[q]).all() and
                              (sub < bs.win_lo[q] + torch.tensor(bs.V)).all())
        check(f'{bs.style}: FFT windows are in-grid and contain their voxels',
              in_grid and contained)


# ---------------------------------------------------------------------------
# Test 3 -- affine fit is exact on an affine field
# ---------------------------------------------------------------------------
def test_affine_fit_exact(torch_dev):
    print('\ntest 3: per-block affine fit on an exactly affine field')
    im_size = (64, 64)
    mask = _disc_mask(im_size, torch_dev)
    rs = gen_grd(im_size).to(torch_dev).double()
    phis = torch.stack([
        0.3 + 0.7 * rs[..., 0] - 0.4 * rs[..., 1],
        -1.1 + 2.0 * rs[..., 1],
    ], dim=0) * mask

    for Q in ((1, 1), (4, 4), (8, 8)):
        bs = make_blocks(mask, Q)
        C, chat, res = fit_block_affine(phis, bs)
        worst = max(float(r.abs().max()) for r in res)
        scale = float(phis[:, mask].abs().max())
        check(f'Q={Q}: residual is zero to machine precision',
              worst / scale < 1e-11, f'max|phi_res| / scale = {worst / scale:.2e}')

        # Every block's residual phase matrix must then be exactly rank 1
        alphas = torch.randn((2, 64), dtype=torch.float64, device=torch_dev) * 3
        svals = block_singular_values(res, alphas)
        ratio = max(float(s[1] / s[0]) for s in svals if len(s) > 1)
        check(f'Q={Q}: every block is exactly rank 1 after absorption',
              ratio < 1e-9, f'max s2/s1 = {ratio:.2e}')


# ---------------------------------------------------------------------------
# Test 1 -- Q_tot = 1 reproduces the global SVD arm
# ---------------------------------------------------------------------------
def test_single_block_identity(torch_dev):
    print('\ntest 1: Q_tot = 1 reproduces the global SVD arm')
    im_size = (96, 96)
    mask = _disc_mask(im_size, torch_dev)
    rs = gen_grd(im_size).to(torch_dev)
    phis = torch.stack([rs[..., 0].square() + rs[..., 1].square(),
                        rs[..., 0] * rs[..., 1]], dim=0)
    alphas = torch.randn((2, 512), device=torch_dev) * 4.0

    # The global remove_linear_terms the baseline already applies
    phis, _, _ = remove_linear_terms(phis, alphas, mask=mask.float())
    phis = (phis * mask).double()
    alphas = alphas.double()

    bs = make_blocks(mask, (1, 1))
    C, chat, res = fit_block_affine(phis, bs)
    scale = float(phis[:, mask].abs().max())
    check('C_q and chat_q come out ~0 after global linear removal',
          float(C.abs().max()) / scale < 1e-5 and float(chat.abs().max()) / scale < 1e-5,
          f'max|C|/scale={float(C.abs().max()) / scale:.2e}, '
          f'max|chat|/scale={float(chat.abs().max()) / scale:.2e}')

    M, n_mask = alphas.shape[1], int(mask.sum())
    P = phase_matrix(phis.reshape((2, -1))[:, torch.argwhere(mask.reshape(-1))[:, 0]],
                     alphas)
    s_glob = torch.linalg.svdvals(P).cpu()
    s_blk = block_singular_values(res, alphas)
    check('the single block reproduces the global singular values',
          torch.allclose(s_blk[0], s_glob, rtol=1e-6, atol=1e-6 * float(s_glob[0])),
          f'max rel dev {float((s_blk[0] - s_glob).abs().max() / s_glob[0]):.2e}')

    S_b, err_b = pooled_error_curve(s_blk, M * n_mask)
    L_g, err_g = global_error_curve(s_glob, M * n_mask)

    # The singular values above already settle the identity. The curve is a tail sum, so
    # near full rank it is roundoff noise around an exact zero and a relative comparison
    # there measures nothing; restrict to the 1e-4 and up band the study operates in.
    n = min(len(err_b), len(err_g) - 1)
    eb, eg = err_b[:n], err_g[1:n + 1]
    live = eg > 1e-4
    dev = float(np.abs(eb[live] / eg[live] - 1).max())
    check('pooled block curve equals the global SVD curve', dev < 1e-5,
          f'max rel dev {dev:.2e} over the {int(live.sum())} ranks with err > 1e-4')


# ---------------------------------------------------------------------------
# Test 6 -- pooled allocation beats a fixed per-block rank
# ---------------------------------------------------------------------------
def test_rank_pooling(torch_dev):
    print('\ntest 6: pooled threshold vs fixed L_q at matched S')
    im_size = (96, 96)
    mask = _disc_mask(im_size, torch_dev)
    rs = gen_grd(im_size).to(torch_dev).double()
    # A deliberately inhomogeneous field: curvature concentrated on one side, so the
    # optimal rank allocation is very much not uniform
    phis = ((rs[..., 0] + 0.5) ** 3 * 6.0)[None] * mask
    alphas = torch.randn((1, 256), dtype=torch.float64, device=torch_dev) * 5.0

    bs = make_blocks(mask, (4, 4))
    _, _, res = fit_block_affine(phis, bs)
    svals = block_singular_values(res, alphas)
    Q = len(svals)
    worse = 0
    for mult in (2, 3, 4, 6):
        S = Q * mult
        _, sq_pool = allocate_ranks(svals, S, floor=1)
        sq_fixed = sum(float(s[mult:].square().sum()) for s in svals)
        worse += int(sq_pool > sq_fixed * (1 + 1e-9))
        check(f'S={S} (L_q={mult} fixed): pooled error is lower',
              sq_pool <= sq_fixed * (1 + 1e-9),
              f'pooled {np.sqrt(sq_pool):.4e} vs fixed {np.sqrt(sq_fixed):.4e} '
              f'({np.sqrt(sq_fixed / max(sq_pool, 1e-300)):.2f}x)')
    check('pooled allocation never loses', worse == 0)


def main():
    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')
    test_block_fft_accuracy(torch_dev)      # validates Sec. 1.2 -- runs first
    test_wrap_freeness(torch_dev)
    test_partition_validity(torch_dev)
    test_affine_fit_exact(torch_dev)
    test_single_block_identity(torch_dev)
    test_rank_pooling(torch_dev)
    print('\n' + ('ALL TESTS PASSED' if not FAILED else f'FAILED: {FAILED}'))
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
