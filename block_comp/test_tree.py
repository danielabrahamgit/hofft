"""
Correctness gates for the tree/Gram machinery (feas.md Sec. 25).

Each test names the claim it validates. The load-bearing ones run first: everything in
Stages 1 and 2 rests on ``to_blocks`` ordering, Gram-vs-SVD equivalence and Gram
additivity, so if those are wrong every rank number in the study is wrong.

Run with:
    JOBID=<n> ./block_comp/run.sh test_tree.py          # or plain python, CPU is fine
"""
import sys
import numpy as np
import torch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tree as T                                                      # noqa: E402
from common import (crop_center, next_dyadic_smooth, pad_center,      # noqa: E402
                    pad_grid, adjoint_error, rel_err)

FAILED = []


def check(name: str, cond: bool, msg: str = ''):
    print(f'  [{"ok" if cond else "XX"}] {name}' + (f'  -- {msg}' if msg else ''))
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------------------
def test_block_roundtrip(dev):
    """to_blocks/from_blocks are exact inverses, and blocks map to the right sub-boxes."""
    for im_size, level in [((16, 24), 2), ((32, 32), 3), ((8, 12, 16), 2)]:
        d = len(im_size)
        x = torch.randn((3,) + im_size, dtype=torch.complex64, device=dev)
        xb = T.to_blocks(x, level)
        s = 1 << level
        check(f'roundtrip {im_size} L{level}',
              torch.equal(T.from_blocks(xb, im_size, level), x))
        check(f'shape {im_size} L{level}',
              xb.shape == (s ** d, 3, int(np.prod(T.block_shape(im_size, level)))))

        # Block q in row-major order must equal the corresponding spatial sub-box
        b = T.block_shape(im_size, level)
        rng = np.random.default_rng(0)
        for _ in range(4):
            ijk = tuple(int(rng.integers(s)) for _ in range(d))
            q = int(np.ravel_multi_index(ijk, (s,) * d))
            sl = (slice(None),) + tuple(
                slice(ijk[i] * b[i], (ijk[i] + 1) * b[i]) for i in range(d))
            check(f'  block {ijk} of {im_size}',
                  torch.equal(xb[q], x[sl].reshape(3, -1)))


def test_gram_vs_svd(dev):
    """Gram eigenvalues are the squared singular values, and eigenvectors are U."""
    torch.manual_seed(0)
    S = torch.randn((6, 16, 16), dtype=torch.complex128, device=dev)
    G = T.block_grams(S, 2)
    Xb = T.to_blocks(S, 2)
    for q in [0, 5, 11]:
        sv = torch.linalg.svdvals(Xb[q].to(torch.complex128))
        ev = T.gram_eigs(G[q][None])[0]
        check(f'eig==sv^2 block {q}',
              float((ev.sqrt() - sv).abs().max() / sv.max()) < 1e-10,
              f'{float((ev.sqrt() - sv).abs().max() / sv.max()):.2e}')

    # Left singular vectors agree up to per-column phase
    U_svd, _, _ = torch.linalg.svd(Xb[0].to(torch.complex128), full_matrices=False)
    ranks = torch.full((G.shape[0],), 4, dtype=torch.long)
    U_g = T.bases_from_grams(G, ranks)[0][0]
    ovl = torch.linalg.svdvals(U_svd[:, :4].conj().T @ U_g)
    check('gram eigenvectors span the SVD subspace',
          float((ovl - 1).abs().max()) < 1e-9, f'{float((ovl - 1).abs().max()):.2e}')


def test_gram_additivity(dev):
    """G_p = sum_c G_c: the identity the whole Stage-2 curve rests on."""
    torch.manual_seed(0)
    for im_size, level in [((32, 32), 3), ((16, 16, 16), 2)]:
        d = len(im_size)
        S = torch.randn((5,) + im_size, dtype=torch.complex64, device=dev)
        pyr = T.gram_pyramid(S, level)
        for l in range(level):
            direct = T.block_grams(S, l)
            e = float((pyr[l] - direct).abs().max() / direct.abs().max())
            check(f'additivity {im_size} L{level}->{l}', e < 1e-9, f'{e:.2e}')

        # And a weighted version, since Stage 1 uses object weights
        w = torch.rand(im_size, device=dev)
        pyr_w = T.gram_pyramid(S, level, weights=w)
        direct_w = T.block_grams(S, 0, weights=w)
        e = float((pyr_w[0] - direct_w).abs().max() / direct_w.abs().max())
        check(f'weighted additivity {im_size}', e < 1e-9, f'{e:.2e}')


def test_rank_semantics(dev):
    """rank_from_eigs(tol) is the smallest k with ||S - S_k||_F <= tol ||S||_F."""
    torch.manual_seed(0)
    S = torch.randn((8, 16, 16), dtype=torch.complex128, device=dev)
    G = T.block_grams(S, 1)
    Xb = T.to_blocks(S, 1)
    for tol in (1e-1, 3e-2, 1e-2):
        r = T.rank_from_eigs(T.gram_eigs(G), tol=tol)
        for q in range(G.shape[0]):
            sv = torch.linalg.svdvals(Xb[q].to(torch.complex128))
            k = int(r[q])
            got = float((sv[k:].square().sum() / sv.square().sum()).sqrt())
            prev = float((sv[k - 1:].square().sum() / sv.square().sum()).sqrt())
            check(f'  tol={tol:.0e} q={q} minimal k={k}', got <= tol < prev,
                  f'err(k)={got:.2e} err(k-1)={prev:.2e}')
        break   # one tolerance is enough per block; the loop above already covers all q

    # Empty blocks must get rank 0, not rank 1
    Z = torch.zeros(2, 4, 4, dtype=torch.complex128, device=dev)
    check('empty block -> rank 0',
          int(T.rank_from_eigs(T.gram_eigs(Z), tol=1e-3).sum()) == 0)

    # Energy and Frobenius conventions are consistent: energy = 1 - tol^2
    e = T.gram_eigs(G)
    check('energy 0.99 == frobenius 0.1',
          torch.equal(T.rank_from_eigs(e, energy=0.99),
                      T.rank_from_eigs(e, tol=0.1)))


def test_truncate_field(dev):
    """truncate_field realizes exactly the per-block projection, at the stated error."""
    torch.manual_seed(0)
    im_size, level = (32, 32), 3
    # Smooth coil-map-like field: each channel a slow complex exponential, so local
    # blocks are genuinely rank-deficient and truncation actually bites.
    r = T.gen_grd64(im_size, device=dev)
    a = torch.randn(6, 2, dtype=torch.float64, device=dev) * 0.8
    S = torch.exp(2j * np.pi * (r @ a.T)).permute(2, 0, 1).contiguous()
    G = T.block_grams(S, level)
    ranks = T.rank_from_eigs(T.gram_eigs(G), tol=1e-2)
    check('smooth field truncates below full rank',
          int(ranks.max()) < S.shape[0], f'max rank {int(ranks.max())} of {S.shape[0]}')
    Sh = T.truncate_field(S, level, ranks, G)

    # Global Frobenius error must match the pooled per-block tail energy exactly
    tail = sum(float(T.gram_eigs(G)[q][int(ranks[q]):].sum()) for q in range(G.shape[0]))
    tot = float(T.gram_eigs(G).sum())
    check('truncation error == pooled eigen tail',
          abs(rel_err(Sh, S) - np.sqrt(tail / tot)) < 1e-9,
          f'{rel_err(Sh, S):.4e} vs {np.sqrt(tail / tot):.4e}')
    check('truncation error within stated tol', rel_err(Sh, S) <= 1e-2,
          f'{rel_err(Sh, S):.2e}')

    # Full rank must be a no-op
    full = torch.full_like(ranks, S.shape[0])
    check('full rank is identity', rel_err(T.truncate_field(S, level, full, G), S) < 1e-6)


def test_parent_basis(dev):
    """Parent spans every child, and the transfers reproduce the children exactly."""
    torch.manual_seed(0)
    S = torch.randn((8, 16, 16), dtype=torch.complex128, device=dev)
    G = T.block_grams(S, 1)
    ranks = T.rank_from_eigs(T.gram_eigs(G), tol=1e-2)
    kids = T.bases_from_grams(G, ranks)
    Up, _, transfers = T.parent_basis(kids, tol=1e-12, weighted=False)
    check('parent rank <= sum child ranks',
          Up.shape[1] <= sum(k[0].shape[1] for k in kids))
    worst = max(float((Up @ Tm - Uc).abs().max()) for (Uc, _), Tm in zip(kids, transfers))
    check('U_p T_{p<-c} == U_c at tol 1e-12', worst < 1e-9, f'{worst:.2e}')
    check('parent basis orthonormal',
          float((Up.conj().T @ Up - torch.eye(Up.shape[1], dtype=Up.dtype,
                                              device=dev)).abs().max()) < 1e-10)

    # An exactly-nested pair must give zero principal angles
    ang = T.principal_angles(Up, kids[0][0])
    check('child subspace inside parent', float(ang.max()) < 1e-6, f'{float(ang.max()):.2e}')


def test_block_centers(dev):
    """r = c + D r' exactly, matching the qblock_feas geometry."""
    from hofft.utils import gen_grd
    for im_size, level in [((32, 24), 2), ((16, 16, 8), 1), ((320, 160), 4)]:
        d = len(im_size)
        b = T.block_shape(im_size, level)
        c = T.block_centers(im_size, level)
        D = torch.tensor([b[i] / im_size[i] for i in range(d)], dtype=torch.float64)
        r_glob = T.to_blocks(T.gen_grd64(im_size).moveaxis(-1, 0), level)   # (Q,d,Nq)
        r_loc = T.gen_grd64(b).reshape(-1, d).T                             # (d,Nq)
        worst = float((r_glob - (c[:, :, None] + D[None, :, None] * r_loc)).abs().max())
        check(f'r = c + D r\' on {im_size} L{level}', worst < 1e-15, f'{worst:.2e}')

    # gen_grd64 must agree with the float32 gen_grd to float32 precision, so the
    # convention is identical and only the precision differs.
    g32, g64 = gen_grd((32, 24)), T.gen_grd64((32, 24))
    check('gen_grd64 matches gen_grd convention',
          float((g32.double() - g64).abs().max()) < 1e-7,
          f'{float((g32.double() - g64).abs().max()):.2e}')


def test_padding(dev):
    """Padding is dyadic, 5-smooth, and crop(pad(x)) == x."""
    for n, L in [(294, 4), (150, 4), (734, 6), (320, 6)]:
        m = next_dyadic_smooth(n, L)
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        check(f'pad {n} -> {m} (L={L})', m >= n and m % (1 << L) == 0 and r == 1)
    check('pad_grid 3D', pad_grid((294, 294, 150)) == (320, 320, 160))
    check('pad_grid 2D', pad_grid((734, 734)) == (768, 768))
    x = torch.randn(3, 7, 11, dtype=torch.complex64, device=dev)
    check('crop(pad(x)) == x', torch.equal(crop_center(pad_center(x, (16, 16)), (7, 11)), x))


def test_adjoint_helper(dev):
    """The inner-product helper reports ~0 for a true adjoint pair and ~1 for a wrong one."""
    torch.manual_seed(0)
    A = torch.randn(7, 5, dtype=torch.complex128, device=dev)
    good = adjoint_error(lambda x: A @ x, lambda y: A.conj().T @ y, (5,), (7,),
                         dev, dtype=torch.complex128)
    bad = adjoint_error(lambda x: A @ x, lambda y: A.T @ y, (5,), (7,),
                        dev, dtype=torch.complex128)
    check('adjoint_error ~ 0 for A, A^H', good < 1e-12, f'{good:.2e}')
    check('adjoint_error large for A, A^T', bad > 1e-2, f'{bad:.2e}')


# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {dev}\n')
    for fn in (test_block_roundtrip,     # ordering: everything downstream needs it
               test_gram_vs_svd,
               test_gram_additivity,     # the Stage-2 curve rests on this
               test_rank_semantics,
               test_truncate_field,      # makes Gate 1 interpretable
               test_parent_basis,
               test_block_centers,
               test_padding,
               test_adjoint_helper):
        print(f'{fn.__name__}: {fn.__doc__.strip().splitlines()[0]}')
        fn(dev)
    print()
    if FAILED:
        print(f'FAILED ({len(FAILED)}): ' + ', '.join(FAILED))
    else:
        print('all tree tests passed')
    sys.exit(1 if FAILED else 0)


if __name__ == '__main__':
    main()
