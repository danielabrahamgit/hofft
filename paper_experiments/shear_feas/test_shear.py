"""
Validation for the shear machinery (tests 1-4, 6 of math_docs/feas_test.md).

These guard the Stage 0 verdict: a negative result is only believable if the same code
reports a large gain on fields that are genuinely locally affine.

Run with:
    PYTHONPATH=src python paper_experiments/shear_feas/test_shear.py
"""
import torch

from hofft.shear import (
    alpha_second_moment,
    spatial_jacobian,
    shear_features,
    weighted_kmeans,
    kaffine_clustering,
    init_label_candidates,
    fit_affine_per_cluster,
    shear_trajectory,
    grid_growth,
    residual_diagnostics,
)
from hofft.utils import gen_grd

FAILED = []


def check(name: str, cond: bool, msg: str = '') -> None:
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f'  ({msg})' if msg else ''))
    if not cond:
        FAILED.append(name)


def _setup(phis, alphas, L, ell, torch_dev, cluster='kmeans'):
    """Cluster, fit, and score both the no-shear baseline and the sheared variant."""
    N = phis[0].numel()
    weights = torch.ones(N, device=torch_dev)
    _, sig_sqrt = alpha_second_moment(alphas)
    jac = spatial_jacobian(phis)

    feats0 = shear_features(phis, jac, sig_sqrt, 0.0)
    labels0 = weighted_kmeans(feats0, weights, L)
    if cluster == 'kaffine':
        inits = init_label_candidates(phis, jac, weights, sig_sqrt, L)
        labels0, consts_c, zeros, _ = kaffine_clustering(
            phis, weights, sig_sqrt, L, inits, include_linear=False)
        labels, consts_a, grads, _ = kaffine_clustering(
            phis, weights, sig_sqrt, L, inits + [labels0], include_linear=True)
    else:
        consts_c, zeros = fit_affine_per_cluster(
            phis, labels0, weights, L, include_linear=False)
        feats = shear_features(phis, jac, sig_sqrt, ell)
        labels = weighted_kmeans(feats, weights, L)
        consts_a, grads = fit_affine_per_cluster(phis, labels, weights, L)

    a_sub = alphas.reshape((alphas.shape[0], -1))
    base = residual_diagnostics(phis, jac, a_sub, labels0, weights, consts_c, zeros)
    shear = residual_diagnostics(phis, jac, a_sub, labels, weights, consts_a, grads)
    return labels, consts_a, grads, base, shear


def test_affine_field(torch_dev):
    """Test 1: an exactly affine field must be annihilated by an L=1 shear."""
    print('\nTest 1: exactly affine field, L=1')
    im_size = (64, 64)
    rs = gen_grd(im_size).to(torch_dev)
    b = torch.tensor([37.0, -21.0], device=torch_dev)
    phis = (4.5 + rs @ b)[None]
    alphas = torch.linspace(-1, 1, 512, device=torch_dev)[None] * 3.0

    _, _, grads, base, shear = _setup(phis, alphas, 1, 0.0, torch_dev)
    check('recovers the exact gradient',
          torch.allclose(grads[0, 0], b, rtol=1e-4, atol=1e-3),
          f'G={grads[0, 0].tolist()} vs b={b.tolist()}')
    check('residual range collapses to ~0',
          float(shear[0].max()) < 1e-3 * float(base[0].max()),
          f'{float(base[0].max()):.4g} -> {float(shear[0].max()):.4g}')
    check('residual gradient collapses to ~0',
          float(shear[2].max()) < 1e-3 * float(base[2].max()),
          f'{float(base[2].max()):.4g} -> {float(shear[2].max()):.4g}')


def test_locally_affine_field(torch_dev):
    """Test: a piecewise-affine field must show a large gain once L matches the pieces."""
    print('\nTest 2: piecewise-affine field (4 tiles), L=4')
    im_size = (64, 64)
    rs = gen_grd(im_size).to(torch_dev)
    # Continuous so the finite-difference Jacobian has no boundary spikes, but exactly
    # affine on each quadrant with gradients (+-60, +-40)
    phis = (60.0 * rs[..., 0].abs() + 40.0 * rs[..., 1].abs())[None]
    alphas = torch.linspace(-1, 1, 512, device=torch_dev)[None] * 2.0

    for ell in (0.25, 1.0, 4.0, 16.0):
        _, _, _, base, shear = _setup(phis, alphas, 4, ell, torch_dev)
        g = float(base[0].max()) / max(float(shear[0].max()), 1e-12)
        print(f'    kmeans ell={ell:<5g} range {float(base[0].max()):8.4f} -> '
              f'{float(shear[0].max()):8.4f}  gain {g:6.2f}x')
    _, _, _, base, shear = _setup(phis, alphas, 4, 0.0, torch_dev, cluster='kaffine')
    gain = float(base[0].max()) / max(float(shear[0].max()), 1e-12)
    print(f'    kaffine        range {float(base[0].max()):8.4f} -> '
          f'{float(shear[0].max()):8.4f}  gain {gain:6.2f}x')
    check('k-affine clustering exceeds 10x on a piecewise-affine field', gain > 10.0,
          f'{float(base[0].max()):.4g} -> {float(shear[0].max()):.4g} = {gain:.1f}x')


def test_quadratic_field(torch_dev):
    """Test: a smooth quadratic field should improve steadily with L."""
    print('\nTest 3: quadratic field, gain vs L')
    im_size = (64, 64)
    rs = gen_grd(im_size).to(torch_dev)
    phis = (120.0 * (rs[..., 0] ** 2 - 0.7 * rs[..., 1] ** 2))[None]
    alphas = torch.linspace(-1, 1, 512, device=torch_dev)[None] * 2.0
    gains = []
    for L in (1, 4, 16):
        _, _, _, base, shear = _setup(phis, alphas, L, 0.0, torch_dev, cluster='kaffine')
        g = float(base[0].max()) / max(float(shear[0].max()), 1e-12)
        gains.append(g)
        print(f'    L={L:<3d} range {float(base[0].max()):8.4f} -> '
              f'{float(shear[0].max()):8.4f}  gain {g:5.2f}x')
    check('gain grows with L on a curved field', gains[-1] > gains[0] > 1.0,
          f'gains={["%.2f" % g for g in gains]}')


def test_global_b0_shear(torch_dev):
    """Test 4: L=1 on a pure linear B0 reproduces the classical trajectory correction."""
    print('\nTest 4: pure-B0 global shear, L=1')
    im_size = (64, 64)
    rs = gen_grd(im_size).to(torch_dev)
    grad_hz = torch.tensor([90.0, -55.0], device=torch_dev)   # Hz per FOV
    b0 = rs @ grad_hz
    ts = torch.arange(2000, device=torch_dev) * 4e-6           # s
    phis, alphas = b0[None], ts[None]

    _, _, grads, _, _ = _setup(phis, alphas, 1, 0.0, torch_dev)
    kappa, delta, tau = shear_trajectory(
        torch.zeros((2000, 2), device=torch_dev), alphas, grads, os=1.25)
    # Classical correction: k(t) += grad_hz * t
    kappa_ref = (grad_hz[None] * ts[:, None])[None]
    check('sheared trajectory equals the hand-coded B0 correction',
          torch.allclose(kappa, kappa_ref, rtol=1e-4, atol=1e-4),
          f'max abs diff {float((kappa - kappa_ref).abs().max()):.3e}')
    check('Delta is integer valued',
          bool(torch.all(delta == delta.round())))


def test_zero_shear_identity(torch_dev):
    """Test 2/6: G = 0 must give Delta = 0, rho = 1, and the unsheared grid."""
    print('\nTest 5: zero shear reproduces standard HOFFT geometry')
    trj = (torch.rand((4096, 2), device=torch_dev) - 0.5) * 200
    alphas = torch.randn((3, 4096), device=torch_dev)
    grads = torch.zeros((5, 3, 2), device=torch_dev)
    kappa, delta, tau = shear_trajectory(trj, alphas, grads, os=1.25)
    rho, leff, _ = grid_growth(trj, kappa, 1.25, (3, 3), (256, 256))
    check('Delta == 0', bool(torch.all(delta == 0)))
    check('kappa == trj', torch.allclose(kappa, trj[None]))
    check('rho == 1 and L_eff == L', torch.allclose(rho, torch.ones_like(rho))
          and abs(leff - 5.0) < 1e-5, f'L_eff={leff:.6f}')


def test_grid_padding_sufficiency(torch_dev):
    """Test 6: the reported span must actually contain every sheared gather point."""
    print('\nTest 6: grid padding covers all sheared samples')
    trj = (torch.rand((4096, 2), device=torch_dev) - 0.5) * 200
    alphas = torch.randn((3, 4096), device=torch_dev) * 5
    grads = torch.randn((6, 3, 2), device=torch_dev) * 4
    os, kern = 1.25, (3, 3)
    kappa, _, _ = shear_trajectory(trj, alphas, grads, os)
    rho, _, _ = grid_growth(trj, kappa, os, kern, (256, 256))
    span_needed = (os * kappa).reshape((6, -1, 2))
    span = span_needed.amax(1) - span_needed.amin(1) + torch.tensor(
        kern, device=torch_dev, dtype=torch.float32)
    base = (os * trj).amax(0) - (os * trj).amin(0) + torch.tensor(
        kern, device=torch_dev, dtype=torch.float32)
    check('rho equals the measured span ratio',
          torch.allclose(rho, (span / base).prod(-1), rtol=1e-5))
    check('sheared spans exceed the nominal span', bool(torch.all(rho >= 1.0)),
          f'rho={[round(float(r), 3) for r in rho]}')


def main() -> None:
    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}')
    test_affine_field(torch_dev)
    test_locally_affine_field(torch_dev)
    test_quadratic_field(torch_dev)
    test_global_b0_shear(torch_dev)
    test_zero_shear_identity(torch_dev)
    test_grid_padding_sufficiency(torch_dev)
    print('\n' + ('ALL TESTS PASSED' if not FAILED else f'FAILED: {FAILED}'))


if __name__ == '__main__':
    main()
