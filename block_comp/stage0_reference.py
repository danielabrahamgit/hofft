"""
Stage 0 -- establish a trusted reference (feas.md Sec. 9).

Builds Baseline A (global coil compression + cuFINUFFT), proves it is a true adjoint
pair at both precisions, checks it against a direct DFT on a cropped problem, and
records the runtime, peak memory and -- the number that matters most -- the measured
FFT / interpolation split.

That split fixes the Amdahl ceiling on the whole study. A hierarchical method still pays
the full ``C' W M`` root interpolation, exactly what Baseline A pays, so ``(1+rho)/rho``
bounds its best possible speedup before any of it is written.

GATE 0: adjointness within feas.md Sec. 9 targets and the DFT check at NUFFT tolerance.

Run with:
    JOBID=<n> ./block_comp/run.sh stage0_reference.py --datasets sos2d hr2d sos3d
"""
import argparse
import sys
import warnings
import numpy as np
import torch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mr_recon.fourier import matrix_nufft                              # noqa: E402

import refop                                                          # noqa: E402
from common import (DATASETS, adjoint_error, gate, load_dataset,      # noqa: E402
                    peak_mib, rel_err, repeat_time, save_results)

OUT = Path(__file__).resolve().parent / 'results'
CPRIME = 16              # the C' the study reports at; Stage 1 sweeps it
N_REPS = 20              # feas.md Sec. 19 wants a warm-up-excluded median
ADJ_TOL_64 = 1e-4        # complex64 target (Sec. 9)
ADJ_TOL_128 = 1e-8       # complex128 target (Sec. 9)
DFT_GRID = (48, 48)      # cropped problem for the direct-DFT comparison
DFT_M = 3000


def dft_check(trj, torch_dev, oversamp, width, seed=0):
    """
    feas.md Sec. 9.5: compare a small cropped problem against a direct DFT.

    Validates sign, ``1/sqrt(N)`` normalization and coordinate scaling all at once by
    driving both operators with the *same* cycles/FOV trajectory, rescaled onto the
    small grid so it spans the same fraction of k-space.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    n = DFT_GRID[0]
    d = trj.shape[-1]
    im_size = (n,) * d
    sub = trj[torch.randperm(trj.shape[0], generator=g)[:DFT_M].to(trj.device)]
    k = sub * (n / max(t for t in trj.abs().max(0).values.tolist()) / 2.5)
    x = torch.randn((1, 1) + im_size, generator=g, dtype=torch.complex64).to(torch_dev)

    ref = matrix_nufft(im_size).forward(x, k[None])
    nuf = refop.cufi_nufft(im_size, oversamp=oversamp, width=width, n_trans=1)
    got = nuf.forward(x, nuf.rescale_trajectory(k)[None])
    return rel_err(got, ref), int(k.shape[0])


def run_one(name, torch_dev, args):
    ds = load_dataset(name, torch_dev).compress(args.cprime)
    print(f'\n=== {name} ===')
    print('  ' + ', '.join(f'{k}={v}' for k, v in ds.summary().items()))

    op = refop.SenseOp(ds.mps, ds.trj, oversamp=args.os, width=args.width)
    x = ds.img_ref.clone()
    y = op.forward(x)

    # --- adjointness, both precisions (Sec. 9.4) ---
    e64 = adjoint_error(op.forward, op.adjoint, ds.im_size, (ds.C, ds.M), torch_dev)
    try:
        op64 = refop.SenseOp64(ds.mps, ds.trj, oversamp=args.os, width=args.width)
        e128 = adjoint_error(op64.forward, op64.adjoint, ds.im_size, (ds.C, ds.M),
                             torch_dev, dtype=torch.complex128)
        del op64
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f'  [warn] complex128 path failed: {type(exc).__name__}: {exc}')
        e128 = float('nan')
    print(f'  adjointness   complex64 {e64:.3e}   complex128 {e128:.3e}')

    # --- direct DFT (Sec. 9.5) ---
    e_dft, m_dft = dft_check(ds.trj, torch_dev, args.os, args.width)
    print(f'  direct DFT    {e_dft:.3e} on {DFT_GRID} with M={m_dft} '
          f'(nufft eps {op.eps:.1e})')

    # --- runtime and memory (Sec. 19) ---
    t_f = repeat_time(lambda: op.forward(x), N_REPS)
    t_a = repeat_time(lambda: op.adjoint(y), N_REPS)
    t_n = repeat_time(lambda: op.normal(x), N_REPS)
    mem, _ = peak_mib(lambda: op.normal(x))
    print(f'  forward {t_f["median"]*1e3:8.2f} ms   adjoint {t_a["median"]*1e3:8.2f} ms'
          f'   normal {t_n["median"]*1e3:8.2f} ms   peak {mem:.0f} MiB')

    # --- the ceiling (why this stage runs first) ---
    split = refop.profile_split(op, ds.trj, n_reps=N_REPS)
    print(f'  fft {split["t_fft"]["median"]*1e3:7.2f} ms   '
          f'interp {split["t_interp"]["median"]*1e3:7.2f} ms   '
          f'rho {split["rho"]:.2f} (flop-pred {split["rho_pred"]:.2f})')
    print(f'  >> Amdahl ceiling on any FFT-only speedup: {split["ceiling"]:.2f}x')

    ok = (e64 < ADJ_TOL_64 and e_dft < max(50 * op.eps, 1e-4)
          and (np.isnan(e128) or e128 < ADJ_TOL_128))
    return dict(dataset=name, summary=ds.summary(), adj_c64=e64, adj_c128=e128,
                dft_err=e_dft, t_fwd=t_f, t_adj=t_a, t_normal=t_n, peak_mib=mem,
                split=split, passed=bool(ok))


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=list(DATASETS))
    ap.add_argument('--cprime', type=int, default=CPRIME)
    ap.add_argument('--os', type=float, default=refop.OS_DEFAULT)
    ap.add_argument('--width', type=int, default=refop.W_DEFAULT)
    ap.add_argument('--tag', type=str, default='')
    args, _ = ap.parse_known_args()

    torch.manual_seed(0)
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {torch_dev}   C\'={args.cprime}   os={args.os} width={args.width}')
    OUT.mkdir(exist_ok=True)

    res = []
    for name in args.datasets:
        res.append(run_one(name, torch_dev, args))
        torch.cuda.empty_cache()

    detail = '\n'.join(
        f'  {r["dataset"]:>6}  adj64 {r["adj_c64"]:.2e}  adj128 {r["adj_c128"]:.2e}  '
        f'dft {r["dft_err"]:.2e}  rho {r["split"]["rho"]:.2f}  '
        f'ceiling {r["split"]["ceiling"]:.2f}x  [{"PASS" if r["passed"] else "FAIL"}]'
        for r in res)
    passed = gate('GATE 0', all(r['passed'] for r in res),
                  f'targets: adj64 < {ADJ_TOL_64:.0e}, adj128 < {ADJ_TOL_128:.0e}, '
                  f'dft at nufft tolerance\n{detail}')

    tag = f'stage0_reference{args.tag}'
    save_results(OUT, tag, dict(results=res, args=vars(args)),
                 slim=dict(results=res, args=vars(args)))
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
