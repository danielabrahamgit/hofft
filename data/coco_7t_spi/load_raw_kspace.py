"""
Load Siemens 7T FLASHSpiral raw k-space to match nufft_FLASHSpiral_v01a_test.m.

Requires pyMapVBVD:
    pip install pyMapVBVD

Raw twix layout (img.squeeze = True):
    Col=2000, Cha=64, Lin=500, Eco=6, Rep=18
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import torch
from einops import rearrange, einsum
from mr_recon.multi_coil.calib import calc_coil_subspace

try:
    import mapvbvd
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'load_raw_kspace requires pyMapVBVD (`pip install pyMapVBVD`).'
    ) from exc


DEFAULT_RAW = (
    '/local_mount/space/mayday/data/users/xc/data/Siemens2026/'
    '20260615_stanford/7t/meas_MID00078_FID01057_FLASHSpiral_v01a.dat'
)
N_ADC = 2000
N_COIL = 64
N_TR = 500
N_ECO = 6
N_SPIRAL = 18
N_POINT = 22136
N_COMP_COIL = 6


@dataclass(frozen=True)
class FlashHeader:
    tr_per_subgrp: int
    opngrp: int
    opacc: int
    n_adc: int
    sequence: str
    sqz_dims: tuple[str, ...]
    sqz_size: tuple[int, ...]

    @property
    def n_coil(self) -> int:
        return int(self.sqz_size[self.sqz_dims.index('Cha')])

    @property
    def n_tr(self) -> int:
        return int(self.sqz_size[self.sqz_dims.index('Lin')])

    @property
    def n_eco(self) -> int:
        return int(self.sqz_size[self.sqz_dims.index('Eco')])

    @property
    def n_spiral(self) -> int:
        return int(self.sqz_size[self.sqz_dims.index('Rep')])

    @property
    def n_adc_samples(self) -> int:
        return int(self.sqz_size[self.sqz_dims.index('Col')])


class TwixLoader:
    """
    Memory-friendly loader that opens the .dat once and reads one Rep
    (spiral interleave, size 18) at a time.
    """

    def __init__(self, dat_path: Union[str, Path]):
        self.dat_path = Path(dat_path)
        self.ds = open_twix(self.dat_path, parse_data=False)
        self.img = self.ds.image
        self.img.squeeze = True
        self.img.removeOS = False
        self.header = parse_flash_header_from_ds(self.ds)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.header.n_spiral))

    def load_rep_sqz(
        self,
        rep: int,
        *,
        dtype: np.dtype = np.complex64,
    ) -> np.ndarray:
        """
        Load one Rep index in squeezed twix order.

        Returns
        -------
        raw : (Col, Cha, Lin, Eco)
        """
        return np.asarray(self.img[:, :, :, :, rep], dtype=dtype)

    def load_rep_matlab(
        self,
        rep: int,
        *,
        dtype: np.dtype = np.complex64,
    ) -> np.ndarray:
        """
        One Rep after the MATLAB permute/reshape (lines 111-116), unfused.

        Returns
        -------
        raw : (Eco, Lin, Col, Cha)
        """
        block = self.load_rep_sqz(rep, dtype=dtype)
        return reshape_rep_unfused(block)


def _last_measurement(twix_obj):
    if isinstance(twix_obj, list):
        return twix_obj[-1]
    return twix_obj


def _alfree_value(hdr, index: int) -> float:
    meas_yaps = hdr['MeasYaps']
    return float(meas_yaps[('sWipMemBlock', 'alFree', str(index))])


def parse_flash_header_from_ds(ds) -> FlashHeader:
    hdr = ds.hdr
    img = ds.image
    return FlashHeader(
        tr_per_subgrp=int(_alfree_value(hdr, 1)),
        opngrp=int(_alfree_value(hdr, 8)),
        opacc=int(_alfree_value(hdr, 10)),
        n_adc=int(_alfree_value(hdr, 15)),
        sequence=str(hdr['Config']['SequenceFileName']),
        sqz_dims=tuple(img.sqzDims),
        sqz_size=tuple(int(x) for x in img.sqzSize),
    )


def parse_flash_header(dat_path: Union[str, Path]) -> FlashHeader:
    twix = mapvbvd.mapVBVD(str(dat_path), parseData=False)
    return parse_flash_header_from_ds(_last_measurement(twix))


def open_twix(dat_path: Union[str, Path], *, parse_data: bool = False):
    twix = mapvbvd.mapVBVD(str(dat_path), parseData=parse_data)
    return _last_measurement(twix)


def reshape_rep_unfused(raw: np.ndarray) -> np.ndarray:
    """
    MATLAB lines 111-116 for a single Rep, keeping Lin separate.

    Input
    -----
    raw : (Col, Cha, Lin, Eco)

    Returns
    -------
    raw : (Eco, Lin, Col, Cha)

    Notes
    -----
    The full MATLAB script fuses Col*Lin into one axis (1e6).  We apply the
    same permute but keep Lin as its own axis so TRs remain addressable and
    ``raw(:,:,:,500)`` in the coil-compression block is well-defined.
    """
    raw = np.squeeze(raw).astype(np.complex64, copy=False)
    if raw.ndim != 4:
        raise ValueError(f'expected (Col, Cha, Lin, Eco), got {raw.shape}')

    # permute(raw, [1,4,2,3,5]) on (Col, Cha, Lin, Eco, Rep) -> (Col, Eco, Cha, Lin)
    raw = np.transpose(raw, (0, 3, 1, 2))
    # equivalent of reshape+permute without fusing Lin into Col
    return np.transpose(raw, (1, 3, 0, 2))  # (Eco, Lin, Col, Cha)


def reshape_raw_matlab(
    raw: np.ndarray,
    *,
    opn_subgrp: int = 1,
    opn_dummy: int = 0,
    n_tr: int = N_TR,
) -> np.ndarray:
    """
    Literal MATLAB lines 111-118 on full (Col, Cha, Lin, Eco, Rep) volume.

    Returns (Eco, Col*Lin, Rep, Cha) after TR selection on the fused axis.
    """
    raw = np.squeeze(raw).astype(np.complex64, copy=False)
    raw = np.transpose(raw, (0, 3, 1, 2, 4))
    raw = raw.reshape(
        raw.shape[0] * raw.shape[3],
        raw.shape[1],
        raw.shape[2],
        raw.shape[4],
    )
    raw = np.transpose(
        raw.reshape(
            raw.shape[0],
            raw.shape[1],
            raw.shape[2] * opn_subgrp,
            raw.shape[3] // opn_subgrp,
        ),
        (1, 0, 3, 2),
    )
    return raw[:, :, opn_dummy:, :n_tr]


def assemble_matlab_raw(
    loader: TwixLoader,
    rep_indices: Sequence[int],
    *,
    n_point: int = N_POINT,
    n_tr: int = N_TR,
    opn_dummy: int = 0,
    num_coils_compress: Optional[int] = None,
    torch_dev: torch.device = torch.device('cpu'),
) -> np.ndarray:
    """
    Load selected Rep groups one at a time and assemble the MATLAB ``raw`` array.

    Returns
    -------
    raw : (ncoil, nreadout, nrep, ntr)
    """
    rep_indices = list(rep_indices)
    n_point = min(n_point, loader.header.n_adc_samples) 
    n_tr = min(n_tr, loader.header.n_tr)
    
    if num_coils_compress is None:
        coil_comp = torch.eye(N_COIL, dtype=torch.complex64, device=torch_dev)
        ksp = torch.zeros((N_COIL, n_point * 6, len(rep_indices), n_tr), dtype=torch.complex64, device=torch_dev)
    else:
        coil_comp = None
        ksp = torch.zeros((num_coils_compress, n_point * 6, len(rep_indices), n_tr), dtype=torch.complex64, device=torch_dev)

    for i, rep in enumerate(rep_indices):
        block = loader.load_rep_matlab(rep)  # (Eco, Lin, Col, Cha)
        block = block[:, opn_dummy:n_tr, :n_point, :]
        block = torch.from_numpy(block).to(torch_dev)
        block = rearrange(block, 'E T R C -> C (E R) T')
        
        if coil_comp is None:
            coil_comp = calc_coil_subspace(block[:, :1000], num_coils_compress)
            
        block = einsum(block, coil_comp, 'C R T, C Co -> Co R T')
        ksp[..., i, :] = block

    return ksp


def trim_matlab_raw(
    raw: np.ndarray,
    n_point: int,
    spiral_select: Optional[Sequence[int] | slice] = None,
) -> np.ndarray:
    """MATLAB prep lines 175-178: ``raw(:, 1:n_point, spiral_select, :, :)``."""
    if spiral_select is None:
        spiral_select = slice(None)
    n_point = min(n_point, raw.shape[1])
    return raw[:, :n_point, spiral_select, :, :]


def collapse_cha_svd(
    raw: np.ndarray,
    n_compcoil: Optional[int] = None,
    ref_tr: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SVD-compress the physical Cha axis.

    Input : (Eco, P, Rep, Lin, Cha)
    Output: (Eco, P, Rep, Lin) with Cha collapsed.
    """
    if raw.ndim != 5:
        raise ValueError(f'expected (Eco, P, Rep, Lin, Cha), got {raw.shape}')

    n_eco, n_point, n_rep, n_lin, n_cha = raw.shape
    if n_compcoil is None:
        n_compcoil = n_cha

    ref = raw[..., ref_tr % n_lin, :]  # (Eco, P, Rep, Cha)
    mat = ref.reshape(n_eco * n_point * n_rep, n_cha)
    _, _, vh = np.linalg.svd(mat, full_matrices=False)
    vc = vh.conj().T[:, :n_compcoil]

    out = np.empty((n_eco, n_point, n_rep, n_lin), dtype=raw.dtype)
    for t in range(n_lin):
        slab = raw[..., t, :].reshape(n_eco * n_point * n_rep, n_cha)
        collapsed = (slab @ vc).reshape(n_eco, n_point, n_rep)
        out[..., t] = collapsed
    return out, vc


def coil_compress_matlab(
    raw: np.ndarray,
    n_compcoil: int = N_COMP_COIL,
    ref_tr: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SVD compression matching nufft_FLASHSpiral_v01a_test.m lines 187-203.

    Parameters
    ----------
    raw : (Eco, P, Rep, Lin, Cha) or (Eco, P, Rep, Lin)

    The MATLAB script uses ``n_coil = size(raw, 1)`` (Eco = 6) and
    ``temp_raw = raw(:, :, :, end)`` on the Lin/TR axis.
    """
    breakpoint()
    if raw.ndim == 5:
        raw, _ = collapse_cha_svd(raw, ref_tr=ref_tr)

    if raw.ndim != 4:
        raise ValueError(f'expected (Eco, P, Rep, Lin), got {raw.shape}')

    n_eco, n_point, n_rep, n_lin = raw.shape
    ref = raw[..., ref_tr % n_lin]  # (Eco, P, Rep)
    mat = ref.reshape(n_eco, n_point * n_rep).T  # (P*Rep, Eco)
    _, _, vh = np.linalg.svd(mat, full_matrices=False)
    vc = vh.conj().T[:, :n_compcoil]

    out = np.empty((n_compcoil, n_point, n_rep, n_lin), dtype=raw.dtype)
    for t in range(n_lin):
        slice_t = raw[..., t]  # (Eco, P, Rep)
        slice_pr = np.transpose(slice_t, (1, 0, 2))  # (P, Eco, Rep)
        compressed = slice_pr @ vc  # (P, n_compcoil, Rep)
        out[..., t] = np.transpose(compressed, (1, 0, 2))
    return out, vc


def load_matlab_kspace(
    dat_path: Union[str, Path] = DEFAULT_RAW,
    rep_indices: Optional[Sequence[int]] = None,
    *,
    n_point: int = N_POINT,
    n_tr: int = N_TR,
    coil_compress: bool = True,
    n_compcoil: int = N_COMP_COIL,
) -> dict[str, np.ndarray | FlashHeader]:
    """
    End-to-end MATLAB-mirrored loader (one Rep at a time).

    Returns
    -------
    dict with raw (before coil compression), ksp (after), Vc, header
    """
    loader = TwixLoader(dat_path)
    if rep_indices is None:
        rep_indices = range(loader.header.n_spiral)

    raw = assemble_matlab_raw(
        loader, rep_indices, n_point=n_point, n_tr=n_tr,
    )
    raw = trim_matlab_raw(raw, n_point)

    # (Eco, n_point, n_rep, n_tr, Cha) -> (Eco, n_point, n_rep, n_tr, Cha)
    out: dict[str, np.ndarray | FlashHeader] = {
        'header': loader.header,
        'raw': raw,
    }

    if coil_compress:
        ksp, vc = coil_compress_matlab(raw, n_compcoil=n_compcoil)
        out['ksp'] = ksp
        out['Vc'] = vc

    return out


def select_acc_kspace(
    k_3d: np.ndarray,
    dcf: np.ndarray,
    *,
    opacc: int = 1,
    opngrp: int = N_SPIRAL,
    n_tr: int = N_TR,
) -> tuple[np.ndarray, np.ndarray]:
    """Match MATLAB ``acc kspace`` block (lines 133-167)."""
    n_point = k_3d.shape[0]
    n_rot = k_3d.shape[3]
    point_sel = np.arange(n_point)
    ini_temp = (np.mod(np.arange(opacc, 0, -1) - 1, opacc) + 1).astype(int)

    k_out = np.empty((n_point, 3, opngrp, n_tr), dtype=k_3d.dtype)
    dcf_out = np.empty((n_point, opngrp, n_tr), dtype=dcf.dtype)

    for ii_fh in range(n_tr):
        frame_sel = ii_fh + 1
        index_help = (frame_sel - 1) % n_rot
        start = ini_temp[(frame_sel - 1) % opacc] - 1
        spiral_sel = start + opacc * np.arange(opngrp)

        k_out[:, :, :, ii_fh] = k_3d[point_sel, :, spiral_sel, index_help]
        dcf_out[:, :, ii_fh] = dcf[point_sel[:, None], spiral_sel, index_help]

    return k_out, dcf_out


if __name__ == '__main__':
    header = parse_flash_header(DEFAULT_RAW)
    print('Header:', header)

    loader = TwixLoader(DEFAULT_RAW)
    rep0 = loader.load_rep_matlab(0)
    print('rep 0 matlab layout (Eco, Lin, Col, Cha):', rep0.shape)

    out = load_matlab_kspace(DEFAULT_RAW, rep_indices=[0, 1], n_point=512, n_tr=4)
    print('raw shape:', out['raw'].shape)
    print('ksp shape:', out['ksp'].shape)
