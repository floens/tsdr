"""MBE parameter decoder -- layer 4 of the AMBE+2 2450 pipeline.

Takes the 49-bit ``ambe_d`` vector (produced by layers 1-3) and fills
``cur_mp`` with the fundamental frequency ``w0``, harmonic count ``L``,
voiced/unvoiced decisions ``Vl[1..L]``, overall gain ``gamma``, and
spectral amplitudes ``Ml[1..L]``. Returns:

    0 -- ok (including silence, which is a valid frame type)
    2 -- erasure (caller should output silence + reinit state)
    3 -- tone    (caller should output silence + reinit state)

Two numerical subtleties worth knowing:

  - We use float32 for the array math but float64 for the
    transcendental `log`/`exp` calls -- going pure float32
    drifts audibly over long sequences.
  - ``prev_mp.log2_ml[0]`` is a sentinel set to ``prev_mp.log2_ml[1]``
    before the log2_ml reconstruction loop. It covers the
    ``intkl[harm] == 0`` case in the interpolation -- skipping it reads
    the zero-initialised slot and breaks log2_ml intermittently.
  - Silence frames still run the PRBA/HOC/log2_ml pipeline with
    ``Vl[harm] = 0`` for every harmonic. Only erasure and tone short-circuit.
"""

from __future__ import annotations

import math

import numpy as np

from tsdr.radio.vocoder.ambe._constants import (
    AMBE_DG,
    AMBE_HOC_B5,
    AMBE_HOC_B6,
    AMBE_HOC_B7,
    AMBE_HOC_B8,
    AMBE_L_TABLE,
    AMBE_LMPRBL,
    AMBE_PRBA24,
    AMBE_PRBA58,
    AMBE_VUV,
    AMBE_W0_TABLE,
)
from tsdr.radio.vocoder.ambe._param_kernels import (
    inverse_dct_hoc_kernel,
    inverse_dct_prba_kernel,
    reconstruct_log2ml_kernel,
)
from tsdr.radio.vocoder.ambe.params import MbeParams

F32 = np.float32

# Precomputed constants, cast to float32 for single-precision math.
_TWO_PI = F32(2.0 * math.pi)
_PI = F32(math.pi)
_ZERO = F32(0.0)
_ONE = F32(1.0)
_HALF = F32(0.5)
_RCONST = F32(1.0 / (2.0 * math.sqrt(2.0)))
_UNVC_NUM = F32(0.2046)
_ZEROSIX_FIVE = F32(0.65)
_POINT_SIX_NINE_THREE = F32(0.693)
_SILENCE_W0 = F32(2.0 * math.pi / 32.0)
_SILENCE_F0 = F32(1.0 / 32.0)


# Bit-field positions within ambe_d.
_B0_BITS = (0, 1, 2, 3, 37, 38, 39)
_B1_BITS = (4, 5, 6, 7, 35)
_B2_BITS = (8, 9, 10, 11, 36)
_B3_BITS = (12, 13, 14, 15, 16, 17, 18, 19, 40)
_B4_BITS = (20, 21, 22, 23, 41, 42, 43)
_B5_BITS = (24, 25, 26, 27, 44)
_B6_BITS = (28, 29, 30, 45)
_B7_BITS = (31, 32, 33, 46)
_B8_BITS = (34, 47, 48)


def _pack_bits(ambe_d: np.ndarray, positions: tuple[int, ...]) -> int:
    """Pack a tuple of ambe_d positions into an integer, MSB first."""
    value = 0
    for pos in positions:
        value = (value << 1) | int(ambe_d[pos])
    return value


def extract_bit_fields(ambe_d: np.ndarray) -> tuple[int, ...]:
    """Extract (b0, b1, ..., b8) from the 49-bit ambe_d vector."""
    return (
        _pack_bits(ambe_d, _B0_BITS),
        _pack_bits(ambe_d, _B1_BITS),
        _pack_bits(ambe_d, _B2_BITS),
        _pack_bits(ambe_d, _B3_BITS),
        _pack_bits(ambe_d, _B4_BITS),
        _pack_bits(ambe_d, _B5_BITS),
        _pack_bits(ambe_d, _B6_BITS),
        _pack_bits(ambe_d, _B7_BITS),
        _pack_bits(ambe_d, _B8_BITS),
    )


def _cosf(x: np.float32) -> np.float32:
    """Call libm's cosf via numpy. `np.cos` on a float32 scalar dispatches
    to the single-precision cosf implementation. Using ``math.cos`` (double
    precision) and rounding drifts by 1 ULP on a handful of inputs."""
    return np.cos(x)  # type: ignore[no-any-return]


def inverse_dct_prba(gm: np.ndarray) -> np.ndarray:
    """8-point inverse DCT turning the PRBA vector gm[1..8] into ri[1..8]."""
    ri = np.zeros(9, dtype=F32)
    inverse_dct_prba_kernel(gm, ri)
    return ri


def inverse_dct_hoc(cik: np.ndarray, ji: tuple[int, ...]) -> np.ndarray:
    """Inverse DCT of each cik block back into tl[0..56]."""
    tl = np.zeros(57, dtype=F32)
    ji_arr = np.array(ji, dtype=np.int32)
    inverse_dct_hoc_kernel(cik, ji_arr, tl)
    return tl


def _reconstruct_log2ml(
    cur_mp: MbeParams,
    prev_mp: MbeParams,
    tl: np.ndarray,
    unvc: float,
) -> None:
    """Apply eqs. 40-43: interpolate prev log2_ml, combine with tl/BigGamma, exp to Ml."""
    reconstruct_log2ml_kernel(
        cur_mp.log2_ml,
        cur_mp.Ml,
        cur_mp.Vl,
        prev_mp.log2_ml,
        prev_mp.Ml,
        tl,
        cur_mp.L,
        prev_mp.L,
        cur_mp.gamma,
        unvc,
    )


def decode_ambe2450_parms(
    ambe_d: np.ndarray,
    cur_mp: MbeParams,
    prev_mp: MbeParams,
) -> int:
    """Decode AMBE+2 2450 parameters from the 49-bit ambe_d vector.

    Mutates ``cur_mp`` in place. Returns 0 on success, 2 for erasure,
    3 for tone.
    """
    if ambe_d.shape != (49,):
        raise ValueError(f"expected ambe_d shape (49,), got {ambe_d.shape}")

    # copy repeat from prev_mp (cur_mp may be overwritten below)
    cur_mp.repeat = prev_mp.repeat

    b0, b1, b2, b3, b4, b5, b6, b7, b8 = extract_bit_fields(ambe_d)

    silence = False
    if 120 <= b0 <= 123:
        return 2  # erasure
    if b0 in (124, 125):
        silence = True
        cur_mp.w0 = float(_SILENCE_W0)
        f0 = float(_SILENCE_F0)
        n_harmonics = 14
        cur_mp.L = 14
        for harm in range(1, n_harmonics + 1):
            cur_mp.Vl[harm] = 0
    elif b0 in (126, 127):
        return 3  # tone

    if not silence:
        # w0 from AMBE_W0_TABLE (spec), L from AMBE_L_TABLE.
        f0 = float(F32(AMBE_W0_TABLE[b0]))
        cur_mp.w0 = float(f0 * _TWO_PI)
        n_harmonics = int(AMBE_L_TABLE[b0])
        cur_mp.L = n_harmonics

    unvc = _UNVC_NUM / F32(math.sqrt(float(cur_mp.w0)))

    # V/UV lookup -- only applied when not silence (silence zeroed Vl above).
    if not silence:
        for harm in range(1, n_harmonics + 1):
            jl = int(float(harm) * 16.0 * float(f0))
            cur_mp.Vl[harm] = int(AMBE_VUV[b1, jl])

    # Gain: deltaGamma + 0.5 * prev_mp.gamma, accumulates across frames.
    delta_gamma = F32(AMBE_DG[b2])
    cur_mp.gamma = float(delta_gamma + _HALF * prev_mp.gamma)

    # PRBA decoding: fill gm[2..8] from the PRBA24 and PRBA58 tables.
    gm = np.zeros(9, dtype=F32)
    gm[1] = _ZERO
    gm[2] = F32(AMBE_PRBA24[b3, 0])
    gm[3] = F32(AMBE_PRBA24[b3, 1])
    gm[4] = F32(AMBE_PRBA24[b3, 2])
    gm[5] = F32(AMBE_PRBA58[b4, 0])
    gm[6] = F32(AMBE_PRBA58[b4, 1])
    gm[7] = F32(AMBE_PRBA58[b4, 2])
    gm[8] = F32(AMBE_PRBA58[b4, 3])

    ri = inverse_dct_prba(gm)

    # First two columns of each cik block are computed from ri.
    cik = np.zeros((5, 18), dtype=F32)
    cik[1, 1] = _HALF * (ri[1] + ri[2])
    cik[1, 2] = _RCONST * (ri[1] - ri[2])
    cik[2, 1] = _HALF * (ri[3] + ri[4])
    cik[2, 2] = _RCONST * (ri[3] - ri[4])
    cik[3, 1] = _HALF * (ri[5] + ri[6])
    cik[3, 2] = _RCONST * (ri[5] - ri[6])
    cik[4, 1] = _HALF * (ri[7] + ri[8])
    cik[4, 2] = _RCONST * (ri[7] - ri[8])

    # ji comes from AMBE_LMPRBL[n_harmonics].
    ji = (
        0,
        int(AMBE_LMPRBL[n_harmonics, 0]),
        int(AMBE_LMPRBL[n_harmonics, 1]),
        int(AMBE_LMPRBL[n_harmonics, 2]),
        int(AMBE_LMPRBL[n_harmonics, 3]),
    )

    # HOC load: cik[i][k] for k in 3..ji[i], clipped to zero when k > 6.
    _hoc_tables = (None, AMBE_HOC_B5[b5], AMBE_HOC_B6[b6], AMBE_HOC_B7[b7], AMBE_HOC_B8[b8])
    for i in range(1, 5):
        for k in range(3, ji[i] + 1):
            if k > 6:
                cik[i, k] = _ZERO
            else:
                cik[i, k] = F32(_hoc_tables[i][k - 3])  # type: ignore[index]

    # Inverse DCT each block -> tl.
    tl = inverse_dct_hoc(cik, ji)

    # Reconstruct log2_ml (and therefore Ml) with interpolation + BigGamma.
    _reconstruct_log2ml(cur_mp, prev_mp, tl, float(unvc))

    return 0
