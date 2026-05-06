"""Numba-accelerated parameter decode and enhancement kernels.

All inputs are plain numpy arrays + scalars.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

_F32 = np.float32
_PI = _F32(math.pi)
_ZERO = _F32(0.0)
_HALF = _F32(0.5)
_ONE = _F32(1.0)
_TWO = _F32(2.0)
_ZEROSIX_FIVE = _F32(0.65)
_ONE_TWO = _F32(1.2)
_POINT_NINE_SIX = _F32(0.96)


@nb.njit(cache=True, fastmath=False)
def inverse_dct_prba_kernel(gm: np.ndarray, ri: np.ndarray) -> None:
    """8-point inverse DCT: gm[1..8] -> ri[1..8]."""
    for i in range(1, 9):
        s = _ZERO
        for m in range(1, 9):
            am = _ONE if m == 1 else _TWO
            angle = _F32(_PI * (m - 1) * (i - 0.5) / 8.0)
            s += am * gm[m] * _F32(math.cos(angle))
        ri[i] = s


@nb.njit(cache=True, fastmath=False)
def inverse_dct_hoc_kernel(
    cik: np.ndarray,
    ji: np.ndarray,
    tl: np.ndarray,
) -> None:
    """Inverse DCT of each cik block -> tl[1..56].

    cik is (5, 18), ji is int32[5], both 1-based.
    """
    harm = 1
    for i in range(1, 5):
        ji_i = ji[i]
        for j in range(1, ji_i + 1):
            s = _ZERO
            for k in range(1, ji_i + 1):
                ak = _ONE if k == 1 else _TWO
                angle = _F32(_PI * (k - 1) * (j - 0.5) / ji_i)
                s += ak * cik[i, k] * _F32(math.cos(angle))
            tl[harm] = s
            harm += 1


@nb.njit(cache=True, fastmath=False)
def reconstruct_log2ml_kernel(
    cur_log2_ml: np.ndarray,
    cur_ml: np.ndarray,
    cur_vl: np.ndarray,
    prev_log2_ml: np.ndarray,
    prev_ml: np.ndarray,
    tl: np.ndarray,
    n_harmonics: int,
    prev_n_harmonics: int,
    gamma: float,
    unvc: float,
) -> None:
    """Eqs. 40-43: interpolate prev log2_ml, combine with tl/BigGamma, exp to ml."""
    # Extend prev by repeating last value
    if prev_n_harmonics < n_harmonics:
        for harm in range(prev_n_harmonics + 1, n_harmonics + 1):
            prev_ml[harm] = prev_ml[prev_n_harmonics]
            prev_log2_ml[harm] = prev_log2_ml[prev_n_harmonics]

    # Sentinel for boundary condition
    prev_log2_ml[0] = prev_log2_ml[1]
    prev_ml[0] = prev_ml[1]

    # Part 1: interpolation coefficients + prediction sum
    ratio = _F32(_F32(prev_n_harmonics) / _F32(n_harmonics))
    sum43 = _ZERO
    # Scratch arrays on stack (numba supports small fixed arrays)
    flokl = np.zeros(57, dtype=np.float32)
    intkl = np.zeros(57, dtype=np.int32)
    deltal = np.zeros(57, dtype=np.float32)

    for harm in range(1, n_harmonics + 1):
        flokl[harm] = ratio * _F32(harm)
        intkl[harm] = int(flokl[harm])
        deltal[harm] = flokl[harm] - _F32(intkl[harm])
        term = (_ONE - deltal[harm]) * prev_log2_ml[intkl[harm]] + deltal[harm] * prev_log2_ml[
            intkl[harm] + 1
        ]
        sum43 += term
    sum43 = (_ZEROSIX_FIVE / _F32(n_harmonics)) * sum43

    # Part 2: BigGamma
    sum42 = _ZERO
    for harm in range(1, n_harmonics + 1):
        sum42 += tl[harm]
    sum42 = sum42 / _F32(n_harmonics)
    log2_n = math.log(float(n_harmonics)) / math.log(2.0)
    big_gamma = _F32(gamma - 0.5 * log2_n - float(sum42))

    # Part 3: combine and exp
    for harm in range(1, n_harmonics + 1):
        c1 = _ZEROSIX_FIVE * (_ONE - deltal[harm]) * prev_log2_ml[intkl[harm]]
        c2 = _ZEROSIX_FIVE * deltal[harm] * prev_log2_ml[intkl[harm] + 1]
        cur_log2_ml[harm] = tl[harm] + c1 + c2 - sum43 + big_gamma
        val = math.exp(0.693 * float(cur_log2_ml[harm]))
        if cur_vl[harm] == 1:
            cur_ml[harm] = _F32(val)
        else:
            cur_ml[harm] = _F32(unvc * val)


@nb.njit(cache=True, fastmath=False)
def spectral_amp_enhance_kernel(
    ml_arr: np.ndarray,
    n_harmonics: int,
    w0: np.float32,
) -> None:
    """Spectral amplitude enhancement in-place on ml_arr[1..n_harmonics]."""
    # Compute Rm0, Rm1
    rm0 = _ZERO
    rm1 = _ZERO
    for harm in range(1, n_harmonics + 1):
        ml_sq = ml_arr[harm] * ml_arr[harm]
        rm0 += ml_sq
        rm1 += ml_sq * _F32(math.cos(w0 * _F32(harm)))

    r2m0 = rm0 * rm0
    r2m1 = rm1 * rm1

    # Compute wl, clamp, apply
    for harm in range(1, n_harmonics + 1):
        ml = ml_arr[harm]
        if ml == _ZERO:
            continue
        cos_term = _F32(math.cos(w0 * _F32(harm)))
        numer = _POINT_NINE_SIX * _PI * ((r2m0 + r2m1) - _TWO * rm0 * rm1 * cos_term)
        denom = w0 * rm0 * (r2m0 - r2m1)
        if denom == _ZERO:
            continue
        ratio = numer / denom
        if ratio <= _ZERO:
            continue
        wl = _F32(math.sqrt(ml)) * _F32(math.sqrt(math.sqrt(ratio)))

        if (8 * harm) <= n_harmonics:
            pass
        elif wl > _ONE_TWO:
            ml_arr[harm] = _ONE_TWO * ml
        elif wl < _HALF:
            ml_arr[harm] = _HALF * ml
        else:
            ml_arr[harm] = wl * ml

    # Re-normalise
    sum_sq = _ZERO
    for harm in range(1, n_harmonics + 1):
        sum_sq += ml_arr[harm] * ml_arr[harm]
    if sum_sq == _ZERO:
        g = _ONE
    else:
        g = _F32(math.sqrt(rm0 / sum_sq))
    for harm in range(1, n_harmonics + 1):
        ml_arr[harm] = g * ml_arr[harm]
