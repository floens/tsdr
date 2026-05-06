"""Numba-accelerated synthesis inner loop.

All inputs are plain numpy arrays + scalar floats -- no dataclasses,
no dynamic dispatch. The arrays are pre-allocated by the caller.

Signature:

    synth_kernel(
        aout,            # float32[160]  -- output buffer (zeroed by caller)
        prev_ml, prev_vl, prev_phi_l,  # float32[58], int32[58], float32[58]
        cur_ml, cur_vl, cur_phi_l,     # same
        pw0, cw0, maxl,  # float32, float32, int
        uvquality, uvstep, uvoffset, qfactor,  # int, float32, float32, float32
        ws,              # float32[321] -- window
        rng_state,       # uint32[1] -- xorshift32 state (mutable)
    ) -> None

The rng_state array is a 1-element uint32 box so the callee can update
it in place -- matches `pyambe._rng.Xorshift32.state` semantics.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

N = 160
_UV_SINE = np.float32(1.3591409 * math.e)
_UV_RAND = np.float32(2.0)
_UV_THRESHOLD = np.float32(2700.0 * math.pi / 4000.0)
_PI_F32 = np.float32(math.pi)
_TWO_PI_F32 = np.float32(2.0 * math.pi)
_RAND_SCALE = np.float32(1.0 / float(0x7FFFFFFF))


@nb.njit(cache=True, fastmath=False)
def _xorshift32(state: np.ndarray) -> np.uint32:
    x = state[0]
    x ^= (x << np.uint32(13)) & np.uint32(0xFFFFFFFF)
    x ^= x >> np.uint32(17)
    x ^= (x << np.uint32(5)) & np.uint32(0xFFFFFFFF)
    state[0] = x
    return x  # type: ignore[no-any-return]


@nb.njit(cache=True, fastmath=False)
def _rand(state: np.ndarray) -> np.float32:
    x = _xorshift32(state) & np.uint32(0x7FFFFFFF)
    return np.float32(np.float32(x) * _RAND_SCALE)


@nb.njit(cache=True, fastmath=False)
def _rand_phase(state: np.ndarray) -> np.float32:
    return np.float32(_rand(state) * _TWO_PI_F32 - _PI_F32)


@nb.njit(cache=True, fastmath=False)
def synth_kernel(
    aout: np.ndarray,
    prev_ml: np.ndarray,
    prev_vl: np.ndarray,
    prev_phi_l: np.ndarray,
    cur_ml: np.ndarray,
    cur_vl: np.ndarray,
    cur_phi_l: np.ndarray,
    pw0: np.float32,
    cw0: np.float32,
    maxl: np.int32,
    uvquality: np.int32,
    uvstep: np.float32,
    uvoffset: np.float32,
    qfactor: np.float32,
    ws: np.ndarray,
    rng_state: np.ndarray,
) -> None:
    """Core synthesis loop. Mutates ``aout`` in place."""
    rphase = np.zeros(64, dtype=np.float32)
    rphase2 = np.zeros(64, dtype=np.float32)

    for harm in range(1, maxl + 1):
        cw0l = cw0 * np.float32(harm)
        pw0l = pw0 * np.float32(harm)

        cur_v = cur_vl[harm]
        prev_v = prev_vl[harm]

        if cur_v == 0 and prev_v == 1:
            # voiced -> unvoiced
            for i in range(uvquality):
                rphase[i] = _rand_phase(rng_state)
            prev_phi = prev_phi_l[harm]
            prev_ml_h = prev_ml[harm]
            cur_ml_h = cur_ml[harm]
            for n in range(N):
                c1 = ws[n + N] * prev_ml_h * np.float32(math.cos(pw0l * np.float32(n) + prev_phi))
                c3 = np.float32(0.0)
                for i in range(uvquality):
                    angle = (
                        cw0 * np.float32(n) * (np.float32(harm) + np.float32(i) * uvstep - uvoffset)
                        + rphase[i]
                    )
                    c3 = c3 + np.float32(math.cos(angle))
                    if cw0l > _UV_THRESHOLD:
                        c3 = c3 + (cw0l - _UV_THRESHOLD) * _UV_RAND * _rand(rng_state)
                c3 = c3 * _UV_SINE * ws[n] * cur_ml_h * qfactor
                aout[n] = aout[n] + c1 + c3

        elif cur_v == 1 and prev_v == 0:
            # unvoiced -> voiced
            for i in range(uvquality):
                rphase[i] = _rand_phase(rng_state)
            cur_phi = cur_phi_l[harm]
            prev_ml_h = prev_ml[harm]
            cur_ml_h = cur_ml[harm]
            for n in range(N):
                c1 = ws[n] * cur_ml_h * np.float32(math.cos(cw0l * np.float32(n - N) + cur_phi))
                c3 = np.float32(0.0)
                for i in range(uvquality):
                    angle = (
                        pw0 * np.float32(n) * (np.float32(harm) + np.float32(i) * uvstep - uvoffset)
                        + rphase[i]
                    )
                    c3 = c3 + np.float32(math.cos(angle))
                    if pw0l > _UV_THRESHOLD:
                        c3 = c3 + (pw0l - _UV_THRESHOLD) * _UV_RAND * _rand(rng_state)
                c3 = c3 * _UV_SINE * ws[n + N] * prev_ml_h * qfactor
                aout[n] = aout[n] + c1 + c3

        elif cur_v == 1 or prev_v == 1:
            # voiced <-> voiced
            prev_phi = prev_phi_l[harm]
            cur_phi = cur_phi_l[harm]
            prev_ml_h = prev_ml[harm]
            cur_ml_h = cur_ml[harm]
            for n in range(N):
                c1 = ws[n + N] * prev_ml_h * np.float32(math.cos(pw0l * np.float32(n) + prev_phi))
                c2 = ws[n] * cur_ml_h * np.float32(math.cos(cw0l * np.float32(n - N) + cur_phi))
                aout[n] = aout[n] + c1 + c2

        else:
            # unvoiced -> unvoiced
            for i in range(uvquality):
                rphase[i] = _rand_phase(rng_state)
            for i in range(uvquality):
                rphase2[i] = _rand_phase(rng_state)
            prev_ml_h = prev_ml[harm]
            cur_ml_h = cur_ml[harm]
            for n in range(N):
                c3 = np.float32(0.0)
                for i in range(uvquality):
                    angle = (
                        pw0 * np.float32(n) * (np.float32(harm) + np.float32(i) * uvstep - uvoffset)
                        + rphase[i]
                    )
                    c3 = c3 + np.float32(math.cos(angle))
                    if pw0l > _UV_THRESHOLD:
                        c3 = c3 + (pw0l - _UV_THRESHOLD) * _UV_RAND * _rand(rng_state)
                c3 = c3 * _UV_SINE * ws[n + N] * prev_ml_h * qfactor
                c4 = np.float32(0.0)
                for i in range(uvquality):
                    angle = (
                        cw0 * np.float32(n) * (np.float32(harm) + np.float32(i) * uvstep - uvoffset)
                        + rphase2[i]
                    )
                    c4 = c4 + np.float32(math.cos(angle))
                    if cw0l > _UV_THRESHOLD:
                        c4 = c4 + (cw0l - _UV_THRESHOLD) * _UV_RAND * _rand(rng_state)
                c4 = c4 * _UV_SINE * ws[n] * cur_ml_h * qfactor
                aout[n] = aout[n] + c3 + c4
