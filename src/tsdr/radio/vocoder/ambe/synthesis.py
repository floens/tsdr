"""Layer 6: speech synthesis.

Overlap-add synthesis of 160 float32 samples from the current and
previous MBE parameter frames.

This wrapper handles state setup (L-mismatch fix-up, PSIl/PHIl phase
continuity) and delegates the per-band inner loops to the
numba-compiled ``pyambe._synth_kernel.synth_kernel``. The kernel
picks one of four paths per band based on the voiced/unvoiced
transition between previous and current frame (v→v, v→u, u→v, u→u),
and drives unvoiced multisine noise through the xorshift32 RNG.

The caller must seed the RNG before each frame if they want
reproducible output.
"""

from __future__ import annotations

import math

import numpy as np

from tsdr.radio.vocoder.ambe._constants import WS
from tsdr.radio.vocoder.ambe._rng import Xorshift32
from tsdr.radio.vocoder.ambe._synth_kernel import synth_kernel
from tsdr.radio.vocoder.ambe.params import MbeParams

F32 = np.float32
N = 160  # 20 ms at 8 kHz
_F32_ZERO = F32(0.0)
_F32_HALF = F32(0.5)

# Convert the constants table to float32 once at import time.
_WS_F32 = np.asarray(WS, dtype=F32)


def synthesize_silencef() -> np.ndarray:
    """160 samples of zero (silence frame)."""
    return np.zeros(N, dtype=F32)


def synthesize_speechf(
    cur_mp: MbeParams,
    prev_mp: MbeParams,
    uvquality: int,
    rng: Xorshift32,
) -> np.ndarray:
    """Synthesize 160 float32 PCM samples for one 20 ms frame.

    Sets up the non-kernel state (L mismatch fix-up, PSIl/PHIl update)
    then delegates the per-band inner loops to the numba-compiled
    ``synth_kernel``.

    ``cur_mp`` holds the current frame's parameters. ``prev_mp`` holds
    the previous frame's parameters. The ``PHIl`` and ``PSIl`` of
    ``cur_mp`` are updated in place -- the caller must copy them back
    into prev state before the next frame.

    ``uvquality`` is the "voices per band" knob for unvoiced
    synthesis, 1..64. 3 is the default.

    ``rng`` must be a seeded ``Xorshift32``; the caller re-seeds it per
    frame to get reproducible output.
    """
    if not (1 <= uvquality <= 64):
        uvquality = 3

    aout = np.zeros(N, dtype=F32)

    if uvquality == 1:
        loguvq = F32(1.0 / math.e)
    else:
        loguvq = F32(math.log(float(uvquality)) / float(uvquality))
    uvstep = F32(1.0) / F32(uvquality)
    qfactor = loguvq
    uvoffset = (uvstep * F32(uvquality - 1)) / F32(2.0)

    # count unvoiced bands in the current frame (up to L)
    num_uv = int(np.sum(cur_mp.Vl[1 : cur_mp.L + 1] == 0))

    cw0 = F32(cur_mp.w0)
    pw0 = F32(prev_mp.w0)

    # eq 128/129: L mismatch fix-up -- zero missing harmonics and mark
    # them voiced.
    if cur_mp.L > prev_mp.L:
        maxl = cur_mp.L
        for harm in range(prev_mp.L + 1, maxl + 1):
            prev_mp.Ml[harm] = _F32_ZERO
            prev_mp.Vl[harm] = 1
    else:
        maxl = prev_mp.L
        for harm in range(cur_mp.L + 1, maxl + 1):
            cur_mp.Ml[harm] = _F32_ZERO
            cur_mp.Vl[harm] = 1

    # PSIl/PHIl updates -- PSIl for every harmonic in 1..56 regardless of L.
    # PHIl is PSIl within the first L/4 harmonics and PSIl + random
    # phase jitter above that.
    phi_split = cur_mp.L // 4
    for harm in range(1, 57):
        cur_mp.PSIl[harm] = prev_mp.PSIl[harm] + (pw0 + cw0) * F32(harm * N) * _F32_HALF
        if harm <= phi_split:
            cur_mp.PHIl[harm] = cur_mp.PSIl[harm]
        else:
            cur_mp.PHIl[harm] = cur_mp.PSIl[harm] + F32(
                (num_uv * rng.rand_phase()) / float(cur_mp.L)
            )

    # Hand off the hot loop to the numba kernel. We pass `rng.state` as
    # a 1-element uint32 array so the kernel can mutate it in place.
    rng_state = np.array([rng.state], dtype=np.uint32)
    synth_kernel(
        aout,
        prev_mp.Ml,
        prev_mp.Vl,
        prev_mp.PHIl,
        cur_mp.Ml,
        cur_mp.Vl,
        cur_mp.PHIl,
        pw0,
        cw0,
        np.int32(maxl),
        np.int32(uvquality),
        uvstep,
        uvoffset,
        qfactor,
        _WS_F32,
        rng_state,
    )
    rng.state = int(rng_state[0])
    return aout


def float_to_short(float_buf: np.ndarray) -> np.ndarray:
    """Gain-scale and clip float32 audio to int16.

    Apply a gain of 7 and clip to ±32760.
    """
    if float_buf.shape != (N,):
        raise ValueError(f"expected length {N}, got {float_buf.shape}")
    gain = F32(7.0)
    scaled = gain * float_buf
    clipped = np.clip(scaled, -32760.0, 32760.0)
    return clipped.astype(np.int16)
