"""Spectral amplitude enhancement

Operates in-place on ``cur_mp`` and leaves every other field alone.

The enhancement sharpens spectral peaks by computing two moments
(``Rm0``, ``Rm1``) of the power spectrum, deriving a per-harmonic
weighting ``Wl``, clamping it to the interval [0.5, 1.2] (outside a
don't-care band at low harmonics ``8l <= L``), and re-normalising so
the total energy is preserved.

Uses float32 throughout with single-precision libm routines
(``np.sqrt`` / ``np.cos`` / ``np.power`` on float32 inputs).
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.vocoder.ambe._param_kernels import spectral_amp_enhance_kernel
from tsdr.radio.vocoder.ambe.params import MbeParams


def spectral_amp_enhance(cur_mp: MbeParams) -> None:
    """Enhance the spectral amplitudes of `cur_mp` in place."""
    spectral_amp_enhance_kernel(cur_mp.Ml, cur_mp.L, np.float32(cur_mp.w0))
