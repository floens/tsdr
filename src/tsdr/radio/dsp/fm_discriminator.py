"""FM demodulation via conjugate-product discriminator.

Output is normalized by deviation so ±deviation Hz maps to ±1.0.
Uses a Numba kernel for single-pass computation without intermediate arrays.
"""

import numpy as np

from tsdr.radio.dsp._kernels import _fm_discriminator_f32


class FMDiscriminator:
    """FM demodulation via conjugate-product discriminator (Numba-accelerated)."""

    def __init__(self, sample_rate: float, deviation: float):
        self._scale = sample_rate / (2 * np.pi * deviation)
        self._prev_iq = np.complex64(0)

    def set_deviation(self, sample_rate: float, deviation: float) -> None:
        """Update the output normalization without disturbing prev-sample state."""
        self._scale = sample_rate / (2 * np.pi * deviation)

    def process(self, iq: np.ndarray) -> np.ndarray:
        if len(iq) == 0:
            return np.array([], dtype=np.float32)

        out, prev_re, prev_im = _fm_discriminator_f32(
            np.ascontiguousarray(iq, dtype=np.complex64),
            self._prev_iq.real,
            self._prev_iq.imag,
            self._scale,
        )
        self._prev_iq = np.complex64(complex(prev_re, prev_im))
        result: np.ndarray = out
        return result

    def reset(self) -> None:
        self._prev_iq = np.complex64(0)
