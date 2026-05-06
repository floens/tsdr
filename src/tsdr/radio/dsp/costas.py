"""Costas loop for carrier phase and frequency recovery.

2nd-order PLL with proportional (alpha) + integral (beta) terms.
Supports BPSK (I*Q error) and QPSK (decision-directed error) modes.
"""

import math

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _costas_bpsk(
    symbols: np.ndarray,
    alpha: float,
    beta: float,
    phase_in: float,
    freq_in: float,
) -> tuple[np.ndarray, float, float]:
    """BPSK Costas loop kernel. Error = re * im."""
    n = symbols.shape[0]
    out = np.empty(n, dtype=np.complex128)
    phase = phase_in
    freq = freq_in
    pi = math.pi
    two_pi = 2.0 * pi

    for i in range(n):
        c = math.cos(phase)
        s = math.sin(phase)
        sym_re = symbols[i].real
        sym_im = symbols[i].imag
        re = sym_re * c + sym_im * s
        im = -sym_re * s + sym_im * c
        out[i] = complex(re, im)

        error = re * im
        freq += beta * error
        phase += freq + alpha * error
        if phase > pi:
            phase -= two_pi
        elif phase < -pi:
            phase += two_pi

    return out, phase, freq


@nb.njit(cache=True)
def _costas_qpsk(
    symbols: np.ndarray,
    alpha: float,
    beta: float,
    phase_in: float,
    freq_in: float,
) -> tuple[np.ndarray, float, float]:
    """QPSK Costas loop kernel. Decision-directed error."""
    n = symbols.shape[0]
    out = np.empty(n, dtype=np.complex128)
    phase = phase_in
    freq = freq_in
    pi = math.pi
    two_pi = 2.0 * pi

    for i in range(n):
        c = math.cos(phase)
        s = math.sin(phase)
        sym_re = symbols[i].real
        sym_im = symbols[i].imag
        re = sym_re * c + sym_im * s
        im = -sym_re * s + sym_im * c
        out[i] = complex(re, im)

        sign_re = 1.0 if re >= 0.0 else -1.0
        sign_im = 1.0 if im >= 0.0 else -1.0
        error = im * sign_re - re * sign_im
        freq += beta * error
        phase += freq + alpha * error
        if phase > pi:
            phase -= two_pi
        elif phase < -pi:
            phase += two_pi

    return out, phase, freq


class CostasLoop:
    """Costas loop for carrier phase and frequency recovery."""

    def __init__(self, alpha: float = 0.1, beta: float = 0.001, mode: str = "bpsk"):
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.phase = 0.0
        self.freq = 0.0

    def process(self, symbols: np.ndarray) -> np.ndarray:
        out: np.ndarray
        if self.mode == "qpsk":
            out, self.phase, self.freq = _costas_qpsk(
                symbols,
                self.alpha,
                self.beta,
                self.phase,
                self.freq,
            )
        else:
            out, self.phase, self.freq = _costas_bpsk(
                symbols,
                self.alpha,
                self.beta,
                self.phase,
                self.freq,
            )
        return out

    def reset(self) -> None:
        self.phase = 0.0
        self.freq = 0.0
