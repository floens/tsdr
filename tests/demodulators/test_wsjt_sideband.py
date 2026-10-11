"""WSJT audio is the upper sideband only: taking the real part of the IQ
alone folded a lower-sideband signal onto the same audio."""

import numpy as np
import pytest

from tsdr.radio.demodulators.wsjt import WSJTDemodulator


def _tone(rate: float, offset_hz: float) -> np.ndarray:
    t = np.arange(int(rate)) / rate
    return np.exp(2j * np.pi * offset_hz * t).astype(np.complex64)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio[len(audio) // 2 :] ** 2)))


@pytest.mark.parametrize("rate", [12_000.0, 12_500.0])
def test_keeps_the_upper_sideband_and_drops_the_lower(rate: float) -> None:
    usb = _rms(WSJTDemodulator(mode="FT8", sample_rate=rate)._usb_audio(_tone(rate, 1000.0)))
    lsb = _rms(WSJTDemodulator(mode="FT8", sample_rate=rate)._usb_audio(_tone(rate, -1000.0)))
    assert usb == pytest.approx(1 / np.sqrt(2), rel=0.02)
    assert 20 * np.log10(lsb / usb) < -50.0


def test_sideband_selection_is_continuous_across_chunks() -> None:
    iq = _tone(12_000.0, 1500.0) + 0.5 * _tone(12_000.0, -700.0)
    whole = WSJTDemodulator(mode="FT8", sample_rate=12_000)._usb_audio(iq)
    demod = WSJTDemodulator(mode="FT8", sample_rate=12_000)
    chunks = np.concatenate([demod._usb_audio(iq[i : i + 997]) for i in range(0, len(iq), 997)])
    assert np.allclose(whole, chunks, atol=1e-5)
