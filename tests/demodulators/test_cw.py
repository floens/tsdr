"""End-to-end tests for the CW (Morse code) demodulator.

All tests run with squelch disabled (the demodulator default) so the chain is
deterministic chunk-to-chunk.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdr.radio.demodulators.cw import CWDemodulator


def _generate_cw_iq(
    sample_rate: float,
    duration: float,
    carrier_offset: float = 0.0,
    keying: np.ndarray | None = None,
) -> np.ndarray:
    """Synthesize a CW signal: a complex carrier at ``carrier_offset`` Hz,
    optionally gated by a real-valued envelope ``keying``.
    """
    n = int(duration * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    iq = np.exp(2j * np.pi * carrier_offset * t)
    if keying is not None:
        iq = iq * keying
    return iq.astype(np.complex64)


def _drain_audio(demod: CWDemodulator) -> np.ndarray:
    return np.concatenate([b.samples for b in demod.get_audio()])


def _peak_freq_hz(audio: np.ndarray, audio_rate: float) -> float:
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / audio_rate)
    return float(freqs[int(np.argmax(spec))])


def _tone_magnitude(audio: np.ndarray, audio_rate: float, target_hz: float) -> float:
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / audio_rate)
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    return float(spec[idx])


def test_recovers_bfo_tone():
    """Carrier at IF=0 with default tone_hz=700 -> audio peak at 700 Hz."""
    sample_rate = 240_000.0
    audio_rate = 48_000.0

    iq = _generate_cw_iq(sample_rate, duration=0.5, carrier_offset=0.0)
    demod = CWDemodulator(sample_rate, audio_rate)
    demod.demodulate(iq, 0.0)
    audio = _drain_audio(demod)

    audio_steady = audio[int(0.2 * demod.decimated_rate) :]
    peak = _peak_freq_hz(audio_steady, demod.decimated_rate)
    assert abs(peak - demod.tone_hz) < 1.0


def test_two_tones_pass():
    """Carriers at IF=+50 and IF=-50 both produce audio in the passband.

    Locks in that the IF LPF is symmetric (both sidebands of the IF passband
    are demodulated; CW reception is symmetric around DC).
    """
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    n = int(0.5 * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    iq = (np.exp(2j * np.pi * 50.0 * t) + np.exp(-2j * np.pi * 50.0 * t)).astype(np.complex64)

    demod = CWDemodulator(sample_rate, audio_rate)
    demod.demodulate(iq, 0.0)
    audio = _drain_audio(demod)
    audio_steady = audio[int(0.2 * demod.decimated_rate) :]

    tone_lower = _tone_magnitude(audio_steady, demod.decimated_rate, demod.tone_hz - 50.0)
    tone_upper = _tone_magnitude(audio_steady, demod.decimated_rate, demod.tone_hz + 50.0)
    # Both sidebands present; their magnitudes should be similar.
    assert tone_lower > 5.0
    assert tone_upper > 5.0
    assert 0.3 < tone_lower / tone_upper < 3.0


def test_image_rejection():
    """Carrier at IF = +2*tone_hz folds onto +tone_hz audio after .real;
    the LPF must reject it by >=40 dB vs the on-channel case."""
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    tone_hz = 700.0

    iq_in = _generate_cw_iq(sample_rate, duration=0.5, carrier_offset=0.0)
    iq_image = _generate_cw_iq(sample_rate, duration=0.5, carrier_offset=2 * tone_hz)

    in_demod = CWDemodulator(sample_rate, audio_rate, tone_hz=tone_hz)
    image_demod = CWDemodulator(sample_rate, audio_rate, tone_hz=tone_hz)
    in_demod.demodulate(iq_in, 0.0)
    image_demod.demodulate(iq_image, 0.0)

    a_in = _drain_audio(in_demod)[int(0.2 * in_demod.decimated_rate) :]
    a_image = _drain_audio(image_demod)[int(0.2 * image_demod.decimated_rate) :]

    tone_in = _tone_magnitude(a_in, in_demod.decimated_rate, tone_hz)
    tone_image = _tone_magnitude(a_image, image_demod.decimated_rate, tone_hz)
    rejection_db = 20.0 * np.log10(tone_in / max(tone_image, 1e-10))
    assert rejection_db > 40.0


def test_rejects_out_of_band():
    """Carrier at IF=+500 Hz with channel_bandwidth=200 must be rejected.

    The LPF cutoff is 100 Hz, deep stopband at 500 Hz.
    """
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    tone_hz = 700.0

    iq_in = _generate_cw_iq(sample_rate, duration=0.5, carrier_offset=0.0)
    iq_out = _generate_cw_iq(sample_rate, duration=0.5, carrier_offset=500.0)

    in_demod = CWDemodulator(sample_rate, audio_rate, channel_bandwidth=200.0, tone_hz=tone_hz)
    out_demod = CWDemodulator(sample_rate, audio_rate, channel_bandwidth=200.0, tone_hz=tone_hz)
    in_demod.demodulate(iq_in, 0.0)
    out_demod.demodulate(iq_out, 0.0)

    a_in = _drain_audio(in_demod)[int(0.2 * in_demod.decimated_rate) :]
    a_out = _drain_audio(out_demod)[int(0.2 * out_demod.decimated_rate) :]

    tone_in = _tone_magnitude(a_in, in_demod.decimated_rate, tone_hz)
    tone_out = _tone_magnitude(a_out, out_demod.decimated_rate, tone_hz + 500.0)
    rejection_db = 20.0 * np.log10(tone_in / max(tone_out, 1e-10))
    assert rejection_db > 30.0


def test_bandwidth_change_at_runtime():
    """Narrowing channel_bandwidth must reduce a near-edge tone."""
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    # Carrier at IF=+150 -> audio at tone+150. Wide BW=400 (cutoff 200) passes;
    # narrow BW=100 (cutoff 50) rejects.
    iq = _generate_cw_iq(sample_rate, duration=0.5, carrier_offset=150.0)

    wide = CWDemodulator(sample_rate, audio_rate, channel_bandwidth=400.0)
    wide.demodulate(iq, 0.0)
    a_wide = _drain_audio(wide)[int(0.2 * wide.decimated_rate) :]
    tone_wide = _tone_magnitude(a_wide, wide.decimated_rate, wide.tone_hz + 150.0)

    narrow = CWDemodulator(sample_rate, audio_rate, channel_bandwidth=400.0)
    narrow.set_channel_bandwidth(100.0)
    narrow.demodulate(iq, 0.0)
    a_narrow = _drain_audio(narrow)[int(0.2 * narrow.decimated_rate) :]
    tone_narrow = _tone_magnitude(a_narrow, narrow.decimated_rate, narrow.tone_hz + 150.0)

    assert tone_wide / max(tone_narrow, 1e-10) > 3.0


def test_keyed_tone():
    """A keyed envelope produces a keyed audio tone at tone_hz."""
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    duration = 1.0
    n = int(duration * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    # 5 Hz on/off keying: 100 ms ON, 100 ms OFF, repeating.
    keying = (np.sin(2 * np.pi * 5.0 * t) > 0).astype(np.float64)
    iq = _generate_cw_iq(sample_rate, duration=duration, keying=keying)

    demod = CWDemodulator(sample_rate, audio_rate)
    demod.demodulate(iq, 0.0)
    audio = _drain_audio(demod)
    audio_steady = audio[int(0.3 * demod.decimated_rate) :]

    # Audio peak should be at the BFO tone.
    peak = _peak_freq_hz(audio_steady, demod.decimated_rate)
    assert abs(peak - demod.tone_hz) < 5.0

    # Envelope of audio should have spectral energy at the keying rate (5 Hz).
    audio_env = np.abs(audio_steady)
    audio_env -= float(np.mean(audio_env))
    env_mag_5hz = _tone_magnitude(audio_env, demod.decimated_rate, 5.0)
    env_mag_50hz = _tone_magnitude(audio_env, demod.decimated_rate, 50.0)
    # 5 Hz keying clearly dominates over neighbouring frequencies.
    assert env_mag_5hz / max(env_mag_50hz, 1e-10) > 5.0


@pytest.mark.parametrize("k", [1, 3, 5, 7, 11, 50])
def test_chunk_continuity(k: int):
    """Chunking input must not change output beyond fastmath drift."""
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    iq = _generate_cw_iq(sample_rate, duration=0.3, carrier_offset=0.0)

    full = CWDemodulator(sample_rate, audio_rate)
    split = CWDemodulator(sample_rate, audio_rate)

    full.demodulate(iq, 0.0)
    a_full = _drain_audio(full)

    for c in np.array_split(iq, k):
        split.demodulate(np.ascontiguousarray(c), 0.0)
    a_split = _drain_audio(split)

    np.testing.assert_allclose(a_full, a_split, rtol=1e-5, atol=1e-6)


def test_reset_restores_initial_state():
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    iq = _generate_cw_iq(sample_rate, duration=0.2, carrier_offset=0.0)

    reused = CWDemodulator(sample_rate, audio_rate)
    fresh = CWDemodulator(sample_rate, audio_rate)

    reused.demodulate(iq, 0.0)
    _ = _drain_audio(reused)
    reused.reset()
    reused.demodulate(iq, 0.0)
    a_reused = _drain_audio(reused)

    fresh.demodulate(iq, 0.0)
    a_fresh = _drain_audio(fresh)

    np.testing.assert_allclose(a_reused, a_fresh, rtol=1e-5, atol=1e-6)
