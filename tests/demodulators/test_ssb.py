"""End-to-end tests for the SSB (USB/LSB) demodulator.

All tests run with squelch disabled (the demodulator default) so that the
chain is deterministic chunk-to-chunk.

The synthesizer below produces a *pure analytic carrier* at the requested
audio frequency: ``exp(+j 2pi f t)`` for USB, ``exp(-j 2pi f t)`` for LSB.
This is sufficient for single-tone recovery and rejection tests; it is not
a general SSB modulator (which would require Hilbert-transformed audio).
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdr.radio.demodulators.ssb import SSBDemodulator


def _generate_ssb_iq(
    sample_rate: float,
    duration: float,
    audio_freq: float,
    mode: str = "USB",
    carrier_offset: float = 0.0,
) -> np.ndarray:
    n = int(duration * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    sign = +1.0 if mode.upper() == "USB" else -1.0
    iq = np.exp(sign * 2j * np.pi * audio_freq * t)
    if carrier_offset != 0.0:
        iq *= np.exp(2j * np.pi * carrier_offset * t)
    return iq.astype(np.complex64)


def _drain_audio(demod: SSBDemodulator) -> np.ndarray:
    return np.concatenate([b.samples for b in demod.get_audio()])


def _tone_magnitude(audio: np.ndarray, audio_rate: float, target_hz: float) -> float:
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / audio_rate)
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    return float(spec[idx])


def _peak_freq_hz(audio: np.ndarray, audio_rate: float) -> float:
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / audio_rate)
    return float(freqs[int(np.argmax(spec))])


def test_usb_recovers_tone():
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    f_audio = 1000.0

    iq = _generate_ssb_iq(sample_rate, duration=0.5, audio_freq=f_audio, mode="USB")
    demod = SSBDemodulator("USB", sample_rate, audio_rate, channel_bandwidth=3_000)
    demod.demodulate(iq, 0.0)
    audio = _drain_audio(demod)

    audio_steady = audio[int(0.2 * demod.decimated_rate) :]
    peak = _peak_freq_hz(audio_steady, demod.decimated_rate)
    assert abs(peak - f_audio) < 5.0


def test_lsb_recovers_tone():
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    f_audio = 1000.0

    iq = _generate_ssb_iq(sample_rate, duration=0.5, audio_freq=f_audio, mode="LSB")
    demod = SSBDemodulator("LSB", sample_rate, audio_rate, channel_bandwidth=3_000)
    demod.demodulate(iq, 0.0)
    audio = _drain_audio(demod)

    audio_steady = audio[int(0.2 * demod.decimated_rate) :]
    peak = _peak_freq_hz(audio_steady, demod.decimated_rate)
    assert abs(peak - f_audio) < 5.0


def test_opposite_sideband_rejection():
    """USB demod fed an LSB analytic signal -> heavily attenuated.

    Catches accidental real/imag swaps and verifies that the asymmetric
    audio butter actually rejects the wrong sideband.
    """
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    f_audio = 1000.0

    iq_correct = _generate_ssb_iq(sample_rate, duration=0.5, audio_freq=f_audio, mode="USB")
    iq_wrong = _generate_ssb_iq(sample_rate, duration=0.5, audio_freq=f_audio, mode="LSB")

    correct = SSBDemodulator("USB", sample_rate, audio_rate, channel_bandwidth=3_000)
    wrong = SSBDemodulator("USB", sample_rate, audio_rate, channel_bandwidth=3_000)
    correct.demodulate(iq_correct, 0.0)
    wrong.demodulate(iq_wrong, 0.0)

    a_correct = _drain_audio(correct)[int(0.2 * correct.decimated_rate) :]
    a_wrong = _drain_audio(wrong)[int(0.2 * wrong.decimated_rate) :]

    tone_correct = _tone_magnitude(a_correct, correct.decimated_rate, f_audio)
    tone_wrong = _tone_magnitude(a_wrong, wrong.decimated_rate, f_audio)
    rejection_db = 20.0 * np.log10(tone_correct / max(tone_wrong, 1e-10))
    assert rejection_db > 20.0


def test_rejects_out_of_band():
    """A USB tone above channel_bandwidth must be heavily attenuated."""
    sample_rate = 240_000.0
    audio_rate = 48_000.0

    iq_in = _generate_ssb_iq(sample_rate, duration=0.5, audio_freq=1000.0, mode="USB")
    iq_out = _generate_ssb_iq(sample_rate, duration=0.5, audio_freq=5000.0, mode="USB")

    in_demod = SSBDemodulator("USB", sample_rate, audio_rate, channel_bandwidth=3_000)
    out_demod = SSBDemodulator("USB", sample_rate, audio_rate, channel_bandwidth=3_000)
    in_demod.demodulate(iq_in, 0.0)
    out_demod.demodulate(iq_out, 0.0)

    a_in = _drain_audio(in_demod)[int(0.2 * in_demod.decimated_rate) :]
    a_out = _drain_audio(out_demod)[int(0.2 * out_demod.decimated_rate) :]

    tone_in = _tone_magnitude(a_in, in_demod.decimated_rate, 1000.0)
    tone_out = _tone_magnitude(a_out, out_demod.decimated_rate, 5000.0)
    rejection_db = 20.0 * np.log10(tone_in / max(tone_out, 1e-10))
    assert rejection_db > 30.0


def test_channel_bandwidth_is_audio_passband():
    """Lock in the new ``channel_bandwidth`` semantics: it's the audio
    passband upper cutoff, not the IF complex BW.

    USB at +2.5 kHz passes when channel_bandwidth=3000; the same tone is
    rejected when channel_bandwidth=2000. If anyone reverts to the old
    "channel_bandwidth = full IF" interpretation (cutoff = bw/2 -> 1500 Hz
    one-sided -> would also pass 2.5 kHz weirdly), this test catches it.
    """
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    iq = _generate_ssb_iq(sample_rate, duration=0.5, audio_freq=2500.0, mode="USB")

    wide = SSBDemodulator("USB", sample_rate, audio_rate, channel_bandwidth=3_000)
    narrow = SSBDemodulator("USB", sample_rate, audio_rate, channel_bandwidth=2_000)
    wide.demodulate(iq, 0.0)
    narrow.demodulate(iq, 0.0)

    a_wide = _drain_audio(wide)[int(0.2 * wide.decimated_rate) :]
    a_narrow = _drain_audio(narrow)[int(0.2 * narrow.decimated_rate) :]

    tone_wide = _tone_magnitude(a_wide, wide.decimated_rate, 2500.0)
    tone_narrow = _tone_magnitude(a_narrow, narrow.decimated_rate, 2500.0)
    # Loose threshold: a 128-tap FIR has ~1.2 kHz transition BW at 48 kHz, so
    # at the wide passband's edge the rolloff is gradual. >3x (~10 dB) is
    # enough to confirm the BW knob actually narrows the audio passband.
    assert tone_wide / max(tone_narrow, 1e-10) > 3.0


def test_bandwidth_change_at_runtime():
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    iq = _generate_ssb_iq(sample_rate, duration=0.5, audio_freq=2500.0, mode="USB")

    wide = SSBDemodulator("USB", sample_rate, audio_rate, channel_bandwidth=3_000)
    wide.demodulate(iq, 0.0)
    a_wide = _drain_audio(wide)[int(0.2 * wide.decimated_rate) :]
    tone_wide = _tone_magnitude(a_wide, wide.decimated_rate, 2500.0)

    narrow = SSBDemodulator("USB", sample_rate, audio_rate, channel_bandwidth=3_000)
    narrow.set_channel_bandwidth(2_000)
    narrow.demodulate(iq, 0.0)
    a_narrow = _drain_audio(narrow)[int(0.2 * narrow.decimated_rate) :]
    tone_narrow = _tone_magnitude(a_narrow, narrow.decimated_rate, 2500.0)

    # Loose threshold: a 128-tap FIR has ~1.2 kHz transition BW at 48 kHz, so
    # at the wide passband's edge the rolloff is gradual. >3x (~10 dB) is
    # enough to confirm the BW knob actually narrows the audio passband.
    assert tone_wide / max(tone_narrow, 1e-10) > 3.0


@pytest.mark.parametrize("k", [1, 3, 5, 7, 11, 50])
def test_chunk_continuity(k: int):
    """Splitting input into chunks must not change output beyond fastmath drift."""
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    iq = _generate_ssb_iq(sample_rate, duration=0.3, audio_freq=1000.0, mode="USB")

    full = SSBDemodulator("USB", sample_rate, audio_rate)
    split = SSBDemodulator("USB", sample_rate, audio_rate)

    full.demodulate(iq, 0.0)
    a_full = _drain_audio(full)

    for c in np.array_split(iq, k):
        split.demodulate(np.ascontiguousarray(c), 0.0)
    a_split = _drain_audio(split)

    np.testing.assert_allclose(a_full, a_split, rtol=1e-5, atol=1e-6)


def test_reset_restores_initial_state():
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    iq = _generate_ssb_iq(sample_rate, duration=0.2, audio_freq=1000.0, mode="USB")

    reused = SSBDemodulator("USB", sample_rate, audio_rate)
    fresh = SSBDemodulator("USB", sample_rate, audio_rate)

    reused.demodulate(iq, 0.0)
    _ = _drain_audio(reused)
    reused.reset()
    reused.demodulate(iq, 0.0)
    a_reused = _drain_audio(reused)

    fresh.demodulate(iq, 0.0)
    a_fresh = _drain_audio(fresh)

    np.testing.assert_allclose(a_reused, a_fresh, rtol=1e-5, atol=1e-6)
