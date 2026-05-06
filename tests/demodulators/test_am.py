from __future__ import annotations

import numpy as np

from tsdr.radio.demodulators.am import AMDemodulator


def _generate_am_iq(
    sample_rate: float,
    duration: float,
    modulation_freq: float,
    carrier_offset: float = 0.0,
    modulation_depth: float = 0.5,
) -> np.ndarray:
    """AM-modulated complex IQ: (1 + m*sin(2pi f_m t)) * exp(2pi j f_c t)."""
    n = int(duration * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    envelope = 1.0 + modulation_depth * np.sin(2.0 * np.pi * modulation_freq * t)
    iq = envelope * np.exp(2j * np.pi * carrier_offset * t)
    return iq.astype(np.complex64)


def _drain_audio(demod: AMDemodulator) -> np.ndarray:
    return np.concatenate([b.samples for b in demod.get_audio()])


def _peak_freq_hz(audio: np.ndarray, audio_rate: float) -> float:
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / audio_rate)
    return float(freqs[int(np.argmax(spec))])


def test_recovers_modulation_tone():
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    f_mod = 1000.0

    iq = _generate_am_iq(sample_rate, duration=0.5, modulation_freq=f_mod)
    demod = AMDemodulator(sample_rate, audio_rate, channel_bandwidth=10_000)
    demod.demodulate(iq, 0.0)
    audio = _drain_audio(demod)

    # Skip startup transient before peak detection.
    audio_steady = audio[int(0.2 * audio_rate) :]
    peak = _peak_freq_hz(audio_steady, audio_rate)
    assert abs(peak - f_mod) < 5.0


def test_rejects_out_of_band_modulation():
    sample_rate = 240_000.0
    audio_rate = 48_000.0

    iq_in = _generate_am_iq(sample_rate, duration=0.5, modulation_freq=1000.0)
    iq_out = _generate_am_iq(sample_rate, duration=0.5, modulation_freq=15_000.0)

    in_demod = AMDemodulator(sample_rate, audio_rate, channel_bandwidth=10_000)
    out_demod = AMDemodulator(sample_rate, audio_rate, channel_bandwidth=10_000)
    in_demod.demodulate(iq_in, 0.0)
    out_demod.demodulate(iq_out, 0.0)

    a_in = _drain_audio(in_demod)[int(0.2 * audio_rate) :]
    a_out = _drain_audio(out_demod)[int(0.2 * audio_rate) :]

    rms_in = float(np.sqrt(np.mean(np.square(a_in))))
    rms_out = float(np.sqrt(np.mean(np.square(a_out))))
    rejection_db = 20.0 * np.log10(rms_in / max(rms_out, 1e-10))
    assert rejection_db > 30.0


def test_bandwidth_change_at_runtime():
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    # 12 kHz tone: passes a wide channel filter, blocked by a narrow one.
    iq = _generate_am_iq(sample_rate, duration=0.5, modulation_freq=12_000.0)

    wide = AMDemodulator(sample_rate, audio_rate, channel_bandwidth=24_000)
    wide.demodulate(iq, 0.0)
    a_wide = _drain_audio(wide)[int(0.2 * audio_rate) :]
    rms_wide = float(np.sqrt(np.mean(np.square(a_wide))))

    narrow = AMDemodulator(sample_rate, audio_rate, channel_bandwidth=24_000)
    narrow.set_channel_bandwidth(5_000)
    narrow.demodulate(iq, 0.0)
    a_narrow = _drain_audio(narrow)[int(0.2 * audio_rate) :]
    rms_narrow = float(np.sqrt(np.mean(np.square(a_narrow))))

    assert rms_wide / max(rms_narrow, 1e-10) > 10.0


def test_chunk_continuity_bit_exact():
    """Splitting the input into chunks must not change the output."""
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    iq = _generate_am_iq(sample_rate, duration=0.3, modulation_freq=1000.0)

    full = AMDemodulator(sample_rate, audio_rate)
    split = AMDemodulator(sample_rate, audio_rate)

    full.demodulate(iq, 0.0)
    a_full = _drain_audio(full)

    chunks = np.array_split(iq, 7)
    for c in chunks:
        split.demodulate(np.ascontiguousarray(c), 0.0)
    a_split = _drain_audio(split)

    np.testing.assert_array_equal(a_full, a_split)
