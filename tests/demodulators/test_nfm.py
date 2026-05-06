"""End-to-end tests for the Narrowband FM demodulator.

All tests run with squelch disabled (the demodulator default) so that the
chain is deterministic chunk-to-chunk and bit-exact streamability holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdr.radio.demodulators.nfm import NarrowbandFMDemodulator


def _generate_fm_iq(
    sample_rate: float,
    duration: float,
    modulation_freq: float,
    deviation: float,
    modulation_depth: float = 1.0,
    carrier_offset: float = 0.0,
) -> np.ndarray:
    """Continuous-phase FM signal modulated by a single sine tone, optionally
    centred at ``carrier_offset`` Hz from DC.
    """
    n = int(duration * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    beta = deviation * modulation_depth / modulation_freq
    phase = beta * np.sin(2.0 * np.pi * modulation_freq * t)
    iq = np.exp(1j * phase) * np.exp(2j * np.pi * carrier_offset * t)
    return iq.astype(np.complex64)


def _drain_audio(demod: NarrowbandFMDemodulator) -> np.ndarray:
    return np.concatenate([b.samples for b in demod.get_audio()])


def _peak_freq_hz(audio: np.ndarray, audio_rate: float) -> float:
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / audio_rate)
    return float(freqs[int(np.argmax(spec))])


def _tone_magnitude(audio: np.ndarray, audio_rate: float, target_hz: float) -> float:
    """Magnitude of the spectral bin closest to ``target_hz``.

    For FM, RMS conflates a recovered tone with the broadband noise that an
    attenuated input produces through the discriminator. The spectral bin at
    the modulation frequency is what actually drops when the channel filter
    rejects an out-of-band signal.
    """
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / audio_rate)
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    return float(spec[idx])


def test_recovers_modulation_tone():
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    f_mod = 1000.0

    iq = _generate_fm_iq(sample_rate, duration=0.5, modulation_freq=f_mod, deviation=2500.0)
    demod = NarrowbandFMDemodulator(sample_rate, audio_rate, channel_bandwidth=12_500)
    demod.demodulate(iq, 0.0)
    audio = _drain_audio(demod)

    audio_steady = audio[int(0.1 * demod.decimated_rate) :]
    peak = _peak_freq_hz(audio_steady, demod.decimated_rate)
    assert abs(peak - f_mod) < 5.0


def test_rejects_out_of_band_signal():
    """Channel selectivity test using carrier offset.

    Modulator frequency just shapes sidebands inside Carson's bandwidth, so
    only a frequency-shifted *carrier* tests the channel filter.
    """
    sample_rate = 240_000.0
    audio_rate = 48_000.0

    iq_in = _generate_fm_iq(
        sample_rate,
        duration=0.5,
        modulation_freq=1000.0,
        deviation=2000.0,
    )
    iq_out = _generate_fm_iq(
        sample_rate,
        duration=0.5,
        modulation_freq=1000.0,
        deviation=2000.0,
        carrier_offset=15_000.0,
    )

    in_demod = NarrowbandFMDemodulator(sample_rate, audio_rate, channel_bandwidth=8_000)
    out_demod = NarrowbandFMDemodulator(sample_rate, audio_rate, channel_bandwidth=8_000)
    in_demod.demodulate(iq_in, 0.0)
    out_demod.demodulate(iq_out, 0.0)

    a_in = _drain_audio(in_demod)[int(0.1 * in_demod.decimated_rate) :]
    a_out = _drain_audio(out_demod)[int(0.1 * out_demod.decimated_rate) :]

    tone_in = _tone_magnitude(a_in, in_demod.decimated_rate, 1000.0)
    tone_out = _tone_magnitude(a_out, out_demod.decimated_rate, 1000.0)
    rejection_db = 20.0 * np.log10(tone_in / max(tone_out, 1e-10))
    assert rejection_db > 30.0


def test_bandwidth_change_at_runtime():
    """Narrowing channel_bandwidth must reject high-frequency modulator sidebands.

    Strategy: a high modulation frequency (8 kHz) creates modulator sidebands
    well outside a narrow channel filter but inside a wide one. Wide BW
    recovers the modulator tone; narrow BW rejects it.

    Uses ``deviation=`` override so the discriminator scale stays constant
    between the wide and narrow runs -- otherwise narrowing the BW also
    rescales the discriminator output and inflates residual noise on a
    rejected signal.
    """
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    f_mod = 8000.0
    iq = _generate_fm_iq(
        sample_rate,
        duration=0.5,
        modulation_freq=f_mod,
        deviation=1000.0,
    )

    wide = NarrowbandFMDemodulator(
        sample_rate,
        audio_rate,
        channel_bandwidth=24_000,
        deviation=1000.0,
        de_emphasis_tc=0.0,  # disable so we measure filter selectivity, not LPF roll-off
    )
    wide.demodulate(iq, 0.0)
    a_wide = _drain_audio(wide)[int(0.1 * wide.decimated_rate) :]
    tone_wide = _tone_magnitude(a_wide, wide.decimated_rate, f_mod)

    narrow = NarrowbandFMDemodulator(
        sample_rate,
        audio_rate,
        channel_bandwidth=24_000,
        deviation=1000.0,
        de_emphasis_tc=0.0,
    )
    narrow.set_channel_bandwidth(5_000)
    narrow.demodulate(iq, 0.0)
    a_narrow = _drain_audio(narrow)[int(0.1 * narrow.decimated_rate) :]
    tone_narrow = _tone_magnitude(a_narrow, narrow.decimated_rate, f_mod)

    assert tone_wide / max(tone_narrow, 1e-10) > 10.0


def test_deviation_override_pins_scale():
    """Explicit ``deviation=`` ctor arg pins discriminator scale; without it
    a bandwidth change re-derives the scale from channel_bandwidth/2."""
    sample_rate = 240_000.0
    audio_rate = 48_000.0

    pinned = NarrowbandFMDemodulator(
        sample_rate,
        audio_rate,
        channel_bandwidth=12_500,
        deviation=5000.0,
    )
    initial_scale = pinned._fm_discrim._scale
    pinned.set_channel_bandwidth(25_000)
    assert pinned._fm_discrim._scale == initial_scale

    derived = NarrowbandFMDemodulator(sample_rate, audio_rate, channel_bandwidth=12_500)
    initial_scale = derived._fm_discrim._scale
    derived.set_channel_bandwidth(25_000)
    assert derived._fm_discrim._scale != initial_scale


def test_de_emphasis_attenuates_high_freq():
    """A 5 kHz tone must come out lower than a 1 kHz tone with default 750 µs
    de-emphasis (single-pole IIR has ~−14 dB at 5 kHz given fc ~ 212 Hz)."""
    sample_rate = 240_000.0
    audio_rate = 48_000.0

    iq_low = _generate_fm_iq(
        sample_rate,
        duration=0.5,
        modulation_freq=1000.0,
        deviation=2000.0,
    )
    iq_high = _generate_fm_iq(
        sample_rate,
        duration=0.5,
        modulation_freq=5000.0,
        deviation=2000.0,
    )

    low = NarrowbandFMDemodulator(sample_rate, audio_rate, channel_bandwidth=20_000)
    high = NarrowbandFMDemodulator(sample_rate, audio_rate, channel_bandwidth=20_000)
    low.demodulate(iq_low, 0.0)
    high.demodulate(iq_high, 0.0)

    a_low = _drain_audio(low)[int(0.1 * low.decimated_rate) :]
    a_high = _drain_audio(high)[int(0.1 * high.decimated_rate) :]
    rms_low = float(np.sqrt(np.mean(np.square(a_low))))
    rms_high = float(np.sqrt(np.mean(np.square(a_high))))
    # 5 kHz is at least 6 dB below 1 kHz with 750 µs de-emphasis.
    attenuation_db = 20.0 * np.log10(rms_low / max(rms_high, 1e-10))
    assert attenuation_db > 6.0


@pytest.mark.parametrize("k", [1, 3, 5, 7, 11, 50])
def test_chunk_continuity(k: int):
    """Splitting input into chunks must not change output beyond fastmath drift.

    The pipeline includes ``fastmath=True`` Numba kernels (FIR, FM
    discriminator) whose SIMD vectorization differs with input length. The
    resulting drift is below float32 epsilon (~1e-7); anything bigger is a
    real state-continuity bug.
    """
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    iq = _generate_fm_iq(
        sample_rate,
        duration=0.3,
        modulation_freq=1000.0,
        deviation=2000.0,
    )

    full = NarrowbandFMDemodulator(sample_rate, audio_rate)
    split = NarrowbandFMDemodulator(sample_rate, audio_rate)

    full.demodulate(iq, 0.0)
    a_full = _drain_audio(full)

    for c in np.array_split(iq, k):
        split.demodulate(np.ascontiguousarray(c), 0.0)
    a_split = _drain_audio(split)

    np.testing.assert_allclose(a_full, a_split, rtol=1e-5, atol=1e-6)


def test_reset_restores_initial_state():
    """After demod -> reset -> demod, the second pass must equal a fresh
    instance's output. Tolerance allows fastmath rounding."""
    sample_rate = 240_000.0
    audio_rate = 48_000.0
    iq = _generate_fm_iq(
        sample_rate,
        duration=0.2,
        modulation_freq=1000.0,
        deviation=2000.0,
    )

    reused = NarrowbandFMDemodulator(sample_rate, audio_rate)
    fresh = NarrowbandFMDemodulator(sample_rate, audio_rate)

    reused.demodulate(iq, 0.0)
    _ = _drain_audio(reused)
    reused.reset()
    reused.demodulate(iq, 0.0)
    a_reused = _drain_audio(reused)

    fresh.demodulate(iq, 0.0)
    a_fresh = _drain_audio(fresh)

    np.testing.assert_allclose(a_reused, a_fresh, rtol=1e-5, atol=1e-6)
