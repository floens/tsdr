import numpy as np

from tsdr.core.sdr.io import load_iq
from tsdr.radio.demodulators.wfm import WidebandFMDemodulator

SAMPLE_RATE = 250_000
AUDIO_RATE = 48000


def _make_fm_signal(
    duration: float,
    *,
    pilot: bool = True,
    stereo_tone_hz: float = 1000.0,
) -> np.ndarray:
    """Synthesize a baseband FM IQ signal with optional stereo pilot and subcarrier.

    Args:
        duration: Signal duration in seconds
        pilot: Whether to include 19 kHz pilot tone
        stereo_tone_hz: Frequency of the L-R test tone (Hz)
    """
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    deviation = 75000.0

    # Baseband audio: L+R mono tone at 1 kHz
    mono = 0.5 * np.sin(2 * np.pi * stereo_tone_hz * t)

    # Composite baseband
    composite = mono

    if pilot:
        # 19 kHz pilot at 10% amplitude (per FM broadcast spec)
        pilot_signal = 0.1 * np.sin(2 * np.pi * 19000 * t)
        # L-R on 38 kHz DSB-SC subcarrier (different tone for L vs R)
        lr_diff = 0.3 * np.sin(2 * np.pi * stereo_tone_hz * 1.5 * t)
        subcarrier = lr_diff * np.sin(2 * np.pi * 38000 * t)
        composite = mono + pilot_signal + subcarrier

    # FM modulate: integrate composite to get phase
    phase = 2 * np.pi * deviation * np.cumsum(composite) / SAMPLE_RATE
    return np.exp(1j * phase).astype(np.complex64)


def test_stereo_detected_with_pilot():
    """Stereo is detected when a 19 kHz pilot tone is present."""
    iq = _make_fm_signal(0.5, pilot=True)
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE, rds_enabled=False)

    # Process in chunks to simulate real streaming
    chunk_size = 65536
    for i in range(0, len(iq), chunk_size):
        demod.demodulate(iq[i : i + chunk_size], 0.0)

    assert demod.stereo_detected is True


def test_mono_when_no_pilot():
    """Stereo is NOT detected when there is no pilot tone."""
    iq = _make_fm_signal(0.5, pilot=False)
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE, rds_enabled=False)

    chunk_size = 65536
    for i in range(0, len(iq), chunk_size):
        demod.demodulate(iq[i : i + chunk_size], 0.0)

    assert demod.stereo_detected is False


def test_output_shape_is_stereo():
    """Output is always (N, 2) regardless of stereo detection."""
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE, rds_enabled=False)

    iq = _make_fm_signal(0.1, pilot=False)
    demod.demodulate(iq, 0.0)
    batches = demod.get_audio()

    assert len(batches) == 1
    audio = batches[0].samples
    assert audio.ndim == 2
    assert audio.shape[1] == 2
    assert audio.dtype == np.float32


def test_empty_input_returns_no_audio():
    """Empty input produces no audio batches."""
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE, rds_enabled=False)
    demod.demodulate(np.array([], dtype=np.complex64), 0.0)
    assert demod.get_audio() == []


def test_stereo_on_real_data():
    """Stereo detection works on real FM broadcast recording."""
    iq = load_iq("tests/samples/freq=98.9M_sr=240k_dur=5s_gain=28_20260423T1733.cu8.zst")
    demod = WidebandFMDemodulator(sample_rate=240_000, audio_rate=AUDIO_RATE, rds_enabled=False)

    chunk_size = 65536
    for i in range(0, min(len(iq), chunk_size * 5), chunk_size):
        demod.demodulate(iq[i : i + chunk_size], 0.0)
        batches = demod.get_audio()
        for batch in batches:
            assert batch.samples.ndim == 2
            assert batch.samples.shape[1] == 2

    assert demod.stereo_detected is True


def test_reset_clears_stereo_state():
    """Reset clears stereo detection and PLL state."""
    iq = _make_fm_signal(0.5, pilot=True)
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE, rds_enabled=False)

    chunk_size = 65536
    for i in range(0, len(iq), chunk_size):
        demod.demodulate(iq[i : i + chunk_size], 0.0)
    assert demod.stereo_detected is True

    demod.reset()
    assert demod.stereo_detected is False
    assert demod.pilot_snr_ema == 0.0
    assert demod.pll_phase == 0.0
