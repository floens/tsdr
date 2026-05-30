import numpy as np
import pytest

from tsdr.core.sdr.datatypes import AudioBatch
from tsdr.radio.dsp.denoise import AudioDenoiser
from tsdr.radio.dsp.rnnoise import FRAME_SIZE, RNNoiseState, rnnoise_available

pytestmark = pytest.mark.skipif(
    not rnnoise_available(), reason="librnnoise not available on this platform"
)


def _batch(samples: np.ndarray, sample_rate: float, stereo: bool = False) -> AudioBatch:
    return AudioBatch(samples=samples.astype(np.float32), sample_rate=sample_rate, stereo=stereo)


def test_rnnoise_state_frame_shape_and_dtype():
    state = RNNoiseState()
    rng = np.random.default_rng(0)
    frame = (rng.standard_normal(FRAME_SIZE) * 3000.0).astype(np.float32)
    out = state.process(frame)
    assert out.shape == (FRAME_SIZE,)
    assert out.dtype == np.float32
    state.close()


def test_state_close_idempotent():
    state = RNNoiseState()
    state.close()
    state.close()  # no error on second close


def test_denoiser_passes_stereo_through_untouched():
    den = AudioDenoiser()
    batch = _batch(np.zeros(1000), 48000.0, stereo=True)
    assert den.process(batch) is batch
    den.close()


def test_denoiser_keeps_48k_and_frames_with_carry():
    den = AudioDenoiser()
    rng = np.random.default_rng(1)

    out1 = den.process(_batch(rng.standard_normal(500) * 0.1, 48000.0))
    assert out1.sample_rate == 48000.0
    assert len(out1.samples) == FRAME_SIZE  # 500 -> one 480 frame, 20 carried
    assert den._resampler is None  # no resampling at 48k

    out2 = den.process(_batch(rng.standard_normal(460) * 0.1, 48000.0))
    assert len(out2.samples) == FRAME_SIZE  # 20 carried + 460 = 480
    den.close()


def test_denoiser_resamples_non_48k_to_48k():
    den = AudioDenoiser()
    rng = np.random.default_rng(2)
    out = den.process(_batch(rng.standard_normal(5000) * 0.1, 50000.0))
    assert out.sample_rate == 48000.0
    assert den._resampler is not None
    # 5000 @ 50k -> ~4800 @ 48k -> 10 frames
    assert len(out.samples) == 10 * FRAME_SIZE
    den.close()


def test_denoiser_attenuates_noise():
    den = AudioDenoiser()
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(4800) * 0.3).astype(np.float32)
    out = den.process(_batch(x, 48000.0))
    in_rms = float(np.sqrt(np.dot(x, x) / len(x)))
    out_rms = float(np.sqrt(np.dot(out.samples, out.samples) / len(out.samples)))
    assert out_rms < in_rms * 0.5  # RNNoise strongly suppresses pure noise
    den.close()


def test_denoiser_output_in_range():
    den = AudioDenoiser()
    rng = np.random.default_rng(4)
    out = den.process(_batch(rng.standard_normal(4800) * 0.2, 48000.0))
    assert np.all(np.isfinite(out.samples))
    assert np.max(np.abs(out.samples)) < 2.0
    den.close()
