"""Streaming RNNoise audio denoiser.

Wraps a per-instance ``RNNoiseState`` for use on a demodulator's audio stream.
RNNoise runs at a fixed 48 kHz on 480-sample mono frames, so this:

- resamples the input to 48 kHz when the demod emits another rate (only then);
- scales ±1.0 audio to the int16 range RNNoise expects, and back;
- accumulates 480-sample frames across calls, carrying the remainder;
- passes stereo batches through untouched (WFM/broadcast is out of scope).

Denoised audio stays at 48 kHz — the audio worker resamples to its output rate
anyway, so there's no reason to convert back to the demod's native rate.
"""

import numpy as np

from tsdr.core.sdr.datatypes import AudioBatch
from tsdr.radio.dsp._kernels import StreamingPolyphaseResampler, make_rational_resampler
from tsdr.radio.dsp.rnnoise import FRAME_SIZE, RNNoiseState

RNNOISE_RATE = 48000
_INT16_SCALE = 32768.0


class AudioDenoiser:
    """Denoise a demodulator's mono audio stream with RNNoise."""

    def __init__(self) -> None:
        self._state = RNNoiseState()
        self._resampler: StreamingPolyphaseResampler | None = None
        self._resample_src_rate: float | None = None
        self._carry = np.empty(0, dtype=np.float32)

    def _resample_to_48k(self, samples: np.ndarray, src_rate: float) -> np.ndarray:
        if src_rate != self._resample_src_rate:
            self._resampler = make_rational_resampler(RNNOISE_RATE, src_rate)
            self._resample_src_rate = src_rate
        assert self._resampler is not None
        return self._resampler.process(samples.reshape(-1, 1))[:, 0]

    def process(self, batch: AudioBatch) -> AudioBatch:
        if batch.stereo:
            return batch

        x = np.ascontiguousarray(batch.samples, dtype=np.float32)
        if batch.sample_rate != RNNOISE_RATE:
            x = self._resample_to_48k(x, batch.sample_rate)

        buf = np.concatenate((self._carry, x * _INT16_SCALE))
        n_frames = len(buf) // FRAME_SIZE
        frames = [
            self._state.process(buf[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]) for i in range(n_frames)
        ]
        self._carry = buf[n_frames * FRAME_SIZE :].copy()

        if frames:
            denoised = np.concatenate(frames) / _INT16_SCALE
        else:
            denoised = np.empty(0, dtype=np.float32)

        return AudioBatch(
            samples=denoised,
            sample_rate=float(RNNOISE_RATE),
            stereo=False,
            prebuffer_seconds=batch.prebuffer_seconds,
        )

    def reset(self) -> None:
        self._carry = np.empty(0, dtype=np.float32)
        if self._resampler is not None:
            self._resampler.reset()

    def close(self) -> None:
        self._state.close()
