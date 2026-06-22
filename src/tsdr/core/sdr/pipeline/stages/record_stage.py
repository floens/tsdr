from __future__ import annotations

import logging
from pathlib import Path
from typing import IO, Any

import numpy as np
import zstandard as zstd

from tsdr.core.events.events import RecordingFinishedEvent
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.samples_batch import SampleFormat, SamplesBatch
from tsdr.radio.dsp import StreamingDecimFilter, firwin
from tsdr.radio.dsp._kernels import StreamingPolyphaseResampler

logger = logging.getLogger(__name__)

_RESAMPLE_TAPS = 65
_DEFAULT_LEVEL = 3


class RecordStage:
    """Sink stage that writes IQ samples to a ``.cu8.zst`` / ``.cf32.zst`` file.

    ``sample_format`` selects the on-disk precision: ``UINT8_IQ`` (compact,
    lossless for 8-bit devices like RTL-SDR) or ``COMPLEX64`` (full float32,
    preserves the dynamic range of high-bit-depth devices like Airspy HF+).

    If ``max_samples`` is set the stage self-terminates once reached and
    publishes ``RecordingFinishedEvent``; otherwise it runs until the pipeline
    is removed from the device config.
    """

    def __init__(
        self,
        output_path: Path | str,
        pipeline_name: str = "recording",
        resample: tuple[int, int] | None = None,
        max_samples: int | None = None,
        sample_format: SampleFormat = SampleFormat.UINT8_IQ,
    ) -> None:
        self._path = Path(output_path)
        self._partial_path = self._path.with_suffix(self._path.suffix + ".partial")
        self._pipeline_name = pipeline_name
        self._resample = resample
        self._max_samples = max_samples
        self._sample_format = sample_format

        self._decim: StreamingDecimFilter | None = None
        self._poly: StreamingPolyphaseResampler | None = None

        self._file: IO[bytes] | None = None
        self._zwriter: Any = None
        self._samples_written = 0
        self._closed = False
        self._finished_event_published = False

    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        if self._closed or data.iq_samples is None:
            return data

        self._ensure_open(data.sample_rate)
        samples = self._resample_chunk(data.iq_samples)

        if self._max_samples is not None:
            remaining = self._max_samples - self._samples_written
            if remaining <= 0:
                self._finalize_duration_mode(context)
                return data
            if len(samples) > remaining:
                samples = samples[:remaining]

        if len(samples) > 0:
            self._write_samples(samples)
            self._samples_written += len(samples)

        if self._max_samples is not None and self._samples_written >= self._max_samples:
            self._finalize_duration_mode(context)
        return data

    def _ensure_open(self, device_rate: float) -> None:
        if self._file is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._partial_path, "wb")  # noqa: SIM115 - owned by stage, closed in close()
        self._zwriter = zstd.ZstdCompressor(level=_DEFAULT_LEVEL).stream_writer(self._file)

        if self._resample is None:
            return
        up, down = self._resample
        if up == down:
            self._resample = None
        elif up == 1 and down > 1:
            cutoff = (device_rate / down) * 0.45
            taps = firwin(_RESAMPLE_TAPS, cutoff, fs=device_rate).astype(np.float32)
            self._decim = StreamingDecimFilter(taps, down, dtype=np.complex64)
        else:
            self._poly = StreamingPolyphaseResampler(up, down, n_taps=_RESAMPLE_TAPS)

    def _resample_chunk(self, samples: np.ndarray) -> np.ndarray:
        if self._decim is not None:
            return self._decim.process(samples)
        if self._poly is not None:
            stacked = np.empty((len(samples), 2), dtype=np.float32)
            stacked[:, 0] = samples.real
            stacked[:, 1] = samples.imag
            out = self._poly.process(stacked)
            recombined = np.empty(len(out), dtype=np.complex64)
            recombined.real = out[:, 0]
            recombined.imag = out[:, 1]
            return recombined
        return samples

    def _write_samples(self, samples: np.ndarray) -> None:
        assert self._zwriter is not None
        if self._sample_format is SampleFormat.COMPLEX64:
            self._zwriter.write(memoryview(np.ascontiguousarray(samples, dtype=np.complex64)))
            return
        n = len(samples)
        cu8 = np.empty(n * 2, dtype=np.uint8)
        cu8[0::2] = np.clip(np.round(samples.real * 127.5 + 127.5), 0, 255).astype(np.uint8)
        cu8[1::2] = np.clip(np.round(samples.imag * 127.5 + 127.5), 0, 255).astype(np.uint8)
        self._zwriter.write(cu8.tobytes())

    def _finalize_duration_mode(self, context: PipelineContext) -> None:
        if self._finished_event_published:
            return
        self._finished_event_published = True
        self.close()
        context.event_bus.publish(
            RecordingFinishedEvent(
                source_id=f"record_{context.device_context.device_id}",
                device_id=context.device_context.device_id,
                pipeline_name=self._pipeline_name,
                path=str(self._path),
                samples_written=self._samples_written,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._zwriter is not None:
            self._zwriter.close()
            self._zwriter = None
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._partial_path.exists():
            self._partial_path.rename(self._path)

    def on_config_change(self, config: Any) -> None:
        pass

    def reset(self) -> None:
        pass

    @property
    def samples_written(self) -> int:
        return self._samples_written

    @property
    def path(self) -> Path:
        return self._path
