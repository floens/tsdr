import logging
import queue
import time
from math import gcd
from typing import Any

import numpy as np

from tsdr.core.events.events import AudioOutputErrorEvent
from tsdr.core.tracing import span
from tsdr.core.workers import WorkerContext
from tsdr.radio.dsp._kernels import StreamingPolyphaseResampler

logger = logging.getLogger(__name__)

BLOCK_SIZE = 2048  # frames per callback invocation
_DEFAULT_PREBUFFER_SECONDS = 0.15
# Cap callback queue depth at this multiple of the pre-buffer target so a
# transient producer overrun doesn't drift playback forward by seconds.
_MAX_DEPTH_FACTOR = 4


class AudioOutputWorker:
    """Audio output worker for playing demodulated audio.

    Uses sounddevice callback-based OutputStream for glitch-free playback.
    PortAudio pulls from a queue of fixed-size blocks; empty queue: silence.

    Thread Safety:
        - Reads from audio_queue (thread-safe Queue)
        - Callback reads from _callback_queue (thread-safe Queue)
        - Emits events via EventBus (thread-safe)
    """

    TARGET_RATE = 48000  # Target sample rate for output
    BLOCK_SIZE = BLOCK_SIZE

    def __init__(
        self,
        source_id: str,
        audio_queue: queue.Queue,
        output_device: str | None = None,
    ) -> None:
        self.source_id = source_id
        self.audio_queue = audio_queue
        self.output_device = output_device

        # Resources initialized in setup()
        self.stream: Any | None = None  # sounddevice.OutputStream
        self.sd: Any | None = None  # sounddevice module

        # Callback-based output state
        self._callback_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._residual: np.ndarray | None = None
        self._started = False
        self._rebuffering = False
        self._pre_buffer_blocks = max(
            1, int(_DEFAULT_PREBUFFER_SECONDS * self.TARGET_RATE / self.BLOCK_SIZE)
        )

        # Streaming polyphase resampler state
        self._resample_source_rate: float = 0.0
        self._resampler: StreamingPolyphaseResampler | None = None

        # Cumulative duration tracking for drift detection
        self._cumulative_input_duration: float = 0.0
        self._cumulative_output_duration: float = 0.0

        self._volume: float = 0.0

        # Debug tracking state
        self._underflow_count = 0
        self._silence_count = 0
        self._callback_count = 0
        self._rebuffer_count = 0
        self._dropped_blocks = 0
        self._last_batch_time: float | None = None
        self._total_frames_in = 0
        self._total_frames_out = 0
        self._stream_start_time: float | None = None
        self._last_stats_time: float | None = None

    def _audio_callback(self, outdata, frames, time_info, status):
        self._callback_count += 1
        if status.output_underflow:
            self._underflow_count += 1
            logger.warning(
                "Audio underflow #%d (callback_queue depth: %d)",
                self._underflow_count,
                self._callback_queue.qsize(),
            )
        if self._rebuffering:
            outdata[:] = 0
            self._silence_count += 1
            return
        try:
            block = self._callback_queue.get_nowait()
            outdata[:] = block * (self._volume**2)
            self._total_frames_out += frames
        except queue.Empty:
            outdata[:] = 0
            self._silence_count += 1
            self._rebuffering = True
            self._rebuffer_count += 1
            logger.warning("Buffer underrun, rebuffering (#%d)", self._rebuffer_count)

    def setup(self, context: WorkerContext) -> None:
        logger.info(f"Audio output worker starting for source {self.source_id}")

        try:
            import sounddevice as sd  # noqa: PLC0415

            self.sd = sd
        except ImportError as e:
            error_msg = f"sounddevice not available: {e}"
            logger.error(f"Audio worker {self.source_id}: {error_msg}")
            context.emit_event(AudioOutputErrorEvent(source_id=self.source_id, error=error_msg))
            raise

        try:
            self.stream = self.sd.OutputStream(
                device=self.output_device,
                channels=2,
                samplerate=self.TARGET_RATE,
                blocksize=self.BLOCK_SIZE,
                dtype=np.float32,
                latency="high",
                callback=self._audio_callback,
            )
            # Don't start yet - wait for pre-buffer in run()
            logger.info(
                f"Audio stream opened for source {self.source_id} "
                f"(device={self.output_device or 'default'}, rate={self.TARGET_RATE})"
            )
        except Exception as e:
            error_msg = f"Failed to open audio device: {e}"
            logger.error(f"Audio worker {self.source_id}: {error_msg}")
            context.emit_event(AudioOutputErrorEvent(source_id=self.source_id, error=error_msg))
            raise

    def run(self, context: WorkerContext) -> None:
        """Main audio output loop."""
        assert self.sd is not None, "sounddevice must be set in setup()"
        assert self.stream is not None, "audio stream must be set in setup()"

        while context.should_continue():
            with span("audio_worker"):
                with span("audio.queue_get"):
                    try:
                        audio_batch = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                now = time.perf_counter()
                audio_samples = audio_batch.samples
                source_rate = audio_batch.sample_rate

                # Monotonic-max: bump pre-buffer if batch requests more
                requested_blocks = max(
                    1,
                    int(audio_batch.prebuffer_seconds * self.TARGET_RATE / self.BLOCK_SIZE),
                )
                if requested_blocks > self._pre_buffer_blocks:
                    self._pre_buffer_blocks = requested_blocks
                    logger.info(
                        "Pre-buffer raised to %d blocks (%.2fs)",
                        self._pre_buffer_blocks,
                        self._pre_buffer_blocks * self.BLOCK_SIZE / self.TARGET_RATE,
                    )

                # Log batch arrival and queue stats
                batch_frames = audio_samples.shape[0]
                batch_duration = batch_frames / source_rate
                if self._last_batch_time is not None:
                    gap = now - self._last_batch_time
                    # Bursty IO produces large instantaneous gaps without
                    # actual underrun risk; only warn when the playback buffer
                    # is also low.
                    if (
                        gap > batch_duration * 1.5
                        and gap > 0.1
                        and self._callback_queue.qsize() < self._pre_buffer_blocks
                    ):
                        logger.warning(
                            "Upstream stall: gap %.3fs >> batch duration %.3fs (cb_q=%d)",
                            gap,
                            batch_duration,
                            self._callback_queue.qsize(),
                        )
                self._last_batch_time = now
                self._total_frames_in += batch_frames

                # Upmix mono to stereo if needed (e.g. NFM, AM demodulators)
                if audio_samples.ndim == 1:
                    audio_samples = np.column_stack([audio_samples, audio_samples])

                if abs(source_rate - self.TARGET_RATE) > 1.0:
                    with span("resample"):
                        try:
                            audio_resampled = self._resample_streaming(audio_samples, source_rate)
                        except ValueError as e:
                            logger.warning(f"Resampling failed for {self.source_id}: {e}")
                            audio_resampled = audio_samples
                else:
                    audio_resampled = audio_samples

                # Track cumulative duration drift
                self._cumulative_input_duration += batch_frames / source_rate
                self._cumulative_output_duration += audio_resampled.shape[0] / self.TARGET_RATE

                audio_output = np.ascontiguousarray(
                    np.clip(audio_resampled, -1.0, 1.0), dtype=np.float32
                )

                # Prepend residual from previous iteration
                if self._residual is not None:
                    audio_output = np.concatenate([self._residual, audio_output])
                    self._residual = None

                n_blocks = len(audio_output) // self.BLOCK_SIZE
                max_depth = self._pre_buffer_blocks * _MAX_DEPTH_FACTOR
                for i in range(n_blocks):
                    block = audio_output[i * self.BLOCK_SIZE : (i + 1) * self.BLOCK_SIZE]
                    while self._callback_queue.qsize() >= max_depth:
                        try:
                            self._callback_queue.get_nowait()
                            self._dropped_blocks += 1
                        except queue.Empty:
                            break
                    self._callback_queue.put_nowait(block)

                remainder = len(audio_output) % self.BLOCK_SIZE
                if remainder:
                    self._residual = audio_output[n_blocks * self.BLOCK_SIZE :]

                # Resume after rebuffering
                if self._rebuffering and self._callback_queue.qsize() >= self._pre_buffer_blocks:
                    self._rebuffering = False
                    logger.info(
                        "Rebuffer complete (#%d), resuming with %d blocks",
                        self._rebuffer_count,
                        self._callback_queue.qsize(),
                    )

                # Start stream once pre-buffer is filled
                if not self._started and self._callback_queue.qsize() >= self._pre_buffer_blocks:
                    self.stream.start()
                    self._started = True
                    self._stream_start_time = time.perf_counter()
                    self._last_stats_time = self._stream_start_time
                    logger.info(
                        "Audio stream started (pre-buffered %d blocks, %.2fs)",
                        self._callback_queue.qsize(),
                        self._callback_queue.qsize() * self.BLOCK_SIZE / self.TARGET_RATE,
                    )

                # Periodic throughput summary (~5s)
                if self._last_stats_time is not None and self._stream_start_time is not None:
                    stats_elapsed = time.perf_counter() - self._last_stats_time
                    if stats_elapsed >= 5.0:
                        total_elapsed = time.perf_counter() - self._stream_start_time
                        effective_rate = (
                            self._total_frames_out / total_elapsed if total_elapsed > 0 else 0
                        )
                        drift = self._cumulative_output_duration - self._cumulative_input_duration
                        residual_size = len(self._residual) if self._residual is not None else 0
                        logger.debug(
                            "Stats: elapsed=%.1fs, frames_in=%d, frames_out=%d, "
                            "rate=%.0fHz, underflows=%d, silences=%d, rebuffers=%d, "
                            "dropped=%d, cb_q=%d, drift=%.6fs, residual=%d",
                            total_elapsed,
                            self._total_frames_in,
                            self._total_frames_out,
                            effective_rate,
                            self._underflow_count,
                            self._silence_count,
                            self._rebuffer_count,
                            self._dropped_blocks,
                            self._callback_queue.qsize(),
                            drift,
                            residual_size,
                        )
                        self._last_stats_time = time.perf_counter()

    def _resample_streaming(self, audio_samples: np.ndarray, source_rate: float) -> np.ndarray:
        """Resample using a streaming polyphase FIR filter."""
        if source_rate != self._resample_source_rate:
            g = gcd(int(self.TARGET_RATE), int(source_rate))
            up = int(self.TARGET_RATE) // g
            down = int(source_rate) // g
            max_rate = max(up, down)
            n_taps = 2 * 10 * max_rate + 1
            self._resampler = StreamingPolyphaseResampler(up, down, n_taps)
            self._resample_source_rate = source_rate

        assert self._resampler is not None
        return self._resampler.process(audio_samples)

    def teardown(self, context: WorkerContext) -> None:
        elapsed = time.perf_counter() - self._stream_start_time if self._stream_start_time else 0
        logger.info(
            "Audio teardown: callbacks=%d, underflows=%d, silences=%d, "
            "rebuffers=%d, dropped=%d, frames_in=%d, frames_out=%d, elapsed=%.1fs",
            self._callback_count,
            self._underflow_count,
            self._silence_count,
            self._rebuffer_count,
            self._dropped_blocks,
            self._total_frames_in,
            self._total_frames_out,
            elapsed,
        )

        # Zero-pad and enqueue any residual
        if self._residual is not None and len(self._residual) > 0:
            padded = np.zeros((self.BLOCK_SIZE, self._residual.shape[1]), dtype=np.float32)
            padded[: len(self._residual)] = self._residual
            try:
                self._callback_queue.put(padded, timeout=0.5)
            except queue.Full:
                pass
            self._residual = None

        # Wait briefly for callback to drain
        if self._started:
            deadline = time.monotonic() + 0.5
            while not self._callback_queue.empty() and time.monotonic() < deadline:
                time.sleep(0.01)

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                logger.info(f"Audio stream closed for source {self.source_id}")
            except Exception as e:  # noqa: BLE001 - cleanup must not fail
                logger.warning(f"Error closing audio stream: {e}")

    def on_config_change(self, config) -> None:
        self._volume = config.audio_volume


def list_audio_devices() -> list[dict[str, Any]]:
    """List available audio output devices."""
    import sounddevice as sd  # noqa: PLC0415

    try:
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": dev["name"],
                        "channels": dev["max_output_channels"],
                        "sample_rate": dev["default_samplerate"],
                    }
                )
        return devices
    except ImportError as e:
        logger.warning(f"sounddevice not available: {e}")
        return []
    except (OSError, RuntimeError) as e:
        logger.error(f"Failed to query audio devices: {e}", exc_info=True)
        return []
