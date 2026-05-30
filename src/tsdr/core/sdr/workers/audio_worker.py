import inspect
import logging
import queue
import time
from math import gcd
from typing import Any

import numpy as np
import soundcard

from tsdr.core.events.events import AudioOutputErrorEvent
from tsdr.core.tracing import span
from tsdr.core.workers import WorkerContext
from tsdr.radio.dsp._kernels import StreamingPolyphaseResampler

logger = logging.getLogger(__name__)

# SoundCard's CoreAudio backend caps blocksize at 512 frames.
BLOCK_SIZE = 512
_DEFAULT_PREBUFFER_SECONDS = 0.15
_MAX_DEPTH_FACTOR = 4
_DEFAULT_CHECK_INTERVAL_SECONDS = 1.0


class AudioOutputWorker:
    TARGET_RATE = 48000
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

        self._player_cm: Any | None = None
        self.player: Any | None = None
        # soundcard's macOS backend takes play(data, wait=...) and defaults to
        # blocking; the Linux/Windows backends take play(data) only. Detected
        # per-player in _open_player.
        self._play_accepts_wait: bool = False
        self._speaker_id: Any | None = None
        self._last_default_check = 0.0

        self._residual: np.ndarray | None = None
        self._pre_buffer_blocks = max(
            1, int(_DEFAULT_PREBUFFER_SECONDS * self.TARGET_RATE / self.BLOCK_SIZE)
        )
        # SoundCard's audio thread starts draining player._queue on __enter__,
        # so we can't defer its start. Hold the first _pre_buffer_blocks here
        # and flush them all at once; the audio thread plays silence meanwhile.
        self._pending_blocks: list[np.ndarray] = []
        self._prebuffered = False
        self._needs_flush = False

        self._resample_source_rate: float = 0.0
        self._resampler: StreamingPolyphaseResampler | None = None

        self._cumulative_input_duration: float = 0.0
        self._cumulative_output_duration: float = 0.0

        self._volume: float = 0.0

        self._push_count = 0
        self._underflow_count = 0
        self._underrunning = False
        self._drop_count = 0
        self._last_batch_time: float | None = None
        self._total_frames_in = 0
        self._total_frames_out = 0
        self._stream_start_time: float | None = None
        self._last_stats_time: float | None = None
        self._last_glitch_underflows = 0

    def setup(self, context: WorkerContext) -> None:
        logger.info("audio_worker_starting source=%s", self.source_id)
        try:
            self._open_player()
        except Exception as e:  # noqa: BLE001 - soundcard surfaces opaque OS errors
            logger.error("audio_stream_open_failed source=%s error=%r", self.source_id, e)
            context.emit_event(
                AudioOutputErrorEvent(
                    source_id=self.source_id,
                    error=f"Failed to open audio device: {e}",
                )
            )
            raise

    def run(self, context: WorkerContext) -> None:
        assert self.player is not None, "player must be set in setup()"
        self._stream_start_time = time.perf_counter()
        self._last_stats_time = self._stream_start_time

        while context.should_continue():
            with span("audio_worker"):
                if self._needs_flush:
                    self._flush_pending_audio()
                self._maybe_follow_default()

                with span("audio.queue_get"):
                    try:
                        audio_batch = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                now = time.perf_counter()
                audio_samples = audio_batch.samples
                source_rate = audio_batch.sample_rate

                requested_blocks = max(
                    1,
                    int(audio_batch.prebuffer_seconds * self.TARGET_RATE / self.BLOCK_SIZE),
                )
                if requested_blocks > self._pre_buffer_blocks:
                    self._pre_buffer_blocks = requested_blocks
                    logger.info(
                        "audio_prebuffer_raised source=%s blocks=%d seconds=%.2f",
                        self.source_id,
                        self._pre_buffer_blocks,
                        self._pre_buffer_blocks * self.BLOCK_SIZE / self.TARGET_RATE,
                    )

                batch_frames = audio_samples.shape[0]
                batch_duration = batch_frames / source_rate
                self._maybe_log_upstream_stall(now, batch_duration)
                self._last_batch_time = now
                self._total_frames_in += batch_frames

                if audio_samples.ndim == 1:
                    audio_samples = np.column_stack([audio_samples, audio_samples])

                if abs(source_rate - self.TARGET_RATE) > 1.0:
                    with span("resample"):
                        try:
                            audio_resampled = self._resample_streaming(audio_samples, source_rate)
                        except ValueError as e:
                            logger.warning(
                                "audio_resampling_failed source=%s error=%r", self.source_id, e
                            )
                            audio_resampled = audio_samples
                else:
                    audio_resampled = audio_samples

                self._cumulative_input_duration += batch_frames / source_rate
                self._cumulative_output_duration += audio_resampled.shape[0] / self.TARGET_RATE

                audio_output = np.ascontiguousarray(
                    np.clip(audio_resampled, -1.0, 1.0), dtype=np.float32
                )

                if self._residual is not None:
                    audio_output = np.concatenate([self._residual, audio_output])
                    self._residual = None

                n_blocks = len(audio_output) // self.BLOCK_SIZE
                max_depth = self._pre_buffer_blocks * _MAX_DEPTH_FACTOR
                gain = self._volume**2
                for i in range(n_blocks):
                    block = audio_output[i * self.BLOCK_SIZE : (i + 1) * self.BLOCK_SIZE]
                    self._push_block(block, max_depth, gain)

                remainder = len(audio_output) % self.BLOCK_SIZE
                if remainder:
                    self._residual = audio_output[n_blocks * self.BLOCK_SIZE :]

                self._maybe_emit_glitch_aggregate()

    def _push_block(self, block: np.ndarray, max_depth: int, gain: float = 1.0) -> None:
        if not self._prebuffered:
            self._pending_blocks.append(block)
            if len(self._pending_blocks) >= self._pre_buffer_blocks:
                for pending in self._pending_blocks:
                    self._raw_push(pending, max_depth, gain)
                self._pending_blocks.clear()
                self._prebuffered = True
                logger.info(
                    "audio_prebuffered source=%s blocks=%d seconds=%.2f",
                    self.source_id,
                    self._pre_buffer_blocks,
                    self._pre_buffer_blocks * self.BLOCK_SIZE / self.TARGET_RATE,
                )
            return

        depth = self._queue_depth()
        if depth == 0:
            self._underflow_count += 1
            if not self._underrunning:
                self._underrunning = True
                logger.warning(
                    "audio_underflow_began source=%s count=%d",
                    self.source_id,
                    self._underflow_count,
                )
        elif self._underrunning:
            self._underrunning = False
            logger.info(
                "audio_underflow_resolved source=%s total=%d depth=%d",
                self.source_id,
                self._underflow_count,
                depth,
            )

        self._raw_push(block, max_depth, gain)

    def _raw_push(self, block: np.ndarray, max_depth: int, gain: float = 1.0) -> None:
        q = getattr(self.player, "_queue", None)
        if q is not None:
            while len(q) >= max_depth:
                try:
                    q.popleft()
                    self._drop_count += 1
                except IndexError:
                    break

        assert self.player is not None
        if self._play_accepts_wait:
            self.player.play(block * gain, wait=False)
        else:
            self.player.play(block * gain)
        self._push_count += 1
        self._total_frames_out += self.BLOCK_SIZE

    def _queue_depth(self) -> int:
        q = getattr(self.player, "_queue", None)
        return len(q) if q is not None else 0

    def _maybe_log_upstream_stall(self, now: float, batch_duration: float) -> None:
        if self._last_batch_time is None:
            return
        gap = now - self._last_batch_time
        if gap <= batch_duration * 1.5 or gap <= 0.1:
            return
        depth = self._queue_depth()
        if depth >= self._pre_buffer_blocks:
            return
        logger.warning(
            "audio_upstream_stall source=%s gap=%.3f batch_duration=%.3f depth=%d",
            self.source_id,
            gap,
            batch_duration,
            depth,
        )

    def _maybe_emit_glitch_aggregate(self) -> None:
        if self._last_stats_time is None:
            return
        stats_elapsed = time.perf_counter() - self._last_stats_time
        if stats_elapsed < 5.0:
            return
        new_underflows = self._underflow_count - self._last_glitch_underflows
        if new_underflows > 0:
            drift = self._cumulative_output_duration - self._cumulative_input_duration
            logger.info(
                "audio_glitch device=%s elapsed=%.1f underflows=%d dropped=%d drift=%.6f",
                self.source_id,
                stats_elapsed,
                new_underflows,
                self._drop_count,
                drift,
            )
            self._last_glitch_underflows = self._underflow_count
        self._last_stats_time = time.perf_counter()

    def _open_player(self) -> None:
        speaker = (
            soundcard.default_speaker()
            if self.output_device is None
            else soundcard.get_speaker(_coerce_speaker_spec(self.output_device))
        )
        self._speaker_id = speaker.id
        self._player_cm = speaker.player(
            samplerate=self.TARGET_RATE,
            channels=2,
            blocksize=self.BLOCK_SIZE,
        )
        self.player = self._player_cm.__enter__()
        self._play_accepts_wait = "wait" in inspect.signature(self.player.play).parameters
        logger.info(
            "audio_stream_opened source=%s device=%r speaker_id=%r rate=%d",
            self.source_id,
            speaker.name,
            self._speaker_id,
            self.TARGET_RATE,
        )

    def _close_player(self) -> None:
        if self._player_cm is None:
            return
        try:
            self._player_cm.__exit__(None, None, None)
            logger.info("audio_stream_closed source=%s", self.source_id)
        except Exception as e:  # noqa: BLE001 - cleanup must not fail
            logger.warning("audio_stream_close_failed source=%s error=%r", self.source_id, e)
        self._player_cm = None
        self.player = None
        self._speaker_id = None

    def _maybe_follow_default(self) -> None:
        if self.output_device is not None:
            return
        now = time.monotonic()
        if now - self._last_default_check < _DEFAULT_CHECK_INTERVAL_SECONDS:
            return
        self._last_default_check = now
        try:
            new_speaker = soundcard.default_speaker()
        except Exception as e:  # noqa: BLE001 - soundcard surfaces opaque OS errors
            logger.warning("audio_default_query_failed source=%s error=%r", self.source_id, e)
            return
        if new_speaker.id == self._speaker_id:
            return
        logger.info(
            "audio_default_changed source=%s old=%s new=%s",
            self.source_id,
            self._speaker_id,
            new_speaker.id,
        )
        self._close_player()
        self._open_player()

    def _resample_streaming(self, audio_samples: np.ndarray, source_rate: float) -> np.ndarray:
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
            "audio_teardown source=%s pushes=%d underflows=%d dropped=%d "
            "frames_in=%d frames_out=%d elapsed=%.1f",
            self.source_id,
            self._push_count,
            self._underflow_count,
            self._drop_count,
            self._total_frames_in,
            self._total_frames_out,
            elapsed,
        )
        self._close_player()

    def on_config_change(self, config) -> None:
        self._volume = config.audio_volume

    def _flush_pending_audio(self) -> None:
        dropped = 0
        q = getattr(self.player, "_queue", None)
        if q is not None:
            while True:
                try:
                    q.popleft()
                    dropped += 1
                except IndexError:
                    break
        self._pending_blocks.clear()
        self._residual = None
        self._prebuffered = False
        self._needs_flush = False
        logger.info("audio_flush source=%s dropped=%d", self.source_id, dropped)

    def request_flush(self) -> None:
        self._needs_flush = True


def _coerce_speaker_spec(spec: str) -> str | int:
    # SoundCard's get_speaker takes either an int id or a name substring;
    # stringified ints ("86") don't match either path.
    try:
        return int(spec)
    except ValueError:
        return spec


def list_audio_devices() -> list[dict[str, Any]]:
    try:
        return [
            {"id": sp.id, "name": sp.name, "channels": sp.channels}
            for sp in soundcard.all_speakers()
        ]
    except Exception as e:  # noqa: BLE001 - soundcard surfaces opaque OS errors
        logger.error("audio_devices_query_failed error=%r", e, exc_info=True)
        return []
