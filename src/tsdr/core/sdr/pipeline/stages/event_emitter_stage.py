import time

import numpy as np

from tsdr.core.events.events import (
    ConstellationUpdateEvent,
    DecoderOutputEvent,
    DemodStatusEvent,
    FFTUpdateEvent,
    StatsUpdateEvent,
)
from tsdr.core.sdr.config import SDRConfig
from tsdr.core.sdr.datatypes import DemodStatus
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.processing import compute_statistics
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.tracing import get_smoothed_stats, traced
from tsdr.radio.dsp._kernels import _iq_metrics_c64


class RateLimiter:
    """Fixed-FPS gate for UI updates."""

    def __init__(self, target_fps: int):
        self.target_fps = target_fps
        self.min_interval = 1.0 / target_fps
        self.last_send_time: float | None = None

    def should_send(self) -> bool:
        now = time.perf_counter()
        if self.last_send_time is None or now - self.last_send_time >= self.min_interval:
            self.last_send_time = now
            return True
        return False

    def set_target_fps(self, fps: int) -> None:
        if fps <= 0:
            raise ValueError("FPS must be positive")

        self.target_fps = fps
        self.min_interval = 1.0 / fps


class EventEmitterStage:
    """Terminal stage that publishes TUI-bound events.

    Publishes:
    - FFTUpdateEvent (rate-limited) when spectrum/frequencies are populated
    - StatsUpdateEvent (rate-limited) when spectrum is populated
    - DecoderOutputEvent when decoded_messages are present
    """

    def __init__(self, config: SDRConfig):
        self.fft_rate_limiter = RateLimiter(target_fps=config.update_rate_fps)
        self.stats_rate_limiter = RateLimiter(target_fps=config.update_rate_fps)
        self.constellation_rate_limiter = RateLimiter(target_fps=20)
        self._last_demod_status: DemodStatus | None = None

    @traced("event_emitter")
    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        dc = context.device_context
        device_id = dc.device_id

        self._emit_fft(data, context, device_id)
        self._emit_stats(data, context, device_id)
        self._emit_demod_status(data, context, device_id)
        self._emit_decoder_messages(data, context, device_id)
        self._emit_constellation(data, context, device_id)

        return data

    def _emit_fft(self, data: SamplesBatch, context: PipelineContext, device_id: str) -> None:
        if data.spectrum is None or data.frequencies is None:
            return
        if not self.fft_rate_limiter.should_send():
            return

        context.event_bus.publish(
            FFTUpdateEvent(
                source_id=f"fft_{device_id}",
                device_id=device_id,
                spectrum=data.spectrum,
                frequencies=data.frequencies,
                center_frequency=data.center_frequency,
                sample_rate=data.sample_rate,
            )
        )

    def _emit_stats(self, data: SamplesBatch, context: PipelineContext, device_id: str) -> None:
        if data.spectrum is None:
            return
        if not self.stats_rate_limiter.should_send():
            return

        dc = context.device_context
        profile = dc.demod_profile if dc else None
        channel_bw = profile.channel_bandwidth if profile else None
        signal_stats = compute_statistics(
            data.spectrum,
            data.center_frequency,
            data.sample_rate,
            channel_bandwidth=channel_bw,
            channel_offset_hz=dc.config.tuned_frequency - data.center_frequency if dc else 0.0,
        )

        sample_queue = dc.sample_queue
        queue_size = sample_queue.qsize() if sample_queue else 0
        queue_capacity = sample_queue.maxsize if sample_queue else 0

        # IQ amplitude metrics (single-pass numba kernel)
        iq_rms = None
        iq_peak = None
        iq_clip_pct = None
        if data.iq_samples is not None and len(data.iq_samples) > 0:
            iq_rms, iq_peak, iq_clip_pct = _iq_metrics_c64(
                np.ascontiguousarray(data.iq_samples, dtype=np.complex64)
            )

        context.event_bus.publish(
            StatsUpdateEvent(
                source_id=f"stats_{device_id}",
                device_id=device_id,
                center_frequency=data.center_frequency,
                sample_rate=data.sample_rate,
                rf_gain=data.rf_gain,
                samples_processed=dc.total_samples_read,
                samples_dropped=dc.dropped_samples,
                queue_size=queue_size,
                queue_capacity=queue_capacity,
                peak_power=signal_stats.peak_power,
                average_power=signal_stats.average_power,
                peak_frequency=signal_stats.peak_frequency,
                peak_bin=signal_stats.peak_bin,
                noise_floor=signal_stats.noise_floor,
                dynamic_range=signal_stats.dynamic_range,
                fft_size=dc.config.fft_size,
                fft_window=dc.config.fft_window,
                spectrum_bins=len(data.spectrum),
                demod_mode=dc.active_mode,
                channel_snr=signal_stats.channel_snr,
                stereo=dc.stereo,
                iq_rms=iq_rms,
                iq_peak=iq_peak,
                iq_clip_pct=iq_clip_pct,
                update_rate_fps=context.config.update_rate_fps,
                performance_stats=get_smoothed_stats(window_seconds=5.0),
            )
        )

    def _emit_demod_status(
        self, data: SamplesBatch, context: PipelineContext, device_id: str
    ) -> None:
        if data.demod_status is None:
            return
        if data.demod_status == self._last_demod_status:
            return
        self._last_demod_status = data.demod_status
        context.event_bus.publish(
            DemodStatusEvent(
                source_id=f"demod_status_{device_id}",
                device_id=device_id,
                demod_status=data.demod_status,
            )
        )

    def _emit_decoder_messages(
        self, data: SamplesBatch, context: PipelineContext, device_id: str
    ) -> None:
        if not data.decoded_messages:
            return

        context.event_bus.publish(
            DecoderOutputEvent(
                source_id=f"demod_{device_id}",
                device_id=device_id,
                protocol=context.device_context.active_mode,
                messages=data.decoded_messages,
            )
        )

    def _emit_constellation(
        self, data: SamplesBatch, context: PipelineContext, device_id: str
    ) -> None:
        if data.constellation_points is None:
            return
        if not self.constellation_rate_limiter.should_send():
            return

        context.event_bus.publish(
            ConstellationUpdateEvent(
                source_id=f"constellation_{device_id}",
                device_id=device_id,
                points=data.constellation_points,
                modulation=data.constellation_modulation,
            )
        )

    def on_config_change(self, config) -> None:
        if isinstance(config, SDRConfig):
            self.fft_rate_limiter.set_target_fps(config.update_rate_fps)
            self.stats_rate_limiter.set_target_fps(config.update_rate_fps)

    def reset(self) -> None:
        self._last_demod_status = None
