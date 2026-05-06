"""Client-side RF Gain AGC stage.

Monitors raw IQ levels and adjusts RF gain to keep the ADC in its optimal range.
Uses threshold-based algorithm: step gain down on clipping, step gain up on weak signal.
"""

import time

import numpy as np

from tsdr.core.events.events import AGCGainChangeEvent
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.samples_batch import SamplesBatch


class AGCStage:
    """RF Gain AGC stage that adjusts tuner gain based on IQ signal levels.

    Operates on raw IQ samples before any other processing. When clipping is
    detected, gain is stepped down. When signal is too weak, gain is stepped up.
    A cooldown prevents oscillation.

    The step is applied in dB and clamped to the device's `gain_range`. Hardware
    quantization (e.g. R820T's discrete gain table) is handled by the device's
    `set_gain` implementation, not here.
    """

    def __init__(
        self,
        clip_threshold: float = 2.0,
        low_rms_threshold: float = 0.05,
        cooldown_s: float = 0.5,
        gain_step_db: float = 2.5,
    ):
        self.clip_threshold = clip_threshold
        self.low_rms_threshold = low_rms_threshold
        self.cooldown_s = cooldown_s
        self.gain_step_db = gain_step_db

        self._current_gain_db = 29.7  # default; overwritten on first config sync
        self._last_adjust_time = 0.0

    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        config = context.device_context.config
        if not config.enable_agc or config.auto_gain:
            # AGC disabled or hardware auto-gain active - sync from config
            self._current_gain_db = config.rf_gain
            return data

        min_db, max_db = context.device_context.device.gain_range
        if min_db == max_db:
            # Device has no controllable gain (file playback, mock).
            return data

        if data.iq_samples is None or len(data.iq_samples) == 0:
            return data

        now = time.monotonic()
        if now - self._last_adjust_time < self.cooldown_s:
            return data

        iq = data.iq_samples
        clipped = (np.abs(iq.real) >= 0.99) | (np.abs(iq.imag) >= 0.99)
        clip_pct = 100.0 * float(np.mean(clipped))

        new_gain_db = self._current_gain_db

        if clip_pct > self.clip_threshold:
            new_gain_db = max(min_db, self._current_gain_db - self.gain_step_db)
        elif clip_pct == 0.0:
            mag = np.abs(iq)
            rms = float(np.sqrt(np.mean(mag**2)))
            if rms < self.low_rms_threshold:
                new_gain_db = min(max_db, self._current_gain_db + self.gain_step_db)

        if new_gain_db != self._current_gain_db:
            self._current_gain_db = new_gain_db
            self._last_adjust_time = now

            context.event_bus.publish(
                AGCGainChangeEvent(
                    source_id=f"agc_{context.device_context.device_id}",
                    device_id=context.device_context.device_id,
                    rf_gain=new_gain_db,
                )
            )

        return data

    def on_config_change(self, config) -> None:
        if isinstance(config, DeviceConfig):
            self._current_gain_db = config.rf_gain

    def reset(self) -> None:
        self._last_adjust_time = 0.0
