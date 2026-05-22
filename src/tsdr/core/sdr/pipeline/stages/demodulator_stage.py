import logging
import queue

from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.tracing import span
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.registry import make_demodulator

logger = logging.getLogger(__name__)


class DemodulatorStage:
    """Unified stage that runs any Demodulator (audio demod or protocol decoder).

    Calls demodulate(), drains audio batches to audio_queue, and puts
    decoded messages on SamplesBatch. Pure data stage, no event publishing.
    """

    def __init__(
        self,
        demodulator: Demodulator,
        mode_name: str = "",
        pipeline_name: str = "",
    ):
        self._demodulator = demodulator
        raw_name = mode_name or demodulator.info().label
        self.mode_name = raw_name.upper().replace(" ", "")
        self._pipeline_name = pipeline_name
        self._last_freq: float | None = None
        self._last_sample_rate: float | None = None

    @property
    def demodulator(self) -> Demodulator:
        return self._demodulator

    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        if data.iq_samples is None:
            return data

        # Pass the currently tuned frequency down so protocol decoders can
        # compare it against network-announced frequencies (e.g. TETRA MCCH).
        if context.device_context is not None:
            self._demodulator.set_tuned_frequency(
                int(context.device_context.config.center_frequency)
            )

        with span("demodulate"):
            self._demodulator.demodulate(data.iq_samples, data.capture_utc_s)

        # Drain audio
        if context.audio_queue is not None:
            for batch in self._demodulator.get_audio():
                try:
                    context.audio_queue.put(batch, block=False)
                except queue.Full:
                    pass

                # Cross-pipeline stereo flag (stats reads device_context)
                if context.device_context is not None:
                    context.device_context.stereo = batch.stereo

        # Put decoded messages and signal info on SamplesBatch
        messages = self._demodulator.get_messages()
        changes: dict = {"stage_name": "demodulator", "signal_info": self._demodulator.info()}
        if messages:
            changes["decoded_messages"] = tuple(messages)

        # Constellation points (gated on device config)
        if (
            context.device_context is not None
            and context.device_context.config.calculate_constellation
        ):
            result = self._demodulator.get_constellation()
            if result is not None:
                points, modulation = result
                changes["constellation_points"] = points
                changes["constellation_modulation"] = modulation

        return data.with_changes(**changes)

    def on_config_change(self, config) -> None:
        if not isinstance(config, DeviceConfig):
            return
        pipeline_config = config.pipelines.get(self._pipeline_name)

        if pipeline_config is not None and pipeline_config.demod_mode:
            new_mode = pipeline_config.demod_mode.upper()
            if new_mode != self.mode_name:
                # Build before assigning so a failed make_demodulator leaves the old demod intact.
                new_demod = make_demodulator(
                    new_mode,
                    config.sample_rate,
                    config.channel_bandwidth,
                    pipeline_config.fm_deviation_hz,
                )
                logger.info("demodulator_swapped old=%s new=%s", self.mode_name, new_mode)
                self._demodulator = new_demod
                self.mode_name = new_mode
                self._last_sample_rate = config.sample_rate
                self._last_freq = config.center_frequency
                self._demodulator.set_squelch(
                    enabled=pipeline_config.squelch_enabled,
                    threshold_db=pipeline_config.squelch_threshold_db,
                    hang_ms=pipeline_config.squelch_hang_ms,
                )
                return

        # Sample-rate first: bandwidth/deviation/squelch hooks below
        # build filters off decimated_rate, which the rebuild updates.
        if config.sample_rate != self._last_sample_rate:
            self._last_sample_rate = config.sample_rate
            logger.debug("demodulator_sample_rate_changed rate=%d", int(config.sample_rate))
            self._demodulator.set_sample_rate(config.sample_rate)
            self._demodulator.reset()
        if config.channel_bandwidth is not None:
            self._demodulator.set_channel_bandwidth(config.channel_bandwidth)
        if config.center_frequency is not None and config.center_frequency != self._last_freq:
            logger.debug("demodulator_reset reason=frequency_changed")
            self._last_freq = config.center_frequency
            self._demodulator.reset()
        if pipeline_config is None:
            return
        if pipeline_config.fm_deviation_hz is not None:
            self._demodulator.set_deviation(pipeline_config.fm_deviation_hz)
        self._demodulator.set_squelch(
            enabled=pipeline_config.squelch_enabled,
            threshold_db=pipeline_config.squelch_threshold_db,
            hang_ms=pipeline_config.squelch_hang_ms,
        )

    def reset(self) -> None:
        self._demodulator.reset()
