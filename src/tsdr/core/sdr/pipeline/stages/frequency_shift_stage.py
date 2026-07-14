from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.tracing import span
from tsdr.radio.dsp._kernels import apply_freq_shift_c64


class FrequencyShiftStage:
    """Shifts the tuned channel to baseband for the demodulator.

    The offset is derived per batch from the gap between the batch's capture
    center and the device's tuned frequency, so a hardware retune that lags
    the dial is compensated automatically. Phase continuity is preserved
    across batches; an offset change resets the phase accumulator.
    """

    def __init__(self) -> None:
        self.frequency_offset = 0.0
        self.phase_accumulator = 0.0  # radians

    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        if data.iq_samples is None or context.device_context is None:
            return data

        tuned = context.device_context.config.tuned_frequency
        offset = data.center_frequency - tuned
        # Tuned signal not in this capture (retune transient, unvalidated
        # tune on a range-less device): pass unshifted rather than alias.
        if abs(offset) >= data.sample_rate / 2:
            return data

        if offset != self.frequency_offset:
            self.set_offset(offset)
        if offset == 0.0:
            return data

        with span("freq_shift"):
            # Kernel convention: positive offset_hz shifts spectrum down (signal at +f → baseband).
            # This stage's convention is the opposite (positive frequency_offset shifts up), so negate.
            iq_shifted, self.phase_accumulator = apply_freq_shift_c64(
                data.iq_samples,
                -self.frequency_offset,
                data.sample_rate,
                self.phase_accumulator,
            )

        return data.with_changes(
            iq_samples=iq_shifted,
            center_frequency=tuned,
            stage_name="frequency_shift",
        )

    def on_config_change(self, config) -> None:
        # Offset is derived per batch in process(); config carries nothing extra.
        pass

    def set_offset(self, frequency_offset: float) -> None:
        """Change frequency offset and reset phase to avoid a discontinuity."""
        self.frequency_offset = frequency_offset
        self.phase_accumulator = 0.0

    def reset(self) -> None:
        self.phase_accumulator = 0.0
