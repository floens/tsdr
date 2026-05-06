from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.tracing import span
from tsdr.radio.dsp._kernels import apply_freq_shift_c64


class FrequencyShiftStage:
    """Stage that shifts IQ samples in frequency domain.

    Multiplies the input by `exp(j * 2pi * f_shift * t)` so a signal at the
    captured center plus `frequency_offset` ends up at baseband. Phase
    continuity is preserved across batches.
    """

    def __init__(self, frequency_offset: float = 0.0):
        self.frequency_offset = frequency_offset
        self.phase_accumulator = 0.0  # radians

    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        if data.iq_samples is None:
            return data

        if self.frequency_offset == 0.0:
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

        new_center_freq = data.center_frequency - self.frequency_offset

        return data.with_changes(
            iq_samples=iq_shifted,
            center_frequency=new_center_freq,
            stage_name="frequency_shift",
        )

    def on_config_change(self, config) -> None:
        # Frequency offset is set directly via set_offset(); SDRConfig is not used here.
        pass

    def set_offset(self, frequency_offset: float) -> None:
        """Change frequency offset and reset phase to avoid a discontinuity."""
        self.frequency_offset = frequency_offset
        self.phase_accumulator = 0.0

    def reset(self) -> None:
        self.phase_accumulator = 0.0
