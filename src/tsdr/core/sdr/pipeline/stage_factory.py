import logging

from tsdr.core.sdr.config import DeviceConfig, PipelineConfig, SDRConfig, StageType
from tsdr.core.sdr.pipeline.stage import PipelineStage
from tsdr.core.sdr.pipeline.stages import (
    AGCStage,
    DemodulatorStage,
    DenoiserStage,
    EventEmitterStage,
    FFTStage,
    FrequencyShiftStage,
    RecordStage,
)
from tsdr.radio.registry import make_demodulator

logger = logging.getLogger(__name__)

_CLASS_TO_STAGE_TYPE: dict[type, StageType] = {
    AGCStage: StageType.AGC,
    FFTStage: StageType.FFT,
    EventEmitterStage: StageType.EVENT_EMITTER,
    DemodulatorStage: StageType.DEMODULATOR,
    DenoiserStage: StageType.DENOISER,
    FrequencyShiftStage: StageType.FREQUENCY_SHIFT,
    RecordStage: StageType.RECORD,
}


def stage_type_of(stage: PipelineStage) -> StageType:
    """Map a stage instance to its StageType enum value."""
    stage_type = _CLASS_TO_STAGE_TYPE.get(type(stage))
    if stage_type is None:
        raise ValueError(f"Unknown stage class: {type(stage).__name__}")
    return stage_type


def create_stage(
    stage_type: StageType,
    pipeline_config: PipelineConfig,
    sdr_config: SDRConfig,
    device_config: DeviceConfig,
    device_id: str,
    pipeline_name: str,
) -> PipelineStage:
    """Create a pipeline stage instance from its type and config."""
    match stage_type:
        case StageType.AGC:
            return AGCStage()
        case StageType.FFT:
            return FFTStage(config=sdr_config)
        case StageType.EVENT_EMITTER:
            return EventEmitterStage(config=sdr_config)
        case StageType.FREQUENCY_SHIFT:
            offset = (
                pipeline_config.audio_spec.frequency_offset
                if pipeline_config.audio_spec is not None
                else 0.0
            )
            return FrequencyShiftStage(frequency_offset=offset)
        case StageType.DEMODULATOR:
            spec = pipeline_config.audio_spec
            if spec is None:
                raise ValueError("DEMODULATOR stage requires audio_spec in PipelineConfig")
            demodulator = make_demodulator(
                spec, device_config.sample_rate, device_config.channel_bandwidth
            )
            demodulator.set_squelch(
                enabled=pipeline_config.squelch_enabled,
                threshold_db=pipeline_config.squelch_threshold_db,
                hang_ms=pipeline_config.squelch_hang_ms,
            )
            return DemodulatorStage(
                demodulator=demodulator,
                mode_name=spec.mode,
                pipeline_name=pipeline_name,
            )
        case StageType.DENOISER:
            return DenoiserStage(enabled=sdr_config.denoise)
        case StageType.RECORD:
            if not pipeline_config.record_path:
                raise ValueError("RECORD stage requires record_path in PipelineConfig")
            return RecordStage(
                output_path=pipeline_config.record_path,
                resample=pipeline_config.record_resample,
                max_samples=pipeline_config.record_max_samples,
                sample_format=pipeline_config.record_sample_format,
            )
