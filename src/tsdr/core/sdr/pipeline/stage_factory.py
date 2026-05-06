import logging

from tsdr.core.sdr.config import DeviceConfig, PipelineConfig, SDRConfig, StageType
from tsdr.core.sdr.pipeline.stage import PipelineStage
from tsdr.core.sdr.pipeline.stages import (
    AGCStage,
    DemodulatorStage,
    EventEmitterStage,
    FFTStage,
    FrequencyShiftStage,
    RecordStage,
)
from tsdr.radio.registry import DEMODULATORS

logger = logging.getLogger(__name__)

_CLASS_TO_STAGE_TYPE: dict[type, StageType] = {
    AGCStage: StageType.AGC,
    FFTStage: StageType.FFT,
    EventEmitterStage: StageType.EVENT_EMITTER,
    DemodulatorStage: StageType.DEMODULATOR,
    FrequencyShiftStage: StageType.FREQUENCY_SHIFT,
    RecordStage: StageType.RECORD,
}


def stage_type_of(stage: PipelineStage) -> StageType:
    """Map a stage instance to its StageType enum value."""
    stage_type = _CLASS_TO_STAGE_TYPE.get(type(stage))
    if stage_type is None:
        raise ValueError(f"Unknown stage class: {type(stage).__name__}")
    return stage_type


def _make_demodulator(
    mode: str,
    sample_rate: float,
    channel_bandwidth: float | None = None,
    fm_deviation_hz: float | None = None,
):
    mode = mode.upper()
    if mode not in DEMODULATORS:
        raise ValueError(f"Unknown demodulator mode: {mode}")
    kw: dict = {}
    if channel_bandwidth is not None:
        kw["channel_bandwidth"] = channel_bandwidth
    if mode == "NFM" and fm_deviation_hz is not None:
        kw["deviation"] = fm_deviation_hz
    return DEMODULATORS[mode](sample_rate=sample_rate, **kw)


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
            return FrequencyShiftStage(frequency_offset=pipeline_config.frequency_offset)
        case StageType.DEMODULATOR:
            if not pipeline_config.demod_mode:
                raise ValueError("DEMODULATOR stage requires demod_mode in PipelineConfig")
            demodulator = _make_demodulator(
                pipeline_config.demod_mode,
                device_config.sample_rate,
                device_config.channel_bandwidth,
                pipeline_config.fm_deviation_hz,
            )
            demodulator.set_squelch(
                enabled=pipeline_config.squelch_enabled,
                threshold_db=pipeline_config.squelch_threshold_db,
                hang_ms=pipeline_config.squelch_hang_ms,
            )
            return DemodulatorStage(
                demodulator=demodulator,
                mode_name=pipeline_config.demod_mode,
                pipeline_name=pipeline_name,
            )
        case StageType.RECORD:
            if not pipeline_config.record_path:
                raise ValueError("RECORD stage requires record_path in PipelineConfig")
            return RecordStage(
                output_path=pipeline_config.record_path,
                resample=pipeline_config.record_resample,
                max_samples=pipeline_config.record_max_samples,
            )
