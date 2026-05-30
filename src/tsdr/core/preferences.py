from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from tsdr.core import storage
from tsdr.core.audio_spec import PreviousTuneState
from tsdr.core.bandplans import get_bandplan_store
from tsdr.core.devices import PersistedDevice, get_device_store
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices import (
    DeviceParams,
    IQFileParams,
    MockParams,
    RTLSDRParams,
    RTLTCPParams,
    SoapySDRParams,
    SpyServerParams,
)
from tsdr.radio.registry import DEMODULATORS

_DEVICE_CONFIG_FIELDS = (
    "center_frequency",
    "sample_rate",
    "rf_gain",
    "buffer_samples",
    "target_fps",
    "network_buffer_seconds",
    "channel_bandwidth",
)

if TYPE_CHECKING:
    from tsdr.core.sdr.device_context import SDRDeviceContext
    from tsdr.core.sdr.engine import SDREngine
    from tsdr.core.tuning_state import TuningState

logger = logging.getLogger(__name__)

PREFERENCES_FILE = "config.toml"


def load_preferences() -> dict[str, Any]:
    return storage.load_toml(PREFERENCES_FILE)


def save_preferences(data: dict[str, Any]) -> None:
    storage.save_toml(PREFERENCES_FILE, data)


def save_engine_config(engine: SDREngine) -> None:
    prefs = load_preferences()
    prefs["engine"] = {
        "audio_volume": engine.config.audio_volume,
        "denoise": engine.config.denoise,
    }
    save_preferences(prefs)


def restore_engine_config(prefs: dict[str, Any]) -> None:
    engine_prefs = prefs.get("engine", {})
    if "audio_volume" in engine_prefs:
        get_engine().update_global_config(audio_volume=float(engine_prefs["audio_volume"]))
    if "denoise" in engine_prefs:
        get_engine().update_global_config(denoise=bool(engine_prefs["denoise"]))


def _persist_params(params: DeviceParams) -> dict[str, Any]:
    result = dataclasses.asdict(params)
    fmt = result.get("sample_format")
    if isinstance(fmt, SampleFormat):
        result["sample_format"] = fmt.value
    return result


def _build_persisted_device(context: SDRDeviceContext) -> PersistedDevice:
    fields: dict[str, Any] = {
        "id": context.device_id,
        "type": context.device_type,
        **_persist_params(context.params),
        "center_frequency": context.config.center_frequency,
        "sample_rate": context.config.sample_rate,
        "rf_gain": context.config.rf_gain,
        "target_fps": context.config.target_fps,
        "buffer_samples": context.config.buffer_samples,
        "network_buffer_seconds": context.config.network_buffer_seconds,
        "channel_bandwidth": context.config.channel_bandwidth,
    }

    audio_config = context.config.pipelines.get("audio")
    if audio_config and audio_config.audio_spec is not None:
        fields["audio_spec"] = audio_config.audio_spec
        if audio_config.squelch_enabled:
            fields["squelch_enabled"] = True
        if audio_config.squelch_threshold_db != -50.0:
            fields["squelch_threshold_db"] = audio_config.squelch_threshold_db
        if audio_config.squelch_hang_ms != 100.0:
            fields["squelch_hang_ms"] = audio_config.squelch_hang_ms

    return PersistedDevice(**{k: v for k, v in fields.items() if v is not None})


def save_device(engine: SDREngine) -> None:
    """Snapshot every device in the engine plus the focused id."""
    devices = [_build_persisted_device(ctx) for ctx in engine.devices.values()]
    get_device_store().snapshot(devices=devices, focused=engine.focused_device)


def save_bandplan(filename: str | None) -> None:
    """Persist the active bandplan filename, or clear it when None."""
    prefs = load_preferences()
    if filename is None:
        prefs.pop("bandplan", None)
    else:
        prefs["bandplan"] = {"active": filename}
    save_preferences(prefs)


def restore_bandplan(prefs: dict[str, Any]) -> None:
    """Activate the saved bandplan. Silent no-op if file no longer exists."""
    filename = prefs.get("bandplan", {}).get("active")
    if not filename:
        return
    get_bandplan_store().set_active(filename)


def save_tuning_state(state: TuningState) -> None:
    prefs = load_preferences()
    tuning: dict[str, Any] = {}
    if state.step is not None:
        tuning["step"] = float(state.step)
    if state.previous is not None:
        tuning["previous"] = state.previous.model_dump(exclude_none=True)
    if state.current_band_key is not None:
        tuning["current_band_key"] = int(state.current_band_key)
    if tuning:
        prefs["tuning"] = tuning
    else:
        prefs.pop("tuning", None)
    save_preferences(prefs)


def restore_tuning_state(state: TuningState, prefs: dict[str, Any]) -> None:
    tuning = prefs.get("tuning", {})
    if "step" in tuning:
        state.step = float(tuning["step"])
    prev = tuning.get("previous")
    if isinstance(prev, dict):
        state.previous = PreviousTuneState.model_validate(prev)
    if "current_band_key" in tuning:
        state.current_band_key = int(tuning["current_band_key"])


def _build_params(device: PersistedDevice) -> DeviceParams | None:
    if device.type == "rtltcp":
        return RTLTCPParams(host=device.host or "localhost", port=device.port or 1234)
    if device.type == "spyserver":
        return SpyServerParams(host=device.host or "localhost", port=device.port or 5555)
    if device.type == "rtlsdr":
        return RTLSDRParams(serial=device.serial or "", device_index=device.device_index or 0)
    if device.type == "soapy":
        return SoapySDRParams(
            driver=device.driver or "",
            serial=device.serial or "",
            antenna=device.antenna or "",
            device_args=device.device_args or "",
        )
    if device.type == "iq-file":
        if not device.path:
            logger.warning("preferences_iq_file_missing_path device=%s", device.id)
            return None
        fmt = SampleFormat(device.sample_format) if device.sample_format else None
        return IQFileParams(path=device.path, sample_format=fmt)
    if device.type == "mock":
        return MockParams(
            signal_freq_offset=device.signal_freq_offset
            if device.signal_freq_offset is not None
            else 10e3,
            noise_level=device.noise_level if device.noise_level is not None else 0.1,
        )
    logger.info("preferences_restore_skipped device=%s type=%s", device.id, device.type)
    return None


def _build_device_config(device: PersistedDevice) -> DeviceConfig | None:
    persisted = device.model_dump(exclude_none=True)
    kwargs = {k: persisted[k] for k in _DEVICE_CONFIG_FIELDS if k in persisted}
    return DeviceConfig(**kwargs) if kwargs else None


def _restore_audio_pipeline(engine: SDREngine, device: PersistedDevice) -> None:
    spec = device.audio_spec
    if spec is None or spec.mode not in DEMODULATORS:
        return
    try:
        engine.set_audio_demod(device.id, spec)
        logger.info("preferences_demod_restored device=%s mode=%s", device.id, spec.mode)
    except (KeyError, ValueError, OSError) as e:
        logger.warning("preferences_demod_restore_failed device=%s error=%r", device.id, e)
        return

    squelch_kwargs: dict[str, Any] = {}
    if device.squelch_enabled is not None:
        squelch_kwargs["enabled"] = device.squelch_enabled
    if device.squelch_threshold_db is not None:
        squelch_kwargs["threshold_db"] = device.squelch_threshold_db
    if device.squelch_hang_ms is not None:
        squelch_kwargs["hang_ms"] = device.squelch_hang_ms
    if squelch_kwargs:
        try:
            engine.update_squelch(device.id, "audio", **squelch_kwargs)
            logger.info(
                "preferences_squelch_restored device=%s fields=%r", device.id, squelch_kwargs
            )
        except (KeyError, ValueError, OSError) as e:
            logger.warning("preferences_squelch_restore_failed device=%s error=%r", device.id, e)


def restore_devices() -> None:
    """Restore every persisted device into the engine in STOPPED state, then apply focus."""
    store = get_device_store()
    engine = get_engine()

    for device in store.all():
        params = _build_params(device)
        if params is None:
            continue
        config = _build_device_config(device)
        try:
            engine.add_device(device.id, device.type, params, config)
            logger.info("preferences_device_restored device=%s type=%s", device.id, device.type)
        except (OSError, ConnectionError, ValueError) as e:
            logger.warning("preferences_device_restore_failed device=%s error=%r", device.id, e)
            continue
        _restore_audio_pipeline(engine, device)

    if store.focused_id and store.focused_id in engine.devices:
        engine.set_focused_device(store.focused_id)
