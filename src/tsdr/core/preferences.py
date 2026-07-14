from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from tsdr.core import storage
from tsdr.core.bandplans import get_bandplan_store
from tsdr.core.demod_spec import PreviousTuneState
from tsdr.core.devices import PersistedDevice, get_device_store
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices import DeviceParams, IQFileParams
from tsdr.devices.registry import BY_NAME
from tsdr.radio.registry import DEMODULATORS

_DEVICE_CONFIG_FIELDS = (
    "tuned_frequency",
    "center_frequency",
    "tuning_mode",
    "sample_rate",
    "rf_gain",
    "buffer_samples",
    "target_fps",
    "network_buffer_seconds",
    "channel_bandwidth",
    "fft_size",
    "fft_window",
    "spectrum_center",
    "spectrum_span",
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
        "tuned_frequency": context.config.tuned_frequency,
        "tuning_mode": context.config.tuning_mode,
        "center_frequency": context.config.center_frequency,
        "sample_rate": context.config.sample_rate,
        "rf_gain": context.config.rf_gain,
        "target_fps": context.config.target_fps,
        "buffer_samples": context.config.buffer_samples,
        "network_buffer_seconds": context.config.network_buffer_seconds,
        "channel_bandwidth": context.config.channel_bandwidth,
        "fft_size": context.config.fft_size,
        "fft_window": context.config.fft_window,
        "spectrum_center": context.config.spectrum_center,
        "spectrum_span": context.config.spectrum_span,
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
    """Reconstruct a device's Params from its persisted row via the registry.

    Every Params field has a default and a matching PersistedDevice field, so the
    non-None persisted values reproduce the per-type constructors (same idiom as
    `_build_device_config`). `sample_format` is stored as a string and coerced
    back to its enum; iq-file additionally requires a path.
    """
    dt = BY_NAME.get(device.type)
    if dt is None:
        logger.info("preferences_restore_skipped device=%s type=%s", device.id, device.type)
        return None
    names = {f.name for f in dataclasses.fields(dt.params_cls)}
    persisted = device.model_dump(exclude_none=True)
    kwargs = {n: persisted[n] for n in names if n in persisted}
    if dt.params_cls is IQFileParams and not kwargs.get("path"):
        logger.warning("preferences_iq_file_missing_path device=%s", device.id)
        return None
    if "sample_format" in kwargs:
        kwargs["sample_format"] = SampleFormat(kwargs["sample_format"])
    return dt.params_cls(**kwargs)


def _build_device_config(device: PersistedDevice) -> DeviceConfig | None:
    persisted = device.model_dump(exclude_none=True)
    kwargs = {k: persisted[k] for k in _DEVICE_CONFIG_FIELDS if k in persisted}
    if "tuned_frequency" not in kwargs and "center_frequency" in kwargs:
        kwargs["tuned_frequency"] = kwargs["center_frequency"]
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
