from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from tsdr.core import storage
from tsdr.core.bandplans import get_bandplan_store
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.devices import DeviceParams, RTLSDRParams, RTLTCPParams, SoapySDRParams
from tsdr.radio.registry import DEMODULATORS

if TYPE_CHECKING:
    from tsdr.core.sdr.engine import SDREngine
    from tsdr.tui.state import UIState

logger = logging.getLogger(__name__)

PREFERENCES_FILE = "config.toml"


def load_preferences() -> dict[str, Any]:
    return storage.load_toml(PREFERENCES_FILE)


def save_preferences(data: dict[str, Any]) -> None:
    storage.save_toml(PREFERENCES_FILE, data)


def save_ui_state(ui_state: UIState) -> None:
    prefs = load_preferences()
    prefs["ui"] = {
        "zoom": ui_state.zoom,
        "db_min": ui_state.db_min,
        "db_max": ui_state.db_max,
        "image_mode": ui_state.image_mode,
        "active_panel": ui_state.active_panel or "",
    }
    save_preferences(prefs)


def save_engine_config(engine: SDREngine) -> None:
    prefs = load_preferences()
    prefs["engine"] = {
        "audio_volume": engine.config.audio_volume,
    }
    save_preferences(prefs)


def restore_engine_config(prefs: dict[str, Any]) -> None:
    engine_prefs = prefs.get("engine", {})
    if "audio_volume" in engine_prefs:
        get_engine().update_global_config(audio_volume=float(engine_prefs["audio_volume"]))


def save_device(engine: SDREngine) -> None:
    """Save first device state by inspecting the engine."""
    if not engine.devices:
        return

    context = next(iter(engine.devices.values()))

    device_dict: dict[str, Any] = {
        "id": context.device_id,
        "type": context.device_type,
        **dataclasses.asdict(context.params),
        "frequency": context.config.center_frequency / 1e6,
        "sample_rate": context.config.sample_rate,
        "rf_gain": context.config.rf_gain,
        "target_fps": context.config.target_fps,
        "buffer_samples": context.config.buffer_samples,
        "network_buffer_seconds": context.config.network_buffer_seconds,
        "channel_bandwidth": context.config.channel_bandwidth,
        "running": context.state == DeviceState.RUNNING,
    }

    audio_config = context.config.pipelines.get("audio")
    if audio_config and audio_config.demod_mode:
        device_dict["demod_mode"] = audio_config.demod_mode
        if audio_config.frequency_offset != 0.0:
            device_dict["demod_offset"] = audio_config.frequency_offset
        if audio_config.squelch_enabled:
            device_dict["squelch_enabled"] = True
        if audio_config.squelch_threshold_db != -50.0:
            device_dict["squelch_threshold_db"] = audio_config.squelch_threshold_db
        if audio_config.squelch_hang_ms != 100.0:
            device_dict["squelch_hang_ms"] = audio_config.squelch_hang_ms
        if audio_config.fm_deviation_hz is not None:
            device_dict["fm_deviation_hz"] = audio_config.fm_deviation_hz

    prefs = load_preferences()
    prefs["device"] = {k: v for k, v in device_dict.items() if v is not None}
    save_preferences(prefs)


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


def restore_ui_state(ui_state: UIState, prefs: dict[str, Any]) -> None:
    ui = prefs.get("ui", {})
    if "zoom" in ui:
        ui_state.zoom = float(ui["zoom"])
    if "db_min" in ui:
        ui_state.db_min = float(ui["db_min"])
    if "db_max" in ui:
        ui_state.db_max = float(ui["db_max"])
    if "image_mode" in ui:
        ui_state.image_mode = bool(ui["image_mode"])
    if ui.get("active_panel"):
        ui_state.active_panel = str(ui["active_panel"])


def restore_device(prefs: dict[str, Any]) -> None:
    """Restore a saved device using engine methods directly."""
    dev = prefs.get("device")
    if not dev:
        return

    engine = get_engine()
    device_id = dev.get("id", "rtl0")
    device_type = dev.get("type", "rtltcp")

    # Build typed params
    params: DeviceParams
    if device_type == "rtltcp":
        params = RTLTCPParams(
            host=dev.get("host", "localhost"),
            port=int(dev.get("port", 1234)),
        )
    elif device_type == "rtlsdr":
        params = RTLSDRParams(
            serial=dev.get("serial", ""),
            device_index=int(dev.get("device_index", 0)),
        )
    elif device_type == "soapy":
        params = SoapySDRParams(
            driver=dev.get("driver", ""),
            serial=dev.get("serial", ""),
            antenna=dev.get("antenna", ""),
            device_args=dev.get("device_args", ""),
        )
    else:
        logger.info("Skipping restore for device type %s", device_type)
        return

    # Build config from saved values
    config_kwargs: dict[str, Any] = {}
    if "frequency" in dev:
        config_kwargs["center_frequency"] = float(dev["frequency"]) * 1e6
    if "sample_rate" in dev:
        config_kwargs["sample_rate"] = float(dev["sample_rate"])
    if "rf_gain" in dev:
        config_kwargs["rf_gain"] = float(dev["rf_gain"])
    if "buffer_samples" in dev:
        config_kwargs["buffer_samples"] = int(dev["buffer_samples"])
    if "target_fps" in dev:
        config_kwargs["target_fps"] = float(dev["target_fps"])
    if "network_buffer_seconds" in dev:
        config_kwargs["network_buffer_seconds"] = float(dev["network_buffer_seconds"])
    if "channel_bandwidth" in dev:
        config_kwargs["channel_bandwidth"] = float(dev["channel_bandwidth"])

    config = DeviceConfig(**config_kwargs) if config_kwargs else None

    should_run = bool(dev.get("running", False))

    try:
        engine.add_device(device_id, device_type, params, config)
        if should_run:
            engine.start_device(device_id)
        logger.info("Restored device %s (%s, running=%s)", device_id, device_type, should_run)
    except (OSError, ConnectionError, ValueError) as e:
        logger.warning("Failed to restore device %s: %s", device_id, e)
        return

    demod_mode = dev.get("demod_mode")
    if not demod_mode or demod_mode not in DEMODULATORS:
        return

    try:
        demod_offset = float(dev.get("demod_offset", 0.0))
        fm_deviation_hz = float(dev["fm_deviation_hz"]) if "fm_deviation_hz" in dev else None
        engine.set_audio_demod(device_id, demod_mode, demod_offset, fm_deviation_hz)
        logger.info("Restored demod %s for %s", demod_mode, device_id)
    except (KeyError, ValueError, OSError) as e:
        logger.warning("Failed to restore demod for %s: %s", device_id, e)
        return

    squelch_kwargs: dict[str, Any] = {}
    if "squelch_enabled" in dev:
        squelch_kwargs["enabled"] = bool(dev["squelch_enabled"])
    if "squelch_threshold_db" in dev:
        squelch_kwargs["threshold_db"] = float(dev["squelch_threshold_db"])
    if "squelch_hang_ms" in dev:
        squelch_kwargs["hang_ms"] = float(dev["squelch_hang_ms"])
    if squelch_kwargs:
        try:
            engine.update_squelch(device_id, "audio", **squelch_kwargs)
            logger.info("Restored squelch for %s: %s", device_id, squelch_kwargs)
        except (KeyError, ValueError, OSError) as e:
            logger.warning("Failed to restore squelch for %s: %s", device_id, e)
