"""Engine-facing operations for directory receivers: map a listed receiver to an
engine device, and add/remove it. Kept in core so both the `directory` command and
the directory widget drive the engine through one path.
"""

from __future__ import annotations

from dataclasses import dataclass

from tsdr.core.directory.favorites import FavoriteDevice
from tsdr.core.directory.model import PublicDevice, Source
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.devices import DeviceParams, KiwiSDRParams, SpyServerParams
from tsdr.devices.base import NetworkDeviceParams

_DEFAULT_PORT: dict[Source, int] = {"spyserver": 5555, "kiwisdr": 8073}
_ID_PREFIX: dict[Source, str] = {"spyserver": "spy", "kiwisdr": "kiwi"}


@dataclass(frozen=True)
class ConnectResult:
    ok: bool
    message: str


def device_endpoint(device: PublicDevice | FavoriteDevice) -> tuple[str, int]:
    return device.host, device.port if device.port is not None else _DEFAULT_PORT[device.source]


def _params_for(source: Source, host: str, port: int) -> DeviceParams:
    if source == "kiwisdr":
        return KiwiSDRParams(host=host, port=port)
    return SpyServerParams(host=host, port=port)


def find_added_device(device: PublicDevice | FavoriteDevice) -> str | None:
    host, port = device_endpoint(device)
    for did, context in get_engine().devices.items():
        params = context.params
        if isinstance(params, NetworkDeviceParams) and params.host == host and params.port == port:
            return did
    return None


def added_endpoints() -> set[tuple[str, int]]:
    return {
        (params.host, params.port)
        for context in get_engine().devices.values()
        if isinstance(params := context.params, NetworkDeviceParams)
    }


def add_directory_device(device: PublicDevice) -> ConnectResult:
    existing = find_added_device(device)
    if existing is not None:
        engine = get_engine()
        starting = engine.devices[existing].state != DeviceState.RUNNING
        if starting:
            engine.start_device(existing)
        engine.set_focused_device(existing)
        return ConnectResult(True, f"{'Started' if starting else 'Focused'} {device.name}")
    if not device.usable:
        return ConnectResult(False, f"{device.name} is not usable ({device.usable_reason})")

    host, port = device_endpoint(device)
    did = _default_device_id(device)
    _stop_running_devices()
    engine = get_engine()
    engine.add_device(
        did,
        device.source,
        _params_for(device.source, host, port),
        DeviceConfig(
            tuned_frequency=_connect_freq(device),
            center_frequency=_connect_freq(device),
        ),
    )
    engine.set_focused_device(did)
    engine.start_device(did)
    return ConnectResult(True, f"Added {device.name} ({host}:{port})")


def start_directory_device(device: PublicDevice | FavoriteDevice) -> ConnectResult:
    """Focus an already-added receiver, retune it to the frequency the user is on,
    and start it (stopping any other running device so it takes audio)."""
    engine = get_engine()
    did = find_added_device(device)
    if did is None:
        return ConnectResult(False, f"{device.name} not added")
    freq = _connect_freq(device)  # read before refocus, while it reflects the active device
    _stop_running_devices()
    engine.update_device_config(did, tuned_frequency=freq)
    engine.set_focused_device(did)
    engine.start_device(did)
    return ConnectResult(True, f"Started {device.name}")


def remove_directory_device(device: PublicDevice | FavoriteDevice) -> ConnectResult:
    """Stop and remove the engine device serving this receiver, if any."""
    engine = get_engine()
    did = find_added_device(device)
    if did is None:
        return ConnectResult(True, f"{device.name} not added")
    if engine.devices[did].state == DeviceState.RUNNING:
        engine.stop_device(did)
    engine.remove_device(did)
    return ConnectResult(True, f"Removed {device.name} ({device.host})")


def _default_device_id(device: PublicDevice | FavoriteDevice) -> str:
    host, port = device_endpoint(device)
    return f"{_ID_PREFIX[device.source]}-{host.replace('.', '-')}-{port}"


def _default_freq(device: PublicDevice | FavoriteDevice) -> float:
    lo, hi = device.freq_min, device.freq_max
    if lo is not None and hi is not None and hi > lo:
        return float(lo + (hi - lo) * 0.3)
    return 100e6


def _clip_to_band(freq: float, lo: float | None, hi: float | None) -> float:
    """Clip a frequency into [lo, hi] when the band is known, else pass through."""
    if lo is not None and hi is not None:
        return min(max(freq, lo), hi)
    return freq


def _connect_freq(device: PublicDevice | FavoriteDevice) -> float:
    """The frequency the user is already tuned to, clipped into this receiver's
    band. Falls back to a default when nothing is tuned yet."""
    focused = get_engine().get_focused_device()
    if focused is None:
        return _default_freq(device)
    return _clip_to_band(focused.config.tuned_frequency, device.freq_min, device.freq_max)


def _stop_running_devices() -> None:
    engine = get_engine()
    for dev_id, context in list(engine.devices.items()):
        if context.state == DeviceState.RUNNING:
            engine.stop_device(dev_id)
