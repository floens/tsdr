from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from tsdr.core import storage
from tsdr.core.demod_spec import DemodSpec

logger = logging.getLogger(__name__)

DEVICES_FILE = "devices.toml"


class PersistedDevice(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: str

    host: str | None = None
    port: int | None = None
    serial: str | None = None
    device_index: int | None = None
    driver: str | None = None
    antenna: str | None = None
    device_args: str | None = None
    path: str | None = None
    sample_format: str | None = None
    signal_freq_offset: float | None = None
    noise_level: float | None = None

    center_frequency: float | None = None
    sample_rate: float | None = None
    rf_gain: float | None = None
    buffer_samples: int | None = None
    target_fps: float | None = None
    network_buffer_seconds: float | None = None
    channel_bandwidth: float | None = None

    audio_spec: DemodSpec | None = None
    squelch_enabled: bool | None = None
    squelch_threshold_db: float | None = None
    squelch_hang_ms: float | None = None


class DeviceStore:
    def __init__(self) -> None:
        self._devices: dict[str, PersistedDevice] = {}
        self._focused_id: str | None = None

    def snapshot(self, devices: list[PersistedDevice], focused: str | None) -> None:
        self._devices = {d.id: d for d in devices}
        self._focused_id = focused if focused in self._devices else None
        self.save()

    def all(self) -> list[PersistedDevice]:
        return list(self._devices.values())

    @property
    def focused_id(self) -> str | None:
        return self._focused_id

    def load(self) -> None:
        data = storage.load_toml(DEVICES_FILE)
        focused = data.get("focused")
        self._focused_id = str(focused) if isinstance(focused, str) else None
        for entry in data.get("device", []):
            try:
                device = PersistedDevice.model_validate(entry)
                self._devices[device.id] = device
            except ValidationError as e:
                logger.warning("device_entry_invalid error=%r", e)
        if self._focused_id and self._focused_id not in self._devices:
            self._focused_id = None

    def save(self) -> None:
        data: dict[str, Any] = {
            "device": [d.model_dump(exclude_none=True) for d in self._devices.values()],
        }
        if self._focused_id:
            data["focused"] = self._focused_id
        storage.save_toml(DEVICES_FILE, data)


_store: DeviceStore | None = None


def get_device_store() -> DeviceStore:
    if _store is None:
        raise RuntimeError("Device store not initialized")
    return _store


def init_device_store() -> DeviceStore:
    global _store
    _store = DeviceStore()
    _store.load()
    return _store
