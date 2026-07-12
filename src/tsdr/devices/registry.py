from dataclasses import dataclass

from tsdr.devices.base import DeviceParams
from tsdr.devices.iq_file import IQFileParams
from tsdr.devices.kiwisdr import KiwiSDRParams
from tsdr.devices.mock import MockParams
from tsdr.devices.rtlsdr import RTLSDRParams
from tsdr.devices.rtltcp import RTLTCPParams
from tsdr.devices.soapy import SoapySDRParams
from tsdr.devices.spyserver import SpyServerParams


@dataclass(frozen=True)
class DeviceType:
    """One device kind: maps the ``--type`` / persisted ``type`` string to its
    Params class and add-time defaults. The single source of truth for device
    dispatch -- adding a device means adding one entry to ``DEVICE_TYPES``
    (mirrors ``radio/registry.py``)."""

    name: str
    params_cls: type[DeviceParams]
    default_port: int | None = None
    default_frequency_hz: float | None = None


DEVICE_TYPES: tuple[DeviceType, ...] = (
    DeviceType("rtltcp", RTLTCPParams, default_port=1234),
    DeviceType("rtlsdr", RTLSDRParams),
    DeviceType("mock", MockParams),
    DeviceType("iq-file", IQFileParams),
    DeviceType("soapy", SoapySDRParams),
    DeviceType("spyserver", SpyServerParams, default_port=5555),
    DeviceType("kiwisdr", KiwiSDRParams, default_port=8073, default_frequency_hz=10e6),
)

BY_NAME: dict[str, DeviceType] = {d.name: d for d in DEVICE_TYPES}
DEVICE_TYPE_NAMES: tuple[str, ...] = tuple(d.name for d in DEVICE_TYPES)
