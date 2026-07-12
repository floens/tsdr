"""SDR device implementations and factory.

Public API:
    - SDRDevice protocol and DeviceParams base class
    - Per-device Params + Device pairs (RTLTCP, Mock, IQFile, SoapySDR)
    - create_device() factory
    - EXTENSION_FORMAT_MAP for file-extension sample-format detection
"""

from tsdr.devices.base import (
    DeviceCapabilities,
    DeviceIdentity,
    DeviceParams,
    NetworkDeviceParams,
    SDRDevice,
)
from tsdr.devices.factory import create_device
from tsdr.devices.iq_file import EXTENSION_FORMAT_MAP, IQFileDevice, IQFileParams
from tsdr.devices.kiwisdr import KiwiSDRDevice, KiwiSDRParams
from tsdr.devices.mock import MockParams, MockSDRDevice
from tsdr.devices.rtlsdr import RTLSDRDevice, RTLSDRParams
from tsdr.devices.rtltcp import RTLTCPDevice, RTLTCPParams
from tsdr.devices.soapy import SoapySDRDevice, SoapySDRParams
from tsdr.devices.spyserver import SpyServerDevice, SpyServerParams

__all__ = [
    "EXTENSION_FORMAT_MAP",
    "DeviceCapabilities",
    "DeviceIdentity",
    "DeviceParams",
    "IQFileDevice",
    "IQFileParams",
    "KiwiSDRDevice",
    "KiwiSDRParams",
    "MockParams",
    "MockSDRDevice",
    "NetworkDeviceParams",
    "RTLSDRDevice",
    "RTLSDRParams",
    "RTLTCPDevice",
    "RTLTCPParams",
    "SDRDevice",
    "SoapySDRDevice",
    "SoapySDRParams",
    "SpyServerDevice",
    "SpyServerParams",
    "create_device",
]
