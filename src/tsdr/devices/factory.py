from pathlib import Path

from tsdr.devices.base import DeviceParams, SDRDevice
from tsdr.devices.iq_file import EXTENSION_FORMAT_MAP, IQFileDevice, IQFileParams
from tsdr.devices.kiwisdr import KiwiSDRDevice, KiwiSDRParams
from tsdr.devices.mock import MockParams, MockSDRDevice
from tsdr.devices.rtlsdr import RTLSDRDevice, RTLSDRParams
from tsdr.devices.rtltcp import RTLTCPDevice, RTLTCPParams
from tsdr.devices.soapy import SoapySDRDevice, SoapySDRParams
from tsdr.devices.spyserver import SpyServerDevice, SpyServerParams


def create_device(params: DeviceParams) -> SDRDevice:
    """Factory function to create SDR devices from typed params.

    Args:
        params: Device-specific parameters (RTLTCPParams, MockParams, etc.)

    Returns:
        SDRDevice instance

    Raises:
        ValueError: If params type is unknown

    Examples:
        >>> device = create_device(RTLTCPParams(host="localhost", port=1234))
        >>> device = create_device(MockParams(signal_freq_offset=10e3))
    """
    match params:
        case RTLTCPParams(host=host, port=port):
            if not (1 <= port <= 65535):
                raise ValueError(f"Port must be between 1 and 65535, got {port}")
            return RTLTCPDevice(host=host, port=port)

        case MockParams(signal_freq_offset=offset, noise_level=noise):
            return MockSDRDevice(signal_freq_offset=offset, noise_level=noise)

        case IQFileParams(path=path_str, sample_format=fmt):
            path = Path(path_str)
            if fmt is None:
                # Strip compression extension to find the format extension
                name = path.name
                if name.endswith(".zst") or name.endswith(".gz"):
                    format_ext = Path(name.rsplit(".", 1)[0]).suffix.lower()
                else:
                    format_ext = path.suffix.lower()
                fmt = EXTENSION_FORMAT_MAP.get(format_ext)
                if fmt is None:
                    raise ValueError(
                        f"Cannot auto-detect format for extension '{format_ext}'. "
                        f"Use --format cu8|cf32"
                    )
            return IQFileDevice(path=path, sample_format=fmt)

        case SoapySDRParams(driver=driver, serial=serial, antenna=antenna, device_args=device_args):
            return SoapySDRDevice(
                driver=driver, serial=serial, antenna=antenna, device_args=device_args
            )

        case RTLSDRParams(serial=serial, device_index=device_index):
            return RTLSDRDevice(serial=serial, device_index=device_index)

        case SpyServerParams(host=host, port=port):
            if not (1 <= port <= 65535):
                raise ValueError(f"Port must be between 1 and 65535, got {port}")
            return SpyServerDevice(host=host, port=port)

        case KiwiSDRParams(host=host, port=port, password=password, user=user):
            if not (1 <= port <= 65535):
                raise ValueError(f"Port must be between 1 and 65535, got {port}")
            return KiwiSDRDevice(host=host, port=port, password=password, user=user)

        case _:
            raise ValueError(f"Unknown device params type: {type(params).__name__}")
