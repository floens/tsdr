"""RTL-SDR device via the pyrtlsdr Python binding.

pyrtlsdr is installed as an optional extra (`pip install tsdr[rtlsdr]`) and
wraps librtlsdr via ctypes. Importing it can fail for several reasons:
the extra isn't installed (ImportError), librtlsdr isn't on the loader
path (ImportError raised by the pyrtlsdr loader), or the installed
librtlsdr is a different ABI than the one pyrtlsdr was built against
(AttributeError when a newer symbol is looked up). All three are treated
as "not available" so tsdr still starts.
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices.base import DeviceParams

logger = logging.getLogger(__name__)


@contextmanager
def _silence_stderr() -> Any:
    """Redirect fd 2 to /dev/null for the duration of the block.

    librtlsdr writes tuner-detection and error messages via C's fprintf(stderr),
    which corrupts the TUI. Python logging is file-backed so nothing Python-side
    is lost during the window.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def import_rtlsdr() -> Any:
    """Try to import pyrtlsdr. Returns the module, or None if unavailable."""
    try:
        import rtlsdr  # noqa: PLC0415

        return rtlsdr
    except (ImportError, OSError, AttributeError) as e:
        logger.debug("pyrtlsdr not available: %s", e)
        return None


_rtlsdr = import_rtlsdr()
_HAS_RTLSDR = _rtlsdr is not None


# Tuner type enum from rtl-sdr.h (rtlsdr_tuner)
_TUNER_TYPES = {
    0: "Unknown",
    1: "E4000",
    2: "FC0012",
    3: "FC0013",
    4: "FC2580",
    5: "R820T",
    6: "R828D",
}


@dataclass(frozen=True)
class RTLSDRParams(DeviceParams):
    """Native RTL-SDR (pyrtlsdr) device parameters."""

    serial: str = ""
    device_index: int = 0


class RTLSDRDevice:
    """RTL-SDR device via the pyrtlsdr library (direct USB access).

    Requires the `rtlsdr` extra (`pip install tsdr[rtlsdr]`) and librtlsdr
    on the loader path. For remote/networked RTL-SDR use `RTLTCPDevice`
    instead.
    """

    def __init__(self, serial: str = "", device_index: int = 0):
        self._serial = serial
        self._device_index = device_index
        self._device: Any = None
        self._is_open = False
        self._supports_bias_tee = False

    def open(self) -> None:
        if not _HAS_RTLSDR:
            raise DeviceError(
                "pyrtlsdr not available. "
                "Install with `pip install tsdr[rtlsdr]` and ensure librtlsdr "
                "is on the library path."
            )

        try:
            with _silence_stderr():
                if self._serial:
                    self._device = _rtlsdr.RtlSdr(serial_number=self._serial)
                else:
                    self._device = _rtlsdr.RtlSdr(device_index=self._device_index)
        except OSError as e:
            raise DeviceError(f"Failed to open RTL-SDR device: {e}")

        self._supports_bias_tee = hasattr(self._device, "set_bias_tee")

        try:
            with _silence_stderr():
                tuner_type = self._device.get_tuner_type()
            tuner_name = _TUNER_TYPES.get(tuner_type, f"Unknown ({tuner_type})")
        except OSError as e:
            logger.debug("RTL-SDR get_tuner_type failed: %s", e)
            tuner_name = "Unknown"

        self._is_open = True
        logger.info(
            "RTL-SDR opened: tuner=%s, serial=%s, index=%d",
            tuner_name,
            self._serial or "(any)",
            self._device_index,
        )

    def close(self) -> None:
        if self._device is not None:
            try:
                with _silence_stderr():
                    self._device.close()
            except OSError as e:
                logger.debug("Error closing RTL-SDR: %s", e)
            self._device = None
        self._is_open = False

    def read_samples(self, count: int) -> bytes:
        if self._device is None:
            raise DeviceError("Device not open")

        # librtlsdr's sync USB read passes count straight to libusb_bulk_transfer.
        # The USB 2.0 HS bulk endpoint max packet size is 512 bytes; unaligned
        # counts cause LIBUSB_ERROR_OVERFLOW and corrupt the driver state.
        aligned = max(512, (count // 512) * 512)

        try:
            # pyrtlsdr reuses its internal buffer across calls, so convert
            # to bytes immediately to get a stable copy.
            buf = self._device.read_bytes(aligned)
            return bytes(buf)
        except OSError as e:
            raise DeviceError(f"RTL-SDR read failed: {e}")

    def _on_setter_failure(self, what: str, exc: OSError) -> None:
        # pyrtlsdr's tuning setters call self.close() on error before raising,
        # so by the time we catch it the dongle is already shut. Sync our state.
        self._device = None
        self._is_open = False
        raise DeviceError(f"RTL-SDR {what} failed: {exc}")

    def set_frequency(self, freq: float) -> None:
        if self._device is None:
            return
        try:
            with _silence_stderr():
                self._device.set_center_freq(freq)
        except OSError as e:
            self._on_setter_failure("set_frequency", e)

    def set_sample_rate(self, rate: float) -> None:
        if self._device is None:
            return
        try:
            with _silence_stderr():
                self._device.set_sample_rate(rate)
        except OSError as e:
            self._on_setter_failure("set_sample_rate", e)

    def set_gain(self, gain: float) -> None:
        if self._device is None:
            return
        try:
            with _silence_stderr():
                self._device.gain = gain
        except OSError as e:
            self._on_setter_failure("set_gain", e)

    def set_auto_gain(self, enable: bool) -> None:
        if self._device is None:
            return
        try:
            with _silence_stderr():
                self._device.set_manual_gain_enabled(not enable)
        except OSError as e:
            self._on_setter_failure("set_auto_gain", e)

    @property
    def gain_range(self) -> tuple[float, float]:
        # librtlsdr only ships the R820T tuner in practice; pyrtlsdr exposes
        # the same dB range (0.0 to 49.6) and clamps internally.
        return (0.0, 49.6)

    @property
    def supports_bias_tee(self) -> bool:
        return self._supports_bias_tee

    def set_bias_tee(self, enable: bool) -> None:
        if self._device is not None and self._supports_bias_tee:
            with _silence_stderr():
                self._device.set_bias_tee(enable)
            logger.debug("RTL-SDR: bias-T %s", "on" if enable else "off")

    def get_sample_format(self) -> SampleFormat:
        return SampleFormat.UINT8_IQ

    def __str__(self) -> str:
        status = "open" if self._is_open else "closed"
        return (
            f"RTLSDRDevice(serial={self._serial or '(any)'}, index={self._device_index}, {status})"
        )
