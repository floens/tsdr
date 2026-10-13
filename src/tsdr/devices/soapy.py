"""SoapySDR device wrapper and SoapySDR module loader.

SoapySDR Python bindings are installed as part of the system C++ library
(e.g. `brew install soapysdr`), not via pip. Isolated venvs created by
pipx / uv won't see system site-packages, so we probe well-known locations
when the regular import fails.
"""

import ctypes.util
import logging
import platform
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from numba.core.errors import NumbaError

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices.base import DeviceCapabilities, DeviceIdentity, DeviceParams
from tsdr.devices.soapy_reader import (
    NativeStreamReader,
    StreamError,
    bind_read_stream,
    compile_read_loop,
)

logger = logging.getLogger(__name__)
_soapy_logger = logging.getLogger("SoapySDR")

_SOAPY_TO_PYTHON_LEVEL = {
    1: logging.CRITICAL,  # SOAPY_SDR_FATAL
    2: logging.CRITICAL,  # SOAPY_SDR_CRITICAL
    3: logging.ERROR,  # SOAPY_SDR_ERROR
    4: logging.WARNING,  # SOAPY_SDR_WARNING
    5: logging.INFO,  # SOAPY_SDR_NOTICE
    6: logging.INFO,  # SOAPY_SDR_INFO
    7: logging.DEBUG,  # SOAPY_SDR_DEBUG
    8: logging.DEBUG,  # SOAPY_SDR_TRACE
    9: logging.DEBUG,  # SOAPY_SDR_SSI
}


def _log_handler(level: int, message: str) -> None:
    py_level = _SOAPY_TO_PYTHON_LEVEL.get(level, logging.DEBUG)
    _soapy_logger.log(py_level, message.rstrip())


def _system_site_packages() -> list[str]:
    """Return candidate system site-package directories for SoapySDR."""
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    # Free-threaded builds (PEP 703) install under e.g. python3.14t, not python3.14.
    # sys.abiflags is POSIX-only; on Windows it is absent (and ABI flags don't apply).
    tagged = f"{ver}{getattr(sys, 'abiflags', '')}"
    candidates: list[str] = []

    if platform.system() == "Darwin":
        # Homebrew (Apple Silicon and Intel)
        for v in (tagged, ver):
            candidates.append(f"/opt/homebrew/lib/python{v}/site-packages")
            candidates.append(f"/usr/local/lib/python{v}/site-packages")
    else:
        # Debian/Ubuntu apt packages
        candidates.append(f"/usr/lib/python{sys.version_info.major}/dist-packages")
        for v in (tagged, ver):
            candidates.append(f"/usr/lib/python{v}/dist-packages")
            candidates.append(f"/usr/local/lib/python{v}/dist-packages")
            candidates.append(f"/usr/local/lib/python{v}/site-packages")

    # De-duplicate while preserving order (tagged == ver on non-free-threaded builds).
    return [c for c in dict.fromkeys(candidates) if Path(c).is_dir()]


def import_soapysdr() -> Any:
    """Try to import SoapySDR, probing system paths if needed.

    Returns the SoapySDR module, or None if not available.
    """
    try:
        import SoapySDR  # noqa: PLC0415

        SoapySDR.registerLogHandler(_log_handler)
        return SoapySDR
    except ImportError, SystemError:
        pass

    added: list[str] = []
    for path in _system_site_packages():
        if path not in sys.path:
            # Append, not insert(0): probe system dirs for SoapySDR without letting
            # them shadow the venv's own libraries.
            sys.path.append(path)
            added.append(path)

    try:
        import SoapySDR  # noqa: PLC0415

        logger.debug("soapy_found_system_path file=%s", SoapySDR.__file__)
        SoapySDR.registerLogHandler(_log_handler)
        return SoapySDR
    except ImportError, SystemError:
        for path in added:
            sys.path.remove(path)
        return None


_SoapySDR = import_soapysdr()
_HAS_SOAPY = _SoapySDR is not None

# Stream-format preference, best-first (CF32 lossless; integer formats fallback).
# Fields: SOAPY_SDR_* string, SampleFormat, numpy read dtype, components per IQ
# sample (1 = complex64, 2 = interleaved 8-bit).
_STREAM_FORMAT_PREFERENCE: tuple[tuple[str, SampleFormat, Any, int], ...] = (
    ("CF32", SampleFormat.COMPLEX64, np.complex64, 1),
    ("CU8", SampleFormat.UINT8_IQ, np.uint8, 2),
    ("CS8", SampleFormat.SINT8_IQ, np.int8, 2),
)


_OVERFLOW_LOG_INTERVAL_S = 5.0
# Native reader: samples it buffers, and how long a read may wait for data.
_RING_SECONDS = 0.5
_READ_IDLE_TIMEOUT_S = 2.0
_READER_STOP_TIMEOUT_S = 5.0
# Device, stream and reader whose loop never left the driver: freeing them
# would pull the device (or the ring it writes) out from under it.
_abandoned_streams: list[tuple[Any, Any, NativeStreamReader]] = []


@cache
def _native_read_stream() -> Any | None:
    """SoapySDR's C `readStream`, its read loop compiled, or None to read in
    Python instead.

    Looked up through the Python extension, whose links name the exact
    library the bindings use. Compiled here, before any stream starts:
    compiling takes ~0.5 s, and some drivers stall if the first read comes
    that late (see compile_read_loop).
    """
    candidates = [getattr(getattr(_SoapySDR, "_SoapySDR", None), "__file__", None)]
    candidates.append(ctypes.util.find_library("SoapySDR"))
    for path in candidates:
        if not path:
            continue
        try:
            read_stream = bind_read_stream(path)
        except (OSError, AttributeError) as e:
            logger.debug("soapy_native_read_stream_lookup_failed path=%s error=%r", path, e)
            continue
        try:
            compile_read_loop(read_stream)
        except NumbaError as e:
            logger.warning("soapy_native_reader_unavailable reason=compile_failed error=%r", e)
            return None
        return read_stream
    logger.warning("soapy_native_reader_unavailable reason=no_read_stream_symbol")
    return None


@dataclass(frozen=True)
class SoapySDRParams(DeviceParams):
    """SoapySDR device parameters."""

    driver: str = ""
    serial: str = ""
    antenna: str = ""
    device_args: str = ""

    def describe(self) -> str:
        return self.serial or self.driver or self.device_args


class SoapySDRDevice:
    """SDR device via SoapySDR.

    Supports any hardware with a SoapySDR driver module (RTL-SDR, HackRF,
    LimeSDR, PlutoSDR, Airspy, etc.). Prefers lossless CF32, falling back to
    CU8/CS8 only if CF32 is unadvertised. For SoapyRemote links, set the wire
    format independently with a `remote:format=CU8` device arg (the server
    converts, so local CF32 does not bloat the network).

    Requires the SoapySDR Python bindings (system package, not pip).
    """

    def __init__(self, driver: str, serial: str, antenna: str, device_args: str):
        self._driver = driver
        self._serial = serial
        self._antenna = antenna
        self._device_args = device_args
        self._device = None
        self._stream = None
        self._is_open = False
        self._sample_rate: float = 0.0
        # Driver-side overflows (samples the radio dropped before we read them),
        # logged as an aggregate so a link that can't sustain the rate shows up.
        self.overflow_count = 0
        self._overflows_logged = 0
        self._overflow_log_at = 0.0
        # Reads run on a native thread when the bindings expose C handles
        # (see soapy_reader); the counters below track what it has reported.
        self._reader: NativeStreamReader | None = None
        self._reader_overflows = 0
        self._reader_dropped = 0
        self.dropped_count = 0
        self._dropped_logged = 0
        self._dropped_log_at = 0.0
        self._interrupted = False
        self._frequency: float | None = None
        self._sample_format: SampleFormat = SampleFormat.COMPLEX64
        self._read_dtype: Any = np.complex64
        self._read_components: int = 1
        self._identity = DeviceIdentity(
            type_label=driver if driver else "Soapy",
            serial=serial or None,
        )
        self._capabilities = DeviceCapabilities(
            frequency_range=None,
            frequency_controllable=True,
            sample_rates=None,
            gain_supported=True,
            gain_range=(0.0, 49.6),
            gain_step=1.0,
            gain_unit="dB",
            bias_tee_supported=False,
        )

    def _build_args(self):
        """Build SoapySDR kwargs dict.

        Uses SoapySDRKwargs instead of a plain dict because
        Device.make() requires this type.
        """
        args = _SoapySDR.SoapySDRKwargs()
        if self._driver:
            args["driver"] = self._driver
        if self._serial:
            args["serial"] = self._serial
        if self._device_args:
            for part in self._device_args.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    args[k.strip()] = v.strip()
        return args

    def _select_stream_format(self) -> str:
        """Choose the stream format, preferring lossless CF32.

        Uses ``getStreamFormats``; ``getNativeStreamFormat`` is uncallable in
        some Python bindings (its ``double &`` out-param has no typemap).
        """
        assert self._device is not None
        try:
            supported = tuple(self._device.getStreamFormats(_SoapySDR.SOAPY_SDR_RX, 0))
        except (RuntimeError, AttributeError) as e:
            logger.debug("soapy_get_stream_formats_failed error=%r", e)
            supported = (_SoapySDR.SOAPY_SDR_CF32,)
        supported_str = ",".join(supported)
        logger.debug("soapy_stream_formats supported=%s", supported_str)

        for fmt, sample_format, dtype, components in _STREAM_FORMAT_PREFERENCE:
            if fmt in supported:
                self._sample_format = sample_format
                self._read_dtype = dtype
                self._read_components = components
                logger.info(
                    "soapy_stream_format chosen=%s sample_format=%s supported=%s",
                    fmt,
                    sample_format.value,
                    supported_str,
                )
                return fmt
        raise DeviceError(f"No supported stream format advertised: {supported_str}")

    def open(self) -> None:
        if not _HAS_SOAPY:
            raise DeviceError(
                "SoapySDR Python bindings not found. "
                "Install via system package manager (e.g. brew install soapysdr)"
            )
        args = self._build_args()
        self._interrupted = False
        try:
            self._device = _SoapySDR.Device(args)
        except RuntimeError as e:
            raise DeviceError(f"Failed to open SoapySDR device: {e}")
        assert self._device is not None

        if self._antenna:
            self._device.setAntenna(_SoapySDR.SOAPY_SDR_RX, 0, self._antenna)

        soapy_format = self._select_stream_format()
        self._stream = self._device.setupStream(_SoapySDR.SOAPY_SDR_RX, soapy_format)
        _native_read_stream()  # compiles the native reader before the stream starts
        self._device.activateStream(self._stream)
        self._is_open = True

        try:
            settings = self._device.getSettingInfo()
            supports_bias_tee = any(s.key == "biastee" for s in settings)
        except (RuntimeError, AttributeError) as e:
            logger.debug("soapy_get_setting_info_failed error=%r", e)
            supports_bias_tee = False

        gain_range = self._capabilities.gain_range
        gain_step = self._capabilities.gain_step
        try:
            r = self._device.getGainRange(_SoapySDR.SOAPY_SDR_RX, 0)
            gain_range = (float(r.minimum()), float(r.maximum()))
            if r.step() > 0:
                gain_step = float(r.step())
        except (RuntimeError, AttributeError) as e:
            logger.debug("soapy_get_gain_range_failed error=%r", e)

        # Diagnostic log of the driver's gain elements + AGC support. The
        # aggregate rf_gain drives every element, so they aren't exposed separately.
        gains: tuple[str, ...] = ()
        has_agc = False
        try:
            gains = tuple(self._device.listGains(_SoapySDR.SOAPY_SDR_RX, 0))
            has_agc = bool(self._device.hasGainMode(_SoapySDR.SOAPY_SDR_RX, 0))
        except (RuntimeError, AttributeError) as e:
            logger.debug("soapy_list_gains_failed error=%r", e)

        hw = self._device.getHardwareKey()
        logger.debug("soapy_gains hw=%s gains=%s has_agc=%s", hw, ",".join(gains), has_agc)
        self._identity = DeviceIdentity(type_label=hw or "Soapy", serial=self._serial or None)
        frequency_range = self._probe_frequency_range()
        sample_rates = self._probe_sample_rates()
        self._capabilities = DeviceCapabilities(
            frequency_range=frequency_range,
            frequency_controllable=True,
            sample_rates=sample_rates,
            gain_supported=True,
            gain_range=gain_range,
            gain_step=gain_step,
            gain_unit="dB",
            bias_tee_supported=supports_bias_tee,
        )
        logger.info(
            "soapy_opened hw=%s driver=%s frequency_range=%s sample_rates=%s gain_range=%s",
            hw,
            self._driver or "auto",
            frequency_range,
            sample_rates,
            gain_range,
        )

    def _probe_frequency_range(self) -> tuple[float, float] | None:
        """Overall tunable span; disjoint driver ranges are merged end to end."""
        assert self._device is not None
        try:
            ranges = list(self._device.getFrequencyRange(_SoapySDR.SOAPY_SDR_RX, 0))
        except (RuntimeError, AttributeError) as e:
            logger.debug("soapy_get_frequency_range_failed error=%r", e)
            return None
        if not ranges:
            return None
        return (
            min(float(r.minimum()) for r in ranges),
            max(float(r.maximum()) for r in ranges),
        )

    def _probe_sample_rates(self) -> tuple[float, ...] | None:
        """Discrete supported rates, or None when the driver accepts a continuous range.

        Drivers with fixed rates (e.g. Airspy: 2.5 / 10 MS/s) report one
        zero-width range per rate; listing them lets the engine snap the
        configured rate to a supported one and offer them as completions.
        """
        assert self._device is not None
        try:
            ranges = list(self._device.getSampleRateRange(_SoapySDR.SOAPY_SDR_RX, 0))
        except (RuntimeError, AttributeError) as e:
            logger.debug("soapy_get_sample_rate_range_failed error=%r", e)
            return None
        if not ranges or any(r.maximum() > r.minimum() for r in ranges):
            return None
        return tuple(sorted({float(r.minimum()) for r in ranges}))

    def interrupt(self) -> None:
        # A native-reader read polls this; a direct Soapy read has a 2 s
        # timeout, so it unblocks on its own.
        self._interrupted = True

    def close(self) -> None:
        if self._device and self._stream:
            if self._stop_reader():
                try:
                    self._device.deactivateStream(self._stream)
                    self._device.closeStream(self._stream)
                except RuntimeError as e:
                    logger.debug("soapy_stream_close_failed error=%r", e)
            else:
                assert self._reader is not None
                _abandoned_streams.append((self._device, self._stream, self._reader))
                self._reader = None
            self._stream = None
        self._device = None
        self._is_open = False

    def read_samples(self, count: int) -> bytes:
        if not self._device or not self._stream:
            raise DeviceError("Device not open")
        total_samples = count // self._sample_format.bytes_per_sample
        reader = self._reader
        if reader is not None and reader.stopping:
            # A stop that timed out: no new reader on this stream until the
            # old loop has left the driver.
            if reader.alive:
                raise DeviceError("SoapySDR native reader is still inside the driver")
            self._reader = reader = None
        reader = reader or self._start_reader()
        if reader is None:
            return self._read_in_this_thread(total_samples)
        try:
            return reader.read(
                total_samples,
                idle_timeout_s=_READ_IDLE_TIMEOUT_S,
                interrupted=lambda: self._interrupted,
            )
        except StreamError as e:
            self._stop_reader()
            raise DeviceError(
                f"SoapySDR readStream error: {_SoapySDR.errToStr(e.code)} ({e.code})"
            ) from None
        finally:
            self._collect_reader_counts(reader)

    def _start_reader(self) -> NativeStreamReader | None:
        """Start the native reader, or None to read in this thread instead."""
        assert self._device is not None
        read_stream = _native_read_stream()
        if read_stream is None:
            return None
        try:
            handles = int(self._device.this), int(self._stream)
        except AttributeError, TypeError:
            # Not SWIG objects (another binding, or a test double).
            return None
        rate = self._sample_rate or float(self._device.getSampleRate(_SoapySDR.SOAPY_SDR_RX, 0))
        self._reader = NativeStreamReader(
            read_stream,
            *handles,
            sample_bytes=self._sample_format.bytes_per_sample,
            sample_rate=rate,
            ring_seconds=_RING_SECONDS,
        )
        if self._frequency is not None:
            self._reader.mark(self._frequency)
        self._reader_overflows = 0
        self._reader_dropped = 0
        logger.info(
            "soapy_native_reader_started sample_rate=%.0f ring_samples=%d",
            rate,
            self._reader.capacity,
        )
        return self._reader

    def _stop_reader(self) -> bool:
        """Stop the native reader. False when its loop is still inside the
        driver: it then stays this stream's reader, so nothing reconfigures,
        reads, or frees the stream under it."""
        reader = self._reader
        if reader is None:
            return True
        if not reader.stop(_READER_STOP_TIMEOUT_S):
            logger.error("soapy_native_reader_stuck timeout_s=%.1f", _READER_STOP_TIMEOUT_S)
            return False
        self._reader = None
        return True

    @property
    def read_backlog_samples(self) -> int:
        """Samples the native reader holds beyond what has been read."""
        return self._reader.backlog if self._reader is not None else 0

    @property
    def read_center_frequency(self) -> float | None:
        """The center frequency of the samples the last read returned."""
        return self._reader.tag if self._reader is not None else None

    def _collect_reader_counts(self, reader: NativeStreamReader) -> None:
        if reader.overflows != self._reader_overflows:
            self.overflow_count += reader.overflows - self._reader_overflows
            self._reader_overflows = reader.overflows
            self._maybe_log_overflows()
        if reader.dropped != self._reader_dropped:
            self.dropped_count += reader.dropped - self._reader_dropped
            self._reader_dropped = reader.dropped
            self._maybe_log_dropped()

    def _maybe_log_dropped(self) -> None:
        now = time.monotonic()
        if now - self._dropped_log_at < _OVERFLOW_LOG_INTERVAL_S:
            return
        logger.warning(
            "soapy_native_reader_overrun dropped=%d total=%d sample_rate=%.0f",
            self.dropped_count - self._dropped_logged,
            self.dropped_count,
            self._sample_rate,
        )
        self._dropped_logged = self.dropped_count
        self._dropped_log_at = now

    def _read_in_this_thread(self, total_samples: int) -> bytes:
        assert self._device is not None
        # complex64 → 1 element/sample; CU8/CS8 → 2 one-byte components/sample.
        components = self._read_components
        buf = np.empty(total_samples * components, dtype=self._read_dtype)
        offset = 0

        # Loop to fill buffer - readStream may return fewer samples than
        # requested (e.g. SoapyRemote is MTU-limited to ~714 per call)
        while offset < total_samples:
            remaining = total_samples - offset
            chunk = buf[offset * components : (offset + remaining) * components]
            sr = self._device.readStream(self._stream, [chunk], remaining, timeoutUs=2_000_000)
            status = sr.ret
            if status == _SoapySDR.SOAPY_SDR_OVERFLOW:
                # Samples were lost in the driver; the stream continues.
                self.overflow_count += 1
                self._maybe_log_overflows()
                continue
            if status < 0:
                raise DeviceError(
                    f"SoapySDR readStream error: {_SoapySDR.errToStr(status)} ({status})"
                )
            if status == 0:
                raise DeviceError("SoapySDR readStream returned 0 samples")
            offset += status

        result: bytes = buf.tobytes()  # type: ignore[assignment]
        return result

    def _maybe_log_overflows(self) -> None:
        now = time.monotonic()
        if now - self._overflow_log_at < _OVERFLOW_LOG_INTERVAL_S:
            return
        logger.warning(
            "soapy_overflow new=%d total=%d sample_rate=%.0f",
            self.overflow_count - self._overflows_logged,
            self.overflow_count,
            self._sample_rate,
        )
        self._overflows_logged = self.overflow_count
        self._overflow_log_at = now

    def set_frequency(self, freq: float) -> None:
        if self._device:
            self._device.setFrequency(_SoapySDR.SOAPY_SDR_RX, 0, freq)
            self._frequency = freq
            # What the ring holds was captured at the old frequency: reads
            # stop at the change and report each side's frequency.
            if self._reader is not None:
                self._reader.mark(freq)

    def set_sample_rate(self, rate: float) -> None:
        if self._device:
            # Samples in the ring were taken at the old rate; the next read
            # starts a reader sized for the new one.
            if not self._stop_reader():
                raise DeviceError("Sample rate unchanged: the native reader is stuck in the driver")
            self._device.setSampleRate(_SoapySDR.SOAPY_SDR_RX, 0, rate)
            self._sample_rate = float(self._device.getSampleRate(_SoapySDR.SOAPY_SDR_RX, 0))

    @property
    def actual_sample_rate(self) -> float:
        return self._sample_rate

    def set_gain(self, gain: float) -> None:
        if self._device:
            self._device.setGainMode(_SoapySDR.SOAPY_SDR_RX, 0, False)
            self._device.setGain(_SoapySDR.SOAPY_SDR_RX, 0, gain)

    def set_auto_gain(self, enable: bool) -> None:
        if not self._device:
            return
        try:
            if self._device.hasGainMode(_SoapySDR.SOAPY_SDR_RX, 0):
                self._device.setGainMode(_SoapySDR.SOAPY_SDR_RX, 0, enable)
                logger.debug("soapy_set_auto_gain enabled=%s", enable)
        except (RuntimeError, AttributeError) as e:
            logger.debug("soapy_set_auto_gain_failed error=%r", e)

    @property
    def identity(self) -> DeviceIdentity:
        return self._identity

    @property
    def capabilities(self) -> DeviceCapabilities:
        return self._capabilities

    def set_bias_tee(self, enable: bool) -> None:
        if self._device and self._capabilities.bias_tee_supported:
            self._device.writeSetting("biastee", "true" if enable else "false")
            logger.debug("soapy_set_bias_tee enabled=%s", enable)

    def set_network_buffer_seconds(self, seconds: float) -> None:
        # Soapy handles its own buffering; no client-side jitter buffer.
        pass

    def get_sample_format(self) -> SampleFormat:
        return self._sample_format

    def __str__(self) -> str:
        status = "open" if self._is_open else "closed"
        return f"SoapySDRDevice(driver={self._driver or 'auto'}, {status})"
