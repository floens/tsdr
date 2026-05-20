import io
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import zstandard as zstd

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices.base import DeviceCapabilities, DeviceIdentity, DeviceParams

_IQFILE_IDENTITY = DeviceIdentity(type_label="IQ file", serial=None)
_IQFILE_CAPABILITIES = DeviceCapabilities(
    frequency_range=None,
    frequency_controllable=False,
    sample_rates=None,
    gain_supported=False,
    gain_range=(0.0, 0.0),
    gain_step=1.0,
    gain_unit="dB",
    bias_tee_supported=False,
)

logger = logging.getLogger(__name__)


EXTENSION_FORMAT_MAP: dict[str, SampleFormat] = {
    ".cu8": SampleFormat.UINT8_IQ,
    ".cf32": SampleFormat.COMPLEX64,
    ".raw": SampleFormat.COMPLEX64,
    ".iq": SampleFormat.COMPLEX64,
}

_SI_MULTIPLIER = {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9}
_SR_PATTERN = re.compile(r"(?:^|[_/])sr=(\d+(?:\.\d+)?)([kMG]?)(?=[_.]|$)")


def parse_sample_rate_from_filename(name: str) -> float | None:
    """Extract sample rate from a filename produced by the record command.

    Matches the `sr=<value><unit>` segment (e.g. `sr=250k`, `sr=2.4M`).
    """
    m = _SR_PATTERN.search(name)
    if not m:
        return None
    return float(m.group(1)) * _SI_MULTIPLIER[m.group(2)]


@dataclass(frozen=True)
class IQFileParams(DeviceParams):
    """IQ file playback parameters."""

    path: str = ""
    sample_format: SampleFormat | None = None

    def describe(self) -> str:
        return Path(self.path).name


class IQFileDevice:
    """IQ file playback device.

    Reads raw IQ samples from a file, looping at EOF. Throttles reads
    to simulate real-time playback at the configured sample rate.
    """

    def __init__(self, path: Path, sample_format: SampleFormat):
        self._path = path
        self._sample_format = sample_format
        self._file: io.BufferedReader | io.BytesIO | None = None
        self._file_size = 0
        self._sample_rate = 2.4e6
        self._playback_time = 0.0
        self._read_count = 0
        self._loop_count = 0

    def open(self) -> None:
        if not self._path.exists():
            raise DeviceError(f"File not found: {self._path}")
        if self._path.stat().st_size == 0:
            raise DeviceError(f"File is empty: {self._path}")
        if self._path.name.endswith(".zst"):
            dctx = zstd.ZstdDecompressor()
            with open(self._path, "rb") as f:
                raw = dctx.stream_reader(f).read()
            self._file = io.BytesIO(raw)
            self._file_size = len(raw)
        else:
            self._file = open(self._path, "rb")  # noqa: SIM115
            self._file_size = self._path.stat().st_size
        self._playback_time = time.monotonic()
        logger.debug(
            "iqfile_opened path=%s bytes=%d format=%s sample_rate=%s",
            self._path,
            self._file_size,
            self._sample_format.value,
            self._sample_rate,
        )

    def interrupt(self) -> None:
        # Paced playback: each read returns within one chunk's worth of
        # wall-clock time, so nothing to break out of.
        pass

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def read_samples(self, count: int) -> bytes:
        if not self._file:
            raise DeviceError("Device not open")

        # Throttle to simulate real-time playback using absolute timing.
        # Sleep overshoot in one read self-corrects in the next, preventing
        # cumulative drift that causes audio jitter.
        num_samples = count / self._sample_format.bytes_per_sample
        expected_duration = num_samples / self._sample_rate
        self._playback_time += expected_duration
        sleep_time = self._playback_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif sleep_time < -0.1:
            # Fell too far behind (e.g. queue backpressure), reset to avoid burst
            self._playback_time = time.monotonic()

        # Read with EOF looping
        data = b""
        remaining = count
        while remaining > 0:
            chunk = self._file.read(remaining)
            if not chunk:
                self._loop_count += 1
                logger.debug("iqfile_eof_loop loop=%d read=%d", self._loop_count, self._read_count)
                self._file.seek(0)
                chunk = self._file.read(remaining)
                if not chunk:
                    raise DeviceError("Failed to read from file after seek")
            data += chunk
            remaining -= len(chunk)

        self._read_count += 1
        if self._read_count <= 3 or self._read_count % 100 == 0:
            logger.debug(
                "iqfile_read read=%d bytes=%d sleep=%.4f rate=%s",
                self._read_count,
                count,
                sleep_time,
                self._sample_rate,
            )
        return data

    def set_frequency(self, freq: float) -> None:
        pass

    def set_sample_rate(self, rate: float) -> None:
        logger.debug("iqfile_set_sample_rate old=%s new=%s", self._sample_rate, rate)
        self._sample_rate = rate

    @property
    def actual_sample_rate(self) -> float:
        return self._sample_rate

    def set_gain(self, gain: float) -> None:
        pass

    def set_auto_gain(self, enable: bool) -> None:
        pass

    def set_bias_tee(self, enable: bool) -> None:
        pass

    @property
    def identity(self) -> DeviceIdentity:
        return _IQFILE_IDENTITY

    @property
    def capabilities(self) -> DeviceCapabilities:
        return _IQFILE_CAPABILITIES

    def set_network_buffer_seconds(self, seconds: float) -> None:
        pass

    def get_sample_format(self) -> SampleFormat:
        return self._sample_format
