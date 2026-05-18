import logging
import socket
import struct
from dataclasses import dataclass

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices._jitter_buffer import JitterBuffer
from tsdr.devices.base import DeviceParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RTLTCPParams(DeviceParams):
    """RTL-TCP connection parameters."""

    host: str = "localhost"
    port: int = 1234


class RTLTCPDevice:
    """RTL-SDR device via rtltcp network protocol.

    rtltcp is a simple TCP protocol for controlling RTL-SDR devices remotely.
    Commands are 5 bytes: 1 byte command + 4 bytes parameter (big-endian uint32).

    Protocol Commands:
        0x01: SET_FREQUENCY (Hz)
        0x02: SET_SAMPLE_RATE (Hz)
        0x03: SET_GAIN_MODE (0=manual, 1=auto)
        0x04: SET_GAIN (tenths of dB, e.g., 300 = 30.0 dB)
        0x05: SET_FREQ_CORRECTION (ppm)

    IQ Data Format:
        Unsigned 8-bit pairs [I, Q, I, Q, ...]
        Convert to float: (value - 127.5) / 127.5

    Header Format (12 bytes on connection):
        Bytes 0-3: Magic string "RTL0"
        Bytes 4-7: Tuner type (big-endian uint32)
        Bytes 8-11: Gain count (big-endian uint32)
    """

    # rtltcp command bytes
    CMD_SET_FREQUENCY = 0x01
    CMD_SET_SAMPLE_RATE = 0x02
    CMD_SET_GAIN_MODE = 0x03
    CMD_SET_GAIN = 0x04
    CMD_SET_FREQ_CORRECTION = 0x05
    CMD_SET_GAIN_INDEX = 0x0D
    CMD_SET_BIAS_TEE = 0x0E

    # R820T tuner gain values in tenths of dB (index 0-28)
    # fmt: off
    R820T_GAINS = [
        0, 9, 14, 27, 37, 77, 87, 125, 144, 157,
        166, 197, 207, 229, 254, 280, 297, 328, 338, 364,
        372, 386, 402, 421, 434, 439, 445, 480, 496,
    ]
    # fmt: on

    # Tuner type constants (from rtl-sdr.h)
    TUNER_TYPES = {
        0: "Unknown",
        1: "E4000",
        2: "FC0012",
        3: "FC0013",
        4: "FC2580",
        5: "R820T",
        6: "R828D",
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1234,
        network_buffer_seconds: float = 0.5,
    ):
        self.host = host
        self.port = port
        self.socket: socket.socket | None = None
        self._is_open = False
        self._sample_rate: float = 0.0

        # Header information (populated on connection)
        self.tuner_type: int | None = None
        self.gain_count: int | None = None

        # sample_rate=0 defers ring allocation until set_sample_rate().
        self.jitter = JitterBuffer(
            prefill_seconds=network_buffer_seconds,
            sample_rate=0.0,
            bytes_per_sample=2,
        )

    def open(self) -> None:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))

            # rtltcp sends a 12-byte header on connection
            # Magic: "RTL0" (4 bytes) + tuner type (4 bytes) + gain count (4 bytes)
            header = self.socket.recv(12)
            if len(header) < 12:
                self.socket.close()
                raise DeviceError("Failed to receive rtltcp header (incomplete)")

            # Verify magic string
            magic = header[0:4]
            if magic != b"RTL0":
                self.socket.close()
                raise DeviceError(f"Invalid rtltcp header magic: expected b'RTL0', got {magic!r}")

            # Parse tuner type and gain count (both big-endian uint32)
            self.tuner_type = struct.unpack(">I", header[4:8])[0]
            self.gain_count = struct.unpack(">I", header[8:12])[0]

            # Clear the handshake timeout: the jitter buffer is responsible
            # for tolerating long stalls, and a recv timeout would surface
            # the very jitter we're trying to absorb.
            self.socket.settimeout(None)

            # Log header information
            tuner_name = self.get_tuner_name()
            logger.info(
                f"RTL-TCP connected to {self.host}:{self.port} - "
                f"Tuner: {tuner_name}, Gain count: {self.gain_count}"
            )
            logger.debug(
                f"RTL-TCP header: magic={magic!r}, tuner_type={self.tuner_type}, "
                f"gain_count={self.gain_count}"
            )

            self._is_open = True

        except OSError as e:
            if self.socket:
                self.socket.close()
                self.socket = None
            raise DeviceError(f"Failed to connect to rtltcp at {self.host}:{self.port}: {e}")

        try:
            self.jitter.start(self._read_raw)
        except Exception:
            assert self.socket is not None
            self.socket.close()
            self.socket = None
            self._is_open = False
            raise

    def interrupt(self) -> None:
        """Half-close the socket to unblock the producer's recv.

        Safe to call from any thread; no resources freed. The producer
        thread is blocked in socket.recv with no timeout (set in open());
        shutdown is the only way to wake it from outside its own thread.
        """
        sock = self.socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError as e:
                logger.debug(f"interrupt: shutdown skipped for {self.host}:{self.port}: {e}")

    def close(self) -> None:
        """Disconnect from rtltcp server.

        Ordering: shutdown → jitter.stop → close. Shutdown is idempotent
        (interrupt() may have already done it); jitter.stop joins the
        producer thread, which by now is unblocked from its recv.
        """
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError as e:
                logger.debug(f"Shutdown skipped for {self.host}:{self.port}: {e}")
            self.jitter.stop()
            try:
                self.socket.close()
            except OSError as e:
                logger.debug(f"Error closing socket to {self.host}:{self.port}: {e}")
            finally:
                self.socket = None
                self._is_open = False
                self.tuner_type = None
                self.gain_count = None
        else:
            self.jitter.stop()

    def _send_command(self, command: int, parameter: int) -> None:
        if not self.socket:
            raise DeviceError("Device not open")

        # Pack command: 1 byte command + 4 bytes big-endian uint32
        data = struct.pack(">BI", command, parameter)

        try:
            self.socket.sendall(data)
        except OSError as e:
            raise DeviceError(f"Failed to send command: {e}")

    def read_samples(self, count: int) -> bytes:
        return self.jitter.read(count)

    def _read_raw(self, count: int) -> bytes:
        """Blocking socket read used by the jitter buffer's producer thread.

        Returns exactly `count` bytes or raises DeviceError. recv() has no
        timeout in this state (set in open()), so this blocks indefinitely
        on a stalled connection — that's correct: the close() path uses
        socket.shutdown to unblock the producer when shutting down.
        """
        sock = self.socket
        if sock is None:
            raise DeviceError("Device not open")
        try:
            data = b""
            while len(data) < count:
                chunk = sock.recv(count - len(data))
                if not chunk:
                    raise DeviceError("Connection closed by server")
                data += chunk
            return data
        except OSError as e:
            raise DeviceError(f"Socket error reading samples: {e}")

    def set_frequency(self, freq: float) -> None:
        self._send_command(self.CMD_SET_FREQUENCY, int(freq))

    @property
    def frequency_range(self) -> tuple[float, float] | None:
        return None

    def set_sample_rate(self, rate: float) -> None:
        self._send_command(self.CMD_SET_SAMPLE_RATE, int(rate))
        self._sample_rate = float(int(rate))
        # Resize the jitter ring so capacity tracks the new bytes/second.
        self.jitter.set_sample_rate(self._sample_rate)

    @property
    def actual_sample_rate(self) -> float:
        return self._sample_rate

    def set_gain(self, gain: float) -> None:
        """Set RF gain via CMD_SET_GAIN_INDEX (0x0d).

        Maps `gain` (dB) to the nearest R820T gain step. Index-based gain is
        more reliable than CMD_SET_GAIN (0x04) for discrete tuner gain values.
        """
        # Set gain mode to manual FIRST (required before gain value is accepted)
        self._send_command(self.CMD_SET_GAIN_MODE, 1)  # 1 = manual

        # Find nearest gain index
        gain_tenths = int(gain * 10)
        best_idx = 0
        best_diff = abs(self.R820T_GAINS[0] - gain_tenths)
        for idx, g in enumerate(self.R820T_GAINS):
            diff = abs(g - gain_tenths)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx

        actual_gain = self.R820T_GAINS[best_idx] / 10.0
        self._send_command(self.CMD_SET_GAIN_INDEX, best_idx)
        logger.debug(f"RTL-TCP: Set gain index {best_idx} ({actual_gain} dB, requested {gain} dB)")

    def set_auto_gain(self, enable: bool) -> None:
        # rtltcp gain mode: 0=automatic, 1=manual (counterintuitive)
        self._send_command(self.CMD_SET_GAIN_MODE, 0 if enable else 1)
        logger.debug(f"RTL-TCP: Set AGC {'enabled' if enable else 'disabled'}")

    @property
    def gain_range(self) -> tuple[float, float]:
        return (self.R820T_GAINS[0] / 10.0, self.R820T_GAINS[-1] / 10.0)

    def set_freq_correction(self, ppm: int) -> None:
        self._send_command(self.CMD_SET_FREQ_CORRECTION, ppm)

    @property
    def supports_bias_tee(self) -> bool:
        return True

    def set_bias_tee(self, enable: bool) -> None:
        self._send_command(self.CMD_SET_BIAS_TEE, 1 if enable else 0)
        logger.debug("RTL-TCP: bias-T %s", "on" if enable else "off")

    def set_network_buffer_seconds(self, seconds: float) -> None:
        self.jitter.set_prefill_seconds(seconds)

    def get_tuner_name(self) -> str | None:
        if self.tuner_type is None:
            return None
        return self.TUNER_TYPES.get(self.tuner_type, f"Unknown ({self.tuner_type})")

    def get_sample_format(self) -> SampleFormat:
        return SampleFormat.UINT8_IQ

    def __str__(self) -> str:
        status = "connected" if self._is_open else "disconnected"
        return f"RTLTCPDevice({self.host}:{self.port}, {status})"
