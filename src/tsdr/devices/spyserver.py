"""SpyServer remote-SDR device driver.

SpyServer is the streaming IQ protocol of the Airspy ecosystem. It exposes an
SDR (Airspy R2, AirspyHF+, RTL-SDR) over a TCP socket using a documented binary
protocol. Designed for thick clients that do their own DSP — the perfect fit
for TSDR's "remote antenna + ADC" use case.

The device negotiates INT16 IQ samples on the wire and converts to complex64
internally, applying SpyServer's per-message digital-gain compensation
(`mflags` field in the message-type word).
"""

import logging
import socket
import struct
from dataclasses import dataclass

import numpy as np

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices._jitter_buffer import JitterBuffer
from tsdr.devices.base import DeviceParams

logger = logging.getLogger(__name__)


PROTO_VERSION = (2 << 24) | (0 << 16) | 1700  # 0x020006A4

# Commands (client → server)
CMD_HELLO = 0
CMD_SET_SETTING = 2
CMD_PING = 3

# Settings
SETTING_STREAMING_MODE = 0
SETTING_STREAMING_ENABLED = 1
SETTING_GAIN = 2
SETTING_IQ_FORMAT = 100
SETTING_IQ_FREQUENCY = 101
SETTING_IQ_DECIMATION = 102
SETTING_IQ_DIGITAL_GAIN = 103

# Stream modes
STREAM_MODE_IQ_ONLY = 1

# IQ wire formats
FORMAT_UINT8 = 1
FORMAT_INT16 = 2
FORMAT_INT24 = 3
FORMAT_FLOAT = 4
FORMAT_DINT4 = 5

# Message types (server → client). Upper 16 bits of the MessageType word carry
# `mflags`, which for IQ messages is digital-gain compensation in dB.
MSG_DEVICE_INFO = 0
MSG_CLIENT_SYNC = 1
MSG_PONG = 2
MSG_INT16_IQ = 101

# DEVICE_INFO body is 12 × uint32 = 48 bytes
_DEVICE_INFO_LAYOUT = "<12I"
# CLIENT_SYNC body is 9 × uint32 = 36 bytes
_CLIENT_SYNC_LAYOUT = "<9I"
# Message header: ProtocolID, MessageType, StreamType, SequenceNumber, BodySize
_MSG_HEADER_LAYOUT = "<IIIII"
_MSG_HEADER_SIZE = 20

# Identifying string sent in HELLO
APP_NAME = "tsdr"


@dataclass(frozen=True)
class SpyServerParams(DeviceParams):
    """SpyServer connection parameters."""

    host: str = "localhost"
    port: int = 5555


class SpyServerDevice:
    """SpyServer remote-SDR device.

    Connects to a SpyServer over TCP, negotiates INT16 IQ streaming at the
    chosen decimation, and delivers complex64 samples through `read_samples`.

    `set_sample_rate` accepts only the device-supported rates (powers-of-two
    divisors of the underlying SDR's native rate). Out-of-range tunes and
    rates raise `ValueError`; transport failures raise `DeviceError`.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        network_buffer_seconds: float = 0.5,
    ) -> None:
        self.host = host
        self.port = port
        self._socket: socket.socket | None = None
        self._is_open = False

        # Populated by open() from DEVICE_INFO + CLIENT_SYNC
        self._device_type: int = 0
        self._max_sample_rate: int = 0
        self._decimation_stage_count: int = 0
        self._max_gain_index: int = 0
        self._freq_range: tuple[float, float] | None = None
        self._supported_rates: list[int] = []

        # Mutable state
        self._actual_sample_rate: float = 0.0
        self._streaming = False
        self._recv_buf = bytearray()
        self._iq_buf = bytearray()
        # mflags is per-message dB gain compensation; the value usually stays
        # constant for long stretches, so cache the float32 scale to skip the
        # expensive 10**x on every message.
        self._scale_cache: dict[int, np.float32] = {}

        # bytes_per_sample=8: we deliver complex64.
        # sample_rate=0 defers ring allocation until set_sample_rate().
        self.jitter = JitterBuffer(
            prefill_seconds=network_buffer_seconds,
            sample_rate=0.0,
            bytes_per_sample=8,
        )

    def open(self) -> None:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(10.0)
            self._socket.connect((self.host, self.port))
            self._send_handshake()

            got_device_info = False
            got_client_sync = False
            while not (got_device_info and got_client_sync):
                msg_type, _mflags, body = self._recv_message()
                if msg_type == MSG_DEVICE_INFO:
                    self._parse_device_info(body)
                    got_device_info = True
                elif msg_type == MSG_CLIENT_SYNC:
                    self._parse_client_sync(body)
                    got_client_sync = True

            # Initial settings the user can't override
            self._send_setting(SETTING_IQ_FORMAT, FORMAT_INT16)
            self._send_setting(SETTING_IQ_DIGITAL_GAIN, 0)
            self._send_setting(SETTING_STREAMING_MODE, STREAM_MODE_IQ_ONLY)

            # Clear the handshake timeout: the jitter buffer is responsible
            # for tolerating long stalls. Without this, recv would surface
            # the jitter we're trying to absorb.
            self._socket.settimeout(None)

            self._is_open = True
            logger.info(
                "SpyServer %s:%d connected — device_type=%d, max_sr=%d, "
                "decim_stages=%d, freq_range=%s, max_gain=%d",
                self.host,
                self.port,
                self._device_type,
                self._max_sample_rate,
                self._decimation_stage_count,
                self._freq_range,
                self._max_gain_index,
            )
        except (OSError, struct.error) as e:
            if self._socket:
                self._socket.close()
                self._socket = None
            raise DeviceError(f"SpyServer connect failed ({self.host}:{self.port}): {e}")

        try:
            self.jitter.start(self._read_raw)
        except Exception:
            assert self._socket is not None
            self._socket.close()
            self._socket = None
            self._is_open = False
            raise

    def close(self) -> None:
        if self._socket:
            try:
                self._send_setting(SETTING_STREAMING_ENABLED, 0)
            except (OSError, DeviceError) as e:
                logger.debug("SpyServer close: streaming-off send failed: %s", e)
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError as e:
                logger.debug("SpyServer close: shutdown skipped: %s", e)
            # Shutdown above unblocks the producer's recv; safe to join now.
            self.jitter.stop()
            try:
                self._socket.close()
            except OSError as e:
                logger.debug("SpyServer close: error: %s", e)
            self._socket = None
        else:
            self.jitter.stop()
        self._is_open = False
        self._streaming = False
        self._recv_buf.clear()
        self._iq_buf.clear()
        self._scale_cache.clear()

    def set_frequency(self, freq: float) -> None:
        self._send_setting(SETTING_IQ_FREQUENCY, int(freq))

    @property
    def frequency_range(self) -> tuple[float, float] | None:
        return self._freq_range

    def set_sample_rate(self, rate: float) -> None:
        if not self._supported_rates:
            raise DeviceError("Device not open; cannot set sample rate")
        # Pick nearest within 1% tolerance
        chosen = min(self._supported_rates, key=lambda r: abs(r - rate))
        if abs(chosen - rate) / rate > 0.01:
            valid = ", ".join(str(r) for r in sorted(self._supported_rates))
            raise ValueError(f"SpyServer rate {int(rate)} not supported; valid: {valid}")
        decim = self._max_sample_rate.bit_length() - chosen.bit_length()
        if decim < 0:
            decim = 0
        self._send_setting(SETTING_IQ_DECIMATION, decim)
        self._actual_sample_rate = float(chosen)
        # Resize the jitter ring so capacity tracks the new bytes/second.
        self.jitter.set_sample_rate(self._actual_sample_rate)
        logger.debug(
            "SpyServer set_sample_rate: requested=%d, chosen=%d, decim=%d",
            int(rate),
            chosen,
            decim,
        )

    @property
    def actual_sample_rate(self) -> float:
        return self._actual_sample_rate

    def set_gain(self, gain: float) -> None:
        idx = max(0, min(int(gain), self._max_gain_index))
        self._send_setting(SETTING_GAIN, idx)

    def set_auto_gain(self, enable: bool) -> None:
        # SpyServer doesn't expose hardware AGC; TSDR's client-side AGC takes over.
        pass

    @property
    def gain_range(self) -> tuple[float, float]:
        return (0.0, float(self._max_gain_index))

    def get_sample_format(self) -> SampleFormat:
        return SampleFormat.COMPLEX64

    @property
    def supports_bias_tee(self) -> bool:
        return False

    def set_bias_tee(self, enable: bool) -> None:
        pass

    def set_network_buffer_seconds(self, seconds: float) -> None:
        self.jitter.set_prefill_seconds(seconds)

    def read_samples(self, count: int) -> bytes:
        # All reads flow through the jitter buffer; the producer thread
        # calls _read_raw to drain the SpyServer message stream.
        return self.jitter.read(count)

    def _read_raw(self, count: int) -> bytes:
        """Blocking SpyServer message-decode loop used by the jitter producer.

        Returns exactly `count` bytes of complex64 IQ or raises DeviceError.
        Side effects: enables streaming on the server on first call;
        updates the client-sync gain cache as MSG_CLIENT_SYNC arrives.
        recv() has no timeout in this state (set in open()); the close()
        path uses socket.shutdown to unblock this.
        """
        if not self._is_open or self._socket is None:
            raise DeviceError("Device not open")

        if not self._streaming:
            self._send_setting(SETTING_STREAMING_ENABLED, 1)
            self._streaming = True

        while len(self._iq_buf) < count:
            msg_type, mflags, body = self._recv_message()
            if msg_type == MSG_INT16_IQ:
                self._iq_buf.extend(self._decode_int16_iq(body, mflags))
            elif msg_type == MSG_CLIENT_SYNC:
                self._parse_client_sync(body)
            elif msg_type == MSG_DEVICE_INFO:
                self._parse_device_info(body)
            # else: ignore (PONG, unknown types)

        out = bytes(self._iq_buf[:count])
        del self._iq_buf[:count]
        return out

    def _decode_int16_iq(self, body: bytes, mflags: int) -> bytes:
        # Per-message digital-gain compensation: float = (int16 / 32768) / 10^(mflags/20).
        # mflags usually stays put for many messages — cache the scale.
        scale = self._scale_cache.get(mflags)
        if scale is None:
            scale = np.float32(1.0 / (32768.0 * (10.0 ** (mflags / 20.0))))
            self._scale_cache[mflags] = scale
        ints = np.frombuffer(body, dtype="<i2")
        if ints.size % 2:
            ints = ints[:-1]
        floats = ints.astype(np.float32) * scale
        return floats.view(np.complex64).tobytes()

    def _send_handshake(self) -> None:
        appname = APP_NAME.encode("utf-8")
        body = struct.pack("<I", PROTO_VERSION) + appname
        self._send_command(CMD_HELLO, body)

    def _send_command(self, cmd_type: int, body: bytes) -> None:
        if self._socket is None:
            raise DeviceError("Device not open")
        header = struct.pack("<II", cmd_type, len(body))
        try:
            self._socket.sendall(header + body)
        except OSError as e:
            raise DeviceError(f"SpyServer send failed: {e}")

    def _send_setting(self, setting_id: int, value: int) -> None:
        body = struct.pack("<II", setting_id, value & 0xFFFFFFFF)
        self._send_command(CMD_SET_SETTING, body)

    def _recv_exact(self, n: int) -> bytes:
        assert self._socket is not None
        # Drain accumulated buffer first
        while len(self._recv_buf) < n:
            try:
                chunk = self._socket.recv(max(4096, n - len(self._recv_buf)))
            except OSError as e:
                raise DeviceError(f"SpyServer recv failed: {e}")
            if not chunk:
                raise DeviceError("SpyServer connection closed by server")
            self._recv_buf.extend(chunk)
        out = bytes(self._recv_buf[:n])
        del self._recv_buf[:n]
        return out

    def _recv_message(self) -> tuple[int, int, bytes]:
        header = self._recv_exact(_MSG_HEADER_SIZE)
        _proto_id, msg_type_word, _stream_type, _seq, body_size = struct.unpack(
            _MSG_HEADER_LAYOUT, header
        )
        msg_type = msg_type_word & 0xFFFF
        mflags = (msg_type_word >> 16) & 0xFFFF
        body = self._recv_exact(body_size) if body_size else b""
        return msg_type, mflags, body

    def _parse_device_info(self, body: bytes) -> None:
        if len(body) < struct.calcsize(_DEVICE_INFO_LAYOUT):
            raise DeviceError(f"DEVICE_INFO too short: {len(body)} bytes")
        fields = struct.unpack(_DEVICE_INFO_LAYOUT, body[:48])
        (
            self._device_type,
            _serial,
            self._max_sample_rate,
            _max_bw,
            self._decimation_stage_count,
            _gain_stage_count,
            self._max_gain_index,
            min_freq,
            max_freq,
            _resolution,
            _min_iq_decim,
            _forced_iq_format,
        ) = fields
        self._freq_range = (float(min_freq), float(max_freq))
        # Supported rates: max_sr >> 0..decimation_stage_count
        self._supported_rates = [
            self._max_sample_rate >> n for n in range(self._decimation_stage_count + 1)
        ]

    def _parse_client_sync(self, body: bytes) -> None:
        if len(body) < struct.calcsize(_CLIENT_SYNC_LAYOUT):
            return
        fields = struct.unpack(_CLIENT_SYNC_LAYOUT, body[:36])
        (
            _can_control,
            _current_gain,
            _device_center_freq,
            _iq_center_freq,
            _fft_center_freq,
            min_iq_freq,
            max_iq_freq,
            _min_fft_freq,
            _max_fft_freq,
        ) = fields
        # CLIENT_SYNC narrows the tunable range to what the *user* can request
        # via IQ_FREQUENCY (vs the DEVICE_INFO range which is hardware-wide).
        # If both are present, use the IQ range — that's what set_frequency
        # actually controls.
        if min_iq_freq and max_iq_freq:
            self._freq_range = (float(min_iq_freq), float(max_iq_freq))

    def __str__(self) -> str:
        status = "connected" if self._is_open else "disconnected"
        return f"SpyServerDevice({self.host}:{self.port}, {status})"
