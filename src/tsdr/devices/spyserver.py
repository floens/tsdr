"""SpyServer remote-SDR device driver.

SpyServer is the streaming IQ protocol of the Airspy ecosystem. It exposes an
SDR (Airspy R2, AirspyHF+, RTL-SDR) over a TCP socket using a documented binary
protocol. Designed for thick clients that do their own DSP.

The device negotiates INT16 IQ samples on the wire and converts to complex64
internally, applying SpyServer's per-message digital-gain compensation
(`mflags` field in the message-type word).
"""

from __future__ import annotations

import logging
import socket
import struct
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.core.units import find_nearest
from tsdr.devices._jitter_buffer import JitterBuffer
from tsdr.devices.base import DeviceCapabilities, DeviceIdentity, DeviceParams

logger = logging.getLogger(__name__)


PROTO_VERSION = (2 << 24) | (0 << 16) | 1700  # 0x020006A4
APP_NAME = "tsdr"

STREAM_MODE_IQ_ONLY = 1


class Command(IntEnum):
    """Client → server command types."""

    HELLO = 0
    SET_SETTING = 2
    PING = 3


class Setting(IntEnum):
    """SET_SETTING parameter IDs."""

    STREAMING_MODE = 0
    STREAMING_ENABLED = 1
    GAIN = 2
    IQ_FORMAT = 100
    IQ_FREQUENCY = 101
    IQ_DECIMATION = 102
    IQ_DIGITAL_GAIN = 103


class MsgType(IntEnum):
    """Server → client message types. Upper 16 bits of the MessageType word
    carry `mflags`, which for IQ messages is digital-gain compensation in dB."""

    DEVICE_INFO = 0
    CLIENT_SYNC = 1
    PONG = 2
    UINT8_IQ = 100
    INT16_IQ = 101
    INT24_IQ = 102
    FLOAT_IQ = 103


class IqFormat(IntEnum):
    """IQ wire format identifiers."""

    DEFAULT = 0
    UINT8 = 1
    INT16 = 2
    INT24 = 3
    FLOAT = 4
    DINT4 = 5


class DeviceType(IntEnum):
    """SpyServer DeviceType enum values, cross-checked SDR++ and
    miweber67/spyserver_client. Used for log readability."""

    INVALID = 0
    AIRSPY_ONE = 1
    AIRSPY_HF = 2
    RTLSDR = 3


_DECODABLE_FORCED_FORMATS = frozenset(
    {IqFormat.UINT8, IqFormat.INT16, IqFormat.INT24, IqFormat.FLOAT}
)

_WIRE_BYTES_PER_IQ_PAIR: dict[int, int] = {
    IqFormat.UINT8: 2,
    IqFormat.INT16: 4,
    IqFormat.INT24: 6,
    IqFormat.FLOAT: 8,
}


# DEVICE_INFO body is 12 × uint32 = 48 bytes
_DEVICE_INFO_LAYOUT = "<12I"
# CLIENT_SYNC body is 9 × uint32 = 36 bytes
_CLIENT_SYNC_LAYOUT = "<9I"
# Message header: ProtocolID, MessageType, StreamType, SequenceNumber, BodySize
_MSG_HEADER_LAYOUT = "<IIIII"
_MSG_HEADER_SIZE = 20


def _enum_name(cls: type[IntEnum], value: int) -> str:
    """Enum member name for `value`, or `UNKNOWN(value)` if not a member."""
    try:
        return cls(value).name
    except ValueError:
        return f"UNKNOWN({value})"


def _format_connect_error(host: str, port: int, exc: BaseException) -> str:
    target = f"{host}:{port}"
    if isinstance(exc, socket.gaierror):
        return f"SpyServer host unresolvable: {host}"
    if isinstance(exc, ConnectionRefusedError):
        return f"SpyServer at {target} refused connection (no server, or wrong port)"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return f"SpyServer at {target} did not respond within 10s (host unreachable or firewalled)"
    if isinstance(exc, DeviceError) and "closed by server" in str(exc):
        return (
            f"SpyServer at {target} dropped the connection during handshake "
            f"(likely server full, IP restricted, or protocol version mismatch)"
        )
    if isinstance(exc, DeviceError):
        # _apply_device_info raises DeviceError with a complete message; don't re-wrap.
        return str(exc)
    if isinstance(exc, struct.error):
        return f"SpyServer at {target} returned malformed handshake: {exc}"
    return f"SpyServer at {target} connect failed: {exc}"


@dataclass(frozen=True)
class SpyServerParams(DeviceParams):
    """SpyServer connection parameters."""

    host: str = "localhost"
    port: int = 5555

    def describe(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class _DeviceInfo:
    """Parsed DEVICE_INFO message body. No derivation — pure view of the wire."""

    device_type: int
    serial: int
    max_sample_rate: int
    max_bandwidth: int
    decimation_stage_count: int
    gain_stage_count: int
    max_gain_index: int
    min_freq: int
    max_freq: int
    resolution: int
    min_iq_decimation: int
    forced_iq_format: int

    @classmethod
    def from_bytes(cls, body: bytes) -> _DeviceInfo:
        if len(body) < struct.calcsize(_DEVICE_INFO_LAYOUT):
            raise DeviceError(f"DEVICE_INFO too short: {len(body)} bytes")
        return cls(*struct.unpack(_DEVICE_INFO_LAYOUT, body[:48]))


@dataclass(frozen=True)
class _ClientSync:
    """Parsed CLIENT_SYNC message body."""

    can_control: bool
    current_gain: int
    device_center_freq: int
    iq_center_freq: int
    fft_center_freq: int
    min_iq_freq: int
    max_iq_freq: int
    min_fft_freq: int
    max_fft_freq: int

    @classmethod
    def from_bytes(cls, body: bytes) -> _ClientSync | None:
        # Returns None on short body — server can send partial sync mid-stream;
        # callers log a warning and skip.
        if len(body) < struct.calcsize(_CLIENT_SYNC_LAYOUT):
            return None
        fields = struct.unpack(_CLIENT_SYNC_LAYOUT, body[:36])
        return cls(bool(fields[0]), *fields[1:])


class _SpyServerCodec:
    """Frame, send, receive, and decode SpyServer protocol bytes.

    Owns the socket and the recv buffer; converts between bytes and structured
    messages but holds no device-level state. Not thread-safe: the device
    assumes send_* runs on the main thread and recv_/decode_iq on the producer
    thread (current TSDR usage), so no internal lock.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._recv_buf = bytearray()
        # mflags-to-scale cache: mflags usually stays put for many messages,
        # so the float32 scale is reused; 10**(x/20) is expensive enough to
        # matter at audio rates. Only one divisor is used per session, so
        # keying solely on mflags is safe.
        self._scale_cache: dict[int, np.float32] = {}

    def set_recv_timeout(self, t: float | None) -> None:
        self._sock.settimeout(t)

    def send_command(self, cmd_type: Command, body: bytes) -> None:
        header = struct.pack("<II", cmd_type, len(body))
        try:
            self._sock.sendall(header + body)
        except OSError as e:
            raise DeviceError(f"SpyServer send failed: {e}")

    def send_hello(self) -> None:
        body = struct.pack("<I", PROTO_VERSION) + APP_NAME.encode("utf-8")
        self.send_command(Command.HELLO, body)
        logger.debug(
            "spyserver_hello proto=0x%08x version=%d.%d.%d app=%s",
            PROTO_VERSION,
            (PROTO_VERSION >> 24) & 0xFF,
            (PROTO_VERSION >> 16) & 0xFF,
            PROTO_VERSION & 0xFFFF,
            APP_NAME,
        )

    def send_setting(self, setting: Setting, value: int) -> None:
        body = struct.pack("<II", setting, value & 0xFFFFFFFF)
        self.send_command(Command.SET_SETTING, body)
        # IQ_FORMAT value is itself an IqFormat — decode it for readability.
        val_str = _enum_name(IqFormat, value) if setting is Setting.IQ_FORMAT else str(value)
        logger.debug("spyserver_setting key=%s value=%s", setting.name, val_str)

    def recv_message(self) -> tuple[int, int, bytes]:
        header = self._recv_exact(_MSG_HEADER_SIZE)
        _proto_id, msg_type_word, _stream_type, _seq, body_size = struct.unpack(
            _MSG_HEADER_LAYOUT, header
        )
        msg_type = msg_type_word & 0xFFFF
        mflags = (msg_type_word >> 16) & 0xFFFF
        body = self._recv_exact(body_size) if body_size else b""
        return msg_type, mflags, body

    def decode_iq(self, msg_type: int, mflags: int, body: bytes) -> bytes:
        """Convert an IQ message body to packed complex64 bytes."""
        if msg_type == MsgType.INT16_IQ:
            scale = self._mflags_scale(mflags, 32768.0)
            ints = np.frombuffer(body, dtype="<i2")
            if ints.size % 2:
                ints = ints[:-1]
            return (ints.astype(np.float32) * scale).view(np.complex64).tobytes()
        if msg_type == MsgType.UINT8_IQ:
            scale = self._mflags_scale(mflags, 128.0)
            u8 = np.frombuffer(body, dtype=np.uint8)
            if u8.size % 2:
                u8 = u8[:-1]
            return ((u8.astype(np.float32) - 128.0) * scale).view(np.complex64).tobytes()
        if msg_type == MsgType.INT24_IQ:
            scale = self._mflags_scale(mflags, 8388608.0)
            u8 = np.frombuffer(body, dtype=np.uint8)
            n_components = (len(u8) // 3) & ~1
            if n_components == 0:
                return b""
            u8 = u8[: n_components * 3].reshape(-1, 3)
            # (u8^0x80)-0x80 sign-extends the top byte; .view(np.int8) would need contiguous memory.
            i32 = (
                u8[:, 0].astype(np.int32)
                | (u8[:, 1].astype(np.int32) << 8)
                | (((u8[:, 2].astype(np.int32) ^ 0x80) - 0x80) << 16)
            )
            return (i32.astype(np.float32) * scale).view(np.complex64).tobytes()
        if msg_type == MsgType.FLOAT_IQ:
            # Wire format is already float32 IQ pairs; mflags still applies.
            scale = self._mflags_scale(mflags, 1.0)
            floats = np.frombuffer(body, dtype="<f4")
            if floats.size % 2:
                floats = floats[:-1]
            return (floats * scale).view(np.complex64).tobytes()
        raise DeviceError(f"decode_iq: unsupported msg_type={msg_type}")

    def shutdown(self) -> None:
        """Half-close the socket to unblock a blocked recv on another thread."""
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError as e:
            logger.debug("spyserver_codec_shutdown_skipped error=%r", e)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError as e:
            logger.debug("spyserver_codec_close_failed error=%r", e)

    def _recv_exact(self, n: int) -> bytes:
        while len(self._recv_buf) < n:
            try:
                chunk = self._sock.recv(max(4096, n - len(self._recv_buf)))
            except OSError as e:
                raise DeviceError(f"SpyServer recv failed: {e}")
            if not chunk:
                raise DeviceError("SpyServer connection closed by server")
            self._recv_buf.extend(chunk)
        out = bytes(self._recv_buf[:n])
        del self._recv_buf[:n]
        return out

    def _mflags_scale(self, mflags: int, divisor: float) -> np.float32:
        # Per-message digital-gain compensation:
        # float = (sample / divisor) / 10^(mflags/20).
        scale = self._scale_cache.get(mflags)
        if scale is None:
            scale = np.float32(1.0 / (divisor * (10.0 ** (mflags / 20.0))))
            self._scale_cache[mflags] = scale
        return scale


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

        # Lifecycle. `_codec is not None` is the source of truth for "is open".
        self._codec: _SpyServerCodec | None = None
        self._streaming = False

        # Device-reported caps (set in _apply_device_info)
        self._device_type: int = 0
        self._max_sample_rate: int = 0
        # `MinimumIQDecimation` from DEVICE_INFO: lowest decimation stage the
        # server is willing to deliver. Non-zero when the operator has set
        # `maximum_bandwidth` in spyserver.config (common for AirspyHF+ public
        # servers); decimation values below this are silently refused and
        # streaming never starts. Must clamp `IQ_DECIMATION` >= this.
        self._min_iq_decim: int = 0
        # `ForcedIQFormat` from DEVICE_INFO: if non-zero, the server overrides
        # `SETTING_IQ_FORMAT` and sends this format regardless of what the
        # client requests. We must honor it or the IQ message-type stream is
        # one we can't decode (UINT8_IQ=100, FLOAT_IQ=103 vs INT16_IQ=101).
        self._iq_format: int = IqFormat.INT16
        self._max_gain_index: int = 0
        self._supported_rates: list[int] = []

        # DEVICE_INFO range: full hardware tunable range. Stays fixed for the session.
        self._device_freq_range: tuple[float, float] | None = None
        # CLIENT_SYNC IQ-center window: the bounds the server will honor as IQ_CENTER
        # without retuning the hardware. Narrower than _device_freq_range, but only
        # gates tuning when we lack CanControl (the controlling client retunes hw on
        # an out-of-window IQ_FREQUENCY request).
        self._iq_window: tuple[float, float] | None = None
        # Effective tunable range surfaced via capabilities. Derived from the two above.
        self._freq_range: tuple[float, float] | None = None
        # Default True to avoid a "Locked" flash before the first CLIENT_SYNC;
        # the handshake corrects it. `set_gain` no-ops while `_codec is None`.
        self._can_control: bool = True
        self._actual_sample_rate: float = 0.0

        self._identity = DeviceIdentity(type_label="SpyServer", serial=None)
        self._capabilities = DeviceCapabilities(
            frequency_range=None,
            frequency_controllable=True,
            sample_rates=None,
            gain_supported=True,
            gain_range=(0.0, 0.0),
            gain_step=1.0,
            gain_unit="index",
            bias_tee_supported=False,
        )

        # read_raw accumulator — variable-size server messages → fixed-size reads.
        self._iq_buf = bytearray()

        # Logging state. `_last_mflags is None` doubles as "first IQ not seen yet".
        # `_seen_unknown` is bounded by the set of distinct unknown msg types
        # the server actually emits (in practice 0 or 1) — used to warn once.
        self._last_mflags: int | None = None
        self._seen_unknown: set[int] = set()
        # Last CLIENT_SYNC applied; used to log INFO only on change since the
        # server can emit identical sync messages periodically.
        self._last_client_sync: _ClientSync | None = None

        # sample_rate=0 defers ring allocation until set_sample_rate().
        self.jitter = JitterBuffer(
            prefill_seconds=network_buffer_seconds,
            sample_rate=0.0,
            bytes_per_sample=8,
        )

    def open(self) -> None:
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((self.host, self.port))
            self._codec = _SpyServerCodec(sock)
            self._codec.send_hello()

            got_device_info = False
            got_client_sync = False
            while not (got_device_info and got_client_sync):
                msg_type, _mflags, body = self._codec.recv_message()
                if msg_type == MsgType.DEVICE_INFO:
                    self._apply_device_info(_DeviceInfo.from_bytes(body))
                    got_device_info = True
                elif msg_type == MsgType.CLIENT_SYNC:
                    sync = _ClientSync.from_bytes(body)
                    if sync is not None:
                        self._apply_client_sync(sync)
                    else:
                        logger.warning("spyserver_client_sync_short_handshake bytes=%d", len(body))
                    got_client_sync = True

            # Initial settings the user can't override. If the server set
            # `ForcedIQFormat` we must use it — sending a different format is
            # accepted but ignored, and the server then emits IQ messages with
            # a message-type we wouldn't recognise.
            self._codec.send_setting(Setting.IQ_FORMAT, self._iq_format)
            self._codec.send_setting(Setting.IQ_DIGITAL_GAIN, 0)
            self._codec.send_setting(Setting.STREAMING_MODE, STREAM_MODE_IQ_ONLY)

            # Clear the handshake timeout: the jitter buffer is responsible
            # for tolerating long stalls. Without this, recv would surface
            # the jitter we're trying to absorb.
            self._codec.set_recv_timeout(None)

            logger.info(
                "spyserver_ready host=%s:%d device=%s iq_format=%s",
                self.host,
                self.port,
                _enum_name(DeviceType, self._device_type),
                _enum_name(IqFormat, self._iq_format),
            )
        except (OSError, struct.error, DeviceError) as e:
            if self._codec is not None:
                self._codec.close()
                self._codec = None
            elif sock is not None:
                sock.close()
            raise DeviceError(_format_connect_error(self.host, self.port, e)) from e

        try:
            self.jitter.start(self._read_raw)
        except Exception:
            self.close()
            raise

    def interrupt(self) -> None:
        """Half-close the codec socket to unblock the producer's recv.

        Safe to call from any thread; no resources freed. The producer
        is blocked in a no-timeout recv (see open()); shutdown is the
        only wake mechanism from outside its own thread.
        """
        codec = self._codec
        if codec is not None:
            codec.shutdown()

    def close(self) -> None:
        codec = self._codec
        self._codec = None
        if codec is not None:
            try:
                codec.send_setting(Setting.STREAMING_ENABLED, 0)
            except (OSError, DeviceError) as e:
                logger.debug("spyserver_close_streaming_off_failed error=%r", e)
            # Shutdown is idempotent — interrupt() may have already done it.
            # Either way, the producer is now unblocked and jitter.stop joins.
            codec.shutdown()
        self.jitter.stop()
        if codec is not None:
            codec.close()
        self._streaming = False
        self._iq_buf.clear()
        self._seen_unknown.clear()
        self._last_mflags = None
        self._last_client_sync = None

    def set_frequency(self, freq: float) -> None:
        if self._codec is None:
            raise DeviceError("Device not open")
        self._codec.send_setting(Setting.IQ_FREQUENCY, int(freq))

    def set_sample_rate(self, rate: float) -> None:
        if self._codec is None:
            raise DeviceError("Device not open; cannot set sample rate")
        chosen = find_nearest(self._supported_rates, rate)
        if abs(chosen - rate) / rate > 0.01:
            logger.warning(
                "spyserver_sample_rate_snapped requested=%d chosen=%d",
                int(rate),
                chosen,
            )
        decim = self._max_sample_rate.bit_length() - chosen.bit_length()
        # Server silently refuses decim below MinimumIQDecimation.
        decim = max(decim, self._min_iq_decim)
        self._codec.send_setting(Setting.IQ_DECIMATION, decim)
        self._actual_sample_rate = float(chosen)
        # Resize the jitter ring so capacity tracks the new bytes/second.
        self.jitter.set_sample_rate(self._actual_sample_rate)
        logger.debug(
            "spyserver_set_sample_rate requested=%d chosen=%d decim=%d",
            int(rate),
            chosen,
            decim,
        )

    @property
    def actual_sample_rate(self) -> float:
        return self._actual_sample_rate

    @property
    def wire_bytes_per_sec(self) -> float:
        return self._actual_sample_rate * _WIRE_BYTES_PER_IQ_PAIR.get(self._iq_format, 4)

    def set_gain(self, gain: float) -> None:
        if not self._can_control or self._codec is None:
            return
        idx = max(0, min(int(gain), self._max_gain_index))
        self._codec.send_setting(Setting.GAIN, idx)

    def set_auto_gain(self, enable: bool) -> None:
        # SpyServer doesn't expose hardware AGC; TSDR's client-side AGC takes over.
        pass

    def get_sample_format(self) -> SampleFormat:
        return SampleFormat.COMPLEX64

    def set_bias_tee(self, enable: bool) -> None:
        pass

    @property
    def identity(self) -> DeviceIdentity:
        return self._identity

    @property
    def capabilities(self) -> DeviceCapabilities:
        return self._capabilities

    def set_network_buffer_seconds(self, seconds: float) -> None:
        self.jitter.set_prefill_seconds(seconds)

    def read_samples(self, count: int) -> bytes:
        return self.jitter.read(count)

    def __str__(self) -> str:
        status = "connected" if self._codec is not None else "disconnected"
        return f"SpyServerDevice({self.host}:{self.port}, {status})"

    def _read_raw(self, count: int) -> bytes:
        """Blocking SpyServer message-decode loop used by the jitter producer.

        Returns exactly `count` bytes of complex64 IQ or raises DeviceError.
        Side effects: enables streaming on the server on first call;
        applies CLIENT_SYNC / DEVICE_INFO mid-stream as they arrive.
        recv() has no timeout in this state (cleared in open()); the close()
        path uses socket.shutdown to unblock this.
        """
        if self._codec is None:
            raise DeviceError("Device not open")

        if not self._streaming:
            self._codec.send_setting(Setting.STREAMING_ENABLED, 1)
            self._streaming = True
            logger.info(
                "spyserver_streaming_enabled host=%s:%d sample_rate=%.0f format=%s",
                self.host,
                self.port,
                self._actual_sample_rate,
                _enum_name(IqFormat, self._iq_format),
            )

        while len(self._iq_buf) < count:
            msg_type, mflags, body = self._codec.recv_message()
            if msg_type in (MsgType.INT16_IQ, MsgType.INT24_IQ, MsgType.UINT8_IQ, MsgType.FLOAT_IQ):
                self._note_iq_message(msg_type, mflags, len(body))
                self._iq_buf.extend(self._codec.decode_iq(msg_type, mflags, body))
            elif msg_type == MsgType.CLIENT_SYNC:
                sync = _ClientSync.from_bytes(body)
                if sync is not None:
                    self._apply_client_sync(sync)
                else:
                    logger.warning("spyserver_client_sync_short bytes=%d", len(body))
            elif msg_type == MsgType.DEVICE_INFO:
                self._apply_device_info(_DeviceInfo.from_bytes(body))
            elif msg_type == MsgType.PONG:
                logger.debug("spyserver_pong")
            elif msg_type not in self._seen_unknown:
                # Warn once per unknown msg_type so a protocol mismatch (e.g.
                # server forces a format we don't decode) doesn't silently
                # stall the read loop.
                self._seen_unknown.add(msg_type)
                logger.warning(
                    "spyserver_unknown_message msg_type=%d mflags=0x%04x bytes=%d",
                    msg_type,
                    mflags,
                    len(body),
                )

        out = bytes(self._iq_buf[:count])
        del self._iq_buf[:count]
        return out

    def _note_iq_message(self, msg_type: int, mflags: int, body_size: int) -> None:
        """Log the first IQ message and any mflags transitions thereafter."""
        if self._last_mflags is None:
            logger.info(
                "spyserver_first_iq type=%s mflags=%d bytes=%d",
                _enum_name(MsgType, msg_type),
                mflags,
                body_size,
            )
            self._last_mflags = mflags
            return
        if mflags != self._last_mflags:
            logger.debug("spyserver_mflags_changed old=%d new=%d", self._last_mflags, mflags)
            self._last_mflags = mflags

    def _apply_device_info(self, info: _DeviceInfo) -> None:
        self._device_type = info.device_type
        self._max_sample_rate = info.max_sample_rate
        self._max_gain_index = info.max_gain_index
        self._min_iq_decim = info.min_iq_decimation
        if info.forced_iq_format and info.forced_iq_format not in _DECODABLE_FORCED_FORMATS:
            raise DeviceError(
                f"SpyServer forces unsupported IQ format "
                f"{_enum_name(IqFormat, info.forced_iq_format)}; "
                f"only UINT8/INT16/INT24/FLOAT are decodable"
            )
        if info.forced_iq_format:
            self._iq_format = info.forced_iq_format
        self._device_freq_range = (float(info.min_freq), float(info.max_freq))
        self._recompute_freq_range()
        # Supported rates: max_sr >> n for n in min_iq_decim..decimation_stage_count
        self._supported_rates = [
            info.max_sample_rate >> n
            for n in range(info.min_iq_decimation, info.decimation_stage_count + 1)
        ]
        self._identity = DeviceIdentity(
            type_label=_enum_name(DeviceType, info.device_type),
            serial=f"0x{info.serial:08x}",
        )
        self._rebuild_capabilities()
        fmt_name = _enum_name(IqFormat, info.forced_iq_format) if info.forced_iq_format else "NONE"
        logger.info(
            "spyserver_device_info type=%s serial=0x%08x max_sr=%d max_bw=%d "
            "decim_stages=%d decim_min=%d gain_stages=%d max_gain_index=%d "
            "freq_min=%d freq_max=%d resolution=%d forced_iq_format=%s",
            _enum_name(DeviceType, info.device_type),
            info.serial,
            info.max_sample_rate,
            info.max_bandwidth,
            info.decimation_stage_count,
            info.min_iq_decimation,
            info.gain_stage_count,
            info.max_gain_index,
            info.min_freq,
            info.max_freq,
            info.resolution,
            fmt_name,
        )
        logger.info(
            "spyserver_supported_rates rates=%s",
            ",".join(str(r) for r in self._supported_rates),
        )

    def _apply_client_sync(self, sync: _ClientSync) -> None:
        if sync == self._last_client_sync:
            return
        self._last_client_sync = sync
        self._can_control = sync.can_control
        if sync.min_iq_freq and sync.max_iq_freq:
            self._iq_window = (float(sync.min_iq_freq), float(sync.max_iq_freq))
        self._recompute_freq_range()
        self._rebuild_capabilities()
        logger.debug(
            "spyserver_client_sync can_control=%s current_gain=%d "
            "device_center=%d iq_center=%d fft_center=%d "
            "iq_min=%d iq_max=%d fft_min=%d fft_max=%d",
            sync.can_control,
            sync.current_gain,
            sync.device_center_freq,
            sync.iq_center_freq,
            sync.fft_center_freq,
            sync.min_iq_freq,
            sync.max_iq_freq,
            sync.min_fft_freq,
            sync.max_fft_freq,
        )

    def _recompute_freq_range(self) -> None:
        # With CanControl we can request any frequency in the hardware range —
        # the server retunes the hardware to accommodate. Without it, we're
        # confined to wherever the controlling client has parked the IQ window.
        if self._can_control or self._iq_window is None:
            self._freq_range = self._device_freq_range
        else:
            self._freq_range = self._iq_window

    def _rebuild_capabilities(self) -> None:
        controller_freq: float | None = None
        controller_gain: int | None = None
        if not self._can_control and self._last_client_sync is not None:
            controller_freq = float(self._last_client_sync.iq_center_freq)
            controller_gain = int(self._last_client_sync.current_gain)
        self._capabilities = DeviceCapabilities(
            frequency_range=self._freq_range,
            # A locked client can still retune its own IQ sub-window inside
            # the controller's band (that is what min/max_iq_freq delimit);
            # the lock only narrows frequency_range and takes away gain.
            frequency_controllable=True,
            sample_rates=tuple(float(r) for r in self._supported_rates)
            if self._supported_rates
            else None,
            gain_supported=self._can_control,
            gain_range=(0.0, float(self._max_gain_index)),
            gain_step=1.0,
            gain_unit="index",
            bias_tee_supported=False,
            controller_center_frequency=controller_freq,
            controller_gain=controller_gain,
        )
