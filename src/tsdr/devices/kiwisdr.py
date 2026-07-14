"""KiwiSDR networked HF-receiver device driver.

A KiwiSDR is an HF (0-30 MHz) receiver reachable over WebSocket. Unlike the
other network devices it exposes two independent streams on two sockets that
share a session id (``tstamp``):

- **SND**: a narrowband IQ channel (12 kHz, or 20.25 kHz on rx3.wf3 firmware)
  tuned to a dial frequency. Drives audio/demod and the narrowband FFT.
- **W/F**: a wideband, server-computed FFT (``wf_fft_size`` dB bins over
  0..bandwidth) with server-side zoom/pan. Decoded frames queue as
  ``SpectrumFrame``s for the I/O worker (``SpectrumSource``); view changes
  map to ``SET zoom= cf=`` on the W/F socket.

The wire protocol is messy: server frames are always WebSocket *binary* even
when textual, the SND header mixes endianness (seq LE, s-meter BE, samples
LE only after ``SET little-endian``), audio and waterfall each gate on a
required command set, and the setup ``MSG`` frames arrive in a
non-deterministic order because the server emits them from separate tasks. The
pure functions below (``parse_msg`` / ``decode_snd`` / ``decode_wf`` /
``format_*`` / ``collect_handshake``) are socket-free and unit-tested with
hand-built frames; ``KiwiSDRDevice`` owns the two sockets, the keepalive, and
the jitter buffer.
"""

from __future__ import annotations

import logging
import math
import secrets
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlsplit

import httpx
import numpy as np
from numpy.typing import NDArray
from websocket import WebSocket, WebSocketException, create_connection

from tsdr.core.http import make_client
from tsdr.core.sdr.exceptions import DeviceError
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices._jitter_buffer import JitterBuffer
from tsdr.devices.base import (
    DeviceCapabilities,
    DeviceIdentity,
    DeviceParams,
    SpectrumFrame,
    SpectrumViewStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8073
_CONNECT_TIMEOUT_S = 10.0
_HANDSHAKE_DEADLINE_S = 20.0
_KEEPALIVE_INTERVAL_S = 5.0
_THREAD_JOIN_TIMEOUT_S = 2.0
_IQ_PASSBAND_HZ = 6000
_WF_DC_BINS = 4  # server zeros 2-4 DC bins; floor 4 for clean autoscaling
_DB_FLOOR = -200.0
# Jitter producer chunk. The 64 KB default is ~0.7 s at the 12 kHz IQ rate,
# coarser than the prefill window, so the buffer starves between chunks. Two
# SND packets (~0.05-0.09 s) keeps the ring topped up.
_SND_PRODUCER_CHUNK_BYTES = 8192

# SND flag bits (rx/rx_sound.cpp:468, web/openwebrx/audio.js:37).
SND_FLAG_STEREO = 0x08
SND_FLAG_RESTART = 0x20
SND_FLAG_ADC_OVFL = 0x02
SND_FLAG_NEW_FREQ = 0x04
SND_FLAG_LITTLE_ENDIAN = 0x80

# W/F flags live in the upper 16 bits of flags_x_zoom_server (rx/rx_waterfall.h:166).
WF_FLAG_COMPRESSION = 0x00010000
# The virtual start-bin space is wf_fft_size << MAX_ZOOM regardless of the
# server's zoom_cap (rx/rx_waterfall.h:149-153).
_WF_MAX_ZOOM = 14
# Server fold modes (rx/rx_waterfall.h:207-208): how the server folds its
# internal FFT down to wf_fft_size output bins. MAX=0 MIN=1 LAST=2 DROP=3
# CMA=4; +10 adds CIC droop compensation. DROP+comp (13) is the KiwiSDR web
# client's own default (web/openwebrx/openwebrx.js:258); trace smoothing
# there is done client-side, not by the fold mode.
_WF_INTERP_DROP = 3
_WF_INTERP_CIC_COMP = 10
# ~2.7 s of frames at the 23 fps W/F max; drops oldest under backpressure.
_WF_QUEUE_FRAMES = 64

# badp auth verdict codes (rx/rx_cmd.h:32).
_BADP_REASONS = {
    "1": "wrong or missing password",
    "2": "server still determining local IP, retry shortly",
    "3": "admin login not allowed from this IP",
    "4": "no admin password set and not on local network",
    "5": "duplicate IP not allowed (already connected from this address)",
    "6": "database update in progress",
    "7": "admin connection already open",
}


@dataclass(frozen=True)
class SndPacket:
    """Decoded SND packet: header fields plus packed complex64 IQ bytes."""

    flags: int
    seq: int
    smeter_dbm: float
    iq_bytes: bytes


@dataclass(frozen=True)
class WfFrame:
    """Decoded W/F line: dB bins (float32) plus header fields."""

    db_bins: NDArray[np.float32]
    x_bin_server: int
    zoom: int
    flags: int
    seq: int


def parse_msg(frame: bytes) -> dict[str, str]:
    """Parse a ``MSG `` frame into a token dict, URI-decoding values.

    Values are URI-encoded on the wire (no literal spaces), so a plain space
    split is safe even for the multi-KB ``load_cfg`` JSON blob. Bare keys
    (e.g. ``wf_setup``) map to ``""``.
    """
    text = frame[4:].decode("latin1", "replace")
    out: dict[str, str] = {}
    for tok in text.split():
        key, sep, val = tok.partition("=")
        out[key] = unquote(val) if sep else ""
    return out


def decode_snd(frame: bytes) -> SndPacket:
    """Decode a binary SND packet into header fields + packed complex64 bytes.

    The header mixes endianness: ``seq`` is u32 LE (rx_sound.cpp:1352),
    ``smeter`` is u16 BE (:1324). IQ packets carry the 20-byte header (the
    STEREO flag selects it). Samples are interleaved s16 LE pairs (I then Q)
    because we send ``SET little-endian``; scaled 1/32768 to complex64.
    """
    if len(frame) < 10 or frame[0:3] != b"SND":
        raise DeviceError(f"not a valid SND frame: {frame[0:4]!r} len={len(frame)}")
    flags = frame[3]
    header_len = 20 if (flags & SND_FLAG_STEREO) else 10
    if len(frame) < header_len:
        raise DeviceError(f"SND frame too short for header: {len(frame)} < {header_len}")
    seq = struct.unpack_from("<I", frame, 4)[0]
    smeter = struct.unpack_from(">H", frame, 8)[0]
    samples = np.frombuffer(frame, dtype="<i2", offset=header_len)
    if samples.size % 2:
        samples = samples[:-1]
    iq = (samples.astype(np.float32) / np.float32(32768.0)).view(np.complex64)
    return SndPacket(
        flags=flags,
        seq=seq,
        smeter_dbm=smeter / 10.0 - 127.0,
        iq_bytes=iq.tobytes(),
    )


def decode_wf(frame: bytes, wf_fft_size: int, wf_cal: int) -> WfFrame:
    """Decode a binary W/F packet: 16-byte LE header + ``wf_fft_size`` dB bytes.

    ``dBm = -(255 - byte) + wf_cal`` (rx_util.cpp:1140). Compression is never
    on in Phase 1 (``wf_comp=0`` and zoom 0 are both uncompressed), so a
    COMPRESSION flag is refused rather than ADPCM-decoded.
    """
    if len(frame) < 16 or frame[0:4] != b"W/F ":
        raise DeviceError(f"not a valid W/F frame: {frame[0:4]!r} len={len(frame)}")
    x_bin_server, flags_x_zoom, seq = struct.unpack_from("<III", frame, 4)
    if flags_x_zoom & WF_FLAG_COMPRESSION:
        raise DeviceError("W/F frame is ADPCM-compressed (unexpected with wf_comp=0)")
    payload = frame[16 : 16 + wf_fft_size]
    if len(payload) < wf_fft_size:
        raise DeviceError(f"W/F payload short: {len(payload)} < {wf_fft_size}")
    bins = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
    bins += np.float32(wf_cal - 255)
    bins[:_WF_DC_BINS] = _DB_FLOOR
    return WfFrame(
        db_bins=bins,
        x_bin_server=x_bin_server,
        zoom=flags_x_zoom & 0xFFFF,
        flags=(flags_x_zoom >> 16) & 0xFFFF,
        seq=seq,
    )


def format_auth(password: str) -> str:
    """Auth command. Empty password is the literal ``p=#`` (rx_cmd.cpp:313)."""
    if not password:
        return "SET auth t=kiwi p=#"
    return f"SET auth t=kiwi p={quote(password, safe='')}"


def format_tune(freq_hz: float) -> str:
    """Tune to ``freq_hz`` in iq mode with the passband widened to the full rate.

    ``freq`` is in kHz; ``low_cut``/``high_cut`` in Hz. The iq default passband
    is only +/-6 kHz, so we set it explicitly.
    """
    return (
        f"SET mod=iq low_cut=-{_IQ_PASSBAND_HZ} high_cut={_IQ_PASSBAND_HZ} "
        f"freq={freq_hz / 1000.0:.3f}"
    )


def snd_gating_commands(audio_rate: int, freq_hz: float, user: str) -> list[str]:
    """CMD_SND_ALL: everything audio gates on, plus little-endian and ident.

    Audio never flows until FREQ, MODE, PASSBAND, AGC, and AR_OK have all been
    received (rx_sound.cpp:422). AGC is disabled because its gain is applied to
    the IQ samples (rx_sound.cpp:1142).
    """
    return [
        f"SET AR OK in={audio_rate} out=44100",
        format_tune(freq_hz),
        "SET agc=0 hang=0 thresh=-100 slope=6 decay=1000 manGain=50",
        "SET little-endian",
        f"SET ident_user={quote(user, safe='')}",
    ]


def format_wf_view(zoom: int, cf_baseband_hz: float) -> str:
    """W/F view command: ``cf`` is baseband kHz (rx_waterfall_cmd.cpp:153-158);
    the server derives and clamps the start bin from it."""
    return f"SET zoom={zoom} cf={cf_baseband_hz / 1000.0:.3f}"


def covering_zoom(bandwidth_hz: float, span_hz: float, zoom_cap: int) -> int:
    """Largest server zoom whose span (bandwidth / 2^z) still covers ``span_hz``."""
    z = math.floor(math.log2(bandwidth_hz / max(span_hz, 1.0)))
    return max(0, min(z, zoom_cap))


def wf_frame_geometry(
    x_bin: int, zoom: int, bandwidth_hz: float, wf_fft_size: int, freq_offset_hz: float
) -> tuple[float, float]:
    """(center_hz, span_hz) of a W/F frame.

    The start bin lives in a virtual ``wf_fft_size << _WF_MAX_ZOOM`` bin space
    over 0..bandwidth (rx_waterfall.h:149-153); span halves per zoom level.
    """
    span = bandwidth_hz / (1 << zoom)
    hz_per_start = bandwidth_hz / (wf_fft_size << _WF_MAX_ZOOM)
    center = freq_offset_hz + x_bin * hz_per_start + span / 2
    return center, span


def wf_gating_commands() -> list[str]:
    """CMD_WF_ALL plus compression-off and interpolation.

    Frames never flow until zoom/start, maxdb/mindb, and wf_speed arrive
    (rx_waterfall.cpp:456). ``wf_comp=0`` skips waterfall ADPCM.
    """
    return [
        "SET zoom=0 start=0",
        "SET maxdb=0 mindb=-100",
        "SET wf_speed=4",
        "SET wf_comp=0",
        f"SET interp={_WF_INTERP_CIC_COMP + _WF_INTERP_DROP}",
    ]


def _badp_reason(code: str) -> str:
    return _BADP_REASONS.get(code, f"auth rejected (badp={code})")


def _raise_on_admission(acc: dict[str, str]) -> None:
    """Raise DeviceError on any connection-admission denial in the accumulator.

    These arrive before or instead of ``badp`` (rx_server.cpp:456-730).
    """
    if "too_busy" in acc:
        raise DeviceError(f"KiwiSDR busy: all {acc['too_busy']} channels in use")
    if "redirect" in acc:
        raise DeviceError(f"KiwiSDR redirected to {acc['redirect']}")
    if "down" in acc:
        reason = acc.get("reason_disabled") or f"down={acc['down']}"
        raise DeviceError(f"KiwiSDR unavailable: {reason}")
    if "wb_only" in acc:
        raise DeviceError("KiwiSDR accepts wideband clients only")
    if "exclusive_use" in acc:
        raise DeviceError("KiwiSDR is in exclusive use")
    if "ip_limit" in acc:
        raise DeviceError(f"KiwiSDR IP time limit reached: {acc['ip_limit']}")
    if "password_timeout" in acc:
        raise DeviceError("KiwiSDR dropped connection: authentication timeout")


def _log_rx(socket: str, frame: bytes) -> None:
    """Debug-log a received control frame; truncate long config blobs."""
    logger.debug("kiwi_rx socket=%s msg=%r", socket, frame[4:204].decode("latin1", "replace"))


def collect_handshake(
    recv: Callable[[], bytes],
    ready: Callable[[dict[str, str]], bool],
    *,
    deadline_s: float,
    clock: Callable[[], float] = time.monotonic,
    label: str = "?",
) -> dict[str, str]:
    """Drain MSG frames via ``recv()`` until ``ready(acc)`` or the deadline.

    Order-agnostic: the server emits setup MSGs from separate cooperatively
    scheduled tasks, so their wire order is not fixed. We accumulate tokens
    into a dict and gate on *presence*, never arrival order. Non-MSG frames
    are ignored because no data flows before we send the gating commands.
    Raises DeviceError on admission denial, ``badp != 0``, EOF, or timeout.
    """
    acc: dict[str, str] = {}
    end = clock() + deadline_s
    while clock() < end:
        frame = recv()
        if not frame:
            raise DeviceError("KiwiSDR closed the connection during handshake")
        if frame[0:4] != b"MSG ":
            continue
        _log_rx(label, frame)
        toks = parse_msg(frame)
        acc.update(toks)
        _raise_on_admission(acc)
        if "badp" in toks and toks["badp"] != "0":
            raise DeviceError(f"KiwiSDR auth failed: {_badp_reason(toks['badp'])}")
        if ready(acc):
            return acc
    raise DeviceError("KiwiSDR handshake timed out")


def _snd_ready(acc: dict[str, str]) -> bool:
    return "badp" in acc and "audio_rate" in acc and "bandwidth" in acc


def _wf_ready(acc: dict[str, str]) -> bool:
    return "badp" in acc and "wf_setup" in acc


def resolve_endpoint(host: str, port: int, timeout: float = 8.0) -> tuple[str, int]:
    """Resolve a possibly-proxied Kiwi to its real ``host, port``.

    A proxied Kiwi (``*.proxy.kiwisdr.com``) may 307-redirect on ``/status`` to
    ``*.proxy2...`` (frp load-balancing). When it does, take only the Location's
    host and keep the original port, where the Kiwi serves both ``/status`` and
    the WebSocket (the ``http`` Location targets ``/status`` and need not carry
    the WS port). Hosts that answer 200 (a direct Kiwi, or a proxy that didn't
    redirect) are used unchanged. websocket-client cannot follow an http-scheme
    redirect, so this HTTP pre-flight handles the ones that do.
    """
    for _ in range(3):
        status, location = _http_status(host, port, timeout)
        if status is None or not (300 <= status < 400) or not location:
            return host, port
        new_host = urlsplit(location).hostname
        if not new_host or new_host == host:
            return host, port
        logger.info("kiwi_endpoint_resolved host=%s resolved=%s port=%d", host, new_host, port)
        host = new_host
    return host, port


def _http_status(host: str, port: int, timeout: float) -> tuple[int | None, str | None]:
    url = f"http://{host}:{port}/status"
    try:
        with make_client(timeout) as client:
            resp = client.get(url, follow_redirects=False)
            return resp.status_code, resp.headers.get("Location")
    except httpx.HTTPError:
        return None, None


def _format_connect_error(host: str, port: int, exc: BaseException) -> str:
    target = f"{host}:{port}"
    if isinstance(exc, socket.gaierror):
        return f"KiwiSDR host unresolvable: {host}"
    if isinstance(exc, ConnectionRefusedError):
        return f"KiwiSDR at {target} refused connection (no server, or wrong port)"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return f"KiwiSDR at {target} did not respond within {int(_CONNECT_TIMEOUT_S)}s"
    if isinstance(exc, DeviceError):
        # collect_handshake / admission raise a complete message; don't re-wrap.
        return str(exc)
    return f"KiwiSDR at {target} connect failed: {exc}"


@dataclass(frozen=True)
class KiwiSDRParams(DeviceParams):
    """KiwiSDR connection parameters."""

    host: str = "localhost"
    port: int = DEFAULT_PORT
    password: str = ""
    user: str = "tsdr"

    def describe(self) -> str:
        return f"{self.host}:{self.port}"


class KiwiSDRDevice:
    """KiwiSDR two-socket HF receiver.

    ``open()`` resolves the endpoint, opens SND and W/F on a shared tstamp,
    runs both order-agnostic handshakes, and starts the SND jitter producer,
    the W/F reader, and a shared keepalive. Both sockets are required; any
    handshake failure tears down everything and raises DeviceError. SND IQ is
    delivered as complex64 through ``read_samples``; W/F frames are decoded and
    logged (Phase 1 does not display them).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        password: str = "",
        user: str = "tsdr",
        network_buffer_seconds: float = 0.5,
    ) -> None:
        self.host = host
        self.port = port
        self.password = password
        self.user = user

        self._snd_ws: WebSocket | None = None
        self._wf_ws: WebSocket | None = None
        self._snd_send_lock = threading.Lock()
        self._wf_send_lock = threading.Lock()

        self._stop = threading.Event()
        self._wf_thread: threading.Thread | None = None
        self._keepalive_thread: threading.Thread | None = None

        self._wf_lock = threading.Lock()
        self._spectrum_frames: deque[SpectrumFrame] = deque(maxlen=_WF_QUEUE_FRAMES)
        self._wf_view_sent: tuple[int, float] | None = None  # (zoom, absolute center Hz)
        self._wf_last_frame: tuple[int, float, float, int] | None = None  # zoom, center, span, bins
        self._fatal_reason: str | None = None
        self._inactivity_ack_pending = False

        self._snd_buf = bytearray()
        self._wf_frames = 0
        self._last_wf_log_ts = 0.0

        self._actual_sample_rate = 0.0
        self._audio_rate = 12000
        self._bandwidth_hz = 30_000_000.0
        self._center_freq_hz = 15_000_000.0
        self._freq_offset_hz = 0.0
        self._wf_fft_size = 1024
        self._wf_cal = 0
        self._wf_chans: int | None = None
        self._zoom_cap = 14
        self._wf_fps_expected = 23.0
        self._wf_fps_measured = 0.0

        self._identity = DeviceIdentity(type_label="KiwiSDR", serial=None)
        self._capabilities = DeviceCapabilities(
            frequency_range=None,
            frequency_controllable=True,
            sample_rates=None,
            gain_supported=False,
            gain_range=(0.0, 0.0),
            gain_step=0.0,
            gain_unit="dB",
            bias_tee_supported=False,
        )

        # sample_rate=0 defers ring allocation until open()/set_sample_rate.
        self.jitter = JitterBuffer(
            prefill_seconds=network_buffer_seconds,
            sample_rate=0.0,
            bytes_per_sample=8,
            producer_chunk_bytes=_SND_PRODUCER_CHUNK_BYTES,
        )

    def open(self) -> None:
        try:
            host, port = resolve_endpoint(self.host, self.port)
            tstamp = secrets.randbits(50)

            self._snd_ws = self._connect(host, port, tstamp, "SND")
            self._send(self._snd_ws, self._snd_send_lock, format_auth(self.password))
            snd_info = collect_handshake(
                self._snd_ws.recv, _snd_ready, deadline_s=_HANDSHAKE_DEADLINE_S, label="snd"
            )
            self._apply_snd_setup(snd_info)
            for cmd in snd_gating_commands(self._audio_rate, self._center_freq_hz, self.user):
                self._send(self._snd_ws, self._snd_send_lock, cmd)
            self._snd_ws.settimeout(None)

            self._wf_ws = self._connect(host, port, tstamp, "W/F")
            self._send(self._wf_ws, self._wf_send_lock, format_auth(self.password))
            wf_info = collect_handshake(
                self._wf_ws.recv, _wf_ready, deadline_s=_HANDSHAKE_DEADLINE_S, label="wf"
            )
            self._apply_wf_setup(wf_info)
            for cmd in wf_gating_commands():
                self._send(self._wf_ws, self._wf_send_lock, cmd)
            self._wf_ws.settimeout(None)

            self._rebuild_capabilities()
            self.jitter.set_sample_rate(self._actual_sample_rate)

            self._stop.clear()
            self._fatal_reason = None
            self._keepalive_thread = self._spawn(self._keepalive_loop, "kiwi-keepalive")
            self._wf_thread = self._spawn(self._wf_loop, "kiwi-wf")
            logger.info(
                "kiwi_ready host=%s:%d sample_rate=%.3f bandwidth=%.0f wf_chans=%s",
                host,
                port,
                self._actual_sample_rate,
                self._bandwidth_hz,
                self._wf_chans,
            )
        except (OSError, WebSocketException, DeviceError, struct.error) as e:
            self._teardown_sockets()
            raise DeviceError(_format_connect_error(self.host, self.port, e)) from e

        try:
            self.jitter.start(self._read_snd_raw)
        except Exception:
            self.close()
            raise

    def interrupt(self) -> None:
        """Half-close both sockets to unblock the parked recvs from any thread."""
        for ws in (self._snd_ws, self._wf_ws):
            if ws is not None and ws.sock is not None:
                try:
                    ws.sock.shutdown(socket.SHUT_RDWR)
                except OSError as e:
                    logger.debug("kiwi_shutdown_skipped error=%r", e)

    def close(self) -> None:
        self._stop.set()
        self.interrupt()
        self.jitter.stop()
        for thread in (self._wf_thread, self._keepalive_thread):
            if thread is not None:
                thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
        self._teardown_sockets()
        self._wf_thread = None
        self._keepalive_thread = None
        self._snd_buf.clear()
        with self._wf_lock:
            self._spectrum_frames.clear()

    def set_frequency(self, freq: float) -> None:
        ws = self._snd_ws
        if ws is None:
            raise DeviceError("KiwiSDR not open")
        self._center_freq_hz = float(freq)
        self._send(ws, self._snd_send_lock, format_tune(freq))

    def set_sample_rate(self, rate: float) -> None:
        # Rate is firmware-fixed; keep the exact handshake value and just size
        # the jitter ring to the true bytes/second.
        if self._actual_sample_rate > 0:
            self.jitter.set_sample_rate(self._actual_sample_rate)

    @property
    def actual_sample_rate(self) -> float:
        return self._actual_sample_rate

    @property
    def wire_bytes_per_sec(self) -> float:
        return self._actual_sample_rate * 8

    def set_gain(self, gain: float) -> None:
        pass

    def set_auto_gain(self, enable: bool) -> None:
        pass

    def set_bias_tee(self, enable: bool) -> None:
        pass

    def get_sample_format(self) -> SampleFormat:
        return SampleFormat.COMPLEX64

    @property
    def identity(self) -> DeviceIdentity:
        return self._identity

    @property
    def capabilities(self) -> DeviceCapabilities:
        return self._capabilities

    def set_network_buffer_seconds(self, seconds: float) -> None:
        self.jitter.set_prefill_seconds(seconds)

    def drain_spectrum_frames(self) -> list[SpectrumFrame]:
        with self._wf_lock:
            frames = list(self._spectrum_frames)
            self._spectrum_frames.clear()
        return frames

    def set_spectrum_view(self, center_hz: float, span_hz: float) -> None:
        ws = self._wf_ws
        if ws is None:
            return
        zoom = covering_zoom(self._bandwidth_hz, span_hz, self._zoom_cap)
        self._send(ws, self._wf_send_lock, format_wf_view(zoom, center_hz - self._freq_offset_hz))
        self._wf_view_sent = (zoom, center_hz)
        logger.info("kiwi_wf_view zoom=%d cf=%.3f span=%.0f", zoom, center_hz, span_hz)

    def spectrum_view_status(self) -> SpectrumViewStatus | None:
        sent = self._wf_view_sent
        if sent is None:
            return None
        last = self._wf_last_frame
        return SpectrumViewStatus(
            requested_zoom=sent[0],
            requested_center_hz=sent[1],
            zoom_cap=self._zoom_cap,
            frame_zoom=last[0] if last else None,
            frame_center_hz=last[1] if last else None,
            frame_span_hz=last[2] if last else None,
            frame_bins=last[3] if last else None,
            expected_fps=self._wf_fps_expected,
            measured_fps=self._wf_fps_measured if self._wf_fps_measured > 0 else None,
        )

    def read_samples(self, count: int) -> bytes:
        return self.jitter.read(count)

    def __str__(self) -> str:
        status = "connected" if self._snd_ws is not None else "disconnected"
        return f"KiwiSDRDevice({self.host}:{self.port}, {status})"

    def _connect(self, host: str, port: int, tstamp: int, stream: str) -> WebSocket:
        # The kiwirecorder URL form; <tstamp> pairs SND and W/F server-side. The
        # W/F stream literally contains a slash; websocket-client sends it
        # un-encoded so the server's stream parse recovers it.
        url = f"ws://{host}:{port}/{tstamp}/{stream}"
        return create_connection(url, timeout=_CONNECT_TIMEOUT_S, enable_multithread=True)

    def _send(self, ws: WebSocket, lock: threading.Lock, cmd: str) -> None:
        logger.debug("kiwi_tx socket=%s cmd=%r", "snd" if ws is self._snd_ws else "wf", cmd)
        try:
            with lock:
                ws.send(cmd)
        except (OSError, WebSocketException) as e:
            # A concurrent close() (e.g. rapid stop/start) can tear the socket
            # down mid-send; surface it as DeviceError like the recv path rather
            # than leaking a raw websocket exception out of set_frequency etc.
            raise DeviceError(self._fatal_reason or f"KiwiSDR send failed: {e}") from e

    def _spawn(self, target: Callable[[], None], name: str) -> threading.Thread:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        return thread

    def _apply_snd_setup(self, info: dict[str, str]) -> None:
        self._audio_rate = int(info["audio_rate"])
        self._bandwidth_hz = float(info["bandwidth"])
        self._center_freq_hz = float(info.get("center_freq") or self._bandwidth_hz / 2)
        # sample_rate is the exact GPS-corrected float; it usually arrives during
        # the handshake but the producer refines it if it comes later.
        self._actual_sample_rate = float(info.get("sample_rate") or self._audio_rate)
        if "freq_offset" in info:
            self._freq_offset_hz = float(info["freq_offset"]) * 1000.0  # freq_offset is kHz
        logger.info(
            "kiwi_snd_setup audio_rate=%d sample_rate=%.3f bandwidth=%.0f center=%.0f",
            self._audio_rate,
            self._actual_sample_rate,
            self._bandwidth_hz,
            self._center_freq_hz,
        )

    def _apply_wf_setup(self, info: dict[str, str]) -> None:
        self._wf_fft_size = int(info.get("wf_fft_size") or 1024)
        self._wf_cal = int(info.get("wf_cal") or 0)
        self._zoom_cap = int(info.get("zoom_cap") or 14)
        # Advertised rate of the fast wf_speed setting (which we request).
        self._wf_fps_expected = float(info.get("wf_fps") or 23.0)
        wf_chans = info.get("wf_chans")
        self._wf_chans = int(wf_chans) if wf_chans else None
        if "bandwidth" in info:
            self._bandwidth_hz = float(info["bandwidth"])
        logger.info(
            "kiwi_wf_setup wf_fft_size=%d wf_cal=%d wf_chans=%s zoom_cap=%d wf_share=%s",
            self._wf_fft_size,
            self._wf_cal,
            self._wf_chans,
            self._zoom_cap,
            info.get("wf_share"),
        )
        if self._wf_chans == 0:
            logger.info("kiwi_wf_unavailable wf_chans=0")

    def _rebuild_capabilities(self) -> None:
        self._capabilities = DeviceCapabilities(
            frequency_range=(self._freq_offset_hz, self._freq_offset_hz + self._bandwidth_hz),
            frequency_controllable=True,
            sample_rates=(self._actual_sample_rate,),
            gain_supported=False,
            gain_range=(0.0, 0.0),
            gain_step=0.0,
            gain_unit="dB",
            bias_tee_supported=False,
            # wf_chans=0 (rx14.wf0 builds) never sends a W/F frame; keep the
            # engine-side IQ FFT as the spectrum source there.
            provides_spectrum=self._wf_chans != 0,
        )

    def _read_snd_raw(self, count: int) -> bytes:
        """Jitter producer: decode SND packets until ``count`` bytes accumulate.

        Returns exactly ``count`` bytes of complex64 IQ or raises DeviceError.
        Non-SND frames (control MSG, AF-spectrum DAT) are handled inline and
        never fed to the IQ decoder. recv() blocks (no timeout, cleared in
        open()); close() uses socket.shutdown to unblock it.
        """
        ws = self._snd_ws
        if ws is None:
            raise DeviceError("KiwiSDR SND socket not open")
        while len(self._snd_buf) < count:
            try:
                frame = ws.recv()
            except (OSError, WebSocketException) as e:
                raise DeviceError(self._fatal_reason or f"KiwiSDR SND recv failed: {e}") from e
            if not frame:
                raise DeviceError(self._fatal_reason or "KiwiSDR SND connection closed")
            if isinstance(frame, str):
                frame = frame.encode("latin1", "replace")
            if frame[0:3] == b"SND":
                packet = decode_snd(frame)
                if packet.flags & SND_FLAG_RESTART:
                    logger.info("kiwi_snd_restart seq=%d", packet.seq)
                if packet.flags & SND_FLAG_NEW_FREQ:
                    logger.debug("kiwi_snd_new_freq seq=%d flags=0x%02x", packet.seq, packet.flags)
                self._snd_buf.extend(packet.iq_bytes)
            elif frame[0:4] == b"MSG ":
                _log_rx("snd", frame)
                self._handle_snd_msg(parse_msg(frame))
        out = bytes(self._snd_buf[:count])
        del self._snd_buf[:count]
        return out

    def _handle_snd_msg(self, toks: dict[str, str]) -> None:
        rate = toks.get("sample_rate")
        if rate is not None:
            value = float(rate)
            if value > 0 and value != self._actual_sample_rate:
                self._actual_sample_rate = value
                self.jitter.set_sample_rate(value)
                self._rebuild_capabilities()
        self._check_stream_msg(toks)

    def _wf_loop(self) -> None:
        ws = self._wf_ws
        if ws is None:
            return
        while not self._stop.is_set():
            try:
                frame = ws.recv()
            except (OSError, WebSocketException) as e:
                if not self._stop.is_set():
                    logger.debug("kiwi_wf_recv_stopped error=%r", e)
                return
            if not frame:
                return
            if isinstance(frame, str):
                frame = frame.encode("latin1", "replace")
            if frame[0:4] == b"W/F ":
                self._handle_wf_frame(frame)
            elif frame[0:4] == b"MSG ":
                _log_rx("wf", frame)
                self._check_stream_msg(parse_msg(frame))

    def _handle_wf_frame(self, frame: bytes) -> None:
        try:
            wf = decode_wf(frame, self._wf_fft_size, self._wf_cal)
        except DeviceError as e:
            logger.debug("kiwi_wf_decode_failed error=%r", e)
            return
        center_hz, span_hz = wf_frame_geometry(
            wf.x_bin_server, wf.zoom, self._bandwidth_hz, self._wf_fft_size, self._freq_offset_hz
        )
        self._wf_last_frame = (wf.zoom, center_hz, span_hz, wf.db_bins.size)
        with self._wf_lock:
            self._spectrum_frames.append(
                SpectrumFrame(db_bins=wf.db_bins, center_hz=center_hz, span_hz=span_hz, seq=wf.seq)
            )
        self._wf_frames += 1
        now = time.monotonic()
        if self._last_wf_log_ts == 0.0:
            self._last_wf_log_ts = now
            return
        elapsed = now - self._last_wf_log_ts
        if elapsed >= 1.0:
            self._wf_fps_measured = self._wf_frames / elapsed
            logger.info(
                "kiwi_wf_frame bins=%d db_min=%.1f db_max=%.1f fps=%.1f",
                wf.db_bins.size,
                float(wf.db_bins.min()),
                float(wf.db_bins.max()),
                self._wf_fps_measured,
            )
            self._last_wf_log_ts = now
            self._wf_frames = 0

    def _check_stream_msg(self, toks: dict[str, str]) -> None:
        """Handle inactivity/kick MSGs that can arrive on either socket."""
        if "inactivity_timeout" in toks:
            logger.info("kiwi_inactivity_timeout min=%s", toks["inactivity_timeout"])
            self._inactivity_ack_pending = True
        if "kiwi_kick" in toks:
            self._fatal("KiwiSDR kicked the connection")
        if "ip_limit" in toks:
            self._fatal(f"KiwiSDR IP time limit reached ({toks['ip_limit']})")

    def _keepalive_loop(self) -> None:
        while not self._stop.wait(_KEEPALIVE_INTERVAL_S):
            for ws, lock in (
                (self._snd_ws, self._snd_send_lock),
                (self._wf_ws, self._wf_send_lock),
            ):
                if ws is None:
                    continue
                try:
                    self._send(ws, lock, "SET keepalive")
                except (OSError, WebSocketException, DeviceError) as e:
                    if not self._stop.is_set():
                        self._fatal(f"KiwiSDR keepalive failed: {e}")
                    return
            if self._inactivity_ack_pending and self._snd_ws is not None:
                self._inactivity_ack_pending = False
                try:
                    self._send(self._snd_ws, self._snd_send_lock, "SET inactivity_ack")
                except (OSError, WebSocketException, DeviceError):
                    pass

    def _fatal(self, reason: str) -> None:
        if self._fatal_reason is None:
            self._fatal_reason = reason
            logger.warning("kiwi_fatal reason=%s", reason)
        self.interrupt()

    def _teardown_sockets(self) -> None:
        for ws in (self._snd_ws, self._wf_ws):
            if ws is not None:
                try:
                    ws.close()
                except (OSError, WebSocketException) as e:
                    logger.debug("kiwi_ws_close_failed error=%r", e)
        self._snd_ws = None
        self._wf_ws = None
