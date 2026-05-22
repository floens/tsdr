"""SNTP-based clock-offset monitor.

Sends a 48-byte SNTPv3 packet to a user-chosen server every N seconds in a
background daemon thread, measures the round-trip offset, and exposes the
last result to UI code. Opt-in: no probing happens until `set_server()`
is called with a host.

Why inline instead of ntplib: the SNTP query is ~25 lines of stdlib socket
code (RFC 5905 §A.5.1), no external dep. The features ntplib adds
(kiss-of-death handling, leap-second parsing, stratum validation) don't
apply at the 15 min poll cadence we use against public time servers.

Why SNTP rather than the kernel's `ntp_gettime`: macOS's `timed` does not
discipline the kernel NTP state machine through `adjtimex` the way
chrony/ntpd/systemd-timesyncd do on Linux, so `ntp_gettime` reports
`TIME_OK` regardless of actual sync. An out-of-band SNTP query gives the
same answer on every platform.
"""

import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from typing import Literal

logger = logging.getLogger(__name__)

SyncState = Literal["synced", "drift", "unsynced", "unknown"]

# Seconds between Jan 1 1900 (NTP epoch) and Jan 1 1970 (Unix epoch).
_NTP_EPOCH_OFFSET = 2208988800

# Anything beyond this is "drift" — for FT8/WSPR slot timing, sub-100 ms
# offset is comfortably synced; 0.1–0.5 s is concerning; >0.5 s is broken.
_GREEN_THRESHOLD_S = 0.1
_RED_THRESHOLD_S = 0.5


@dataclass(frozen=True)
class SyncResult:
    """Snapshot of the last SNTP probe."""

    state: SyncState
    offset_s: float | None
    server: str | None
    measured_at: float | None  # monotonic seconds

    def state_for(self, server: str | None) -> SyncState:
        """Return `unknown` if probing is disabled or server changed since last probe."""
        if not self.is_current(server):
            return "unknown"
        return self.state

    def is_current(self, server: str | None) -> bool:
        """True if this snapshot was measured against the currently configured server."""
        return server is not None and self.server == server


def query_sntp(host: str, timeout: float = 3.0) -> float | None:
    """Send one SNTPv3 packet to `host:123`. Returns offset in seconds, or None on failure.

    Offset is (server_clock - local_clock). Positive means our clock is behind.
    Computed per RFC 5905: offset = ((T2-T1) + (T3-T4)) / 2.
    """
    # LI=0, VN=3, Mode=3 (client). Remaining 47 bytes stay zero — server fills its own.
    packet = bytes([0b00_011_011]) + bytes(47)
    sock: socket.socket | None = None
    try:
        info = socket.getaddrinfo(host, 123, type=socket.SOCK_DGRAM)
        family, socktype, proto, _, sockaddr = info[0]
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        t1 = time.time()
        sock.sendto(packet, sockaddr)
        data, _ = sock.recvfrom(1024)
        t4 = time.time()
    except (OSError, IndexError):
        return None
    finally:
        if sock is not None:
            sock.close()
    if len(data) < 48:
        return None
    # T2 (server receive) at offset 32, T3 (server transmit) at offset 40.
    # Each is uint32 seconds + uint32 fraction, NTP epoch.
    t2_int, t2_frac = struct.unpack("!II", data[32:40])
    t3_int, t3_frac = struct.unpack("!II", data[40:48])
    t2 = float(t2_int - _NTP_EPOCH_OFFSET) + float(t2_frac) / 2**32
    t3 = float(t3_int - _NTP_EPOCH_OFFSET) + float(t3_frac) / 2**32
    return ((t2 - t1) + (t3 - t4)) / 2


def _classify(offset_s: float | None) -> SyncState:
    if offset_s is None:
        return "unsynced"
    mag = abs(offset_s)
    if mag <= _GREEN_THRESHOLD_S:
        return "synced"
    if mag <= _RED_THRESHOLD_S:
        return "drift"
    return "unsynced"


class ClockSyncMonitor:
    """Background SNTP poller. Cheap to leave running; cheap to disable.

    Default poll interval 15 min (900 s) — well above pool.ntp.org's 64 s
    minimum, matches the steady-state cadence of chronyd/ntpd/timesyncd
    once they've settled. A clock that drifts faster than that is broken
    in ways an SNTP probe won't help you diagnose. The immediate probe
    on `set_server()` covers the "I just enabled this, give me an answer
    now" case.
    """

    def __init__(self, poll_interval_s: float = 900.0) -> None:
        self._poll_interval_s = poll_interval_s
        self._server: str | None = None
        self._lock = threading.Lock()
        self._last = SyncResult(state="unknown", offset_s=None, server=None, measured_at=None)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the polling thread. Idempotent."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="clock-sync-monitor", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def server(self) -> str | None:
        return self._server

    def set_server(self, host: str | None) -> None:
        """Set or clear the probe target. Wakes the loop for an immediate probe."""
        self._server = host
        if host is None:
            with self._lock:
                self._last = SyncResult(
                    state="unknown", offset_s=None, server=None, measured_at=None
                )
        self._wake.set()

    def get(self) -> SyncResult:
        with self._lock:
            return self._last

    def _loop(self) -> None:
        while not self._stop.is_set():
            host = self._server
            if host is not None:
                offset = query_sntp(host)
                state = _classify(offset)
                with self._lock:
                    self._last = SyncResult(
                        state=state,
                        offset_s=offset,
                        server=host,
                        measured_at=time.monotonic(),
                    )
                if offset is None:
                    logger.info("clock_sync_probe_failed server=%s", host)
                else:
                    logger.debug(
                        "clock_sync_probe_ok server=%s offset_ms=%.1f state=%s",
                        host,
                        offset * 1000,
                        state,
                    )
            self._wake.wait(self._poll_interval_s)
            self._wake.clear()


_monitor: ClockSyncMonitor | None = None


def init_clock_sync_monitor() -> ClockSyncMonitor:
    global _monitor
    if _monitor is not None:
        return _monitor
    _monitor = ClockSyncMonitor()
    _monitor.start()
    return _monitor


def get_clock_sync_monitor() -> ClockSyncMonitor:
    if _monitor is None:
        raise RuntimeError("ClockSyncMonitor not initialized")
    return _monitor


def _ntp_offset_seconds() -> float:
    monitor = _monitor
    if monitor is None:
        return 0.0
    snap = monitor.get()
    if snap.offset_s is None or not snap.is_current(monitor.server):
        return 0.0
    return snap.offset_s


def now_utc_seconds() -> float:
    """NTP-corrected UTC seconds since the Unix epoch.

    Equivalent to ``now(UTC).timestamp()`` but skips the datetime allocation.
    """
    return time.time() + _ntp_offset_seconds()


def now(tz: tzinfo | None = None) -> datetime:
    """Current time, NTP-corrected when the SNTP monitor has a result.

    Falls through to system time when the monitor is uninitialized, has no
    probe target, or hasn't produced a valid result yet. Always returns a
    tz-aware datetime.

    tz=None → system local timezone (mirrors `datetime.now().astimezone()`).
    tz=UTC  → UTC.
    tz=ZoneInfo("...") → that zone.
    """
    aware = datetime.fromtimestamp(now_utc_seconds(), UTC)
    return aware.astimezone() if tz is None else aware.astimezone(tz)
