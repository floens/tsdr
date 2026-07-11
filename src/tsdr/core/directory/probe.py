"""Live reachability probes for public receivers.

Directory listings go stale, so a receiver marked online is often firewalled or
offline by the time you try it. These probes check a single receiver *now*:
- SpyServer: a plain TCP connect to its streaming port (the daemon accepts
  connections only while live).
- KiwiSDR: an unauthenticated GET of `http://host:port/status`, which reports the
  Kiwi's current `status`/`offline`/`users` (see local/KIWISDR_PROTOCOL.md §10.1).
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from tsdr.core.directory.model import PublicDevice, as_int
from tsdr.core.http import HttpError, http_get

_DEFAULT_TIMEOUT = 1.5


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a live probe. `reachable` is transport success (TCP connected or
    HTTP 200). `rtt_ms` is set for SpyServer; `users`/`users_max`/`active` for a
    KiwiSDR `/status` reply (`active` is None for SpyServer, which has no such flag)."""

    reachable: bool
    rtt_ms: float | None = None
    users: int | None = None
    users_max: int | None = None
    active: bool | None = None


def tcp_probe(host: str, port: int, timeout: float) -> float | None:
    """TCP-connect to host:port. Return the round-trip time in ms, or None if the
    connection fails (refused, filtered, timed out, unresolvable)."""
    try:
        t0 = time.monotonic()
        with socket.create_connection((host, port), timeout=timeout):
            return (time.monotonic() - t0) * 1000
    except (TimeoutError, OSError):
        return None


def parse_kiwi_status(body: bytes) -> ProbeResult:
    """Parse a KiwiSDR `/status` body (plain `key=value` lines) into a ProbeResult."""
    fields: dict[str, str] = {}
    for line in body.decode("utf-8", "replace").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    users = as_int(fields.get("users"))
    users_max = as_int(fields.get("users_max"))
    active = fields.get("status") == "active" and fields.get("offline", "no").lower() != "yes"
    return ProbeResult(reachable=True, users=users, users_max=users_max, active=active)


def kiwi_probe(status_url: str, timeout: float) -> ProbeResult:
    """GET a KiwiSDR `/status` endpoint; unreachable (any transport error) → down."""
    try:
        body = http_get(status_url, timeout=timeout, max_bytes=64_000)
    except HttpError:
        return ProbeResult(reachable=False)
    return parse_kiwi_status(body)


def probe_device(device: PublicDevice, *, timeout: float = _DEFAULT_TIMEOUT) -> ProbeResult:
    """Probe a receiver's live reachability, dispatching on its source protocol."""
    if device.source == "kiwisdr":
        base = (device.url or f"http://{device.host}:{device.port or 8073}").rstrip("/")
        return kiwi_probe(f"{base}/status", timeout)
    if device.port is None:
        return ProbeResult(reachable=False)
    rtt = tcp_probe(device.host, device.port, timeout)
    return ProbeResult(reachable=rtt is not None, rtt_ms=rtt)
