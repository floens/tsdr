from __future__ import annotations

from tsdr.core.directory.geo import country_code
from tsdr.core.directory.model import PublicDevice, Source, at_capacity
from tsdr.core.directory.probe import ProbeResult
from tsdr.core.units import format_hz

SOURCE_LABEL: dict[Source, str] = {"spyserver": "spy", "kiwisdr": "kiwi"}
SOURCE_COLOR: dict[Source, str] = {"spyserver": "blue", "kiwisdr": "green"}
SOURCE_ORDER: dict[Source, int] = {"spyserver": 0, "kiwisdr": 1}


def source_label(source: Source) -> str:
    return SOURCE_LABEL[source]


def source_color(source: Source) -> str:
    return SOURCE_COLOR[source]


def source_order(source: Source) -> int:
    """Sort rank — SpyServer before KiwiSDR."""
    return SOURCE_ORDER[source]


def default_sort_key(d: PublicDevice) -> tuple[bool, int, str]:
    """Usable receivers first, then SpyServer before KiwiSDR, then by name."""
    return not d.usable, source_order(d.source), d.name.lower()


def is_full(d: PublicDevice) -> bool:
    return at_capacity(d.users, d.users_max)


def _freq(hz: float) -> str:
    if hz >= 1e9:
        return f"{hz / 1e9:.1f}G".replace(".0G", "G")
    if hz >= 1e6:
        return f"{hz / 1e6:.0f}M"
    if hz >= 1e3:
        return f"{hz / 1e3:.0f}k"
    return f"{hz:.0f}"


def range_label(d: PublicDevice) -> str:
    """Tuning coverage as a compact `lo-hi` (e.g. `0-30M`, `24M-1.8G`)."""
    if d.freq_min is None or d.freq_max is None:
        return "-"
    return f"{_freq(d.freq_min):>4}-{_freq(d.freq_max)}"


def snr_color(snr: int) -> str:
    """Signal-strength colour: readable blue (good), yellow (fair), red (poor)."""
    if snr >= 20:
        return "#4c9aff"
    if snr >= 10:
        return "yellow"
    return "red"


def country_first_location(location: str | None) -> str:
    """Reorder a comma-separated location so the last segment (usually the country)
    comes first, the most important part to keep when the field is truncated."""
    if not location:
        return ""
    parts = [p.strip() for p in location.split(",") if p.strip()]
    if len(parts) <= 1:
        return location.strip()
    return ", ".join([parts[-1], *parts[:-1]])


def location_display(d: PublicDevice) -> str:
    """Best short location: KiwiSDR's country-first text, else lat/lon coordinates
    (SpyServer has no text location, only `antennaLocation`)."""
    if d.location:
        return country_first_location(d.location)
    if d.lat is not None and d.lon is not None:
        return country_code(d.lat, d.lon) or f"{d.lat:.1f}, {d.lon:.1f}"
    return ""


def bandwidth_hz(d: PublicDevice) -> float | None:
    """Best available bandwidth figure: streamed bandwidth, else IQ sample rate."""
    return d.bandwidth if d.bandwidth is not None else d.sample_rate


def bw_label(hz: float | None) -> str:
    return "- BW" if hz is None else format_hz(hz)


def users_label(d: PublicDevice) -> str:
    if d.users is None or d.users_max is None:
        return "N/A"
    return f"{d.users}/{d.users_max}"


def status_text(d: PublicDevice) -> tuple[str, str]:
    """(label, color) for the receiver's usability verdict."""
    if d.usable:
        return d.usable_reason or "ok", "yellow" if d.usable_reason else "green"
    return d.usable_reason or "unusable", "red"


def probe_label(result: ProbeResult | None, *, probing: bool) -> tuple[str, str]:
    """(text, color) for a row's live-reachability line: RTT for a reachable
    SpyServer, live user count for a KiwiSDR, else 'down'. Empty until probed."""
    if probing:
        return "checking…", "dim"
    if result is None:
        return "", "dim"
    if not result.reachable:
        return "down", "red"
    if result.rtt_ms is not None:
        color = "green" if result.rtt_ms < 200 else "yellow" if result.rtt_ms < 500 else "red"
        return f"{result.rtt_ms:.0f} ms", color
    if result.active is False:
        return "offline", "red"
    users = f"{result.users}/{result.users_max}" if result.users_max is not None else "up"
    if at_capacity(result.users, result.users_max):
        return f"full {users}", "yellow"
    return f"up {users}", "green"
