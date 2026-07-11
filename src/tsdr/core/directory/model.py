from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Source = Literal["spyserver", "kiwisdr"]


class PublicDevice(BaseModel):
    """A remote receiver listed in a public directory (SpyServer or KiwiSDR).

    The shared datamodel both directory sources produce. Fields absent for a
    given source stay None (e.g. sample_rate/bandwidth for KiwiSDR, snr for
    SpyServer). `usable`/`usable_reason` are the source-computed verdict on
    whether the receiver can be used right now.
    """

    model_config = ConfigDict(frozen=True)

    source: Source
    id: str
    name: str
    host: str
    port: int | None = None
    url: str | None = None

    lat: float | None = None
    lon: float | None = None
    location: str | None = None
    grid: str | None = None

    device_hw: str | None = None
    freq_min: float | None = None
    freq_max: float | None = None
    sample_rate: float | None = None
    bandwidth: float | None = None
    channels: int | None = None

    users: int | None = None
    users_max: int | None = None
    online: bool = False
    snr: int | None = None
    last_seen: int | None = None

    usable: bool = False
    usable_reason: str = ""


def at_capacity(users: int | None, users_max: int | None) -> bool:
    """A receiver is full: it reports both a current and a max user count and has
    reached it. The one rule for "full", shared by the source verdicts and display."""
    return users is not None and users_max is not None and users >= users_max


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
