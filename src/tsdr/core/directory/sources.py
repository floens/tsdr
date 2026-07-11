from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Annotated, Any
from urllib.parse import urlparse

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from tsdr.core.directory.model import PublicDevice, Source, as_float, as_int, as_str, at_capacity
from tsdr.core.http import HttpError, get_capped, http_get, make_client

logger = logging.getLogger(__name__)

SPYSERVER_URL = "https://airspy.com/directory/status.json"
KIWISDR_URL = "http://kiwisdr.com/public/"

LenientInt = Annotated[int | None, BeforeValidator(as_int)]
LenientFloat = Annotated[float | None, BeforeValidator(as_float)]
LenientStr = Annotated[str | None, BeforeValidator(as_str)]


class DirectoryError(Exception):
    """A directory response could not be parsed into any receivers."""


class PublicDeviceSource(ABC):
    """A public directory of remote receivers, fetched over plain HTTP and mapped
    to the shared PublicDevice model. `fetch()` raises `HttpError` on a transport
    failure and `DirectoryError` when the response yields no receivers; a single
    malformed entry is skipped, not fatal."""

    name: Source
    url: str

    def fetch(self) -> list[PublicDevice]:
        return self.parse(self._get())

    def _get(self) -> bytes:
        return http_get(self.url)

    @abstractmethod
    def parse(self, raw: bytes) -> list[PublicDevice]: ...


def _parse_entries(
    source: str, entries: list[Any], build: Callable[[Any], PublicDevice | None]
) -> list[PublicDevice]:
    """Build a device per entry, skipping (and logging) any that fail validation."""
    devices: list[PublicDevice] = []
    for entry in entries:
        try:
            device = build(entry)
        except ValidationError as e:
            logger.debug("directory_entry_skipped source=%s error=%r", source, e)
            continue
        if device is not None:
            devices.append(device)
    return devices


# --- SpyServer -------------------------------------------------------------

_PLACEHOLDER_NAMES = {"no description", "anonymous"}


class _AntennaLocation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lat: LenientFloat = None
    lon: LenientFloat = Field(default=None, alias="long")


class _SpyServerRaw(BaseModel):
    """A SpyServer directory entry, validated straight off the JSON via aliases."""

    model_config = ConfigDict(extra="ignore")

    host: str = Field(alias="streamingHost", min_length=1)
    port: Annotated[int, BeforeValidator(as_int)] = Field(alias="streamingPort")
    online: bool = False
    users: LenientInt = Field(default=None, alias="currentClientCount")
    users_max: LenientInt = Field(default=None, alias="maxClients")
    full_control: bool = Field(default=True, alias="fullControlAllowed")
    device_type: LenientStr = Field(default=None, alias="deviceType")
    general_description: LenientStr = Field(default=None, alias="generalDescription")
    owner_name: LenientStr = Field(default=None, alias="ownerName")
    location: _AntennaLocation | None = Field(default=None, alias="antennaLocation")
    freq_min: LenientFloat = Field(default=None, alias="minimumFrequency")
    freq_max: LenientFloat = Field(default=None, alias="maximumFrequency")
    sample_rate: LenientFloat = Field(default=None, alias="maximumIQSampleRate")
    bandwidth: LenientFloat = Field(default=None, alias="maximumStreamedBandwidth")
    last_seen: LenientInt = Field(default=None, alias="lastSeen")


class SpyServerDirectory(PublicDeviceSource):
    name = "spyserver"
    url = SPYSERVER_URL

    def parse(self, raw: bytes) -> list[PublicDevice]:
        try:
            doc = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            raise DirectoryError(f"spyserver: invalid JSON ({e})") from e
        servers = doc.get("servers") if isinstance(doc, dict) else None
        if not isinstance(servers, list):
            raise DirectoryError("spyserver: response missing a 'servers' list")
        devices = _parse_entries("spyserver", servers, _spyserver_device)
        if servers and not devices:
            raise DirectoryError("spyserver: no valid receivers in response")
        return devices


def _spyserver_name(raw: _SpyServerRaw) -> str:
    for value in (raw.general_description, raw.owner_name, raw.device_type):
        if value and value.casefold() not in _PLACEHOLDER_NAMES:
            return value
    return raw.host


def _spyserver_verdict(
    online: bool, users: int | None, users_max: int | None, full_control: bool
) -> tuple[bool, str]:
    if not online:
        return False, "offline"
    if at_capacity(users, users_max):
        return False, f"full {users}/{users_max}"
    return True, "" if full_control else "view-only"


def _spyserver_device(entry: Any) -> PublicDevice:
    raw = _SpyServerRaw.model_validate(entry)
    usable, reason = _spyserver_verdict(raw.online, raw.users, raw.users_max, raw.full_control)
    return PublicDevice(
        source="spyserver",
        id=f"spyserver:{raw.host}:{raw.port}",
        name=_spyserver_name(raw),
        host=raw.host,
        port=raw.port,
        url=f"sdr://{raw.host}:{raw.port}",
        lat=raw.location.lat if raw.location else None,
        lon=raw.location.lon if raw.location else None,
        device_hw=raw.device_type,
        freq_min=raw.freq_min,
        freq_max=raw.freq_max,
        sample_rate=raw.sample_rate,
        bandwidth=raw.bandwidth,
        users=raw.users,
        users_max=raw.users_max,
        online=raw.online,
        last_seen=raw.last_seen,
        usable=usable,
        usable_reason=reason,
    )


# --- KiwiSDR ---------------------------------------------------------------

# The public list embeds each receiver's /status fields as `<!-- key=value -->`
# comments. That key=value format has been append-only in the Kiwi firmware
# since ~2016, so we read known keys and ignore the rest.
_COMMENT_RE = re.compile(r"<!--\s*(\w+)=(.*?)\s*-->", re.DOTALL)
_HREF_RE = re.compile(r"href=['\"]([^'\"]+)['\"]")
_GPS_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
_RX_RE = re.compile(r"rx(\d+)")
_HW_SEP = "⁣"  # invisible separator between "KiwiSDR N vN.N" and the badge glyphs
# kiwisdr.com/public serves the list directly, but under rapid repeat requests it
# returns a click-through captcha stub that unlocks via an x-kiwi-auth token
# embedded in the stub's JS. We re-request with that header, then reload.
_KIWI_TOKEN_RE = re.compile(rb"""x-kiwi-auth['"]\s*,\s*['"]([0-9a-f]{16,})['"]""")


class _KiwiRaw(BaseModel):
    """The `key=value` comment fields of a KiwiSDR listing entry. Values arrive as
    strings; only users/users_max are coerced here, the rest are parsed downstream."""

    model_config = ConfigDict(extra="ignore")

    name: LenientStr = None
    status: str | None = None
    offline: str | None = None
    users: LenientInt = None
    users_max: LenientInt = None
    ant_connected: str | None = None
    bands: str | None = None
    gps: str | None = None
    loc: LenientStr = None
    grid: LenientStr = None
    sdr_hw: str | None = None
    mode: str | None = None
    snr: str | None = None


class KiwiSDRDirectory(PublicDeviceSource):
    name = "kiwisdr"
    url = KIWISDR_URL

    def _get(self) -> bytes:
        with make_client() as client:
            raw = get_capped(client, self.url)
            if b"cl-entry" in raw:
                return raw
            token = _KIWI_TOKEN_RE.search(raw)
            if token is None:
                raise HttpError("kiwisdr: click-through page returned no auth token")
            get_capped(client, self.url, headers={"x-kiwi-auth": token.group(1).decode()})
            return get_capped(client, self.url)

    def parse(self, raw: bytes) -> list[PublicDevice]:
        text = raw.decode("utf-8", errors="replace")
        chunks = text.split("<div class='cl-entry")[1:]
        devices = _parse_entries("kiwisdr", chunks, _kiwi_device)
        if chunks and not devices:
            raise DirectoryError("kiwisdr: no valid receivers in response")
        return devices


def _kiwi_bands(value: str | None) -> tuple[float | None, float | None]:
    if not value:
        return None, None
    parts = value.split("-")
    if len(parts) != 2:
        return None, None
    return as_float(parts[0]), as_float(parts[1])


def _kiwi_gps(value: str | None) -> tuple[float | None, float | None]:
    if not value:
        return None, None
    match = _GPS_RE.search(value)
    if match is None:
        return None, None
    return as_float(match.group(1)), as_float(match.group(2))


def _kiwi_snr(value: str | None) -> int | None:
    if not value:
        return None
    for part in value.split(","):
        snr = as_int(part.strip())
        if snr is not None:
            return snr
    return None


def _kiwi_channels(mode: str | None) -> int | None:
    if not mode:
        return None
    match = _RX_RE.search(mode)
    return as_int(match.group(1)) if match else None


def _kiwi_hw(value: str | None) -> str | None:
    if not value:
        return None
    return value.split(_HW_SEP)[0].strip() or None


def _kiwi_verdict(
    online: bool, users: int | None, users_max: int | None, ant_ok: bool
) -> tuple[bool, str]:
    if not online:
        return False, "offline"
    if at_capacity(users, users_max):
        return False, f"full {users}/{users_max}"
    if not ant_ok:
        return False, "no antenna"
    return True, ""


def _kiwi_device(chunk: str) -> PublicDevice | None:
    href = _HREF_RE.search(chunk)
    if href is None:
        return None
    url = href.group(1).strip()
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return None

    fields = {m.group(1): m.group(2).strip() for m in _COMMENT_RE.finditer(chunk)}
    raw = _KiwiRaw.model_validate(fields)
    offline = (raw.offline or "").casefold() == "yes"
    online = (not offline) and raw.status == "active"
    ant_ok = raw.ant_connected != "0"
    freq_min, freq_max = _kiwi_bands(raw.bands)
    lat, lon = _kiwi_gps(raw.gps)
    usable, reason = _kiwi_verdict(online, raw.users, raw.users_max, ant_ok)
    return PublicDevice(
        source="kiwisdr",
        id=f"kiwisdr:{url}",
        name=raw.name or url,
        host=host,
        port=parsed.port,
        url=url,
        lat=lat,
        lon=lon,
        location=raw.loc,
        grid=raw.grid,
        device_hw=_kiwi_hw(raw.sdr_hw),
        freq_min=freq_min,
        freq_max=freq_max,
        channels=_kiwi_channels(raw.mode),
        users=raw.users,
        users_max=raw.users_max,
        online=online,
        snr=_kiwi_snr(raw.snr),
        usable=usable,
        usable_reason=reason,
    )


ALL_SOURCES: tuple[PublicDeviceSource, ...] = (SpyServerDirectory(), KiwiSDRDirectory())
