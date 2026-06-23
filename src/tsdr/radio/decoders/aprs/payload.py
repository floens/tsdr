"""APRS application-layer payload parsing.

Dispatch on the data-type indicator (first info byte): position (`!=/@`),
Mic-E (`` ` ``/`'`), message (`:`), status (`>`), third-party (`}`, recursed).
Everything else keeps `raw_info` only. Structured fields land in `APRSPacket`,
which the decoder attaches to `DecodedMessage.data` for a future map.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from tsdr.radio.decoders.aprs import mic_e
from tsdr.radio.decoders.aprs.ax25 import AX25Frame

_POS = re.compile(
    r"(\d{2})([0-7 ][0-9 ]\.[0-9 ]{2})([NS])(.)(\d{3})([0-7 ][0-9 ]\.[0-9 ]{2})([EW])(.)(.*)",
    re.DOTALL,
)
_CSE_SPD = re.compile(r"^(\d{3})/(\d{3})")
_ALT = re.compile(r"/A=(-?\d{6})")
_MIC_E_TYPES = ("`", "'", "\x1c", "\x1d")


@dataclass(frozen=True)
class APRSPacket:
    source: str
    dest: str
    digis: tuple[str, ...]
    info_type: str
    raw_info: str
    latitude: float | None = None
    longitude: float | None = None
    symbol: str | None = None  # symbol-table char + symbol code, e.g. "/-"
    course: int | None = None  # degrees
    speed: float | None = None  # knots
    altitude: float | None = None  # feet
    comment: str | None = None
    timestamp: str | None = None  # raw APRS timestamp field, if present
    addressee: str | None = None
    message_text: str | None = None
    status: str | None = None
    third_party: APRSPacket | None = None


def parse(frame: AX25Frame) -> APRSPacket:
    digis = tuple(str(d) + ("*" if d.has_been_repeated else "") for d in frame.digis)
    return _parse_info(
        str(frame.src), str(frame.dest), digis, frame.info.decode("latin1", "replace")
    )


def _parse_info(source: str, dest: str, digis: tuple[str, ...], info: str) -> APRSPacket:
    base = APRSPacket(source=source, dest=dest, digis=digis, info_type="other", raw_info=info)
    if not info:
        return base

    dti = info[0]
    if dti in ("!", "=", "/", "@"):
        return _parse_position(base, info)
    if dti in _MIC_E_TYPES:
        return _parse_mic_e(base, dest, info)
    if dti == ":":
        return _parse_message(base, info)
    if dti == ">":
        return replace(base, info_type="status", status=info[1:].strip())
    if dti == "}":
        return _parse_third_party(base, info)
    return base


def _f(num: str, dot_minutes: str) -> float:
    """Degrees + decimal minutes -> decimal degrees. Ambiguity spaces -> 0."""
    return int(num) + float(dot_minutes.replace(" ", "0")) / 60.0


def _parse_position(base: APRSPacket, info: str) -> APRSPacket:
    body = info[1:]
    timestamp = None
    if info[0] in ("/", "@"):
        timestamp, body = body[:7], body[7:]
    m = _POS.match(body)
    if not m:
        return replace(base, info_type="position", timestamp=timestamp)
    lat = _f(m.group(1), m.group(2)) * (1 if m.group(3) == "N" else -1)
    lon = _f(m.group(5), m.group(6)) * (1 if m.group(7) == "E" else -1)
    symbol = m.group(4) + m.group(8)
    comment = m.group(9)

    course = speed = altitude = None
    cse = _CSE_SPD.match(comment)
    if cse:
        course, speed, comment = int(cse.group(1)), float(cse.group(2)), comment[7:]
    alt = _ALT.search(comment)
    if alt:
        altitude = float(alt.group(1))
    return replace(
        base,
        info_type="position",
        latitude=lat,
        longitude=lon,
        symbol=symbol,
        course=course,
        speed=speed,
        altitude=altitude,
        comment=comment.strip() or None,
        timestamp=timestamp,
    )


def _parse_mic_e(base: APRSPacket, dest: str, info: str) -> APRSPacket:
    decoded = mic_e.decode(dest, info)
    if decoded is None:
        return replace(base, info_type="mic_e")
    return replace(base, info_type="mic_e", **decoded)  # type: ignore[arg-type]


def _parse_message(base: APRSPacket, info: str) -> APRSPacket:
    # ":ADDRESSEE:message{seq"  (addressee is 9 chars, space-padded)
    if len(info) < 11 or info[10] != ":":
        return replace(base, info_type="message")
    addressee = info[1:10].strip()
    text = info[11:]
    return replace(base, info_type="message", addressee=addressee, message_text=text)


def _parse_third_party(base: APRSPacket, info: str) -> APRSPacket:
    inner = _parse_tnc2(info[1:])
    if inner is None:
        return replace(base, info_type="third_party")
    return replace(base, info_type="third_party", third_party=inner)


def _parse_tnc2(text: str) -> APRSPacket | None:
    """Parse a `SRC>DEST,path:info` third-party string and recurse the payload."""
    header, sep, info = text.partition(":")
    if not sep or ">" not in header:
        return None
    src, _, rest = header.partition(">")
    parts = rest.split(",")
    dest = parts[0]
    digis = tuple(parts[1:])
    return _parse_info(src, dest, digis, info)
