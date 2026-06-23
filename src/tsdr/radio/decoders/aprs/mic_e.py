"""Mic-E position decode.

The position is split across the AX.25 destination callsign (latitude digits +
N/S, longitude offset, E/W) and the binary info field (longitude degrees/minutes,
speed/course, symbol). Offsets cross-checked against direwolf
`decode_aprs.c:aprs_mic_e`. Returns None (-> raw fallback) on any range failure
rather than emit garbage coordinates.
"""

from __future__ import annotations

_FEET_PER_METER = 3.28084


def _digit(c: str) -> int | None:
    """Mic-E destination char -> latitude digit (0-9), or None if invalid."""
    if "0" <= c <= "9":
        return ord(c) - ord("0")
    if "A" <= c <= "J":  # custom message bit
        return ord(c) - ord("A")
    if "P" <= c <= "Y":  # standard message bit
        return ord(c) - ord("P")
    if c in ("K", "L", "Z"):  # ambiguity / space -> 0
        return 0
    return None


def _is_south_or_zero(c: str) -> bool:
    return ("0" <= c <= "9") or c == "L"


def _b91_3(s: str) -> int:
    return (ord(s[0]) - 33) * 8281 + (ord(s[1]) - 33) * 91 + (ord(s[2]) - 33)


def decode(dest: str, info: str) -> dict[str, object] | None:
    call = dest.partition("-")[0]
    if len(call) != 6 or len(info) < 9:
        return None

    digits: list[int] = []
    for c in call:
        d = _digit(c)
        if d is None:
            return None
        digits.append(d)
    d0, d1, d2, d3, d4, d5 = digits
    lat = (d0 * 10 + d1) + (d2 * 1000 + d3 * 100 + d4 * 10 + d5) / 6000.0
    if _is_south_or_zero(call[3]):
        lat = -lat

    offset = 1 if "P" <= call[4] <= "Z" else 0
    lon = _decode_longitude(offset, ord(info[1]), ord(info[2]), ord(info[3]))
    if lon is None:
        return None
    if "P" <= call[5] <= "Z":  # West
        lon = -lon

    speed, course = _decode_speed_course(ord(info[4]), ord(info[5]), ord(info[6]))
    result: dict[str, object] = {
        "latitude": lat,
        "longitude": lon,
        "symbol": info[8] + info[7],  # table id + symbol code
        "speed": speed,
        "course": course,
    }

    comment = info[9:]
    if len(comment) >= 4 and comment[3] == "}":
        result["altitude"] = (_b91_3(comment[:3]) - 10000) * _FEET_PER_METER
        comment = comment[4:]
    result["comment"] = comment.strip() or None
    return result


def _decode_longitude(offset: int, b0: int, b1: int, b2: int) -> float | None:
    if offset and 118 <= b0 <= 127:
        deg = b0 - 118
    elif not offset and 38 <= b0 <= 127:
        deg = (b0 - 38) + 10
    elif offset and 108 <= b0 <= 117:
        deg = (b0 - 108) + 100
    elif offset and 38 <= b0 <= 107:
        deg = (b0 - 38) + 110
    else:
        return None

    if 88 <= b1 <= 97:
        minutes = b1 - 88
    elif 38 <= b1 <= 87:
        minutes = (b1 - 38) + 10
    else:
        return None

    if not 28 <= b2 <= 127:
        return None
    return deg + (minutes + (b2 - 28) / 100.0) / 60.0


def _decode_speed_course(b0: int, b1: int, b2: int) -> tuple[float, int | None]:
    speed = (b0 - 28) * 10 + (b1 - 28) // 10
    if speed >= 800:
        speed -= 800
    course = ((b1 - 28) % 10) * 100 + (b2 - 28)
    if course >= 400:
        course -= 400
    if course == 0:
        return float(speed), None  # 0 = unknown course
    return float(speed), 0 if course == 360 else course
