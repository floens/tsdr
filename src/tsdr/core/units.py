from __future__ import annotations

from math import ceil, log10

_SUFFIXES: tuple[tuple[str, float], ...] = (
    ("ghz", 1e9),
    ("mhz", 1e6),
    ("khz", 1e3),
    ("hz", 1.0),
    ("g", 1e9),
    ("m", 1e6),
    ("k", 1e3),
)


def parse_hz(s: str) -> int:
    """Parse a frequency literal to Hz.

    Accepts forms like '100M', '100.1M', '100MHz', '250k', '250kHz', '1.5G',
    or plain integer/float strings. Underscores and commas in digits are ignored.
    Case-insensitive. Raises ValueError on invalid input.
    """
    t = s.strip().lower().replace("_", "").replace(",", "")
    if not t:
        raise ValueError(f"invalid frequency: {s!r}")
    for suffix, mul in _SUFFIXES:
        if t.endswith(suffix):
            num = t[: -len(suffix)]
            if not num:
                raise ValueError(f"invalid frequency: {s!r}")
            return int(float(num) * mul)
    return int(float(t))


def format_hz(
    hz: float,
    *,
    decimals: int | None = None,
    interval: float | None = None,
    long_suffix: bool = False,
) -> str:
    """Format frequency with an auto-picked SI suffix.

    `decimals` forces a fixed precision and strips trailing zeros.
    `interval` (mutually exclusive with `decimals`) picks just enough precision
    for adjacent values spaced by `interval` to render distinctly. `long_suffix`
    selects 'MHz'/'kHz' over 'M'/'k'.
    """
    if hz >= 1e9:
        divisor, short, long_ = 1e9, "G", "GHz"
    elif hz >= 1e6:
        divisor, short, long_ = 1e6, "M", "MHz"
    elif hz >= 1e3:
        divisor, short, long_ = 1e3, "k", "kHz"
    else:
        divisor, short, long_ = 1.0, "", "Hz"

    suffix = long_ if long_suffix else short
    sep = " " if long_suffix and suffix else ""

    if interval is not None and interval > 0:
        ratio = interval / divisor
        d = max(0, ceil(-log10(ratio))) if ratio < 1 else 0
        return f"{hz / divisor:.{d}f}{sep}{suffix}"

    if decimals is not None:
        s = f"{hz / divisor:.{decimals}f}".rstrip("0").rstrip(".")
        return f"{s}{sep}{suffix}"

    d = 1 if divisor >= 1e6 else 0
    return f"{hz / divisor:.{d}f}{sep}{suffix}"
