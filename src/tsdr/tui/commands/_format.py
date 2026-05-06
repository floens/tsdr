"""Rich markup helpers shared by all commands.

Conventions used across the command surface:
    bold cyan   device IDs, primary identifiers
    cyan        frequencies (Hz/kHz/MHz)
    yellow      sample rate, gain, dB values, bandwidth
    green       on/running, success markers, demod modes
    red         errors
    dim         labels, inactive items
    bold        table headers, primary names
"""

from collections.abc import Mapping

from rich.markup import escape


def safe(msg: str) -> str:
    """Escape Rich markup metacharacters. Use for untrusted strings (exception text, file paths)."""
    return str(escape(msg))


def success(msg: str) -> str:
    return f"[green]✓[/] {msg}"


def error(msg: str) -> str:
    return f"[red]Error:[/] {msg}"


def field(label: str, value: str) -> str:
    return f"[dim]{label}=[/]{value}"


def fields(items: Mapping[str, str]) -> str:
    return ", ".join(field(k, v) for k, v in items.items())


def device_id(did: str) -> str:
    return f"[bold cyan]{did}[/]"


def freq_mhz(hz: float, *, precision: int = 3) -> str:
    return f"[cyan]{hz / 1e6:.{precision}f} MHz[/]"


def rate_msps(hz: float, *, precision: int = 2) -> str:
    return f"[yellow]{hz / 1e6:.{precision}f} Msps[/]"


def db(value: float) -> str:
    return f"[yellow]{value:+.1f} dB[/]"


def header(title: str) -> str:
    return f"[bold]{title}[/]:"


_STATE_COLORS = {
    "on": "green",
    "running": "green",
    "off": "dim",
    "stopped": "dim",
    "error": "red",
}


def state(value: str) -> str:
    color = _STATE_COLORS.get(value.lower())
    if color is None:
        return value
    return f"[{color}]{value}[/]"
