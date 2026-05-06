import fcntl
import struct
import termios
from dataclasses import dataclass


@dataclass(frozen=True)
class TTYWindowSpec:
    rows: int
    cols: int
    width_px: int
    height_px: int
    cell_width_px: int
    cell_height_px: int


def tty_window_spec(fd: int = 1) -> TTYWindowSpec | None:
    """Query TTY window dimensions and cell pixel size via TIOCGWINSZ, or None on failure."""
    # TODO: this doesn't work on all terminals, we will need some fallback.
    try:
        buf = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)
        rows, cols, width_px, height_px = struct.unpack("HHHH", buf)
        if cols > 0 and rows > 0 and width_px > 0 and height_px > 0:
            return TTYWindowSpec(
                rows=rows,
                cols=cols,
                width_px=width_px,
                height_px=height_px,
                cell_width_px=width_px // cols,
                cell_height_px=height_px // rows,
            )
    except OSError:
        pass
    return None
