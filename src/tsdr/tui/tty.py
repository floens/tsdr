import ctypes
import logging
import os
import re
import struct
import sys
import time
from base64 import standard_b64encode
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum
from multiprocessing.shared_memory import SharedMemory
from typing import Literal

from textual._xterm_parser import XTermParser
from textual.message import Message

# POSIX terminal control (fcntl/termios/tty/select) and Textual's Linux driver
# exist only on Unix; on Windows the Win32 console API (ctypes) and the win32
# driver stand in. Import each platform's pieces conditionally so neither needs
# stubbing — the APC-aware parser patch below targets whichever driver is live.
if sys.platform == "win32":
    import textual.drivers.win32 as _driver_mod
else:
    import fcntl
    import select
    import termios
    import tty

    import textual.drivers.linux_driver as _driver_mod

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTYWindowSpec:
    rows: int
    cols: int
    width_px: int
    height_px: int
    cell_width_px: int
    cell_height_px: int
    method: Literal["resize", "ioctl", "csi", "csi-cell"] = "ioctl"


def tty_window_spec(fd: int = 1) -> TTYWindowSpec | None:
    """Query TTY window dimensions and cell pixel size via TIOCGWINSZ, or None on failure.

    This is the live, re-queryable getter for window geometry; callers refresh it
    on resize rather than caching, so it never goes stale. It is intentionally not
    part of TerminalCapabilities (which holds only session-stable caps).
    """
    # TIOCGWINSZ is POSIX-only; Windows has no console pixel ioctl, so it stays None.
    if sys.platform != "win32":
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


def window_spec_from_resize(
    cols: int, rows: int, pixel_size: tuple[int, int] | None
) -> TTYWindowSpec | None:
    """Build a window spec from a live Textual Resize event's in-band pixel size.

    Where the terminal supports it, Textual reports terminal pixel dimensions on
    the Resize event (in-band, DEC mode 2048); that is the freshest cell-size
    source and tracks live resizes. Returns None when the event carries no pixels
    — notably Textual's Windows driver, which emits cell dimensions only — so
    callers fall back to the ioctl getter or the CSI startup snapshot.
    """
    if pixel_size is None:
        return None
    width_px, height_px = pixel_size
    if cols <= 0 or rows <= 0 or width_px <= 0 or height_px <= 0:
        return None
    return TTYWindowSpec(
        rows=rows,
        cols=cols,
        width_px=width_px,
        height_px=height_px,
        cell_width_px=width_px // cols,
        cell_height_px=height_px // rows,
        method="resize",
    )


# --- terminal capability detection ----------------------------------------
#
# One raw-tty round-trip at startup, parsed into an immutable
# TerminalCapabilities consumers can read. A primary device-attributes request
# (DA1) terminates the batch: every VT-style terminal answers it, so the read
# is bounded even when the terminal ignores the feature queries we care about.

# 1x1 transparent pixel, action=query. i=31 over direct (inline) transport.
_KITTY_DIRECT_QUERY = "\x1b_Gi=31,a=q,t=d,f=24,s=1,v=1;AAAA\x1b\\"
# DECRQM for synchronized output (mode 2026).
_SYNC_QUERY = "\x1b[?2026$p"
# Kitty keyboard protocol: query current progressive-enhancement flags.
_KBD_QUERY = "\x1b[?u"
# DECRQM for SGR mouse (mode 1006) and SGR-pixels mouse (mode 1016).
_SGR_MOUSE_QUERY = "\x1b[?1006$p"
_SGR_PIXEL_MOUSE_QUERY = "\x1b[?1016$p"
# Text-area size in pixels (fallback for terminals where TIOCGWINSZ reports 0 px).
_WINDOW_QUERY = "\x1b[14t"
# Cell size in pixels (CSI 16t). Needed because some terminals (e.g. Rio) answer
# 14t only *asynchronously* — after the DA1 sentinel, so the probe misses it —
# but answer 16t synchronously. On Windows there is no other source (no
# TIOCGWINSZ, and Textual's Resize event carries no pixel size), so 16t is the
# only way to learn the cell size there. It also gives cell px directly.
_CELL_QUERY = "\x1b[16t"
# Primary device attributes — guaranteed terminator.
_DA1 = "\x1b[c"

_DA1_RE = re.compile(rb"\x1b\[\?[0-9;]*c")
_SYNC_RE = re.compile(rb"\x1b\[\?2026;([0-9]+)\$y")
_SGR_MOUSE_RE = re.compile(rb"\x1b\[\?1006;([0-9]+)\$y")
_SGR_PIXEL_MOUSE_RE = re.compile(rb"\x1b\[\?1016;([0-9]+)\$y")
_KBD_RE = re.compile(rb"\x1b\[\?[0-9]+u")
_WINDOW_RE = re.compile(rb"\x1b\[4;([0-9]+);([0-9]+)t")  # CSI 4 ; height ; width t
_CELL_RE = re.compile(rb"\x1b\[6;([0-9]+);([0-9]+)t")  # CSI 6 ; cell_h ; cell_w t

# Some terminals (e.g. Rio) answer the 14t/16t window/cell-size queries from
# their event loop, slightly *after* the DA1 terminator. When DA1 has arrived
# but no pixel reply has, drain the input this much longer to catch the straggler.
_WINDOW_REPLY_GRACE = 0.1


def _has_pixel_reply(buf: bytes) -> bool:
    """True once a 14t (window) or 16t (cell) pixel reply is present in `buf`."""
    return _WINDOW_RE.search(buf) is not None or _CELL_RE.search(buf) is not None


class TerminalProduct(Enum):
    """Terminal emulator product, detected from the environment."""

    WEZTERM = "WezTerm"
    KITTY = "kitty"
    GHOSTTY = "Ghostty"
    ITERM2 = "iTerm2"
    WINDOWS_TERMINAL = "Windows Terminal"
    VSCODE = "Visual Studio Code"
    APPLE_TERMINAL = "Apple Terminal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Quirk:
    """A known issue that applies to the current terminal session.

    ``severity`` is a plain string ("warn" | "fail") so this module stays free of
    any ``doctor`` import; the doctor maps it onto its own ``Status``.
    """

    id: str  # stable snake_case; the doctor uses it as the check name
    summary: str  # human-readable, actionable message
    group: str  # doctor group: "render" | "protocol" | ...
    severity: str  # "warn" | "fail"


@dataclass(frozen=True)
class TerminalIdentity:
    """Unified terminal-emulator identity: product, version, and applicable quirks."""

    product: TerminalProduct
    name: str  # display label: product.value, else the raw TERM_PROGRAM/TERM string
    version: str | None  # raw version string when the terminal exposes one
    quirks: tuple[Quirk, ...]  # known issues that apply to this session


@dataclass(frozen=True)
class TerminalCapabilities:
    """Session-stable terminal capabilities, probed once at startup.

    ``queried`` records whether the raw-tty escape round-trip ran. When it is
    False (Windows, not a tty, no controlling terminal) the round-trip fields are
    at their defaults and only env-derived fields (``identity``, ``multiplexer``)
    are meaningful.
    """

    queried: bool
    identity: TerminalIdentity  # emulator product/version/quirks (env-derived)
    kitty_graphics: bool  # any positive graphics response
    kitty_transports: frozenset[str]  # subset of {"d", "s"}
    sync_output: bool | None  # mode 2026 (DECRPM); None if not reported
    sgr_mouse: bool | None  # mode 1006 (DECRPM); None if not reported
    sgr_pixel_mouse: bool | None  # mode 1016 (DECRPM); None if not reported
    kitty_keyboard: bool
    multiplexer: str | None  # "tmux" | "screen" | None (env-derived; always set)
    # CSI-14t window snapshot, set only as a startup fallback when TIOCGWINSZ
    # reports no pixels. The live geometry getter stays tty_window_spec().
    window_csi: TTYWindowSpec | None


def supports_kitty_images(caps: TerminalCapabilities | None) -> bool:
    """True if kitty images can actually render: graphics confirmed *and* the
    shared-memory transport (t=s) TSDR uploads with is available.

    The single source of truth for both the doctor's kitty demo gating and the
    ``kitty_transports`` check verdict, so the demo tab and the reported status can
    never disagree.
    """
    return caps is not None and caps.kitty_graphics and "s" in caps.kitty_transports


def parse_kitty_transports(buf: bytes) -> frozenset[str]:
    """Transmission mediums that returned OK: 'd' (direct), 's' (shared memory)."""
    found = set()
    if b"_Gi=31;OK" in buf:
        found.add("d")
    if b"_Gi=32;OK" in buf:
        found.add("s")
    return frozenset(found)


def parse_sync_output(buf: bytes) -> bool | None:
    """Synchronized-output (mode 2026) support, or None if not reported.

    DECRPM reports mode state in $y; value 0 means "not recognized".
    """
    m = _SYNC_RE.search(buf)
    if m is None:
        return None
    return m.group(1) != b"0"


def parse_sgr_mouse(buf: bytes) -> bool | None:
    """SGR mouse (mode 1006) support, or None if not reported.

    DECRPM reports mode state in $y; value 0 means "not recognized".
    """
    m = _SGR_MOUSE_RE.search(buf)
    if m is None:
        return None
    return m.group(1) != b"0"


def parse_sgr_pixel_mouse(buf: bytes) -> bool | None:
    """SGR-pixels mouse (mode 1016) support, or None if not reported."""
    m = _SGR_PIXEL_MOUSE_RE.search(buf)
    if m is None:
        return None
    return m.group(1) != b"0"


def parse_kitty_keyboard(buf: bytes) -> bool:
    """True if the terminal answered the kitty keyboard protocol query."""
    return _KBD_RE.search(buf) is not None


def parse_window_pixels(buf: bytes) -> tuple[int, int] | None:
    """(width_px, height_px) from a CSI 14t reply (CSI 4 ; H ; W t), or None."""
    m = _WINDOW_RE.search(buf)
    if m is None:
        return None
    return int(m.group(2)), int(m.group(1))


def parse_cell_pixels(buf: bytes) -> tuple[int, int] | None:
    """(cell_width_px, cell_height_px) from a CSI 16t reply (CSI 6 ; H ; W t), or None."""
    m = _CELL_RE.search(buf)
    if m is None:
        return None
    return int(m.group(2)), int(m.group(1))


def _csi_window_spec(
    *,
    win_px: tuple[int, int] | None = None,
    cell_px: tuple[int, int] | None = None,
    method: Literal["csi", "csi-cell"],
) -> TTYWindowSpec | None:
    """Build a window spec from a CSI reply and the current cell grid size.

    Exactly one of ``win_px`` (CSI-14t window pixels) or ``cell_px`` (CSI-16t cell
    pixels) is given; the missing dimension is derived from the grid.
    """
    try:
        size = os.get_terminal_size()
    except OSError:
        return None
    cols, rows = size.columns, size.lines
    if cols <= 0 or rows <= 0:
        return None
    if win_px is not None:
        width_px, height_px = win_px
        if width_px <= 0 or height_px <= 0:
            return None
        return TTYWindowSpec(
            rows, cols, width_px, height_px, width_px // cols, height_px // rows, method
        )
    if cell_px is None:
        return None
    cell_width_px, cell_height_px = cell_px
    if cell_width_px <= 0 or cell_height_px <= 0:
        return None
    return TTYWindowSpec(
        rows,
        cols,
        cell_width_px * cols,
        cell_height_px * rows,
        cell_width_px,
        cell_height_px,
        method,
    )


def _window_spec_from_csi(buf: bytes) -> TTYWindowSpec | None:
    """Build a window spec from the CSI-14t pixel reply plus the cell grid size."""
    px = parse_window_pixels(buf)
    if px is None:
        return None
    return _csi_window_spec(win_px=px, method="csi")


def _window_spec_from_cell_csi(buf: bytes) -> TTYWindowSpec | None:
    """Build a window spec from a CSI-16t cell-size reply plus the cell grid size.

    Used when 14t is unavailable (e.g. Rio answers 14t asynchronously, after the
    DA1 sentinel) but 16t is.
    """
    cell = parse_cell_pixels(buf)
    if cell is None:
        return None
    return _csi_window_spec(cell_px=cell, method="csi-cell")


# WezTerm's last tagged stable is 20240203 (Feb 2024); kitty image-protocol fixes
# (incl. the Windows breakage, wezterm/wezterm#5757) exist only in newer nightly
# builds. Flag any build older than this so users know to update.
_WEZTERM_RECENT_BUILD = date(2026, 1, 1)


def _parse_wezterm_build(version: str) -> date | None:
    """Parse the YYYYMMDD build date from a WezTerm version string, or None."""
    head = version.split("-", 1)[0]
    if len(head) == 8 and head.isdigit():
        try:
            return date(int(head[:4]), int(head[4:6]), int(head[6:8]))
        except ValueError:
            return None
    return None


def _identify_product() -> tuple[TerminalProduct, str]:
    """Resolve the emulator product and a display name from the environment.

    Prefers ``TERM_PROGRAM`` (the standard, per-process identity); ``WT_SESSION``
    is only a secondary signal because Windows Terminal leaks it into child
    processes (e.g. VS Code launched from WT still has it set).
    """
    program = os.environ.get("TERM_PROGRAM", "")
    term = os.environ.get("TERM", "")
    if program == "WezTerm" or "WEZTERM_PANE" in os.environ:
        return TerminalProduct.WEZTERM, TerminalProduct.WEZTERM.value
    if "KITTY_WINDOW_ID" in os.environ:
        return TerminalProduct.KITTY, TerminalProduct.KITTY.value
    if "GHOSTTY_RESOURCES_DIR" in os.environ:
        return TerminalProduct.GHOSTTY, TerminalProduct.GHOSTTY.value
    if program == "iTerm.app" or "ITERM_SESSION_ID" in os.environ:
        return TerminalProduct.ITERM2, TerminalProduct.ITERM2.value
    if program == "vscode":
        return TerminalProduct.VSCODE, TerminalProduct.VSCODE.value
    if program == "Apple_Terminal":
        return TerminalProduct.APPLE_TERMINAL, TerminalProduct.APPLE_TERMINAL.value
    if "WT_SESSION" in os.environ:
        return TerminalProduct.WINDOWS_TERMINAL, TerminalProduct.WINDOWS_TERMINAL.value
    return TerminalProduct.UNKNOWN, program or term or "unknown"


def _terminal_quirks(
    product: TerminalProduct, version: str | None, kitty_keyboard: bool | None
) -> tuple[Quirk, ...]:
    """Known issues that apply to this product/version/capability combination.

    The single declarative place for terminal-version/quirk rules. ``kitty_keyboard``
    is the probed protocol result (None when off-tty/unqueried); accepted for
    capability-derived rules even though no current quirk uses it yet.
    """
    quirks: list[Quirk] = []
    if product is TerminalProduct.APPLE_TERMINAL:
        quirks.append(
            Quirk(
                "apple_terminal_limited",
                "Apple Terminal is slow to render and lacks image graphics, "
                "and modern protocol support",
                "render",
                "warn",
            )
        )
    if product is TerminalProduct.WEZTERM:
        build = _parse_wezterm_build(version or "")
        if build is not None and build < _WEZTERM_RECENT_BUILD:
            quirks.append(
                Quirk(
                    "wezterm_old_kitty_graphics",
                    f"{version}: outdated; kitty image fixes are in newer WezTerm nightly builds",
                    "render",
                    "warn",
                )
            )
    return tuple(quirks)


def detect_terminal(*, kitty_keyboard: bool | None = None) -> TerminalIdentity:
    """Unified terminal-emulator identity from the environment.

    Env-derived, so it works without a tty and on every platform. ``kitty_keyboard``
    is the probed protocol result when available, used for capability-derived quirks.
    """
    product, name = _identify_product()
    # WezTerm and iTerm2 set TERM_PROGRAM_VERSION; Windows Terminal exposes no
    # version anywhere (env, XTVERSION, and DA2 all omit it).
    version = os.environ.get("TERM_PROGRAM_VERSION") or None
    return TerminalIdentity(
        product=product,
        name=name,
        version=version,
        quirks=_terminal_quirks(product, version, kitty_keyboard),
    )


def detect_multiplexer() -> str | None:
    """Terminal multiplexer from the environment: 'tmux', 'screen', or None.

    Env-derived, so it works without a tty and on every platform.
    """
    if os.environ.get("TMUX"):
        return "tmux"
    if os.environ.get("STY"):
        return "screen"
    return None


def _create_probe_shm(name: str) -> SharedMemory:
    try:
        return SharedMemory(create=True, name=name, size=4, track=False)
    except FileExistsError:
        stale = SharedMemory(name=name, track=False)
        stale.close()
        stale.unlink()
        return SharedMemory(create=True, name=name, size=4, track=False)


def _shm_transport_query() -> tuple[str, Callable[[], None]]:
    """Build the t=s (shared-memory) graphics query, or skip it if shm is unavailable.

    Mirrors KittyImageWidget's upload: a named shared-memory object holding one
    RGBA pixel, referenced by base64 name. Returns the query string and a cleanup
    callable (run only after the round-trip, so the terminal can read it first).
    """
    name = f"tsdr_probe_{os.getpid()}"
    try:
        shm = _create_probe_shm(name)
    except OSError:
        return "", lambda: None
    shm.buf[:4] = b"\x00\x00\x00\x00"  # type: ignore[index]
    b64 = standard_b64encode(shm_payload_name(name).encode()).decode()

    def cleanup() -> None:
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass  # kitty unlinks t=s objects after reading them (no-op on Windows)

    return f"\x1b_Gi=32,a=q,t=s,f=32,s=1,v=1;{b64}\x1b\\", cleanup


def shm_payload_name(name: str) -> str:
    """The shm object name as the terminal expects it in a t=s graphics payload.

    POSIX shm_open references carry a leading slash; Windows named mappings
    (OpenFileMappingW) take the bare name — matching how
    multiprocessing.shared_memory names the object on each platform. Shared by
    the capability probe and KittyImageWidget so both encode names identically.
    """
    return name if sys.platform == "win32" else f"/{name}"


def _query_fd() -> int | None:
    """The single tty fd to run a query round-trip on, or None when unavailable.

    POSIX only (the Win32 console path lives in ``_query_terminal_windows``): a
    redirected or absent tty (pipes, headless) has nothing to query. The tty
    device is bidirectional, so one fd serves both the raw-mode read and the
    escape write — callers fall back to non-interactive detection when None.
    """
    if sys.platform == "win32":
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    return sys.stdin.fileno()


if sys.platform == "win32":

    def _query_terminal_windows(batch: bytes, timeout: float) -> bytes | None:
        """Win32 console VT round-trip: send the query batch, read the reply to DA1.

        Mirrors the POSIX :func:`_query_terminal` via the console API. Opens
        ``CONIN$``/``CONOUT$`` directly so it works even when stdio is redirected,
        enables VT input/output, writes the queries, and reads the byte stream
        until the DA1 terminator or the timeout, restoring the original console
        modes afterwards. Returns None when no console is attached (pythonw, fully
        headless), so callers fall back to env detection.
        """
        k = ctypes.windll.kernel32
        handle_t = ctypes.c_void_p
        dword = ctypes.c_uint32
        lpdword = ctypes.POINTER(dword)
        k.CreateFileW.restype = handle_t
        k.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            dword,
            dword,
            ctypes.c_void_p,
            dword,
            dword,
            handle_t,
        ]
        k.GetConsoleMode.argtypes = [handle_t, lpdword]
        k.SetConsoleMode.argtypes = [handle_t, dword]
        k.FlushConsoleInputBuffer.argtypes = [handle_t]
        k.WriteFile.argtypes = [handle_t, ctypes.c_char_p, dword, lpdword, ctypes.c_void_p]
        k.ReadFile.argtypes = [handle_t, ctypes.c_void_p, dword, lpdword, ctypes.c_void_p]
        k.WaitForSingleObject.argtypes = [handle_t, dword]
        k.WaitForSingleObject.restype = dword
        k.CloseHandle.argtypes = [handle_t]

        generic_rw = 0x80000000 | 0x40000000
        share_rw = 0x1 | 0x2
        open_existing = 3
        invalid = handle_t(-1).value
        enable_vt_input = 0x0200
        enable_vt_output = 0x0004
        cooked_input = 0x0001 | 0x0002 | 0x0004  # processed | line | echo
        wait_object_0 = 0x0

        h_in = k.CreateFileW("CONIN$", generic_rw, share_rw, None, open_existing, 0, None)
        h_out = k.CreateFileW("CONOUT$", generic_rw, share_rw, None, open_existing, 0, None)
        if not h_in or not h_out or h_in == invalid or h_out == invalid:
            for h in (h_in, h_out):
                if h and h != invalid:
                    k.CloseHandle(h)
            logger.debug("win_console_open_failed reason=create_file")
            return None

        old_in, old_out = dword(), dword()
        if not k.GetConsoleMode(h_in, ctypes.byref(old_in)) or not k.GetConsoleMode(
            h_out, ctypes.byref(old_out)
        ):
            k.CloseHandle(h_in)
            k.CloseHandle(h_out)
            logger.debug("win_console_open_failed reason=not_a_console")
            return None  # handle is not a real console (redirected to a file)

        try:
            k.SetConsoleMode(h_in, (old_in.value | enable_vt_input) & ~cooked_input)
            k.SetConsoleMode(h_out, old_out.value | enable_vt_output)
            k.FlushConsoleInputBuffer(h_in)

            written = dword()
            k.WriteFile(h_out, batch, len(batch), ctypes.byref(written), None)
            logger.debug("win_console_query wrote=%d batch_len=%d", written.value, len(batch))

            chunk = ctypes.create_string_buffer(4096)
            nread = dword()
            buf = b""
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if k.WaitForSingleObject(h_in, int(remaining * 1000)) != wait_object_0:
                    break
                if not k.ReadFile(h_in, chunk, 4096, ctypes.byref(nread), None) or nread.value == 0:
                    break
                buf += chunk.raw[: nread.value]
                if _DA1_RE.search(buf):  # terminator arrived
                    if _has_pixel_reply(buf):
                        break
                    # DA1 is in, but Rio answers 14t/16t from its event loop,
                    # slightly after the terminator — grace-drain for that reply.
                    deadline = min(deadline, time.monotonic() + _WINDOW_REPLY_GRACE)
            return buf
        finally:
            k.SetConsoleMode(h_in, old_in.value)
            k.SetConsoleMode(h_out, old_out.value)
            k.CloseHandle(h_in)
            k.CloseHandle(h_out)

else:

    def _query_terminal(fd: int, batch: bytes, timeout: float) -> bytes:
        """Send the query batch on `fd` in raw mode and read the reply until DA1/timeout."""
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            os.write(fd, batch)
            buf = b""
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ready, _, _ = select.select([fd], [], [], remaining)
                if not ready:
                    break
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                buf += chunk
                if _DA1_RE.search(buf):  # terminator arrived
                    if _has_pixel_reply(buf):
                        break
                    # DA1 is in, but Rio answers 14t/16t from its event loop,
                    # slightly after the terminator — grace-drain for that reply.
                    deadline = min(deadline, time.monotonic() + _WINDOW_REPLY_GRACE)
            return buf
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _build_query_batch() -> tuple[bytes, Callable[[], None]]:
    """Assemble the capability query batch and a cleanup callable for it.

    The kitty shared-memory query allocates a named shm object — a POSIX
    shm_open object on Unix, a named CreateFileMapping object on Windows (which
    WezTerm reads via OpenFileMappingW). The cleanup callable is invoked only
    after the round-trip completes, keeping the mapping alive for the terminal to
    read (Windows frees it once the last handle closes).
    """
    shm_query, cleanup = _shm_transport_query()
    batch = (
        _KITTY_DIRECT_QUERY
        + shm_query
        + _SYNC_QUERY
        + _KBD_QUERY
        + _SGR_MOUSE_QUERY
        + _SGR_PIXEL_MOUSE_QUERY
        + _WINDOW_QUERY
        + _CELL_QUERY
        + _DA1
    ).encode()
    return batch, cleanup


def _run_terminal_query(batch: bytes, timeout: float) -> bytes | None:
    """Run the query round-trip on the platform's console, or None if unavailable."""
    if sys.platform == "win32":
        return _query_terminal_windows(batch, timeout)
    else:
        fd = _query_fd()
        if fd is None:
            return None
        return _query_terminal(fd, batch, timeout)


def detect_capabilities(timeout: float = 0.5) -> TerminalCapabilities:
    """Probe the terminal once and return its capabilities.

    Always returns a value: env-derived caps (``multiplexer``) are detected on
    every platform, while the escape round-trip runs on a POSIX tty or a Win32
    console (see ``_run_terminal_query``). When no console is attached (pipes,
    headless), ``queried`` is False and the round-trip fields stay at their
    defaults. Restores terminal/console modes unconditionally.
    """
    multiplexer = detect_multiplexer()
    batch, cleanup = _build_query_batch()
    logger.debug(
        "terminal_probe_send platform=%s timeout=%.2f batch=%r", sys.platform, timeout, batch
    )
    try:
        buf = _run_terminal_query(batch, timeout)
    finally:
        cleanup()
    if buf is None:
        logger.info("terminal_probe_skipped reason=no_console platform=%s", sys.platform)
        return TerminalCapabilities(
            queried=False,
            identity=detect_terminal(),
            kitty_graphics=False,
            kitty_transports=frozenset(),
            sync_output=None,
            sgr_mouse=None,
            sgr_pixel_mouse=None,
            kitty_keyboard=False,
            multiplexer=multiplexer,
            window_csi=None,
        )

    transports = parse_kitty_transports(buf)
    kitty_keyboard = parse_kitty_keyboard(buf)
    # Cache the CSI window snapshot only when the live ioctl getter can't serve it.
    # Prefer 14t (window px); fall back to 16t (cell px) for terminals that only
    # answer the latter synchronously (e.g. Rio on Windows).
    window_csi = (
        None
        if tty_window_spec() is not None
        else (_window_spec_from_csi(buf) or _window_spec_from_cell_csi(buf))
    )
    logger.debug("terminal_probe_reply len=%d bytes=%r", len(buf), buf)
    logger.info(
        "terminal_probe_result kitty_graphics=%s transports=%s sync_output=%s "
        "sgr_mouse=%s sgr_pixel_mouse=%s kitty_keyboard=%s window_px=%s cell_px=%s "
        "da1_terminated=%s reply_len=%d",
        bool(transports),
        sorted(transports),
        parse_sync_output(buf),
        parse_sgr_mouse(buf),
        parse_sgr_pixel_mouse(buf),
        kitty_keyboard,
        parse_window_pixels(buf),
        parse_cell_pixels(buf),
        _DA1_RE.search(buf) is not None,
        len(buf),
    )
    return TerminalCapabilities(
        queried=True,
        identity=detect_terminal(kitty_keyboard=kitty_keyboard),
        kitty_graphics=bool(transports),
        kitty_transports=transports,
        sync_output=parse_sync_output(buf),
        sgr_mouse=parse_sgr_mouse(buf),
        sgr_pixel_mouse=parse_sgr_pixel_mouse(buf),
        kitty_keyboard=kitty_keyboard,
        multiplexer=multiplexer,
        window_csi=window_csi,
    )


_capabilities: TerminalCapabilities | None = None


def probe_capabilities(timeout: float = 0.5) -> TerminalCapabilities:
    """Run detection once at startup and cache it for consumers.

    Call from the entrypoint before any Textual app grabs the tty (cooked mode),
    and only for terminal-attached runs (not headless).
    """
    global _capabilities
    _capabilities = detect_capabilities(timeout)
    return _capabilities


def capabilities() -> TerminalCapabilities | None:
    """The cached startup probe, or None if probe_capabilities() has not run."""
    return _capabilities


_DEFAULT_CELL_PX = (8, 16)


def resolve_window_spec(resize_spec: TTYWindowSpec | None = None) -> TTYWindowSpec | None:
    """The freshest available window spec, by source priority:

    1. ``resize_spec`` — a spec built from a live Textual Resize event's in-band
       pixels (DEC mode 2048); the caller supplies it since only it sees the event.
    2. ``tty_window_spec()`` — live TIOCGWINSZ ioctl.
    3. ``capabilities().window_csi`` — the startup CSI snapshot from 14t (window
       px) or 16t (cell px). The only source on Windows, where there is no
       TIOCGWINSZ and Textual's Resize event carries no pixel size.

    Returns None when nothing is available. This is the single resolution chain;
    consumers (the doctor checks/app, ``cell_pixel_size``) must go through it
    rather than re-deriving the priority.
    """
    if resize_spec is not None:
        return resize_spec
    spec = tty_window_spec()
    if spec is not None:
        return spec
    caps = capabilities()
    return caps.window_csi if caps is not None else None


def cell_pixel_size(
    cols: int, rows: int, pixel_size: tuple[int, int] | None = None
) -> tuple[int, int]:
    """Best-effort (cell_width_px, cell_height_px) via :func:`resolve_window_spec`,
    falling back to the 8x16 default when no source is available.

    ``pixel_size`` is Textual's live Resize pixels (in-band mode 2048); it becomes
    the top-priority resize tier. Callers re-run this per resize, so a None
    ``pixel_size`` early in startup self-heals once the first in-band report arrives.
    """
    spec = resolve_window_spec(window_spec_from_resize(cols, rows, pixel_size))
    if spec is not None:
        return spec.cell_width_px, spec.cell_height_px
    return _DEFAULT_CELL_PX


class _APCAwareXTermParser(XTermParser):
    """XTermParser subclass that strips kitty graphics APC responses.

    Kitty sends responses as APC sequences: \\x1b_Gi=<id>;OK\\x1b\\
    Textual's parser doesn't handle APC, so it would mangle them into
    key events. We strip them before they reach the parser.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._apc_buffer = ""

    def feed(self, data: str) -> Iterable[Message]:
        data = self._extract_apc_responses(data)
        if not data:
            return ()
        result: Iterable[Message] = super().feed(data)
        return result

    def _extract_apc_responses(self, data: str) -> str:
        if self._apc_buffer:
            data = self._apc_buffer + data
            self._apc_buffer = ""

        result: list[str] = []
        i = 0
        while i < len(data):
            if data[i : i + 3] == "\x1b_G":
                end = data.find("\x1b\\", i + 3)
                if end == -1:
                    self._apc_buffer = data[i:]
                    break
                i = end + 2
            else:
                result.append(data[i])
                i += 1
        return "".join(result)


# Monkey-patch the active driver before it creates its input parser. Both the
# Linux driver and the Windows driver (textual.drivers.win32) instantiate
# XTermParser from their own module namespace, so we patch the live one.
_driver_mod.XTermParser = _APCAwareXTermParser  # type: ignore[attr-defined]


def apc_parser_active() -> bool:
    """True if the APC-aware XTermParser patch is installed on the active driver."""
    return _driver_mod.XTermParser is _APCAwareXTermParser
