"""Programmatic terminal & environment capability checks.

Every check returns a :class:`CheckResult`; :func:`run_all` aggregates them.
These run in both interactive and ``--check`` modes. No Textual imports.

Importing this module pulls in ``tsdr.tui.tty`` (for ``resolve_window_spec`` and
the capability helpers), which installs the APC-aware XTermParser monkeypatch as a
side effect. That is harmless in ``--check`` mode since no Textual driver runs there.
"""

import importlib.metadata
import importlib.util
import locale
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numba
import numba.core.caching
import numpy as np
import psutil
import soundcard

from tsdr.core.storage import config_dir
from tsdr.radio.dsp import _kernels
from tsdr.tui.tty import (
    TerminalCapabilities,
    TerminalIdentity,
    TTYWindowSpec,
    apc_parser_active,
    capabilities,
    detect_multiplexer,
    detect_terminal,
    resolve_window_spec,
    supports_kitty_images,
)

logger = logging.getLogger(__name__)

_MIN_COLS = 80
_MIN_ROWS = 24

_TRANSPORT_LABELS = {"d": "direct", "s": "shared-memory"}


class Status(Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    summary: str
    required: bool
    group: str  # render | protocol | session | runtime | deps | system
    detail: dict[str, str] = field(default_factory=dict)


def check_truecolor() -> CheckResult:
    colorterm = os.environ.get("COLORTERM", "")
    term = os.environ.get("TERM", "")
    detail = {"COLORTERM": colorterm or "(unset)", "TERM": term or "(unset)"}
    if colorterm in ("truecolor", "24bit"):
        return CheckResult("truecolor", Status.OK, "24-bit color", True, "render", detail)
    if "direct" in term:
        return CheckResult("truecolor", Status.OK, "direct-color TERM", True, "render", detail)
    if sys.platform == "win32":
        return _windows_truecolor(detail)
    if "256color" in term:
        return CheckResult(
            "truecolor", Status.WARN, "256 colors only; gradients may band", True, "render", detail
        )
    return CheckResult(
        "truecolor",
        Status.WARN,
        "no COLORTERM; truecolor unconfirmed",
        True,
        "render",
        detail,
    )


def _windows_truecolor(detail: dict[str, str]) -> CheckResult:
    """Truecolor on Windows, which renders 24-bit color without setting COLORTERM.

    Don't branch on WT_SESSION to detect this (Textualize/rich#140): it is not a
    capability API, other pseudoconsole-backed terminals never set it, and every
    Windows Terminal tab is backed by a conhost that already drives 24-bit VT
    color. The robust signal is the console build - 24-bit VT color landed in
    Win10 1703 (build 15063); only the legacy pre-1703 console lacks it.
    """
    if wt := os.environ.get("WT_SESSION"):
        detail["WT_SESSION"] = wt
    win = sys.getwindowsversion()  # type: ignore[attr-defined]  # win32-only; called only on win32
    detail["windows_build"] = str(win.build)
    if (win.major, win.build) >= (10, 15063):
        return CheckResult(
            "truecolor", Status.OK, "24-bit color (modern Windows console)", True, "render", detail
        )
    return CheckResult(
        "truecolor", Status.WARN, "legacy console; truecolor unsupported", True, "render", detail
    )


def check_unicode_locale() -> CheckResult:
    enc = (sys.stdout.encoding or "").lower()
    pref = locale.getpreferredencoding(False).lower()
    detail = {"stdout_encoding": enc or "(none)", "preferred_encoding": pref}
    utf8 = any("utf-8" in v or "utf8" in v for v in (enc, pref))
    if utf8:
        return CheckResult("unicode_locale", Status.OK, "UTF-8", True, "render", detail)
    return CheckResult(
        "unicode_locale",
        Status.FAIL,
        "non-UTF-8 locale; block/braille glyphs will break",
        True,
        "render",
        detail,
    )


# Verbose provenance for the cell pixel size, keyed by the resolved spec's
# `method`. Surfaces which probe produced the px/cell figure, and whether it
# tracks live resizes (resize/ioctl) or is fixed for the session (CSI snapshot).
_CELL_SOURCE = {
    "resize": "live Resize event (in-band pixels)",
    "ioctl": "TIOCGWINSZ ioctl (live)",
    "csi": "CSI 14t window-size query (startup probe)",
    "csi-cell": "CSI 16t cell-size query (startup probe)",
}


def check_window_size(spec: TTYWindowSpec | None) -> CheckResult:
    """Terminal window geometry: the cell grid and its pixel dimensions."""
    if spec is None:
        return CheckResult(
            "window_size", Status.FAIL, "unavailable (no ioctl or CSI reply)", False, "render"
        )
    return CheckResult(
        "window_size",
        Status.OK,
        f"{spec.cols}x{spec.rows} cells, {spec.width_px}x{spec.height_px} px",
        False,
        "render",
        {
            "cells": f"{spec.cols}x{spec.rows}",
            "window_px": f"{spec.width_px}x{spec.height_px}",
            "method": spec.method,
        },
    )


def check_pixel_size(spec: TTYWindowSpec | None) -> CheckResult:
    """Cell pixel size, annotated with the source the figure came from.

    The cell size feeds kitty image sizing. Provenance matters: in-band resize
    pixels and the live ioctl track terminal changes, the CSI startup snapshot is
    fixed for the session, and the 8x16 default is a blind guess used when nothing
    else is available.
    """
    if spec is None:
        return CheckResult(
            "pixel_size",
            Status.WARN,
            "8x16 px/cell (fallback default; no probe data, images may be mis-sized)",
            False,
            "render",
            {"cell_px": "8x16", "source": "default"},
        )
    return CheckResult(
        "pixel_size",
        Status.OK,
        f"{spec.cell_width_px}x{spec.cell_height_px} px/cell (source: {_CELL_SOURCE[spec.method]})",
        False,
        "render",
        {"cell_px": f"{spec.cell_width_px}x{spec.cell_height_px}", "source": spec.method},
    )


def _kitty_graphics_result(caps: TerminalCapabilities | None) -> CheckResult:
    if caps is None or not caps.queried:
        return CheckResult(
            "kitty_graphics",
            Status.UNKNOWN,
            "not a tty; run interactive `tsdr doctor`",
            False,
            "render",
        )
    if caps.kitty_graphics:
        return CheckResult("kitty_graphics", Status.OK, "supported", False, "render")
    return CheckResult("kitty_graphics", Status.FAIL, "no kitty graphics response", False, "render")


def _kitty_transports_result(caps: TerminalCapabilities | None) -> CheckResult:
    if caps is None or not caps.queried:
        return CheckResult(
            "kitty_transports",
            Status.UNKNOWN,
            "not a tty; run interactive `tsdr doctor`",
            False,
            "render",
        )
    ordered = [_TRANSPORT_LABELS[t] for t in ("d", "s") if t in caps.kitty_transports]
    detail = {"transports": " ".join(sorted(caps.kitty_transports))}
    # TSDR uploads images over shared memory (kitty t=s); without it they won't render.
    # Same predicate the doctor app gates its kitty demo on, so the two can't drift.
    if supports_kitty_images(caps):
        return CheckResult(
            "kitty_transports", Status.OK, ", ".join(ordered), False, "render", detail
        )
    summary = ", ".join(ordered) if ordered else "none"
    return CheckResult(
        "kitty_transports",
        Status.FAIL,
        f"{summary}; shared-memory required",
        False,
        "render",
        detail,
    )


def _sync_output_result(caps: TerminalCapabilities | None) -> CheckResult:
    if caps is None or not caps.queried:
        return CheckResult("synchronized_output", Status.UNKNOWN, "not a tty", False, "protocol")
    supported = caps.sync_output
    if supported is None:
        return CheckResult("synchronized_output", Status.WARN, "not reported", False, "protocol")
    if supported:
        return CheckResult(
            "synchronized_output", Status.OK, "supported (less flicker)", False, "protocol"
        )
    return CheckResult(
        "synchronized_output", Status.WARN, "unsupported; expect flicker", False, "protocol"
    )


def _kitty_keyboard_result(caps: TerminalCapabilities | None) -> CheckResult:
    if caps is None or not caps.queried:
        return CheckResult("kitty_keyboard", Status.UNKNOWN, "not a tty", False, "protocol")
    if caps.kitty_keyboard:
        return CheckResult("kitty_keyboard", Status.OK, "supported", False, "protocol")
    return CheckResult(
        "kitty_keyboard", Status.FAIL, "unsupported; basic keys only", False, "protocol"
    )


def _mouse_result(caps: TerminalCapabilities | None) -> CheckResult:
    """SGR mouse reporting (mode 1006) and sub-cell pixel precision (mode 1016).

    Textual drives the mouse, but drag-to-tune in the tuner needs SGR mouse; pixel
    precision (1016) gives sub-cell resolution. A terminal that doesn't answer the
    DECRQM query may still support mouse, so absence is a WARN, not a FAIL.
    """
    if caps is None or not caps.queried:
        return CheckResult("mouse", Status.UNKNOWN, "not a tty", False, "protocol")
    detail = {
        "sgr_1006": "yes"
        if caps.sgr_mouse
        else ("no" if caps.sgr_mouse is False else "unreported"),
        "sgr_pixel_1016": "yes"
        if caps.sgr_pixel_mouse
        else ("no" if caps.sgr_pixel_mouse is False else "unreported"),
    }
    if caps.sgr_mouse:
        summary = (
            "SGR mouse + pixel precision" if caps.sgr_pixel_mouse else "SGR mouse (cell precision)"
        )
        return CheckResult("mouse", Status.OK, summary, False, "protocol", detail)
    if caps.sgr_mouse is None:
        return CheckResult("mouse", Status.WARN, "not reported", False, "protocol", detail)
    return CheckResult("mouse", Status.WARN, "SGR mouse unsupported", False, "protocol", detail)


def check_terminal_size() -> CheckResult:
    try:
        size = os.get_terminal_size()
    except OSError:
        return CheckResult("terminal_size", Status.UNKNOWN, "not a tty", False, "protocol")
    detail = {"cols": str(size.columns), "rows": str(size.lines)}
    if size.columns >= _MIN_COLS and size.lines >= _MIN_ROWS:
        return CheckResult(
            "terminal_size", Status.OK, f"{size.columns}x{size.lines}", False, "protocol", detail
        )
    return CheckResult(
        "terminal_size",
        Status.WARN,
        f"{size.columns}x{size.lines}; want >= {_MIN_COLS}x{_MIN_ROWS}",
        False,
        "protocol",
        detail,
    )


def check_terminal_identity(identity: TerminalIdentity) -> CheckResult:
    # TERM / TERM_PROGRAM keys are consumed by report.py's env line.
    detail = {
        "TERM": os.environ.get("TERM", "(unset)"),
        "TERM_PROGRAM": os.environ.get("TERM_PROGRAM", "(unset)"),
        "product": identity.product.value,
        "version": identity.version or "(unset)",
    }
    for var in ("KITTY_WINDOW_ID", "GHOSTTY_RESOURCES_DIR", "WEZTERM_PANE", "ITERM_SESSION_ID"):
        if var in os.environ:
            detail[var] = os.environ[var]
    summary = f"{identity.name} {identity.version}" if identity.version else identity.name
    return CheckResult("terminal_identity", Status.OK, summary, False, "protocol", detail)


def check_terminal_quirks(identity: TerminalIdentity) -> list[CheckResult]:
    """One CheckResult per applicable terminal quirk; empty when the terminal is clean."""
    return [
        CheckResult(
            q.id,
            Status.WARN if q.severity == "warn" else Status.FAIL,
            q.summary,
            False,
            q.group,
            {"product": identity.product.value, "version": identity.version or "(unset)"},
        )
        for q in identity.quirks
    ]


def check_multiplexer() -> CheckResult:
    mux = detect_multiplexer()
    if mux == "tmux":
        return CheckResult(
            "multiplexer",
            Status.FAIL,
            "tmux; image graphics need allow-passthrough",
            False,
            "session",
        )
    if mux == "screen":
        return CheckResult(
            "multiplexer", Status.FAIL, "screen; image graphics unsupported", False, "session"
        )
    return CheckResult("multiplexer", Status.OK, "none", False, "session")


def check_ssh_session() -> CheckResult:
    if os.environ.get("SSH_TTY") or os.environ.get("SSH_CONNECTION"):
        return CheckResult(
            "ssh_session",
            Status.WARN,
            "remote; pixel size & graphics may be unavailable",
            False,
            "session",
        )
    return CheckResult("ssh_session", Status.OK, "local", False, "session")


def check_python_version() -> CheckResult:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    gil_enabled = getattr(sys, "_is_gil_enabled", None)
    detail = {"version": ver}
    if gil_enabled is not None:
        detail["gil"] = "on" if gil_enabled() else "off (free-threaded)"
    if v >= (3, 13):
        return CheckResult("python_version", Status.OK, ver, True, "runtime", detail)
    return CheckResult(
        "python_version", Status.FAIL, f"{ver}; need >= 3.13", True, "runtime", detail
    )


def check_os_platform() -> CheckResult:
    driver = "windows" if sys.platform == "win32" else "linux"
    apc_active = apc_parser_active()
    detail = {
        "platform": platform.platform(terse=True),
        "machine": platform.machine(),
        "textual_driver": driver,
        "apc_parser": "active" if apc_active else "inactive",
    }
    return CheckResult(
        "os_platform",
        Status.OK,
        f"{platform.system()} {platform.machine()}",
        False,
        "runtime",
        detail,
    )


def check_numba() -> CheckResult:
    return CheckResult(
        "numba", Status.OK, f"v{numba.__version__}", True, "runtime", {"version": numba.__version__}
    )


def check_numba_jit() -> CheckResult:
    """Verify numba can actually JIT-compile, not just import."""
    try:
        start = time.perf_counter()

        @numba.njit(cache=False)
        def _probe(x):
            return x + 1

        result = _probe(np.float32(1.0))
        elapsed_ms = (time.perf_counter() - start) * 1000
    except Exception as e:  # noqa: BLE001 - numba/LLVM surface opaque compile errors
        return CheckResult("numba_jit", Status.FAIL, f"JIT compile failed ({e})", True, "runtime")
    if result != np.float32(2.0):
        return CheckResult(
            "numba_jit", Status.FAIL, f"JIT produced wrong result ({result})", True, "runtime"
        )
    return CheckResult(
        "numba_jit",
        Status.OK,
        f"JIT compile OK ({elapsed_ms:.0f} ms)",
        True,
        "runtime",
        {"compile_ms": f"{elapsed_ms:.0f}"},
    )


def check_numba_cache() -> CheckResult:
    """Check that numba's on-disk JIT cache is writable.

    With @nb.njit(cache=True), a non-writable cache target silently degrades to a
    full LLVM recompile on every launch. The cache location is override-aware
    (NUMBA_CACHE_DIR else in-tree __pycache__); resolve it the way numba itself
    does via the FunctionCache locator, then probe writability like config_dir.
    """
    try:
        fc = numba.core.caching.FunctionCache(_kernels._lfilter_iir.py_func)
        cache_path = Path(fc._impl.locator.get_cache_path())
    except (AttributeError, TypeError) as e:
        # Private numba internals moved; report rather than crash the doctor.
        return CheckResult(
            "numba_cache", Status.UNKNOWN, f"path unresolved ({e})", False, "runtime"
        )
    source = "NUMBA_CACHE_DIR" if os.environ.get("NUMBA_CACHE_DIR") else "in-tree __pycache__"
    detail = {"path": str(cache_path), "source": source}
    probe = cache_path / ".doctor_write_test"
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        probe.write_text("")
        probe.unlink()
    except OSError as e:
        return CheckResult(
            "numba_cache", Status.WARN, f"not writable: {e}", False, "runtime", detail
        )
    return CheckResult(
        "numba_cache", Status.OK, f"{cache_path} ({source})", False, "runtime", detail
    )


def check_numpy() -> CheckResult:
    return CheckResult(
        "numpy", Status.OK, f"v{np.__version__}", True, "runtime", {"version": np.__version__}
    )


def check_audio_backend() -> CheckResult:
    try:
        default = soundcard.default_speaker()
        speakers = soundcard.all_speakers()
    except Exception as e:  # noqa: BLE001 - soundcard surfaces opaque OS errors
        return CheckResult("audio_backend", Status.FAIL, f"no output device ({e})", True, "deps")
    return CheckResult(
        "audio_backend",
        Status.OK,
        f"soundcard, default: {default.name}",
        True,
        "deps",
        {"default": default.name, "devices": str(len(speakers))},
    )


def check_sdr_backends() -> CheckResult:
    available = ["rtl_tcp", "spyserver", "kiwisdr"]
    if importlib.util.find_spec("rtlsdr") is not None:
        available.append("rtlsdr")
    if importlib.util.find_spec("SoapySDR") is not None:
        available.append("SoapySDR")
    return CheckResult(
        "sdr_backends",
        Status.OK,
        ", ".join(available),
        False,
        "deps",
        {"available": ", ".join(available)},
    )


def check_config_dir() -> CheckResult:
    path = config_dir()
    probe = path / ".doctor_write_test"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("")
        probe.unlink()
    except OSError as e:
        return CheckResult(
            "config_dir", Status.FAIL, f"not writable: {e}", True, "system", {"path": str(path)}
        )
    return CheckResult("config_dir", Status.OK, str(path), True, "system", {"path": str(path)})


def check_cpu() -> CheckResult:
    cores = os.cpu_count() or 0
    return CheckResult("cpu", Status.OK, f"{cores} cores", False, "system", {"cores": str(cores)})


def check_memory() -> CheckResult:
    vm = psutil.virtual_memory()
    total_gib = vm.total / (1024**3)
    avail_gib = vm.available / (1024**3)
    return CheckResult(
        "memory",
        Status.OK,
        f"{total_gib:.0f} GiB total, {avail_gib:.1f} GiB available",
        False,
        "system",
        {"total_bytes": str(vm.total), "available_bytes": str(vm.available)},
    )


def os_details() -> dict[str, str]:
    """Operating-system / interpreter details (data domain; shown + exported)."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "(unknown)",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def installed_packages() -> dict[str, str]:
    """Installed distributions -> version, sorted case-insensitively by name."""
    pkgs: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pkgs[name] = dist.version
    return dict(sorted(pkgs.items(), key=lambda kv: kv[0].lower()))


def run_all() -> list[CheckResult]:
    """Run every programmatic check, reading the startup terminal-capability probe."""
    caps = capabilities()
    logger.debug("doctor_checks_start caps_queried=%s", caps.queried if caps else None)
    results = _run_all(caps)
    for r in results:
        logger.info(
            "doctor_check name=%s status=%s required=%s summary=%r",
            r.name,
            r.status.value,
            r.required,
            r.summary,
        )
    return results


def _run_all(caps: TerminalCapabilities | None) -> list[CheckResult]:
    identity = caps.identity if caps else detect_terminal()
    results = [
        check_truecolor(),
        _kitty_graphics_result(caps),
        _kitty_transports_result(caps),
        *check_terminal_quirks(identity),
    ]
    # Resolve window geometry through the single tty chain (live ioctl → CSI startup
    # snapshot). The --check path has no resize event, so it passes no resize_spec;
    # the interactive app adds that top tier via resolve_window_spec(self._resize_spec).
    window_spec = resolve_window_spec()
    results += [
        check_window_size(window_spec),
        check_pixel_size(window_spec),
        check_unicode_locale(),
        _sync_output_result(caps),
        _kitty_keyboard_result(caps),
        _mouse_result(caps),
        check_terminal_size(),
        check_terminal_identity(identity),
        check_multiplexer(),
        check_ssh_session(),
        check_python_version(),
        check_os_platform(),
        check_numba(),
        check_numba_jit(),
        check_numba_cache(),
        check_numpy(),
        check_audio_backend(),
        check_sdr_backends(),
        check_config_dir(),
        check_cpu(),
        check_memory(),
    ]
    return results
