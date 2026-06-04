"""Unit tests for the doctor's programmatic checks and raw-tty parsers."""

from __future__ import annotations

import locale
import sys
import types

from tsdr.tui import tty
from tsdr.tui.doctor import checks
from tsdr.tui.doctor.checks import Status


def test_parse_kitty_transports() -> None:
    assert tty.parse_kitty_transports(b"\x1b_Gi=31;OK\x1b\\") == frozenset({"d"})
    assert tty.parse_kitty_transports(b"\x1b_Gi=32;OK\x1b\\") == frozenset({"s"})
    assert tty.parse_kitty_transports(
        b"\x1b_Gi=31;OK\x1b\\\x1b_Gi=32;OK\x1b\\\x1b[?62;c"
    ) == frozenset({"d", "s"})
    assert tty.parse_kitty_transports(b"\x1b_Gi=31;ENOENT:bad\x1b\\") == frozenset()
    assert tty.parse_kitty_transports(b"\x1b[?62;c") == frozenset()  # DA1 only
    assert tty.parse_kitty_transports(b"") == frozenset()


def test_parse_sync_output() -> None:
    assert tty.parse_sync_output(b"\x1b[?2026;1$y") is True
    assert tty.parse_sync_output(b"\x1b[?2026;2$y") is True
    assert tty.parse_sync_output(b"\x1b[?2026;0$y") is False
    assert tty.parse_sync_output(b"\x1b[?62;c") is None


def test_parse_sgr_mouse() -> None:
    assert tty.parse_sgr_mouse(b"\x1b[?1006;1$y") is True
    assert tty.parse_sgr_mouse(b"\x1b[?1006;2$y") is True
    assert tty.parse_sgr_mouse(b"\x1b[?1006;0$y") is False
    assert tty.parse_sgr_mouse(b"\x1b[?62;c") is None


def test_parse_sgr_pixel_mouse() -> None:
    assert tty.parse_sgr_pixel_mouse(b"\x1b[?1016;1$y") is True
    assert tty.parse_sgr_pixel_mouse(b"\x1b[?1016;0$y") is False
    assert tty.parse_sgr_pixel_mouse(b"\x1b[?62;c") is None


def test_parse_kitty_keyboard() -> None:
    assert tty.parse_kitty_keyboard(b"\x1b[?5u")
    assert tty.parse_kitty_keyboard(b"\x1b[?0u")
    assert not tty.parse_kitty_keyboard(b"\x1b[?62;c")  # not confused with DA1


def test_parse_window_pixels() -> None:
    # CSI 4 ; height ; width t  ->  (width_px, height_px)
    assert tty.parse_window_pixels(b"\x1b[4;640;960t") == (960, 640)
    assert tty.parse_window_pixels(b"\x1b[?62;c") is None
    assert tty.parse_window_pixels(b"") is None


def test_parse_cell_pixels() -> None:
    # CSI 6 ; cell_height ; cell_width t  ->  (cell_width_px, cell_height_px)
    assert tty.parse_cell_pixels(b"\x1b[6;16;8t") == (8, 16)
    assert tty.parse_cell_pixels(b"\x1b[?62;c") is None
    assert tty.parse_cell_pixels(b"") is None


def test_detect_multiplexer(monkeypatch) -> None:
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("STY", raising=False)
    assert tty.detect_multiplexer() is None

    monkeypatch.setenv("TMUX", "/tmp/tmux-x/default,1,0")
    assert tty.detect_multiplexer() == "tmux"

    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("STY", "1234.pts-0.host")
    assert tty.detect_multiplexer() == "screen"


def test_detect_capabilities_off_tty(monkeypatch) -> None:
    # With no console available (POSIX: redirected stdio; Windows: no CONIN$), the
    # round-trip is skipped, but caps are still returned with env-derived fields.
    monkeypatch.setenv("TMUX", "/tmp/tmux-x/default,1,0")
    monkeypatch.setattr(tty, "_run_terminal_query", lambda batch, timeout: None)
    caps = tty.detect_capabilities()
    assert caps.queried is False
    assert caps.kitty_graphics is False
    assert caps.kitty_transports == frozenset()
    assert caps.window_csi is None
    assert caps.multiplexer == "tmux"
    assert isinstance(caps.identity, tty.TerminalIdentity)  # env-derived even off-tty


def test_truecolor(monkeypatch) -> None:
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert checks.check_truecolor().status is Status.OK

    # direct-color TERM is recognized on every platform, ahead of platform logic.
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setenv("TERM", "xterm-direct")
    assert checks.check_truecolor().status is Status.OK

    if sys.platform == "win32":
        # Windows Terminal / modern conhost drive 24-bit color without COLORTERM.
        monkeypatch.setenv("WT_SESSION", "test-session")
        monkeypatch.setenv("TERM", "xterm-256color")
        assert checks.check_truecolor().status is Status.OK
    else:
        monkeypatch.setenv("TERM", "xterm-256color")
        assert checks.check_truecolor().status is Status.WARN

        monkeypatch.setenv("TERM", "dumb")
        assert checks.check_truecolor().status is Status.WARN


def test_unicode_locale_utf8_ok() -> None:
    assert checks.check_unicode_locale().status is Status.OK


def test_unicode_locale_non_utf8_fails(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(encoding="ascii"))
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _do_setlocale=True: "ANSI_X3.4")
    assert checks.check_unicode_locale().status is Status.FAIL


def test_multiplexer(monkeypatch) -> None:
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("STY", raising=False)
    assert checks.check_multiplexer().status is Status.OK

    monkeypatch.setenv("TMUX", "/tmp/tmux-x/default,1,0")
    assert checks.check_multiplexer().status is Status.FAIL

    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("STY", "1234.pts-0.host")
    assert checks.check_multiplexer().status is Status.FAIL


def test_ssh_session(monkeypatch) -> None:
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    assert checks.check_ssh_session().status is Status.OK

    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5 6.7.8.9 22")
    assert checks.check_ssh_session().status is Status.WARN


def test_python_version_ok() -> None:
    r = checks.check_python_version()
    assert r.status is Status.OK
    assert r.required


def _identity(
    product: tty.TerminalProduct = tty.TerminalProduct.UNKNOWN,
    *,
    version: str | None = None,
    quirks: tuple[tty.Quirk, ...] = (),
) -> tty.TerminalIdentity:
    return tty.TerminalIdentity(product=product, name=product.value, version=version, quirks=quirks)


def _queried_caps(
    transports: frozenset[str],
    *,
    kitty_keyboard: bool = True,
    sgr_mouse: bool | None = None,
    sgr_pixel_mouse: bool | None = None,
) -> tty.TerminalCapabilities:
    return tty.TerminalCapabilities(
        queried=True,
        identity=_identity(),
        kitty_graphics=bool(transports),
        kitty_transports=transports,
        sync_output=None,
        sgr_mouse=sgr_mouse,
        sgr_pixel_mouse=sgr_pixel_mouse,
        kitty_keyboard=kitty_keyboard,
        multiplexer=None,
        window_csi=None,
    )


def test_kitty_transports_fails_without_shared_memory() -> None:
    assert checks._kitty_transports_result(_queried_caps(frozenset({"d", "s"}))).status is Status.OK
    direct_only = checks._kitty_transports_result(_queried_caps(frozenset({"d"})))
    assert direct_only.status is Status.FAIL
    assert "shared-memory" in direct_only.summary
    assert checks._kitty_transports_result(_queried_caps(frozenset())).status is Status.FAIL


def test_kitty_keyboard_fails_when_unsupported() -> None:
    ok = _queried_caps(frozenset({"d", "s"}), kitty_keyboard=True)
    no_kbd = _queried_caps(frozenset({"d", "s"}), kitty_keyboard=False)
    assert checks._kitty_keyboard_result(ok).status is Status.OK
    assert checks._kitty_keyboard_result(no_kbd).status is Status.FAIL


def test_mouse_result() -> None:
    full = _queried_caps(frozenset(), sgr_mouse=True, sgr_pixel_mouse=True)
    assert checks._mouse_result(full).status is Status.OK
    assert "pixel precision" in checks._mouse_result(full).summary

    cell_only = _queried_caps(frozenset(), sgr_mouse=True, sgr_pixel_mouse=None)
    assert checks._mouse_result(cell_only).status is Status.OK
    assert "cell precision" in checks._mouse_result(cell_only).summary

    unreported = _queried_caps(frozenset(), sgr_mouse=None)
    assert checks._mouse_result(unreported).status is Status.WARN
    assert "not reported" in checks._mouse_result(unreported).summary

    unsupported = _queried_caps(frozenset(), sgr_mouse=False)
    assert checks._mouse_result(unsupported).status is Status.WARN
    assert "unsupported" in checks._mouse_result(unsupported).summary

    assert checks._mouse_result(None).status is Status.UNKNOWN


def test_check_numba_jit_ok() -> None:
    result = checks.check_numba_jit()
    assert result.name == "numba_jit"
    assert result.status is Status.OK
    assert result.required is True
    assert "compile_ms" in result.detail


def test_check_numba_jit_failure(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("llvm broken")

    monkeypatch.setattr(checks.numba, "njit", _boom)
    result = checks.check_numba_jit()
    assert result.status is Status.FAIL
    assert "llvm broken" in result.summary


def test_check_numba_cache_ok() -> None:
    result = checks.check_numba_cache()
    assert result.name == "numba_cache"
    assert result.status is Status.OK
    assert result.required is False
    assert "path" in result.detail


def test_check_numba_cache_not_writable(monkeypatch, tmp_path) -> None:
    readonly = tmp_path / "ro"
    readonly.mkdir()
    readonly.chmod(0o500)

    class _Locator:
        def get_cache_path(self) -> str:
            return str(readonly)

    class _Impl:
        locator = _Locator()

    class _FakeCache:
        def __init__(self, _func) -> None:
            self._impl = _Impl()

    monkeypatch.setattr(checks.numba.core.caching, "FunctionCache", _FakeCache)
    result = checks.check_numba_cache()
    assert result.status is Status.WARN
    assert "not writable" in result.summary


def test_detect_terminal_wezterm_old_build_quirk(monkeypatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "WezTerm")

    monkeypatch.setenv("TERM_PROGRAM_VERSION", "20240203-110809-5046fc22")
    old = tty.detect_terminal()
    assert old.product is tty.TerminalProduct.WEZTERM
    assert old.version == "20240203-110809-5046fc22"
    assert {q.id for q in old.quirks} == {"wezterm_old_kitty_graphics"}

    monkeypatch.setenv("TERM_PROGRAM_VERSION", "20260201-000000-deadbeef")
    assert tty.detect_terminal().quirks == ()  # recent build: no quirk

    monkeypatch.setenv("TERM_PROGRAM_VERSION", "not-a-date")
    assert tty.detect_terminal().quirks == ()  # unparseable build: can't tell, no quirk


def test_check_terminal_quirks_renders_warn() -> None:
    quirk = tty.Quirk("wezterm_old_kitty_graphics", "outdated build", "render", "warn")
    identity = _identity(tty.TerminalProduct.WEZTERM, version="20240203", quirks=(quirk,))
    results = checks.check_terminal_quirks(identity)
    assert len(results) == 1
    assert results[0].name == "wezterm_old_kitty_graphics"
    assert results[0].status is Status.WARN
    assert results[0].group == "render"
    # clean terminal emits no quirk lines
    assert checks.check_terminal_quirks(_identity()) == []


def test_window_and_pixel_size_split() -> None:
    # rows, cols, width_px, height_px, cell_w, cell_h, method
    spec = tty.TTYWindowSpec(40, 120, 960, 640, 8, 16, method="csi-cell")

    win = checks.check_window_size(spec)
    assert win.name == "window_size" and win.status is Status.OK
    assert "120x40 cells" in win.summary and "960x640 px" in win.summary
    assert "px/cell" not in win.summary  # cell size lives on the pixel_size check

    px = checks.check_pixel_size(spec)
    assert px.name == "pixel_size" and px.status is Status.OK
    assert "8x16 px/cell" in px.summary
    assert "CSI 16t" in px.summary  # provenance of the cell size is surfaced
    assert px.detail["source"] == "csi-cell"


def test_pixel_size_source_labels() -> None:
    for method, marker in (
        ("resize", "live Resize event"),
        ("ioctl", "TIOCGWINSZ"),
        ("csi", "CSI 14t"),
    ):
        spec = tty.TTYWindowSpec(40, 120, 960, 640, 8, 16, method=method)
        assert marker in checks.check_pixel_size(spec).summary


def test_window_and_pixel_size_none() -> None:
    assert checks.check_window_size(None).status is Status.FAIL
    fallback = checks.check_pixel_size(None)
    assert fallback.status is Status.WARN
    assert "8x16" in fallback.summary and fallback.detail["source"] == "default"


def test_run_all_kitty_unknown_off_tty() -> None:
    results = checks.run_all()
    assert all(isinstance(r, checks.CheckResult) for r in results)
    kitty = next(r for r in results if r.name == "kitty_graphics")
    assert kitty.status is Status.UNKNOWN  # not a tty under pytest
