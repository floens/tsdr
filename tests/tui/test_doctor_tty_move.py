"""Locks in the platform.py -> tui/tty.py move (no back-compat shim)."""

from __future__ import annotations

import importlib
import os

import pytest

from tsdr.tui import tty
from tsdr.tui.tty import TTYWindowSpec, tty_window_spec


def test_tty_window_spec_importable() -> None:
    assert TTYWindowSpec is not None


def test_tty_window_spec_none_on_non_tty() -> None:
    fd = os.open(os.devnull, os.O_RDONLY)
    try:
        assert tty_window_spec(fd) is None
    finally:
        os.close(fd)


def test_core_platform_deleted() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tsdr.core.platform")


def _caps(window_csi: TTYWindowSpec | None) -> tty.TerminalCapabilities:
    return tty.TerminalCapabilities(
        queried=True,
        identity=tty.TerminalIdentity(tty.TerminalProduct.UNKNOWN, "unknown", None, ()),
        kitty_graphics=False,
        kitty_transports=frozenset(),
        sync_output=None,
        sgr_mouse=None,
        sgr_pixel_mouse=None,
        kitty_keyboard=False,
        multiplexer=None,
        window_csi=window_csi,
    )


def test_cell_pixel_size_prefers_event_pixels() -> None:
    # 800x480 px over an 80x24 grid -> 10x20 px/cell, regardless of ioctl/caps.
    assert tty.cell_pixel_size(80, 24, (800, 480)) == (10, 20)


def test_cell_pixel_size_ioctl_fallback(monkeypatch) -> None:
    monkeypatch.setattr(tty, "tty_window_spec", lambda: TTYWindowSpec(24, 80, 800, 480, 10, 20))
    assert tty.cell_pixel_size(80, 24, None) == (10, 20)


def test_cell_pixel_size_caps_csi_fallback(monkeypatch) -> None:
    monkeypatch.setattr(tty, "tty_window_spec", lambda: None)
    csi = TTYWindowSpec(24, 80, 960, 480, 12, 20, method="csi")
    monkeypatch.setattr(tty, "capabilities", lambda: _caps(csi))
    assert tty.cell_pixel_size(80, 24, None) == (12, 20)


def test_cell_pixel_size_default_when_nothing_available(monkeypatch) -> None:
    monkeypatch.setattr(tty, "tty_window_spec", lambda: None)
    monkeypatch.setattr(tty, "capabilities", lambda: _caps(None))
    assert tty.cell_pixel_size(80, 24, None) == (8, 16)
