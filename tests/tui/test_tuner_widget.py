"""TunerWidget's device line: stats events repaint it only when it changes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from tsdr.tui.widgets.tuner_widget import TunerWidget


class _Static:
    def __init__(self) -> None:
        self.content = ""
        self.updates = 0

    def update(self, content: str) -> None:
        self.content = content
        self.updates += 1


def _widget() -> tuple[TunerWidget, _Static]:
    w = TunerWidget.__new__(TunerWidget)
    device_line = _Static()
    meter = MagicMock()
    w.query_one = lambda selector, _kind=None: (  # type: ignore[method-assign]
        device_line if selector == "#tuner-device" else meter
    )
    w._sr_line = "[bold]10.0[/bold] [dim]MS/s[/dim]"
    w._gain_line = "[yellow]40.0[/yellow] [dim]dB[/dim]"
    return w, device_line


def _stats(clip: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(channel_snr=20.0, iq_clip_pct=clip)


def test_unchanged_stats_leave_the_device_line_alone() -> None:
    # Static.update re-runs the screen layout, and stats arrive ~16x/s.
    w, device_line = _widget()
    for _ in range(5):
        w.update_stats(_stats())  # type: ignore[arg-type]
    assert device_line.updates == 1
    assert device_line.content.endswith("VOL 50%")


def test_clipping_still_updates_the_device_line() -> None:
    w, device_line = _widget()
    w.update_stats(_stats())  # type: ignore[arg-type]
    w.update_stats(_stats(clip=3.0))  # type: ignore[arg-type]
    assert "CLIP 3%" in device_line.content
    w.update_stats(_stats())  # type: ignore[arg-type]
    assert "CLIP" not in device_line.content
    assert device_line.updates == 3
