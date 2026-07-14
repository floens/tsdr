"""Image-mode strip filling: a line whose rows straddle a strip boundary must
carry the remainder into the next strip, not drop it (a dropped row per
rotation shows up as a gap between strips)."""

import numpy as np
import pytest

from tsdr.core.events.events import FFTUpdateEvent
from tsdr.core.sdr.engine import SDREngine
from tsdr.tui.widgets.waterfall_widget import _STRIP_HEIGHT, WaterfallWidget


@pytest.fixture(autouse=True)
def _engine():
    # _project_line reads the focused device's view via the engine singleton;
    # constructing SDREngine registers it (no devices -> event range is used).
    SDREngine()


def _event(n: int = 64) -> FFTUpdateEvent:
    return FFTUpdateEvent(
        device_id="test",
        spectrum=np.full(n, -50.0, dtype=np.float32),
        frequencies=np.linspace(99e6, 101e6, n, dtype=np.float32),
        center_frequency=100e6,
        sample_rate=2e6,
    )


def _total_rows(widget: WaterfallWidget) -> int:
    return sum(s.fill for s in widget._image_strips)


def test_rows_carry_over_at_strip_boundary():
    widget = WaterfallWidget()
    widget._pixel_scale = 3  # 64 % 3 != 0 -> boundary lands mid-line
    event = _event()

    lines = 30
    for _ in range(lines):
        widget._fill_active_strip(event, w=48)

    assert _total_rows(widget) == lines * 3  # no row lost at the rotation
    frozen = [s for s in widget._image_strips if s.frozen]
    assert len(frozen) == 1
    assert frozen[0].fill == _STRIP_HEIGHT
    assert widget._image_strips[0].fill == lines * 3 - _STRIP_HEIGHT


def test_exact_boundary_rotates_to_empty_strip():
    widget = WaterfallWidget()
    event = _event()

    for _ in range(_STRIP_HEIGHT // 2):  # scale 2: fills exactly one strip
        widget._fill_active_strip(event, w=48)

    assert _total_rows(widget) == _STRIP_HEIGHT
    assert widget._image_strips[1].frozen
    assert widget._image_strips[0].fill == 0
