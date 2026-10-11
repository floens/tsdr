"""Click-to-tune: where a click on the spectrum puts the dial."""

from types import SimpleNamespace

import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr import engine as engine_module
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams
from tsdr.tui.widgets import spectrum_widget
from tsdr.tui.widgets.spectrum_widget import SpectrumWidget

DIAL = 151e6
VIEW = (150e6, 152e6)
WIDTH = 200  # columns: 10 kHz each


class _Spectrum(SpectrumWidget):
    @property
    def size(self):  # type: ignore[override]
        return SimpleNamespace(width=WIDTH, height=10)

    def _display_range(self) -> tuple[float, float]:
        return VIEW

    def watch_image_mode(self, *_args: object) -> None:
        pass  # no kitty host off-screen; reading the reactive would run the real watcher


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> SDREngine:
    engine = SDREngine()
    engine.add_device(
        "m", "mock", MockParams(), DeviceConfig(tuned_frequency=DIAL, center_frequency=DIAL)
    )
    engine.set_audio_demod("m", DemodSpec(mode="NFM"))
    monkeypatch.setattr(engine_module, "_active_engine", engine)
    # A coarse keyboard step, as when crossing a band by key.
    monkeypatch.setattr(spectrum_widget, "get_tuning_state", lambda: SimpleNamespace(step=1e6))
    return engine


def _click(column: int) -> list[float]:
    saved: list[float] = []
    w = _Spectrum()  # image mode off: one column per cell
    original = spectrum_widget.save_previous_tune_state
    spectrum_widget.save_previous_tune_state = lambda d: saved.append(d.config.tuned_frequency)
    try:
        w.on_click(SimpleNamespace(x=column, stop=lambda: None))  # type: ignore[arg-type]
    finally:
        spectrum_widget.save_previous_tune_state = original
    return saved


def test_click_lands_on_the_channel_raster(engine: SDREngine) -> None:
    saved = _click(131)  # 151.31 MHz
    assert engine.devices["m"].config.tuned_frequency == 151.3125e6
    assert saved == [DIAL]


def test_click_on_the_dial_keeps_the_ab_slot(engine: SDREngine) -> None:
    saved = _click(100)  # 151.00 MHz, the dial
    assert engine.devices["m"].config.tuned_frequency == DIAL
    assert saved == []
