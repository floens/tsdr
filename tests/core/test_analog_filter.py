"""The radio's analog filter: which width tsdr asks for, and when the I/O
worker sets it."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from tsdr.core.sdr import engine as engine_module
from tsdr.core.sdr.analog_filter import target_bandwidth
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.workers.io_worker import IOWorker
from tsdr.devices import MockParams
from tsdr.devices.base import DeviceCapabilities
from tsdr.tui.commands.sdr import config as config_command


def _caps(*, supported: bool = True, static: bool = True) -> DeviceCapabilities:
    return DeviceCapabilities(
        frequency_range=(70e6, 6e9),
        frequency_controllable=True,
        sample_rates=None,
        gain_supported=True,
        gain_range=(0.0, 73.0),
        gain_step=1.0,
        gain_unit="dB",
        bias_tee_supported=False,
        analog_filter_supported=supported,
        static_analog_filter=static,
    )


def _target(setting: str = "auto", *, width: float | None = None, **caps_kw) -> float | None:
    return target_bandwidth(
        setting=setting,  # type: ignore[arg-type]
        width_hz=width,
        fraction=0.8,
        sample_rate=2e6,
        caps=_caps(**caps_kw),
    )


def test_auto_tracks_the_rate_on_a_static_filter() -> None:
    assert _target() == pytest.approx(1.6e6)


def test_auto_leaves_a_driver_managed_filter_alone() -> None:
    assert _target(static=False) is None


def test_explicit_modes() -> None:
    assert _target("track", static=False) == pytest.approx(1.6e6)
    assert _target("fixed", width=5e6) == 5e6
    assert _target("off") is None


def test_unsupported_filter_is_never_set() -> None:
    assert _target("track", supported=False) is None


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="needs analog_filter_hz"):
        DeviceConfig(analog_filter="fixed").validate()
    with pytest.raises(ValueError, match="analog_filter_fraction"):
        DeviceConfig(analog_filter_fraction=0.0).validate()
    with pytest.raises(ValueError, match="analog_filter must be one of"):
        DeviceConfig(analog_filter="wide").validate()  # type: ignore[arg-type]


class _FilterDevice:
    """Driver whose analog filter starts at 18 MHz (the Pluto at boot)."""

    def __init__(self, caps: DeviceCapabilities) -> None:
        self.capabilities = caps
        self.bandwidth = 18e6
        self.requests: list[float] = []

    def analog_bandwidth(self) -> float:
        return self.bandwidth

    def set_analog_bandwidth(self, hz: float) -> float:
        self.requests.append(hz)
        self.bandwidth = hz
        return hz

    def close(self) -> None:
        self.closed = True


def _worker(device: _FilterDevice) -> IOWorker:
    context = MagicMock()
    context.device = device
    context.device_id = "pluto"
    context.analog_bandwidth_hz = None
    worker = IOWorker(context)
    worker._filter_at_open = device.analog_bandwidth()
    return worker


def _config(**kw) -> DeviceConfig:
    return DeviceConfig(sample_rate=2e6, analog_filter_fraction=0.8, **kw)


def test_worker_tracks_sample_rate_changes() -> None:
    device = _FilterDevice(_caps())
    worker = _worker(device)
    worker._apply_analog_filter(device, _config())  # type: ignore[arg-type]
    worker._apply_analog_filter(device, _config())  # type: ignore[arg-type]
    worker._apply_analog_filter(device, replace(_config(), sample_rate=10e6))  # type: ignore[arg-type]
    assert device.requests == pytest.approx([1.6e6, 8e6])
    assert worker.device_context.analog_bandwidth_hz == pytest.approx(8e6)


def test_worker_retries_a_width_the_driver_rejected() -> None:
    device = _FilterDevice(_caps())
    worker = _worker(device)
    accept = device.set_analog_bandwidth
    failures = [RuntimeError("busy")]

    def flaky(hz: float) -> float:
        if failures:
            raise failures.pop()
        return accept(hz)

    device.set_analog_bandwidth = flaky  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        worker._apply_analog_filter(device, _config())  # type: ignore[arg-type]
    worker._apply_analog_filter(device, _config())  # type: ignore[arg-type]
    assert device.requests == pytest.approx([1.6e6])


def test_worker_hands_back_the_drivers_width() -> None:
    device = _FilterDevice(_caps())
    worker = _worker(device)
    worker._apply_analog_filter(device, _config())  # type: ignore[arg-type]
    worker._apply_analog_filter(device, _config(analog_filter="off"))  # type: ignore[arg-type]
    assert device.requests == pytest.approx([1.6e6, 18e6])


def test_worker_restores_the_drivers_width_on_close() -> None:
    device = _FilterDevice(_caps())
    worker = _worker(device)
    worker._apply_analog_filter(device, _config())  # type: ignore[arg-type]
    worker.teardown(MagicMock())
    assert device.requests == pytest.approx([1.6e6, 18e6])
    assert device.closed


def test_worker_leaves_an_untouched_filter_alone_on_close() -> None:
    device = _FilterDevice(_caps(static=False))
    worker = _worker(device)
    worker._apply_analog_filter(device, _config())  # type: ignore[arg-type]
    worker.teardown(MagicMock())
    assert device.requests == []


def test_worker_leaves_a_driver_managed_filter_alone() -> None:
    device = _FilterDevice(_caps(static=False))
    worker = _worker(device)
    worker._apply_analog_filter(device, _config())  # type: ignore[arg-type]
    assert device.requests == []
    worker._apply_analog_filter(  # type: ignore[arg-type]
        device, _config(analog_filter="fixed", analog_filter_hz=5e6)
    )
    assert device.requests == [5e6]


def test_worker_skips_devices_without_a_filter() -> None:
    device = _FilterDevice(_caps(supported=False))
    worker = _worker(device)
    worker._apply_analog_filter(device, _config(analog_filter="track"))  # type: ignore[arg-type]
    assert device.requests == []


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> SDREngine:
    engine = SDREngine()
    engine.add_device("pluto", "mock", MockParams(), DeviceConfig(sample_rate=2e6))
    monkeypatch.setattr(engine_module, "_active_engine", engine)
    monkeypatch.setattr(config_command, "get_engine", lambda: engine)
    return engine


def test_config_command_sets_a_fixed_width(engine: SDREngine) -> None:
    out = config_command.SDRConfigCommand().execute(
        ["--device", "pluto", "--analog-filter", "1.5M"]
    )
    config = engine.devices["pluto"].config
    assert (config.analog_filter, config.analog_filter_hz) == ("fixed", 1.5e6)
    assert "1.500 MHz" in out


def test_config_command_sets_mode_and_fraction(engine: SDREngine) -> None:
    config_command.SDRConfigCommand().execute(
        ["--device", "pluto", "--analog-filter", "track", "--analog-filter-fraction", "0.9"]
    )
    config = engine.devices["pluto"].config
    assert (config.analog_filter, config.analog_filter_fraction) == ("track", 0.9)


def test_config_command_rejects_a_bad_fraction(engine: SDREngine) -> None:
    out = config_command.SDRConfigCommand().execute(
        ["--device", "pluto", "--analog-filter-fraction", "5"]
    )
    assert "analog_filter_fraction must be" in out
