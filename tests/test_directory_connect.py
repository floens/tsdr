"""find_added_device matches a directory receiver to an engine device by its
host:port endpoint, not by device_id — device ids are user-renamable."""

from types import SimpleNamespace

from tsdr.core.directory import connect
from tsdr.core.directory.model import PublicDevice
from tsdr.core.sdr.device_context import DeviceState
from tsdr.devices import KiwiSDRParams, SpyServerParams

_DEVICE = PublicDevice(source="spyserver", id="x", name="n", host="1.2.3.4", port=5555)


def _fake_engine(monkeypatch, devices: dict) -> None:
    monkeypatch.setattr(connect, "get_engine", lambda: SimpleNamespace(devices=devices))


def test_find_added_device_matches_renamed_id(monkeypatch) -> None:
    ctx = SimpleNamespace(params=SpyServerParams(host="1.2.3.4", port=5555))
    _fake_engine(monkeypatch, {"my-renamed-rx": ctx})
    assert connect.find_added_device(_DEVICE) == "my-renamed-rx"


def test_find_added_device_none_on_different_port(monkeypatch) -> None:
    ctx = SimpleNamespace(params=SpyServerParams(host="1.2.3.4", port=5001))
    _fake_engine(monkeypatch, {"rx": ctx})
    assert connect.find_added_device(_DEVICE) is None


def test_find_added_device_ignores_non_network_devices(monkeypatch) -> None:
    ctx = SimpleNamespace(params=object())  # e.g. an iq-file device, no host/port
    _fake_engine(monkeypatch, {"file": ctx})
    assert connect.find_added_device(_DEVICE) is None


def test_device_endpoint_defaults_missing_port() -> None:
    device = PublicDevice(source="spyserver", id="x", name="n", host="h")
    assert connect.device_endpoint(device) == ("h", 5555)


def test_default_device_id_includes_port() -> None:
    assert connect._default_device_id(_DEVICE) == "spy-1-2-3-4-5555"


def test_device_endpoint_defaults_kiwisdr_port() -> None:
    device = PublicDevice(source="kiwisdr", id="k", name="n", host="h")
    assert connect.device_endpoint(device) == ("h", 8073)


def test_default_device_id_kiwisdr_prefix() -> None:
    device = PublicDevice(source="kiwisdr", id="k", name="n", host="1.2.3.4", port=8073)
    assert connect._default_device_id(device) == "kiwi-1-2-3-4-8073"


def test_add_directory_device_adds_kiwisdr(monkeypatch) -> None:
    device = PublicDevice(
        source="kiwisdr",
        id="k",
        name="Kiwi NL",
        host="kiwi.example.com",
        url="http://kiwi.example.com:8073/",
        freq_min=0.0,
        freq_max=30e6,
        usable=True,
    )
    calls: list = []
    engine = SimpleNamespace(
        devices={},
        get_focused_device=lambda: None,
        add_device=lambda did, dtype, params, cfg: calls.append(("add", did, dtype, params, cfg)),
        set_focused_device=lambda did: calls.append(("focus", did)),
        start_device=lambda did: calls.append(("start", did)),
    )
    monkeypatch.setattr(connect, "get_engine", lambda: engine)

    result = connect.add_directory_device(device)
    assert result.ok
    _, did, dtype, params, cfg = next(c for c in calls if c[0] == "add")
    assert dtype == "kiwisdr"
    assert isinstance(params, KiwiSDRParams)
    assert (params.host, params.port) == ("kiwi.example.com", 8073)
    assert did == "kiwi-kiwi-example-com-8073"
    assert 0.0 <= cfg.center_frequency <= 30e6  # band-aware, in-range default


def test_add_directory_device_rejects_unusable(monkeypatch) -> None:
    _fake_engine(monkeypatch, {})
    unusable = PublicDevice(
        source="spyserver", id="x", name="n", host="1.2.3.4", port=5555, usable_reason="offline"
    )
    result = connect.add_directory_device(unusable)
    assert result.ok is False
    assert "offline" in result.message


def test_start_directory_device_not_added(monkeypatch) -> None:
    _fake_engine(monkeypatch, {})
    result = connect.start_directory_device(_DEVICE)
    assert result.ok is False
    assert "not added" in result.message


def test_start_directory_device_retunes_focuses_and_starts(monkeypatch) -> None:
    device = PublicDevice(
        source="spyserver",
        id="x",
        name="n",
        host="1.2.3.4",
        port=5555,
        freq_min=88e6,
        freq_max=108e6,
        usable=True,
    )
    calls: list = []
    ctx = SimpleNamespace(
        params=SpyServerParams(host="1.2.3.4", port=5555), state=DeviceState.STOPPED
    )
    engine = SimpleNamespace(
        devices={"rx": ctx},
        get_focused_device=lambda: SimpleNamespace(config=SimpleNamespace(tuned_frequency=100e6)),
        update_device_config=lambda did, **kw: calls.append(("cfg", did, kw)),
        set_focused_device=lambda did: calls.append(("focus", did)),
        start_device=lambda did: calls.append(("start", did)),
        stop_device=lambda did: calls.append(("stop", did)),
    )
    monkeypatch.setattr(connect, "get_engine", lambda: engine)

    result = connect.start_directory_device(device)
    assert result.ok
    assert ("cfg", "rx", {"tuned_frequency": 100e6}) in calls  # retuned to the active freq
    assert ("focus", "rx") in calls
    assert ("start", "rx") in calls
