"""Device persistence must round-trip every device type through devices.toml.
KiwiSDR was dropped on restart because `_build_params` had no case for it (and
`PersistedDevice` had no password/user fields), so added Kiwis vanished.
"""

from tsdr.core.devices import PersistedDevice
from tsdr.core.preferences import _build_device_config, _build_params, _persist_params
from tsdr.devices import KiwiSDRParams


def test_kiwisdr_params_round_trip():
    params = KiwiSDRParams(host="h.example.com", port=8073, password="sec", user="me")
    persisted = PersistedDevice(id="kiwi-x", type="kiwisdr", **_persist_params(params))
    assert (persisted.password, persisted.user) == ("sec", "me")
    assert _build_params(persisted) == params


def test_kiwisdr_restore_defaults_when_auth_absent():
    persisted = PersistedDevice(id="kiwi-x", type="kiwisdr", host="h", port=8073)
    assert _build_params(persisted) == KiwiSDRParams(host="h", port=8073, password="", user="tsdr")


def test_unknown_device_type_still_skipped():
    assert _build_params(PersistedDevice(id="x", type="bogus")) is None


def test_fft_geometry_restores_per_device():
    persisted = PersistedDevice(id="d", type="mock", fft_size=4096, fft_window="blackman")
    config = _build_device_config(persisted)
    assert config is not None
    assert (config.fft_size, config.fft_window) == (4096, "blackman")


def test_spectrum_view_restores_per_device():
    persisted = PersistedDevice(id="d", type="mock", spectrum_center=7.1e6, spectrum_span=100e3)
    config = _build_device_config(persisted)
    assert config is not None
    assert (config.spectrum_center, config.spectrum_span) == (7.1e6, 100e3)


def test_spectrum_view_absent_restores_none():
    config = _build_device_config(PersistedDevice(id="d", type="mock", center_frequency=1e6))
    assert config is not None
    assert (config.spectrum_center, config.spectrum_span) == (None, None)


def test_tuned_frequency_round_trips():
    persisted = PersistedDevice(id="d", type="mock", tuned_frequency=7.2e6, center_frequency=7e6)
    config = _build_device_config(persisted)
    assert config is not None
    assert (config.tuned_frequency, config.center_frequency) == (7.2e6, 7e6)


def test_legacy_prefs_without_tuned_frequency_fall_back_to_center():
    config = _build_device_config(PersistedDevice(id="d", type="mock", center_frequency=1e6))
    assert config is not None
    assert config.tuned_frequency == 1e6


def test_tuning_mode_round_trips():
    persisted = PersistedDevice(id="d", type="mock", tuning_mode="free")
    config = _build_device_config(persisted)
    assert config is not None
    assert config.tuning_mode == "free"


def test_tuning_mode_absent_defaults_to_center():
    config = _build_device_config(PersistedDevice(id="d", type="mock", center_frequency=1e6))
    assert config is not None
    assert config.tuning_mode == "center"
