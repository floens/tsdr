"""Device persistence must round-trip every device type through devices.toml.
KiwiSDR was dropped on restart because `_build_params` had no case for it (and
`PersistedDevice` had no password/user fields), so added Kiwis vanished.
"""

from tsdr.core.devices import PersistedDevice
from tsdr.core.preferences import _build_params, _persist_params
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
