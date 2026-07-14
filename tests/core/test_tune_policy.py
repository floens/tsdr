"""derive_center_frequency: when the hardware center follows the dial."""

from tsdr.core.sdr.tune_policy import derive_center_frequency
from tsdr.devices.base import DeviceCapabilities


def _caps(*, controllable: bool = True, provides_spectrum: bool = False) -> DeviceCapabilities:
    return DeviceCapabilities(
        frequency_range=(24e6, 1766e6),
        frequency_controllable=controllable,
        sample_rates=None,
        gain_supported=False,
        gain_range=(0.0, 0.0),
        gain_step=0.0,
        gain_unit="dB",
        bias_tee_supported=False,
        provides_spectrum=provides_spectrum,
    )


def _derive(tuned: float, center: float = 100e6, **kwargs) -> float:
    defaults = {
        "sample_rate": 2.4e6,
        "channel_bandwidth": 200e3,
        "caps": _caps(),
        "running": True,
        "mode": "free",
    }
    return derive_center_frequency(tuned=tuned, center=center, **{**defaults, **kwargs})


def test_channel_fits_keeps_center() -> None:
    assert _derive(100.5e6) == 100e6


def test_channel_past_edge_recenters_on_dial() -> None:
    assert _derive(101.2e6) == 101.2e6


def test_margin_shrinks_usable_band() -> None:
    # Usable half-band = 1.2M - 5% margin (120k) - bw/2 (100k) = 980k.
    assert _derive(100e6 + 979e3) == 100e6
    assert _derive(100e6 + 981e3) == 100e6 + 981e3


def test_wide_channel_forces_recenter_sooner() -> None:
    assert _derive(100.5e6, channel_bandwidth=1.5e6) == 100.5e6


def test_non_controllable_keeps_center() -> None:
    assert _derive(101.5e6, caps=_caps(controllable=False)) == 100e6


def test_provides_spectrum_always_follows_dial() -> None:
    assert _derive(100.001e6, caps=_caps(provides_spectrum=True)) == 100.001e6


def test_stopped_device_recenters() -> None:
    assert _derive(100.001e6, running=False) == 100.001e6


def test_center_mode_follows_dial_mid_band() -> None:
    assert _derive(100.001e6, mode="center") == 100.001e6


def test_free_mode_keeps_center_mid_band() -> None:
    assert _derive(100.001e6, mode="free") == 100e6


def test_center_mode_on_non_controllable_keeps_center() -> None:
    assert _derive(100.001e6, mode="center", caps=_caps(controllable=False)) == 100e6


def test_provides_spectrum_ignores_mode() -> None:
    caps = _caps(provides_spectrum=True)
    assert _derive(100.001e6, mode="free", caps=caps) == 100.001e6
    assert _derive(100.001e6, mode="center", caps=caps) == 100.001e6
