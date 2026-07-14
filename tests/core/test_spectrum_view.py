"""View resolution: the (center, span) every render and device view request
uses, clamped against the device's displayable band."""

from tsdr.core.sdr.config import MIN_SPECTRUM_SPAN_HZ, DeviceConfig
from tsdr.core.sdr.spectrum_view import adjusted_span, full_view_range, resolve_view
from tsdr.devices.base import DeviceCapabilities


def _caps(*, provides_spectrum: bool = False, frequency_range=None) -> DeviceCapabilities:
    return DeviceCapabilities(
        frequency_range=frequency_range,
        frequency_controllable=True,
        sample_rates=None,
        gain_supported=False,
        gain_range=(0.0, 0.0),
        gain_step=1.0,
        gain_unit="dB",
        bias_tee_supported=False,
        provides_spectrum=provides_spectrum,
    )


def test_defaults_give_full_iq_band():
    config = DeviceConfig(tuned_frequency=100e6, center_frequency=100e6, sample_rate=2.4e6)
    assert resolve_view(config, None) == (100e6, 2.4e6)


def test_span_set_center_tracks_dial():
    config = DeviceConfig(
        tuned_frequency=100e6, center_frequency=100e6, sample_rate=2.4e6, spectrum_span=200e3
    )
    assert resolve_view(config, None) == (100e6, 200e3)


def test_center_clamped_to_band_edges():
    config = DeviceConfig(
        tuned_frequency=100e6,
        center_frequency=100e6,
        tuning_mode="free",
        sample_rate=2.4e6,
        spectrum_span=200e3,
        spectrum_center=200e6,
    )
    center, span = resolve_view(config, None)
    assert span == 200e3
    assert center == 100e6 + 2.4e6 / 2 - 100e3  # upper edge minus half span


def test_center_mode_ignores_view_pan():
    # A pinned view would hide every tune in center mode; panning only
    # applies in free mode.
    config = DeviceConfig(
        tuned_frequency=100e6,
        center_frequency=100e6,
        tuning_mode="center",
        sample_rate=2.4e6,
        spectrum_span=200e3,
        spectrum_center=100.5e6,
    )
    assert resolve_view(config, None) == (100e6, 200e3)


def test_span_wider_than_band_clamps_to_full():
    config = DeviceConfig(
        tuned_frequency=100e6, center_frequency=100e6, sample_rate=2.4e6, spectrum_span=10e6
    )
    assert resolve_view(config, None) == (100e6, 2.4e6)


def test_provides_spectrum_uses_capability_range():
    caps = _caps(provides_spectrum=True, frequency_range=(0.0, 30e6))
    config = DeviceConfig(tuned_frequency=7.1e6, center_frequency=7.1e6, sample_rate=12_000.0)
    assert full_view_range(config, caps) == (0.0, 30e6)
    center, span = resolve_view(config, caps)
    assert span == 30e6
    assert center == 15e6  # full-band view centers the band, not the 12 kHz dial


def test_provides_spectrum_zoomed_view_tracks_dial():
    caps = _caps(provides_spectrum=True, frequency_range=(0.0, 30e6))
    config = DeviceConfig(
        tuned_frequency=7.1e6, center_frequency=7.1e6, sample_rate=12_000.0, spectrum_span=100e3
    )
    assert resolve_view(config, caps) == (7.1e6, 100e3)


def test_non_spectrum_device_ignores_capability_range():
    caps = _caps(provides_spectrum=False, frequency_range=(24e6, 1.7e9))
    config = DeviceConfig(tuned_frequency=100e6, center_frequency=100e6, sample_rate=2.4e6)
    assert full_view_range(config, caps) == (100e6 - 1.2e6, 100e6 + 1.2e6)


def test_adjusted_span_steps_down_by_1_5():
    config = DeviceConfig(tuned_frequency=100e6, center_frequency=100e6, sample_rate=2.4e6)
    assert adjusted_span(config, None, 1) == 2.4e6 / 1.5


def test_adjusted_span_returns_none_at_full():
    config = DeviceConfig(
        tuned_frequency=100e6, center_frequency=100e6, sample_rate=2.4e6, spectrum_span=2e6
    )
    assert adjusted_span(config, None, -1) is None


def test_adjusted_span_floors_at_min():
    config = DeviceConfig(
        tuned_frequency=100e6,
        center_frequency=100e6,
        sample_rate=2.4e6,
        spectrum_span=MIN_SPECTRUM_SPAN_HZ,
    )
    assert adjusted_span(config, None, 1) == MIN_SPECTRUM_SPAN_HZ


def test_track_dial_view_follows_tuned_not_center():
    config = DeviceConfig(
        tuned_frequency=100.5e6,
        center_frequency=100e6,
        sample_rate=2.4e6,
        spectrum_span=200e3,
    )
    assert resolve_view(config, None) == (100.5e6, 200e3)
