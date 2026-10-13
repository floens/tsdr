"""derive_center_frequency: when the hardware center follows the dial, and where
the LO offset puts it."""

from dataclasses import replace

from tsdr.core.sdr.tune_policy import MIN_LO_OFFSET_HZ, derive_center_frequency, lo_offset_for
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


# LO offset: keep a zero-IF radio's LO beside the demodulated channel.


def _lo(
    setting: str = "auto",
    *,
    zero_if: bool = True,
    size_hz: float | None = None,
    channel_bandwidth: float | None = 12.5e3,
    sample_rate: float = 2.4e6,
) -> float:
    caps = replace(_caps(), zero_if=zero_if)
    return lo_offset_for(
        setting=setting,  # type: ignore[arg-type]
        size_hz=size_hz,
        channel_bandwidth=channel_bandwidth,
        sample_rate=sample_rate,
        caps=caps,
    )


def test_lo_offset_auto_sits_below_on_zero_if() -> None:
    assert _lo() == -MIN_LO_OFFSET_HZ


def test_lo_offset_auto_is_off_on_low_if() -> None:
    assert _lo(zero_if=False) == 0.0


def test_lo_offset_zero_without_a_demodulator() -> None:
    assert _lo("below", channel_bandwidth=None) == 0.0


def test_lo_offset_explicit_sides() -> None:
    assert _lo("above", zero_if=False) == MIN_LO_OFFSET_HZ
    assert _lo("below", zero_if=False) == -MIN_LO_OFFSET_HZ
    assert _lo("off") == 0.0


def test_lo_offset_grows_with_wide_channels() -> None:
    assert _lo(channel_bandwidth=200e3, sample_rate=1e6) == -200e3


def test_lo_offset_size_override() -> None:
    assert _lo(size_hz=30e3) == -30e3


def test_lo_offset_skipped_when_channel_fills_capture() -> None:
    # DAB: 1.536 MHz channel in a 2.048 MS/s capture has no room beside the LO.
    assert _lo(channel_bandwidth=1.536e6, sample_rate=2.048e6) == 0.0


def test_center_mode_puts_lo_beside_dial() -> None:
    assert _derive(100e6, mode="center", lo_offset=-50e3) == 100e6 - 50e3


def test_lo_offset_mirrors_at_bottom_of_range() -> None:
    # Range starts at 24 MHz: LO below a 24.02 MHz dial would leave it.
    assert _derive(24.02e6, mode="center", lo_offset=-50e3) == 24.02e6 + 50e3


def test_free_mode_keeps_fitting_center_despite_offset() -> None:
    assert _derive(100.5e6, lo_offset=-50e3) == 100e6


def test_free_mode_recenter_lands_beside_dial() -> None:
    assert _derive(101.2e6, lo_offset=-50e3) == 101.2e6 - 50e3


def test_free_mode_recenter_flag_moves_lo() -> None:
    assert _derive(100.5e6, lo_offset=50e3, recenter=True) == 100.5e6 + 50e3


def test_stopped_device_recenters_beside_dial() -> None:
    assert _derive(100.001e6, running=False, lo_offset=-50e3) == 100.001e6 - 50e3


def test_lo_offset_ignored_for_fixed_or_server_tuned_devices() -> None:
    assert _derive(100.001e6, mode="center", caps=_caps(controllable=False), lo_offset=-50e3) == (
        100e6
    )
    assert _derive(100.001e6, caps=_caps(provides_spectrum=True), lo_offset=-50e3) == 100.001e6
