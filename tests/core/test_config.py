"""DeviceConfig read-sizing: the per-read sample count must track the rate the
device actually delivers, not the configured rate, so a low-rate device (e.g. a
12 kHz KiwiSDR against the 2.4 MHz default) doesn't size its first read past its
jitter-buffer capacity before the engine snaps the rate.
"""

import pytest

from tsdr.core.sdr.config import DeviceConfig


def test_buffer_samples_for_uses_delivered_rate():
    config = DeviceConfig(sample_rate=2_400_000.0, target_fps=20.0)
    assert config.buffer_samples_for(12_000.0) == 600
    assert config.effective_buffer_samples == 120_000


def test_buffer_samples_for_falls_back_when_no_delivered_rate():
    config = DeviceConfig(sample_rate=48_000.0, target_fps=20.0)
    assert config.buffer_samples_for(0.0) == 2_400


def test_buffer_samples_for_respects_explicit_override():
    config = DeviceConfig(sample_rate=2_400_000.0, target_fps=20.0, buffer_samples=4096)
    assert config.buffer_samples_for(12_000.0) == 4096


def test_validate_accepts_default_fft_geometry():
    DeviceConfig().validate()


def test_validate_rejects_non_power_of_two_fft_size():
    with pytest.raises(ValueError, match="power of 2"):
        DeviceConfig(fft_size=3000).validate()


@pytest.mark.parametrize("bad", [32, 131_072])
def test_validate_rejects_out_of_range_fft_size(bad: int):
    with pytest.raises(ValueError, match="between 64 and 65536"):
        DeviceConfig(fft_size=bad).validate()


def test_validate_rejects_unknown_fft_window():
    with pytest.raises(ValueError, match="fft_window"):
        DeviceConfig(fft_window="gaussian").validate()


def test_validate_rejects_non_positive_spectrum_span():
    with pytest.raises(ValueError, match="spectrum_span"):
        DeviceConfig(spectrum_span=0.0).validate()


def test_validate_rejects_non_positive_spectrum_center():
    with pytest.raises(ValueError, match="spectrum_center"):
        DeviceConfig(spectrum_center=-1.0).validate()


def test_validate_accepts_none_view():
    DeviceConfig(spectrum_center=None, spectrum_span=None).validate()
