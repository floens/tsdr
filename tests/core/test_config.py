"""DeviceConfig read-sizing: the per-read sample count must track the rate the
device actually delivers, not the configured rate, so a low-rate device (e.g. a
12 kHz KiwiSDR against the 2.4 MHz default) doesn't size its first read past its
jitter-buffer capacity before the engine snaps the rate.
"""

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
