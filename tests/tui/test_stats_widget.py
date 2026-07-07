"""StatsWidget rendering: smoothed dBFS level, actual/requested sample rate,
audio-underrun surfacing, and no Queue section."""

from __future__ import annotations

import math

from tsdr.core.events.events import AudioOutputStatsEvent, StatsUpdateEvent
from tsdr.tui.widgets.stats_widget import StatsWidget


def _widget(**fields) -> StatsWidget:
    w = StatsWidget.__new__(StatsWidget)
    w.current_event = None
    w._channel_bandwidth = None
    w._device_id = "rtl0"
    w._jitter = None
    w._network_address = None
    w._identity_label = None
    w._identity_serial = None
    w._frequency_range = None
    w._frequency_controllable = False
    w._controller_center_frequency = None
    w._controller_gain = None
    w._wire_bytes_per_sec = 0.0
    w._device_state = None
    w._requested_sample_rate = None
    w._level_dbfs = None
    w._audio_underflows = 0
    w._audio_drops = 0
    for k, v in fields.items():
        setattr(w, k, v)
    return w


def _event(**over) -> StatsUpdateEvent:
    base = {
        "device_id": "rtl0",
        "center_frequency": 1e8,
        "sample_rate": 2_400_000.0,
        "rf_gain": 20.0,
        "samples_processed": 1,
        "samples_dropped": 0,
        "queue_size": 1,
        "queue_capacity": 8,
        "peak_power": -10.0,
        "average_power": -40.0,
        "peak_frequency": 1e8,
        "peak_bin": 1,
        "noise_floor": -80.0,
        "dynamic_range": 70.0,
        "fft_size": 4096,
        "fft_window": "blackman",
        "spectrum_bins": 4096,
        "demod_mode": "WFM",
        "channel_snr": 25.0,
        "stereo": True,
        "iq_rms": 0.24,
        "iq_peak": 0.5,
        "iq_clip_pct": 0.0,
        "update_rate_fps": 20,
    }
    base.update(over)
    return StatsUpdateEvent(**base)


def test_no_queue_section() -> None:
    w = _widget(current_event=_event())
    bodies = "\n".join(s.body for s in w.build_sections())
    assert "Queue" not in bodies
    assert "Utilization" not in bodies


def test_level_line_is_single_smoothed_dbfs() -> None:
    w = _widget()
    # Feed a steady rms; the EMA should converge to its dBFS and stay one line.
    for _ in range(200):
        w._update_level(_event(iq_rms=0.1))
    expected = 20.0 * math.log10(0.1)  # -20 dBFS
    assert w._level_dbfs == expected  # converged
    lines = w._signal_lines(_event(iq_rms=0.1))
    body = [ln for ln in lines if "Level:" in ln]
    assert len(body) == 1 and "dBFS" in body[0]
    assert "peak" not in body[0] and "rms" not in body[0]  # not the old dual display


def test_sample_rate_shows_both_and_flags_mismatch() -> None:
    w = _widget(_requested_sample_rate=2_400_000.0)
    matched = w._sample_rate_value(_event(sample_rate=2_400_000.0))
    assert "2.40 / 2.40" in matched and "yellow" not in matched

    w2 = _widget(_requested_sample_rate=2_400_000.0)
    mismatch = w2._sample_rate_value(_event(sample_rate=2_048_000.0))
    assert "2.05" in mismatch and "2.40" in mismatch and "yellow" in mismatch


def test_audio_underruns_rendered() -> None:
    healthy = _widget(_audio_underflows=0)._audio_lines(_event())
    ur = [ln for ln in healthy if "Underruns:" in ln]
    assert len(ur) == 1 and "green" in ur[0]
    assert not any("Drops:" in ln for ln in healthy)  # hidden when zero

    bad = _widget(_audio_underflows=5, _audio_drops=2)._audio_lines(_event())
    ur = [ln for ln in bad if "Underruns:" in ln]
    dr = [ln for ln in bad if "Drops:" in ln]
    assert "red" in ur[0] and "5" in ur[0]
    assert len(dr) == 1 and "red" in dr[0] and "2" in dr[0]


def test_update_audio_stats_filters_by_focused_device() -> None:
    w = _widget(_device_id="rtl0")
    w.refresh = lambda *a, **k: None  # type: ignore[method-assign]
    w.update_audio_stats(AudioOutputStatsEvent(device_id="other", underflow_count=9, drop_count=1))
    assert w._audio_underflows == 0  # ignored: not the focused device
    w.update_audio_stats(AudioOutputStatsEvent(device_id="rtl0", underflow_count=9, drop_count=1))
    assert w._audio_underflows == 9 and w._audio_drops == 1
