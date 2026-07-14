"""project_spectrum maps an event's bins into an arbitrary view range;
out-of-event bins become -inf so they normalize to 0 (blank bars)."""

import numpy as np

from tsdr.tui.widgets.dsp_utils import (
    _TRACE_IIR_MIN_RATE,
    iir_trace_filter,
    project_spectrum,
    transient_view_shift,
)


def test_identity_range_decimates():
    spectrum = np.arange(100, dtype=np.float32)
    out = project_spectrum(spectrum, 0.0, 100.0, 0.0, 100.0, 50)
    assert len(out) == 50
    assert np.all(np.isfinite(out))


def test_shifted_view_maps_bins():
    spectrum = np.arange(100, dtype=np.float32)
    # View is the upper half of the event range; bins sample at their centers,
    # so output bin 0 covers view freq 52.5 -> source bin 52.
    out = project_spectrum(spectrum, 0.0, 100.0, 50.0, 100.0, 10)
    assert np.all(out >= 50.0)
    assert out[0] == spectrum[52]


def test_zoomed_view_stretches():
    spectrum = np.arange(10, dtype=np.float32)
    out = project_spectrum(spectrum, 0.0, 100.0, 40.0, 60.0, 20)
    # 2 source bins stretched over 20 output bins
    assert set(out.tolist()) <= {4.0, 5.0, 6.0}


def test_non_overlapping_view_is_blank():
    spectrum = np.ones(10, dtype=np.float32)
    out = project_spectrum(spectrum, 0.0, 10.0, 100.0, 110.0, 8)
    assert np.all(np.isneginf(out))


def test_partial_overlap_blanks_edges():
    spectrum = np.ones(10, dtype=np.float32)
    out = project_spectrum(spectrum, 0.0, 10.0, 5.0, 15.0, 10)
    assert np.all(out[:5] == 1.0)
    assert np.all(np.isneginf(out[5:]))


def test_transient_shift_anchors_zoomed_view_to_stale_capture():
    # Dial (and view) moved to 110 while the hardware still captured at 100:
    # the zoomed crop stays where it was relative to the data.
    shifted = transient_view_shift((90.0, 130.0), event_center_hz=100.0, capture_center_hz=110.0)
    assert shifted == (80.0, 120.0)


def test_transient_shift_makes_full_band_render_identity():
    # Unzoomed: the shifted window equals the event range, so projection hits
    # the 1:1 fast path — the pre-VFO "bars lag the axis" behavior.
    spectrum = np.arange(100, dtype=np.float32)
    view = (110.0 - 50.0, 110.0 + 50.0)  # tracks the dial at 110
    shifted = transient_view_shift(view, event_center_hz=100.0, capture_center_hz=110.0)
    assert shifted == (50.0, 150.0)
    out = project_spectrum(spectrum, 50.0, 150.0, *shifted, 50)
    assert np.array_equal(out, project_spectrum(spectrum, 50.0, 150.0, 50.0, 150.0, 50))


def test_transient_shift_is_identity_in_steady_state():
    assert transient_view_shift((90.0, 130.0), 100.0, 100.0) == (90.0, 130.0)


def test_empty_event_is_blank():
    out = project_spectrum(np.empty(0, dtype=np.float32), 0.0, 0.0, 0.0, 10.0, 4)
    assert np.all(np.isneginf(out))


_DT = 1.0 / 23.0  # kiwi's design frame rate


def test_iir_trace_filter_seeds_from_first_frame():
    z = np.full(8, 0.5, dtype=np.float32)
    avg = iir_trace_filter(None, z, _DT)
    assert np.allclose(avg, 0.5)
    assert avg is not z  # own copy; caller's buffer stays untouched


def test_iir_trace_filter_attack_outpaces_decay():
    # Rise 0->1 and fall 1->0 over the same elapsed time: the value-dependent
    # rate must make the rise land closer to its target than the fall
    # (holds whenever the full-scale rate exceeds the floor rate, regardless
    # of the tuning values).
    up = iir_trace_filter(None, np.zeros(4, dtype=np.float32), _DT)
    down = iir_trace_filter(None, np.ones(4, dtype=np.float32), _DT)
    for _ in range(10):
        up = iir_trace_filter(up, np.ones(4, dtype=np.float32), _DT)
        down = iir_trace_filter(down, np.zeros(4, dtype=np.float32), _DT)
    assert np.all(up > 1.0 - down)


def test_iir_trace_filter_decay_follows_min_rate():
    # At z=0 the trace decays as exp(-_TRACE_IIR_MIN_RATE * t) exactly, in
    # wall-clock seconds — pins the per-second semantics without pinning the
    # tuning value itself.
    avg = iir_trace_filter(None, np.ones(4, dtype=np.float32), _DT)
    for _ in range(10):
        avg = iir_trace_filter(avg, np.zeros(4, dtype=np.float32), 0.1)
    assert np.allclose(avg, np.exp(-_TRACE_IIR_MIN_RATE), atol=1e-5)


def test_iir_trace_filter_rate_is_frame_rate_independent():
    # Same elapsed time at 23 fps and at 10 fps-ish must decay equally: the
    # wf_share servers deliver ~10 fps and must not double the time constant.
    fast = iir_trace_filter(None, np.ones(4, dtype=np.float32), _DT)
    slow = fast.copy()
    for _ in range(20):
        fast = iir_trace_filter(fast, np.zeros(4, dtype=np.float32), 0.05)
    for _ in range(10):
        slow = iir_trace_filter(slow, np.zeros(4, dtype=np.float32), 0.1)
    assert np.allclose(fast, slow, atol=1e-5)


def test_iir_trace_filter_reseeds_on_shape_change():
    avg = iir_trace_filter(None, np.zeros(4, dtype=np.float32), _DT)
    z = np.full(8, 0.3, dtype=np.float32)
    avg = iir_trace_filter(avg, z, _DT)
    assert avg.shape == (8,)
    assert np.allclose(avg, 0.3)
