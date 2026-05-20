"""Diagnostic preservation tests for the SoundCard-based AudioOutputWorker.

These exercise the worker's underflow detection, max-depth cap, default-follow,
and stall heuristics without touching real audio hardware. We drive a fake
``_Player._queue`` directly to simulate the audio thread's drain behavior.
"""

from __future__ import annotations

import logging
import queue
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tsdr.core.sdr.workers import audio_worker as audio_worker_module
from tsdr.core.sdr.workers.audio_worker import AudioOutputWorker, _coerce_speaker_spec


def _make_worker(output_device: str | None = None) -> AudioOutputWorker:
    w = AudioOutputWorker(source_id="rtl0", audio_queue=queue.Queue(), output_device=output_device)
    # Stand in a fake player with a real deque; no real audio thread.
    fake_queue: deque = deque()
    fake_player = SimpleNamespace(
        _queue=fake_queue, play=lambda data, wait=True: fake_queue.append(None)
    )
    w.player = fake_player  # type: ignore[assignment]
    return w


def test_underflow_logged_on_rising_edge_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """First underflow logs WARN; subsequent depth==0 pushes increment count
    silently until depth recovers (rising-edge only)."""
    w = _make_worker()
    w._prebuffered = True
    w._push_count = w._pre_buffer_blocks + 5
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        # 3 consecutive depth==0 pushes; only the first should WARN.
        for _ in range(3):
            w.player._queue.clear()  # type: ignore[union-attr]
            w._push_block(block, max_depth=w._pre_buffer_blocks * 4)
    assert w._underflow_count == 3
    warn_lines = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert sum("audio_underflow_began" in r.getMessage() for r in warn_lines) == 1


def test_underflow_resolved_logged_on_recovery(caplog: pytest.LogCaptureFixture) -> None:
    """When depth comes back above 0, an INFO resolved line is emitted once."""
    w = _make_worker()
    w._prebuffered = True
    w._push_count = w._pre_buffer_blocks + 5
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    # Enter underflow state.
    with caplog.at_level(logging.INFO):
        w._push_block(block, max_depth=w._pre_buffer_blocks * 4)
    assert w._underrunning
    caplog.clear()
    # Next push: queue is non-empty (we filled it via _push_block's fake play()
    # and additional state setup), so we recover.
    w.player._queue.extend([None] * 3)  # type: ignore[union-attr]
    with caplog.at_level(logging.INFO):
        w._push_block(block, max_depth=w._pre_buffer_blocks * 4)
    assert not w._underrunning
    assert any("audio_underflow_resolved" in r.getMessage() for r in caplog.records)


def test_no_underflow_during_prebuffer_phase(caplog: pytest.LogCaptureFixture) -> None:
    """While _prebuffered is False, depth==0 doesn't count as underflow — we're
    accumulating in _pending_blocks; the audio thread plays silence."""
    w = _make_worker()
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        w._push_block(block, max_depth=w._pre_buffer_blocks * 4)
    assert w._underflow_count == 0
    assert not w._prebuffered
    assert len(w._pending_blocks) == 1


def test_prebuffer_flushes_all_at_once_when_full(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When _pending_blocks reaches _pre_buffer_blocks, all are pushed in one
    burst and the worker transitions to steady-state."""
    w = _make_worker()
    w._pre_buffer_blocks = 4  # small for the test
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    with caplog.at_level(logging.INFO):
        for _ in range(4):
            w._push_block(block, max_depth=16)
    assert w._prebuffered
    assert w._pending_blocks == []
    assert len(w.player._queue) == 4  # type: ignore[union-attr]
    assert any("audio_prebuffered" in r.getMessage() for r in caplog.records)


def test_no_underflow_when_queue_non_empty(caplog: pytest.LogCaptureFixture) -> None:
    w = _make_worker()
    w._prebuffered = True
    w._push_count = w._pre_buffer_blocks + 5
    # Queue has stuff → no underflow.
    w.player._queue.extend([None] * 3)  # type: ignore[union-attr]
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        w._push_block(block, max_depth=w._pre_buffer_blocks * 4)
    assert w._underflow_count == 0


def test_max_depth_cap_drops_oldest() -> None:
    w = _make_worker()
    w._prebuffered = True
    max_depth = 4
    # Pre-fill queue past the cap.
    w.player._queue.extend([f"old-{i}" for i in range(10)])  # type: ignore[union-attr]
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    w._push_block(block, max_depth=max_depth)
    # After push: trim down to max_depth-1 oldest, then append this new push.
    assert len(w.player._queue) == max_depth  # type: ignore[union-attr]
    assert w._drop_count == 10 - (max_depth - 1)


def test_max_depth_no_drop_when_under_cap() -> None:
    w = _make_worker()
    w._prebuffered = True
    w.player._queue.extend([None] * 2)  # type: ignore[union-attr]
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    w._push_block(block, max_depth=10)
    assert w._drop_count == 0
    # Queue grew by 1 from the push, no drops.
    assert len(w.player._queue) == 3  # type: ignore[union-attr]


def test_queue_depth_returns_zero_when_unavailable() -> None:
    w = _make_worker()
    w.player = SimpleNamespace(play=MagicMock())  # no _queue attribute
    assert w._queue_depth() == 0


def test_maybe_follow_default_noop_when_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _make_worker(output_device="pinned-device")
    # Even if soundcard reports a new default, we don't reopen.
    fake_sc = SimpleNamespace(
        default_speaker=lambda: SimpleNamespace(id="new", name="new"),
    )
    monkeypatch.setattr(audio_worker_module, "soundcard", fake_sc)
    open_spy = MagicMock()
    close_spy = MagicMock()
    w._open_player = open_spy  # type: ignore[method-assign]
    w._close_player = close_spy  # type: ignore[method-assign]
    w._last_default_check = 0.0
    w._speaker_id = "old"
    w._maybe_follow_default()
    open_spy.assert_not_called()
    close_spy.assert_not_called()


def test_maybe_follow_default_noop_on_unchanged_id(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _make_worker(output_device=None)
    w._speaker_id = "same"
    fake_sc = SimpleNamespace(
        default_speaker=lambda: SimpleNamespace(id="same", name="same"),
    )
    monkeypatch.setattr(audio_worker_module, "soundcard", fake_sc)
    open_spy = MagicMock()
    close_spy = MagicMock()
    w._open_player = open_spy  # type: ignore[method-assign]
    w._close_player = close_spy  # type: ignore[method-assign]
    w._last_default_check = 0.0
    w._maybe_follow_default()
    open_spy.assert_not_called()
    close_spy.assert_not_called()


def test_maybe_follow_default_reopens_on_change(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = _make_worker(output_device=None)
    w._speaker_id = "old"
    fake_sc = SimpleNamespace(
        default_speaker=lambda: SimpleNamespace(id="new", name="new"),
    )
    monkeypatch.setattr(audio_worker_module, "soundcard", fake_sc)
    open_spy = MagicMock()
    close_spy = MagicMock()
    w._open_player = open_spy  # type: ignore[method-assign]
    w._close_player = close_spy  # type: ignore[method-assign]
    w._last_default_check = 0.0
    with caplog.at_level(logging.INFO):
        w._maybe_follow_default()
    close_spy.assert_called_once()
    open_spy.assert_called_once()
    assert any("audio_default_changed" in r.getMessage() for r in caplog.records)


def test_maybe_follow_default_throttled_by_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _make_worker(output_device=None)
    w._speaker_id = "old"
    calls = []
    fake_sc = SimpleNamespace(
        default_speaker=lambda: (
            calls.append(None) or SimpleNamespace(id="old", name="x")  # type: ignore[func-returns-value]
        ),
    )
    monkeypatch.setattr(audio_worker_module, "soundcard", fake_sc)
    w._last_default_check = time.monotonic()  # just checked
    w._maybe_follow_default()
    assert calls == []  # within interval, didn't call soundcard


def test_open_player_uses_get_speaker_when_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _make_worker(output_device="my-headphones")
    captured: list[str] = []

    def fake_get_speaker(spec: str) -> object:
        captured.append(spec)
        cm = MagicMock()
        cm.__enter__ = lambda self: SimpleNamespace(_queue=deque(), play=MagicMock())
        cm.__exit__ = lambda self, *a: None
        return SimpleNamespace(
            id="hp-id",
            name="my-headphones",
            player=lambda **_kw: cm,
        )

    fake_sc = SimpleNamespace(
        default_speaker=lambda: None,
        get_speaker=fake_get_speaker,
    )
    monkeypatch.setattr(audio_worker_module, "soundcard", fake_sc)
    w._open_player()
    assert captured == ["my-headphones"]
    assert w._speaker_id == "hp-id"


def test_open_player_uses_default_when_unpinned(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _make_worker(output_device=None)
    cm = MagicMock()
    cm.__enter__ = lambda self: SimpleNamespace(_queue=deque(), play=MagicMock())
    cm.__exit__ = lambda self, *a: None
    fake_sc = SimpleNamespace(
        default_speaker=lambda: SimpleNamespace(
            id="default-id", name="default", player=lambda **_kw: cm
        ),
        get_speaker=lambda _: None,
    )
    monkeypatch.setattr(audio_worker_module, "soundcard", fake_sc)
    w._open_player()
    assert w._speaker_id == "default-id"


def test_upstream_stall_logged_on_large_gap(caplog: pytest.LogCaptureFixture) -> None:
    """gap > batch_duration*1.5 AND queue under prebuffer → warn."""
    w = _make_worker()
    w._last_batch_time = time.perf_counter() - 0.5
    with caplog.at_level(logging.WARNING):
        w._maybe_log_upstream_stall(time.perf_counter(), batch_duration=0.1)
    assert any("audio_upstream_stall" in r.getMessage() for r in caplog.records)


def test_upstream_stall_silent_when_buffer_full(caplog: pytest.LogCaptureFixture) -> None:
    """Large gap but queue still high → no warning."""
    w = _make_worker()
    w._last_batch_time = time.perf_counter() - 0.5
    w.player._queue.extend([None] * (w._pre_buffer_blocks + 5))  # type: ignore[union-attr]
    with caplog.at_level(logging.WARNING):
        w._maybe_log_upstream_stall(time.perf_counter(), batch_duration=0.1)
    assert not any("audio_upstream_stall" in r.getMessage() for r in caplog.records)


def test_glitch_aggregate_logged_after_window(caplog: pytest.LogCaptureFixture) -> None:
    w = _make_worker()
    w._underflow_count = 4
    w._last_glitch_underflows = 1
    w._drop_count = 2
    w._cumulative_input_duration = 1.0
    w._cumulative_output_duration = 1.001
    w._last_stats_time = time.perf_counter() - 5.5
    with caplog.at_level(logging.INFO):
        w._maybe_emit_glitch_aggregate()
    assert any("audio_glitch" in r.getMessage() for r in caplog.records)
    assert w._last_glitch_underflows == w._underflow_count


def test_glitch_silent_when_no_new_underflows(caplog: pytest.LogCaptureFixture) -> None:
    w = _make_worker()
    w._underflow_count = 4
    w._last_glitch_underflows = 4
    w._last_stats_time = time.perf_counter() - 5.5
    with caplog.at_level(logging.INFO):
        w._maybe_emit_glitch_aggregate()
    assert not any("audio_glitch" in r.getMessage() for r in caplog.records)


def test_glitch_silent_before_window(caplog: pytest.LogCaptureFixture) -> None:
    w = _make_worker()
    w._underflow_count = 4
    w._last_glitch_underflows = 1
    w._last_stats_time = time.perf_counter() - 2.0  # < 5s
    with caplog.at_level(logging.INFO):
        w._maybe_emit_glitch_aggregate()
    assert not any("audio_glitch" in r.getMessage() for r in caplog.records)


def test_coerce_speaker_spec_int_passthrough() -> None:
    assert _coerce_speaker_spec("86") == 86
    assert _coerce_speaker_spec("MacBook") == "MacBook"
    assert _coerce_speaker_spec("0") == 0


def test_teardown_logs_summary(caplog: pytest.LogCaptureFixture) -> None:
    w = _make_worker()
    w._push_count = 100
    w._underflow_count = 3
    w._drop_count = 5
    w._total_frames_in = 50_000
    w._total_frames_out = 51_200
    w._stream_start_time = time.perf_counter() - 1.0
    with caplog.at_level(logging.INFO):
        w.teardown(MagicMock())
    summary = next(r for r in caplog.records if "audio_teardown" in r.getMessage())
    msg = summary.getMessage()
    assert "pushes=100" in msg
    assert "underflows=3" in msg
    assert "dropped=5" in msg
    assert "frames_in=50000" in msg
    assert "frames_out=51200" in msg
