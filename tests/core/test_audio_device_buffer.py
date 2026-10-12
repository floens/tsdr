"""AudioOutputWorker on backends that buffer in the device (PulseAudio, WASAPI).

SoundCard sizes their device buffer from the player's blocksize and blocks in
play() until there is room. A 512-frame block left ~10 ms of headroom, so any
longer stall of the worker thread (a GC pause, the UI holding the GIL) was an
audible gap: the TUI at 1 MS/s dropped audio several times per second.
"""

from __future__ import annotations

import queue
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tsdr.core.sdr.workers import audio_worker as audio_worker_module
from tsdr.core.sdr.workers.audio_worker import AudioOutputWorker, _device_room_reader

RATE = AudioOutputWorker.TARGET_RATE
BATCH = np.zeros((2400, 2), dtype=np.float32)  # 50 ms


@pytest.fixture(autouse=True)
def _device_buffer_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_worker_module, "_CLIENT_QUEUE_BACKEND", False)


class _DevicePlayer:
    """A device buffer of `capacity` frames that starts playing in real time
    once it first fills (PulseAudio's prebuffer) and frees room a `quantum`
    at a time; play() blocks until there is room, and reports the room free
    the way SoundCard's WASAPI player does. `stall_next` delays the next call
    although there may be room: a GC pause, the GIL. Waits busy-wait, since
    tests patch time.sleep to observe the worker's pacing."""

    def __init__(self, capacity_frames: int, *, quantum: int = 512) -> None:
        self.capacity = capacity_frames
        self.quantum = quantum
        self.stall_next = 0.0
        self.queued = 0
        self.playing_since: float | None = None
        self.calls: list[int] = []
        self.waited: list[bool] = []

    def _played(self) -> int:
        if self.playing_since is None:
            return 0
        frames = int((time.perf_counter() - self.playing_since) * RATE)
        return min(self.queued, frames // self.quantum * self.quantum)

    def _render_available_frames(self) -> int:
        return self.capacity - (self.queued - self._played())

    def play(self, data) -> None:
        self.calls.append(len(data))
        until = time.perf_counter() + self.stall_next
        self.stall_next = 0.0
        while time.perf_counter() < until:
            pass
        self.waited.append(self._render_available_frames() < len(data))
        while self._render_available_frames() < len(data):
            pass
        self.queued += len(data)
        if self.playing_since is None and self.queued + 512 > self.capacity:
            self.playing_since = time.perf_counter()


class _Speaker:
    def __init__(self, player) -> None:
        self.id = "fake"
        self.name = "fake"
        self._player = player
        self.requested_blocksize: int | None = None

    def player(self, *, samplerate, channels, blocksize):
        self.requested_blocksize = blocksize
        player = self._player
        return SimpleNamespace(__enter__=lambda: player, __exit__=lambda *a: None)


def _open(
    monkeypatch: pytest.MonkeyPatch,
    capacity_s: float,
    *,
    quantum: int = 512,
) -> tuple[AudioOutputWorker, _DevicePlayer]:
    player = _DevicePlayer(int(capacity_s * RATE), quantum=quantum)
    _install_speaker(monkeypatch, _Speaker(player))
    w = AudioOutputWorker(source_id="d", audio_queue=queue.Queue())
    w._open_player()
    return w, player


def _install_speaker(monkeypatch: pytest.MonkeyPatch, speaker: _Speaker) -> None:
    monkeypatch.setattr(
        audio_worker_module,
        "soundcard",
        SimpleNamespace(default_speaker=lambda: speaker, get_speaker=lambda spec: speaker),
    )


def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    real_sleep = time.sleep

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        real_sleep(seconds)

    monkeypatch.setattr(audio_worker_module.time, "sleep", sleep)
    return sleeps


def test_open_requests_a_deep_device_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    _open(monkeypatch, capacity_s=0.25)
    speaker = audio_worker_module.soundcard.default_speaker()
    assert speaker.requested_blocksize / RATE >= audio_worker_module._MIN_DEVICE_BUFFER_SECONDS


def test_open_uses_the_granted_capacity_and_starts_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """PipeWire grants ~60% of the request; pacing must use what it really holds."""
    w, player = _open(monkeypatch, capacity_s=0.24)
    assert w._device_frames == player.capacity
    assert player.queued == player.capacity
    assert w._prebuffered


@pytest.mark.parametrize("quantum", [512, 2048], ids=["quantum-512", "pipewire-2048"])
def test_write_into_full_buffer_sleeps_instead_of_polling(
    monkeypatch: pytest.MonkeyPatch, quantum: int
) -> None:
    w, player = _open(monkeypatch, capacity_s=0.24, quantum=quantum)
    sleeps = _record_sleeps(monkeypatch)

    w._raw_push(BATCH, max_depth=0)

    assert sum(sleeps) == pytest.approx(0.05, abs=quantum / RATE)
    assert player.waited[-1] is False


def test_write_with_room_does_not_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    w, _ = _open(monkeypatch, capacity_s=0.24)
    time.sleep(0.08)  # the device plays 80 ms
    sleeps = _record_sleeps(monkeypatch)

    w._raw_push(BATCH, max_depth=0)

    assert sleeps == []


def test_a_slow_write_is_not_taken_for_a_full_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the UI holding the GIL, play() is often slow although the device
    has room. Read as "the buffer is full", that held audio back while the
    device ran dry: live, 30-40 ms gaps several times a second, uncounted."""
    w, player = _open(monkeypatch, capacity_s=0.24)
    time.sleep(0.16)  # the device plays 160 ms
    player.stall_next = 0.01
    w._raw_push(BATCH, max_depth=0)  # a slow write, with room
    sleeps = _record_sleeps(monkeypatch)

    w._raw_push(BATCH, max_depth=0)

    assert sleeps == []


def test_underflow_counted_when_device_buffer_ran_dry(monkeypatch: pytest.MonkeyPatch) -> None:
    w, _ = _open(monkeypatch, capacity_s=0.1)
    w._push_block(BATCH, max_depth=0)
    assert w._underflow_count == 0

    time.sleep(0.25)  # the worker stalled; the device played everything
    w._push_block(BATCH, max_depth=0)
    assert w._underflow_count == 1


def _batch_queue(n: int) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    for _ in range(n):
        q.put(object())
    return q


def test_backlog_within_two_buffers_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    w, _ = _open(monkeypatch, capacity_s=0.25)
    w.audio_queue = _batch_queue(9)  # 0.45 s of 50 ms batches
    w._trim_backlog(0.05)
    assert w.audio_queue.qsize() == 9
    assert w._drop_count == 0


def test_excess_backlog_is_trimmed_to_half_a_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    w, _ = _open(monkeypatch, capacity_s=0.25)
    w.audio_queue = _batch_queue(20)  # 1 s backlog after a long upstream stall
    w._trim_backlog(0.05)
    assert w.audio_queue.qsize() * 0.05 <= w._device_buffer_seconds / 2
    assert w._drop_count > 0


def test_a_stall_of_this_thread_is_not_backlog(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stall grows the queue and drains the device by the same amount; it
    once threw away 0.4 s of audio every few seconds under load."""
    w, _ = _open(monkeypatch, capacity_s=0.25)
    time.sleep(0.2)  # the worker stalled 200 ms while batches kept coming
    w.audio_queue = _batch_queue(9 + 4)
    w._trim_backlog(0.05)
    assert w.audio_queue.qsize() == 13
    assert w._drop_count == 0


class _BlockingPlayer:
    """A device that doesn't report its room: play() simply blocks."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def play(self, data) -> None:
        self.calls.append(len(data))


def test_a_backend_without_room_reporting_falls_back_to_blocking_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player = _BlockingPlayer()
    _install_speaker(monkeypatch, _Speaker(player))
    w = AudioOutputWorker(source_id="d", audio_queue=queue.Queue())
    w._open_player()
    assert w._device_room is None
    assert player.calls == []  # no priming: the prebuffer path fills it
    assert not w._prebuffered

    sleeps = _record_sleeps(monkeypatch)
    w._prebuffered = True
    w._push_block(BATCH, max_depth=0)
    assert sleeps == []
    assert w._underflow_count == 0


class _FakePulseLib:
    def __init__(self, writable: int) -> None:
        self.writable = writable
        self.locked = 0

    def pa_threaded_mainloop_lock(self, mainloop) -> None:
        self.locked += 1

    def pa_threaded_mainloop_unlock(self, mainloop) -> None:
        self.locked -= 1

    def pa_stream_writable_size(self, stream) -> int:
        assert self.locked == 1
        return self.writable


def test_pulseaudio_room_is_the_writable_size_in_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = _FakePulseLib(writable=2048 * 8)
    pulse = SimpleNamespace(_pa=lib, _pulse=SimpleNamespace(mainloop=object()))
    monkeypatch.setitem(sys.modules, "soundcard.pulseaudio", pulse)
    room = _device_room_reader(SimpleNamespace(stream=object(), channels=2))
    assert room is not None
    assert room() == 2048
    assert lib.locked == 0

    lib.writable = 2**64 - 1  # (size_t)-1: an error
    assert room() == 0
