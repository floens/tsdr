"""Shared-memory lifetime rules for KittyImageWidget (platform-specific).

Both platforms transmit over shared memory (kitty t=s) for performance, but the
producer side differs because the terminal unlinks on POSIX and only closes on
Windows (kitty spec):

* POSIX creates a unique object for each frame and closes our handle immediately.
  Queued frames remain intact until the terminal reads and unlinks each object.
* Windows keeps one mapping alive and overwrites it in place; capacity grows
  geometrically so a steadily growing image reallocates O(log) times, and a
  superseded mapping is retired (closed a few frames later) to avoid racing the
  terminal's async read.

See ``kitty_image._write_shm`` / ``_release_shm``.
"""

from __future__ import annotations

import sys
from collections import deque
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from tsdr.tui.widgets import kitty_image
from tsdr.tui.widgets.kitty_image import KittyImageWidget, _ImageEntry


def _entry() -> _ImageEntry:
    e = _ImageEntry(key="k", image_id=1, shm_name="tsdr_1")
    e.width, e.height = 2, 2
    return e


def _flat(n: int, start: int = 0) -> np.ndarray:
    return np.arange(start, start + n, dtype=np.uint8)


def test_windows_reuses_one_mapping_overwritten_in_place() -> None:
    if sys.platform != "win32":
        return
    widget = KittyImageWidget(id="img")
    entry = _entry()
    a, b = _flat(16), _flat(16, 16)
    try:
        widget._write_shm(entry, a)
        first = entry.pending_shm
        first_name = entry.shm_name
        assert first is not None
        assert bytes(first.buf[:16]) == a.tobytes()

        # Same size next frame: the mapping (and name) is reused, overwritten —
        # no reallocation, nothing retired.
        widget._write_shm(entry, b)
        assert entry.pending_shm is first
        assert entry.shm_name == first_name
        assert bytes(first.buf[:16]) == b.tobytes()
        assert widget._retired_shms == []

        # _release_shm keeps the Windows handle open (the terminal can't unlink).
        widget._release_shm(entry)
        assert entry.pending_shm is first
    finally:
        if entry.pending_shm is not None:
            entry.pending_shm.close()


def test_windows_grows_geometrically_and_retires_old_mapping() -> None:
    if sys.platform != "win32":
        return
    widget = KittyImageWidget(id="img")
    entry = _entry()
    try:
        widget._write_shm(entry, _flat(16))
        small_name = entry.shm_name
        assert entry.shm_capacity == 16

        widget._write_shm(entry, _flat(20))  # grow: capacity doubles past 16 -> 32
        assert entry.shm_name != small_name  # fresh name avoids collision
        assert entry.shm_capacity == 32  # geometric (max(20, 16*2))
        assert len(widget._retired_shms) == 1  # old mapping retired, not closed yet

        # A larger frame that still fits the 32-byte buffer must NOT reallocate.
        name_after_grow = entry.shm_name
        widget._write_shm(entry, _flat(32))
        assert entry.shm_name == name_after_grow
        assert entry.shm_capacity == 32

        # Retired mappings close after _SHM_RETIRE_FRAMES render passes.
        for _ in range(widget._SHM_RETIRE_FRAMES):
            widget._age_retired_shms()
        assert widget._retired_shms == []
    finally:
        if entry.pending_shm is not None:
            entry.pending_shm.close()


def test_posix_recreates_and_releases_each_frame() -> None:
    if sys.platform == "win32":
        return
    widget = KittyImageWidget(id="img")
    entry = _entry()
    reader = None
    try:
        widget._write_shm(entry, _flat(16))
        assert entry.pending_shm is not None
        # _release_shm closes our handle immediately (the terminal unlinks).
        widget._release_shm(entry)
        assert entry.pending_shm is None
        reader = SharedMemory(name=entry.shm_name, track=False)
        assert bytes(reader.buf) == _flat(16).tobytes()
    finally:
        if reader is not None:
            reader.close()
            reader.unlink()
        if entry.pending_shm is not None:
            entry.pending_shm.close()
            entry.pending_shm.unlink()


def test_posix_queued_frames_keep_distinct_buffers(monkeypatch) -> None:
    """A slow terminal can read both frames after the producer has resized."""
    if sys.platform == "win32":
        return
    widget = KittyImageWidget(id="img")
    captured: list[str] = []
    monkeypatch.setattr(widget, "_queue_cmd", captured.append)
    frames = [np.full((32, 32, 4), 127, dtype=np.uint8), np.zeros((1, 1, 4), dtype=np.uint8)]
    names: list[str] = []
    try:
        for frame in frames:
            widget.update_image("k", frame)
            names.append(widget._images["k"].shm_name)
        assert len(set(names)) == 2
        assert len(captured) == 2
        for name, frame in zip(names, frames, strict=True):
            reader = SharedMemory(name=name, track=False)
            try:
                assert reader.size == frame.nbytes
                assert bytes(reader.buf) == frame.tobytes()
            finally:
                reader.close()
    finally:
        for name in set(names):
            reader = SharedMemory(name=name, track=False)
            reader.close()
            reader.unlink()


def test_update_image_transmits_t_s_shared_memory(monkeypatch) -> None:
    """Both platforms emit a t=s (shared memory) escape — never inline base64."""
    widget = KittyImageWidget(id="img")
    captured: list[str] = []
    monkeypatch.setattr(widget, "_queue_cmd", lambda cmd: captured.append(cmd))
    widget.update_image("k", np.zeros((2, 2, 4), dtype=np.uint8))
    try:
        assert len(captured) == 1
        assert "t=s" in captured[0]
        assert "t=d" not in captured[0]  # no inline base64 on the hot path
        assert "f=32" in captured[0]
        # Exact byte count (2*2*4=16) so an over-allocated / page-rounded mapping
        # isn't read past the image data.
        assert "S=16," in captured[0]
    finally:
        entry = widget._images["k"]
        if entry.pending_shm is not None:
            entry.pending_shm.close()
            if sys.platform != "win32":
                entry.pending_shm.unlink()
        elif sys.platform != "win32":
            reader = SharedMemory(name=entry.shm_name, track=False)
            reader.close()
            reader.unlink()


def _exists(name: str) -> bool:
    try:
        SharedMemory(name=name, track=False).close()
    except FileNotFoundError:
        return False
    return True


def test_posix_unlinks_frames_the_terminal_never_reads(monkeypatch) -> None:
    """Only the terminal's read unlinks a frame; frames it never reads (no
    terminal here) are unlinked once they fall behind, and on unmount."""
    if sys.platform == "win32":
        return
    monkeypatch.setattr(kitty_image, "_unread_frames", deque())
    widget = KittyImageWidget(id="img")
    monkeypatch.setattr(widget, "_queue_cmd", lambda cmd: None)
    data = np.zeros((2, 2, 4), dtype=np.uint8)
    names = []
    for _ in range(kitty_image._MAX_UNREAD_FRAMES + 10):
        widget.update_image("k", data)
        names.append(widget._images["k"].shm_name)
    assert [n for n in names if _exists(n)] == names[-kitty_image._MAX_UNREAD_FRAMES :]
    widget.on_unmount()
    assert not any(_exists(n) for n in names)


def test_unlink_tolerates_the_terminal_unlinking_first(monkeypatch) -> None:
    """The terminal can unlink a frame between our open and our unlink."""
    if sys.platform == "win32":
        return
    shm = SharedMemory(create=True, size=16, track=False)
    real_unlink = SharedMemory.unlink

    def terminal_got_there_first(self: SharedMemory) -> None:
        real_unlink(self)
        real_unlink(self)  # ours, now of a name that is gone

    monkeypatch.setattr(SharedMemory, "unlink", terminal_got_there_first)
    try:
        kitty_image._unlink_frame(shm.name)
    finally:
        shm.close()
    assert not _exists(shm.name)
