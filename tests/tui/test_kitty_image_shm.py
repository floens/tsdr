"""Shared-memory lifetime rules for KittyImageWidget (platform-specific).

Both platforms transmit over shared memory (kitty t=s) for performance, but the
producer side differs because the terminal unlinks on POSIX and only closes on
Windows (kitty spec):

* POSIX recreates the named object every frame (the terminal unlinks it after
  reading, so the name is gone next frame) and closes our handle immediately.
* Windows keeps one mapping alive and overwrites it in place; capacity grows
  geometrically so a steadily growing image reallocates O(log) times, and a
  superseded mapping is retired (closed a few frames later) to avoid racing the
  terminal's async read.

See ``kitty_image._write_shm`` / ``_release_shm``.
"""

from __future__ import annotations

import sys

import numpy as np

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
    try:
        widget._write_shm(entry, _flat(16))
        assert entry.pending_shm is not None
        assert entry.shm_name == "tsdr_1"  # fixed name reused on POSIX

        # _release_shm closes our handle immediately (the terminal unlinks).
        widget._release_shm(entry)
        assert entry.pending_shm is None
    finally:
        if entry.pending_shm is not None:
            entry.pending_shm.close()
            entry.pending_shm.unlink()


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
