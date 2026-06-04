"""Audible test tone for the interactive doctor.

Plays a plain sine tone once per trigger through the default output device on a
daemon thread, so the user can confirm audio works. Gated: each keypress plays
the tone a single time (a second press while it is sounding stops it); it does
not loop. Reuses the same backend (``soundcard``, 48 kHz, stereo) as the real
audio worker.
"""

import logging
import threading
from collections.abc import Callable

import numpy as np
import soundcard

logger = logging.getLogger(__name__)

_RATE = 48000
_AMP = 0.18
_FREQ = 440.0  # A4
_DURATION = 1.5  # seconds
_CHUNK = 2048  # playback granularity (frames) so a stop request interrupts promptly


def _make_block() -> np.ndarray:
    n = int(_RATE * _DURATION)
    wave = _AMP * np.sin(2 * np.pi * _FREQ * np.arange(n) / _RATE)
    fade = int(_RATE * 0.01)  # 10 ms raised-cosine fades, no clicks
    ramp = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, fade)))
    wave[:fade] *= ramp
    wave[-fade:] *= ramp[::-1]
    mono = wave.astype(np.float32)
    return np.column_stack([mono, mono])


class TonePlayer:
    def __init__(self) -> None:
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_finish: Callable[[], None] | None = None

    @property
    def active(self) -> bool:
        return self._running.is_set()

    def toggle(self, on_finish: Callable[[], None] | None = None) -> bool:
        """Start one playback, or stop one in progress. Returns True if now playing."""
        if self._running.is_set():
            self.stop()
            return False
        self.play(on_finish)
        return True

    def play(self, on_finish: Callable[[], None] | None = None) -> None:
        """Play the tone once. ``on_finish`` fires (on the worker thread) only
        when it plays through to the end, not when stopped early."""
        self.stop()
        self._on_finish = on_finish
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="doctor-tone", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        block = _make_block()
        completed = False
        try:
            with soundcard.default_speaker().player(samplerate=_RATE, channels=2) as player:
                for start in range(0, len(block), _CHUNK):
                    if not self._running.is_set():
                        break
                    player.play(block[start : start + _CHUNK])
                else:
                    completed = True
        except Exception as e:  # noqa: BLE001 - soundcard surfaces opaque OS errors
            logger.warning("tone_player_failed reason=playback error=%r", e)
        finally:
            self._running.clear()
            if completed and self._on_finish is not None:
                self._on_finish()
