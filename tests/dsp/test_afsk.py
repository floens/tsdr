from __future__ import annotations

import numpy as np

from tsdr.radio.dsp import AFSK1200Demod

_FS = 48_000.0


def _tone(freq: float, n: int) -> np.ndarray:
    return np.sin(2 * np.pi * freq * np.arange(n) / _FS).astype(np.float32)


def test_mark_space_decision_sign() -> None:
    demod = AFSK1200Demod(_FS)
    mark = demod.process(_tone(1200.0, 4800))
    demod.reset()
    space = demod.process(_tone(2200.0, 4800))
    # After the LPF settles, mark -> |mark|>|space| (positive), space -> negative.
    assert np.mean(mark[400:]) > 0
    assert np.mean(space[400:]) < 0


def test_chunk_invariance() -> None:
    # Alternating mark/space at the baud rate, fed whole vs in odd chunks.
    fs = _FS
    spb = int(fs / 1200)
    levels = np.repeat(np.arange(200) % 2, spb)
    audio = np.where(levels == 1, _tone(1200.0, len(levels)), _tone(2200.0, len(levels))).astype(
        np.float32
    )
    whole = AFSK1200Demod(fs).process(audio)
    demod = AFSK1200Demod(fs)
    streamed = np.concatenate(
        [demod.process(audio[i : i + 333]) for i in range(0, len(audio), 333)]
    )
    assert len(whole) == len(streamed)
    assert np.allclose(whole, streamed, atol=1e-4)
