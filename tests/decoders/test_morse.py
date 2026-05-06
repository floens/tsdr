"""Tests for the CW Morse decoder.

Synthetic envelope generator constructs an idealised on/off keying envelope
(0.0 / 1.0), with optional jitter and additive noise, so the tests can probe
the algorithm independently of the audio demodulator.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdr.radio.decoders.morse import MORSE_TABLE, MorseDecoder

# Inverse table for synthesizing.
_CHAR_TO_MORSE: dict[str, str] = {v: k for k, v in MORSE_TABLE.items()}


def _build_envelope(
    sample_rate: float,
    wpm: float,
    text: str,
    jitter: float = 0.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    settling_seconds: float = 0.2,
) -> np.ndarray:
    """Build an idealised CW keying envelope for ``text`` at ``wpm``.

    - Spaces in text become 7-dit gaps (word break).
    - Per-element duration is multiplied by ``1 + jitter*N(0,1)`` clipped to >0
      to simulate hand keying.
    - Adds optional Gaussian noise with std ``noise_std``.
    - Prepends ``settling_seconds`` of silence so the trackers have time to
      establish a noise floor before the first mark.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    dit_samples = (1200.0 / wpm) * 1e-3 * sample_rate

    def jitter_mul() -> float:
        if jitter <= 0.0:
            return 1.0
        return max(0.1, 1.0 + jitter * rng.standard_normal())

    chunks: list[np.ndarray] = []
    chunks.append(np.zeros(int(settling_seconds * sample_rate), dtype=np.float32))

    for word_idx, word in enumerate(text.split(" ")):
        if word_idx > 0:
            chunks.append(np.zeros(int(round(7.0 * dit_samples * jitter_mul())), dtype=np.float32))
        for letter_idx, ch in enumerate(word):
            pattern = _CHAR_TO_MORSE.get(ch.upper())
            if pattern is None:
                continue
            if letter_idx > 0:
                chunks.append(
                    np.zeros(int(round(3.0 * dit_samples * jitter_mul())), dtype=np.float32)
                )
            for sym_idx, sym in enumerate(pattern):
                if sym_idx > 0:
                    chunks.append(
                        np.zeros(int(round(dit_samples * jitter_mul())), dtype=np.float32)
                    )
                length = 1.0 if sym == "." else 3.0
                chunks.append(
                    np.ones(int(round(length * dit_samples * jitter_mul())), dtype=np.float32)
                )

    # Trailing silence so the idle-flush timeout fires.
    chunks.append(np.zeros(int(20.0 * dit_samples), dtype=np.float32))

    env = np.concatenate(chunks)
    if noise_std > 0.0:
        env = env + rng.standard_normal(env.shape).astype(np.float32) * noise_std
        env = np.maximum(env, 0.0)
    return env.astype(np.float32)


def _decode(env: np.ndarray, sample_rate: float) -> str:
    decoder = MorseDecoder(sample_rate)
    decoder.process(env, 0.0)
    return " ".join(m.text for m in decoder.get_messages())


def _accuracy(decoded: str, expected: str) -> float:
    """Character-overlap accuracy after stripping spaces and unknown markers."""
    a = decoded.replace(" ", "").replace("*", "")
    b = expected.replace(" ", "")
    if not b:
        return 1.0
    n = min(len(a), len(b))
    matches = sum(1 for x, y in zip(a[:n], b[:n], strict=False) if x == y)
    return matches / len(b)


def test_machine_keying_perfect():
    fs = 8000.0
    text = "HELLO WORLD"
    env = _build_envelope(fs, wpm=12.0, text=text)
    decoded = _decode(env, fs)
    assert decoded == text


def test_machine_keying_fast():
    """30 WPM with no jitter. The first letter is unavoidably wrong because
    ``dit_samples`` starts at the 12 WPM default and only adapts after the
    first 1:3 pair is observed -- this matches fldigi behavior at cold
    start. Subsequent letters recover.
    """
    fs = 8000.0
    text = "CQ DE TEST"
    env = _build_envelope(fs, wpm=30.0, text=text)
    decoded = _decode(env, fs)
    assert _accuracy(decoded, text) >= 0.8


def test_hand_keying_jitter():
    """Realistic hand-keying jitter (~15% per element)."""
    fs = 8000.0
    text = "CQ DE TEST"
    env = _build_envelope(fs, wpm=15.0, text=text, jitter=0.15, rng=np.random.default_rng(1))
    decoded = _decode(env, fs)
    assert _accuracy(decoded, text) >= 0.8


def test_noise_spike_rejected():
    """Marks shorter than ``dit/2`` are dropped (fldigi cw.cxx:921).

    Spikes are inserted into silence regions of an otherwise-clean signal;
    the surrounding "OK" must still decode correctly.
    """
    fs = 8000.0
    text = "OK"
    env = _build_envelope(fs, wpm=12.0, text=text).copy()
    # Inject 30-sample spikes (~3.75 ms, well below half a dit at 12 WPM
    # which is ~333 samples). These are noise-spike-region.
    rng = np.random.default_rng(7)
    spike_len = 30
    spike_starts = rng.integers(int(0.25 * fs), len(env) - spike_len - 100, size=5)
    for s in spike_starts:
        if env[s] < 0.1 and env[s + spike_len + 50] < 0.1:
            env[s : s + spike_len] = 1.0
    decoded = _decode(env, fs)
    assert decoded == text


def test_word_emission():
    fs = 8000.0
    decoder = MorseDecoder(fs)
    env = _build_envelope(fs, wpm=12.0, text="AB CD")
    decoder.process(env, 0.0)
    msgs = decoder.get_messages()
    texts = [m.text for m in msgs]
    assert texts == ["AB", "CD"]


def test_idle_flush_emits_partial_word():
    fs = 8000.0
    # Plain "AB" (no trailing space). The synth helper appends 20-dit silence,
    # which exceeds the 14-dit idle-flush threshold.
    env = _build_envelope(fs, wpm=12.0, text="AB")
    decoder = MorseDecoder(fs)
    decoder.process(env, 0.0)
    msgs = decoder.get_messages()
    assert [m.text for m in msgs] == ["AB"]


def test_unknown_rep_marked_with_asterisk():
    """A pattern not in the Morse table emits '*' as a placeholder."""
    fs = 8000.0
    decoder = MorseDecoder(fs)

    # ......- is not in the table; place between two valid letters.
    dit = (1200.0 / 12.0) * 1e-3 * fs
    settle = np.zeros(int(0.2 * fs), dtype=np.float32)
    pieces: list[np.ndarray] = [settle]
    # Letter 'E' = ".".
    pieces.append(np.ones(int(round(dit)), dtype=np.float32))
    pieces.append(np.zeros(int(round(3.0 * dit)), dtype=np.float32))
    # Unknown rep: 6 dits + 1 dah, with 1-dit gaps.
    for i in range(6):
        if i > 0:
            pieces.append(np.zeros(int(round(dit)), dtype=np.float32))
        pieces.append(np.ones(int(round(dit)), dtype=np.float32))
    pieces.append(np.zeros(int(round(dit)), dtype=np.float32))
    pieces.append(np.ones(int(round(3.0 * dit)), dtype=np.float32))
    # Letter 'T' = "-".
    pieces.append(np.zeros(int(round(3.0 * dit)), dtype=np.float32))
    pieces.append(np.ones(int(round(3.0 * dit)), dtype=np.float32))
    pieces.append(np.zeros(int(20.0 * dit), dtype=np.float32))

    env = np.concatenate(pieces).astype(np.float32)
    decoder.process(env, 0.0)
    msgs = decoder.get_messages()
    assert [m.text for m in msgs] == ["E*T"]


def test_reset_clears_state():
    fs = 8000.0
    decoder = MorseDecoder(fs)
    decoder.process(_build_envelope(fs, wpm=12.0, text="A"), 0.0)
    decoder.reset()
    decoder.process(_build_envelope(fs, wpm=12.0, text="B"), 0.0)
    msgs = decoder.get_messages()
    assert [m.text for m in msgs] == ["B"]


@pytest.mark.parametrize("k", [1, 3, 5, 7, 50])
def test_chunk_continuity(k: int):
    """Splitting envelope across chunks must not change decoded text."""
    fs = 8000.0
    text = "HELLO"
    env = _build_envelope(fs, wpm=12.0, text=text)

    full = MorseDecoder(fs)
    split = MorseDecoder(fs)
    full.process(env, 0.0)
    for c in np.array_split(env, k):
        split.process(np.ascontiguousarray(c), 0.0)
    full_text = [m.text for m in full.get_messages()]
    split_text = [m.text for m in split.get_messages()]
    assert full_text == split_text
