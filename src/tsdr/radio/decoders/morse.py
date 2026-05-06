"""CW Morse code envelope-to-text decoder.

Pipeline:

- Single-pole envelope smoothing (~10 ms TC).
- Asymmetric AGC trackers: slow ``noise_floor`` updated only during silences;
  faster ``sig_avg`` tracking on every sample.
- Schmitt hysteresis around ``sig_avg`` (1.05 / 0.95) to debounce noisy edges.
- Run-length encoding into on/off durations.
- Adaptive WPM tracker gated by 1:3-ratio pairs of consecutive marks: feeds
  ``(short + long) / 2`` (an estimate of ``2 * dit``) into a 16-tap boxcar
  moving average. Robust to jitter because the 1:3 ratio is invariant under
  operator drift.
- Hard threshold at ``2 * dit_samples`` for element classification, ``2..4 *
  dit`` for letter end, ``> 4 * dit`` for word end.
- Reject marks shorter than ``dit/2`` as noise spikes.

Targets ~80% character accuracy on real hand-keyed signals (10-15% jitter
on real keying), near 100% on machine-keyed at the seeded WPM. v1 hardcodes
the initial estimate at 12 WPM; the tracker adapts immediately on first
1:3 pair.
"""

from __future__ import annotations

import collections
import math

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.radio.dsp._kernels import _morse_envelope_kernel

# Standard ITU Morse code: letters, digits, common punctuation.
MORSE_TABLE: dict[str, str] = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
    "--..--": ",",
    ".-.-.-": ".",
    "..--..": "?",
    "-..-.": "/",
    "-...-": "=",
    ".-.-.": "+",
    "-....-": "-",
}


class MorseDecoder:
    """fldigi-style CW decoder.

    Input: real-valued envelope of the IF complex stream (e.g. ``np.abs(iq_filt)``).
    Output: one ``DecodedMessage`` per completed word (or per idle-flush).
    """

    DEFAULT_WPM = 12.0
    SMOOTH_TC_S = 0.010
    NOISE_FLOOR_TC_S = 1.0
    SIG_AVG_TC_S = 0.05
    HYST_HIGH = 1.05
    HYST_LOW = 0.95
    MAX_WORD_CHARS = 80
    IDLE_FLUSH_DITS = 8
    TRACKING_LEN = 16
    # Absolute upper bound on the noise-spike threshold. fldigi uses ``dit/2``,
    # but with no user-set WPM hint the seeded ``dit_samples`` may be too high
    # for the actual signal, killing legitimate fast CW dits. Cap at 30 ms so
    # a ~12 WPM seed (dit/2 = 50 ms) doesn't reject 30 WPM dits (40 ms).
    NOISE_SPIKE_MAX_S = 0.030

    def __init__(self, sample_rate: float):
        self._fs = float(sample_rate)
        self._dit_samples = self._wpm_to_dit_samples(self.DEFAULT_WPM)
        self._noise_spike_max = self.NOISE_SPIKE_MAX_S * self._fs
        self._tracking: collections.deque[float] = collections.deque(maxlen=self.TRACKING_LEN)

        self._smooth_alpha = 1.0 - math.exp(-1.0 / (self.SMOOTH_TC_S * self._fs))
        self._noise_alpha = 1.0 - math.exp(-1.0 / (self.NOISE_FLOOR_TC_S * self._fs))
        self._sig_alpha = 1.0 - math.exp(-1.0 / (self.SIG_AVG_TC_S * self._fs))

        self._smooth_state = np.zeros(1, dtype=np.float32)
        self._noise_floor = np.zeros(1, dtype=np.float32)
        self._sig_avg = np.zeros(1, dtype=np.float32)
        self._is_on = np.zeros(1, dtype=np.int32)

        self._sample_idx = 0
        self._last_transition_idx = 0

        self._last_on_duration = 0.0
        self._current_letter: list[str] = []
        self._current_word_chars: list[str] = []
        self._messages: list[DecodedMessage] = []
        self._timestamp = 0.0

    def _wpm_to_dit_samples(self, wpm: float) -> float:
        # 1 dit @ N WPM = 1200/N ms (PARIS standard).
        return (1200.0 / wpm) * 1e-3 * self._fs

    def process(self, envelope: np.ndarray, timestamp: float) -> None:
        """Feed envelope samples (real float32, |iq_filt|).

        Maintains continuity across chunks via the persistent state arrays.
        """
        if len(envelope) == 0:
            return
        self._timestamp = float(timestamp)

        offsets, signs = _morse_envelope_kernel(
            np.ascontiguousarray(envelope, dtype=np.float32),
            self._smooth_state,
            self._noise_floor,
            self._sig_avg,
            self._is_on,
            self._smooth_alpha,
            self._noise_alpha,
            self._sig_alpha,
            self.HYST_HIGH,
            self.HYST_LOW,
        )

        for off, sign in zip(offsets, signs, strict=True):
            global_idx = self._sample_idx + int(off)
            duration = global_idx - self._last_transition_idx
            if sign < 0:
                self._on_keyup(duration)
            else:
                self._on_keydown(duration)
            self._last_transition_idx = global_idx

        self._sample_idx += len(envelope)
        self._maybe_flush_idle()

    def _on_keyup(self, duration_samples: float) -> None:
        """Tone went off after a mark of ``duration_samples`` samples."""
        # Reject noise spikes shorter than ``dit/2``, capped at
        # NOISE_SPIKE_MAX_S so a stale ``dit_samples`` seed doesn't kill
        # legitimate fast-CW dits.
        spike_threshold = min(0.5 * self._dit_samples, self._noise_spike_max)
        if duration_samples < spike_threshold:
            return
        # Update tracker on the raw duration first so this element's own
        # classification can benefit if the speed has changed.
        self._maybe_update_tracker(duration_samples)
        is_dit = duration_samples <= 2.0 * self._dit_samples
        self._current_letter.append("." if is_dit else "-")
        self._last_on_duration = duration_samples

    def _on_keydown(self, duration_samples: float) -> None:
        """Tone went on after a gap of ``duration_samples`` samples."""
        d = self._dit_samples
        if duration_samples < 2.0 * d:
            return  # in-character gap
        self._emit_letter()
        if duration_samples > 4.0 * d:
            self._emit_word()

    def _maybe_update_tracker(self, dur: float) -> None:
        """Adaptive WPM tracker -- updates on consecutive marks with 1:3 ratio.

        The 1:3 dit-dah ratio is invariant under operator drift, so gating
        on the raw durations (regardless of how they're classified) keeps
        the tracker stable. ``(short + long) / 2`` is an estimate of
        ``2 * dit``; we feed it into a 16-tap boxcar and the actual dit
        length is half the boxcar mean.
        """
        prev_dur = self._last_on_duration
        if prev_dur <= 0.0:
            return
        short = prev_dur if prev_dur <= dur else dur
        long_ = dur if prev_dur <= dur else prev_dur
        if not (2.0 * short < long_ < 4.0 * short):
            return
        two_dits = (short + long_) / 2.0
        self._tracking.append(two_dits)
        self._dit_samples = (sum(self._tracking) / len(self._tracking)) / 2.0

    def _emit_letter(self) -> None:
        if not self._current_letter:
            return
        pattern = "".join(self._current_letter)
        char = MORSE_TABLE.get(pattern, "*")
        self._current_word_chars.append(char)
        self._current_letter.clear()
        if len(self._current_word_chars) >= self.MAX_WORD_CHARS:
            self._emit_word()

    def _emit_word(self) -> None:
        # Flush any pending letter so a word ending mid-letter still surfaces.
        self._emit_letter()
        if not self._current_word_chars:
            return
        word = "".join(self._current_word_chars)
        self._messages.append(DecodedMessage(text=word, timestamp=self._timestamp))
        self._current_word_chars.clear()

    def _maybe_flush_idle(self) -> None:
        """Flush any pending word after extended silence (operator paused / signal lost)."""
        if self._is_on[0]:
            return
        gap = self._sample_idx - self._last_transition_idx
        if gap > self.IDLE_FLUSH_DITS * self._dit_samples:
            self._emit_word()

    def get_messages(self) -> list[DecodedMessage]:
        msgs, self._messages = self._messages, []
        return msgs

    def reset(self) -> None:
        self._smooth_state[0] = 0.0
        self._noise_floor[0] = 0.0
        self._sig_avg[0] = 0.0
        self._is_on[0] = 0
        self._sample_idx = 0
        self._last_transition_idx = 0
        self._last_on_duration = 0.0
        self._current_letter.clear()
        self._current_word_chars.clear()
        self._messages.clear()
        self._tracking.clear()
        self._dit_samples = self._wpm_to_dit_samples(self.DEFAULT_WPM)
