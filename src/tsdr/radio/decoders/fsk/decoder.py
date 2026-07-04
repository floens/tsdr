"""Streaming FSK teleprinter decoders (NAVTEX, RTTY, generic FSK).

Each decoder decimates device IQ to the profile's internal rate, runs the shared
`FSKFrontEnd` to recover soft bits, and feeds the profile's framer (SITOR-B FEC or
async start/stop) to emit text. Modes differ only by an `FSKProfile` + which axes
are pinned. `RTTYDecoder`/`FSKGenericDecoder` auto-acquire any of baud / shift /
polarity left unspecified: they buffer a few seconds, estimate the shift from the two
tones, then sweep the standard bauds x polarity and lock whichever frames best.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Literal, cast

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import DemodStatus
from tsdr.core.units import find_nearest
from tsdr.radio.decoders.fsk.framers import Framer, make_framer
from tsdr.radio.decoders.fsk.profile import (
    GENERIC_PROFILE,
    NAVTEX_PROFILE,
    RTTY_PROFILE,
    STANDARD_BAUDS,
    STANDARD_SHIFTS,
    FSKProfile,
)
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.dsp import FSKFrontEnd, StreamingDecimFilter, estimate_fsk_shift, firwin

logger = logging.getLogger(__name__)

_ACQ_SECONDS = 5.0  # IQ buffered before an auto-acquisition attempt (more = steadier estimate)
_ACQ_MIN_FRAMES = 6  # framing attempts a sweep candidate needs before its ratio counts
_ACQ_MIN_QUALITY = 0.8  # framing valid-ratio for a confident (early) lock
_ACQ_MAX_ATTEMPTS = 3  # after this many attempts, lock to the best guess regardless
_REACQ_FRAMES = 150  # frames over which a locked decoder's fit is re-checked (~20 s)
_REACQ_RATIO = 0.5  # below this windowed valid-ratio, a locked decoder re-acquires


def _snap_shift(shift: float) -> float:
    """Snap an estimated shift to the nearest standard value within tolerance.

    Overlapping FSK spectral humps pull the measured tone peaks a few Hz inward, so the
    raw estimate reads slightly low; teleprinter shifts are a small known set, so snapping
    both corrects the value and places the front-end tones exactly.
    """
    nearest = find_nearest(STANDARD_SHIFTS, shift)
    return nearest if abs(nearest - shift) <= max(15.0, 0.1 * nearest) else shift


class FSKDecoder(Demodulator):
    PROFILE: ClassVar[FSKProfile]
    AUTO_ACQUIRE: ClassVar[bool] = False
    MESSAGE_TYPE = "text"
    HAS_TEXT = True

    def __init__(
        self,
        sample_rate: float = 12_000.0,
        *,
        baud: float | None = None,
        shift_hz: float | None = None,
        reverse: bool | None = None,
        alphabet: str | None = None,
        framing: str | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        p = self.PROFILE
        self._framing = cast('Literal["sitor_b", "start_stop"]', framing or p.framing)
        if alphabet is not None:
            alpha = alphabet
        elif framing is not None:
            alpha = "ccir476" if framing == "sitor_b" else "ita2"
        else:
            alpha = p.alphabet
        self._alphabet = cast('Literal["ccir476", "ita2"]', alpha)
        self._data_bits = p.data_bits

        self._decim = max(1, round(sample_rate / p.internal_rate))
        self._internal_rate = sample_rate / self._decim
        if self._decim > 1:
            ntaps = max(31, (self._decim * 8) | 1)
            taps = firwin(ntaps, self._internal_rate * 0.45, fs=sample_rate)
            self._decimator: StreamingDecimFilter | None = StreamingDecimFilter(
                taps, self._decim, dtype=np.complex64
            )
        else:
            self._decimator = None

        # None on an axis means auto-acquire it; a value pins it.
        if self.AUTO_ACQUIRE:
            self._pin_baud = baud
            self._pin_shift = shift_hz
            self._pin_invert = reverse
        else:
            self._pin_baud = baud if baud is not None else p.baud
            self._pin_shift = shift_hz if shift_hz is not None else p.shift_hz
            self._pin_invert = reverse if reverse is not None else (p.polarity == "reverse")

        self._auto = self._pin_baud is None or self._pin_shift is None
        if not self._auto and self._pin_invert is None:
            self._pin_invert = False
        self.reset()

    def reset(self) -> None:
        super().reset()
        if self._decimator is not None:
            self._decimator.reset()
        self._pending: list[DecodedMessage] = []
        self._messages = 0
        self._enter_acquisition()
        if not self._auto:
            assert self._pin_baud is not None and self._pin_shift is not None
            self._build(self._pin_baud, self._pin_shift, bool(self._pin_invert), 0.0)

    def _enter_acquisition(self) -> None:
        self._baud: float | None = None
        self._shift: float | None = None
        self._invert: bool | None = None
        self._acq_buf: list[np.ndarray] = []
        self._acq_len = 0
        self._acq_attempts = 0
        self._frontend: FSKFrontEnd | None = None
        self._framer: Framer | None = None
        self._check_frames = 0
        self._check_valid = 0

    @property
    def acquired(self) -> bool:
        return self._frontend is not None

    def _build(self, baud: float, shift: float, invert: bool, center_hz: float) -> None:
        self._baud, self._shift, self._invert = baud, shift, invert
        self._frontend = FSKFrontEnd(self._internal_rate, baud, shift, center_hz=center_hz)
        self._framer = self._new_framer()

    def _new_framer(self) -> Framer:
        return make_framer(self._framing, self._alphabet, self._data_bits)

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        iq = iq_samples if self._decimator is None else self._decimator.process(iq_samples)
        if self._frontend is None:
            self._acq_buf.append(np.asarray(iq).copy())
            self._acq_len += len(iq)
            if self._acq_len >= _ACQ_SECONDS * self._internal_rate:
                self._acquire(capture_utc_s)
            return
        assert self._framer is not None
        soft = self._frontend.process(iq)
        if self._invert:
            np.negative(soft, out=soft)
        self._collect(self._framer.process(soft, capture_utc_s))
        if self._auto:
            self._maybe_reacquire()

    def _maybe_reacquire(self) -> None:
        """Drop back to acquisition if a locked decoder frames a lot but validates little,
        so a wrong lock on a live signal isn't pinned to bad params forever."""
        assert self._framer is not None
        valid, frames = self._framer.fit
        df = frames - self._check_frames
        if df < _REACQ_FRAMES:
            return
        dv = valid - self._check_valid
        self._check_valid, self._check_frames = valid, frames
        if dv / df < _REACQ_RATIO:
            self._enter_acquisition()

    def _acquire(self, ts: float) -> None:
        buf = np.concatenate(self._acq_buf)
        self._acq_buf = []
        self._acq_len = 0
        self._acq_attempts += 1

        if self._pin_shift is not None:
            center, shift = 0.0, self._pin_shift
        else:
            center, shift = estimate_fsk_shift(
                buf, self._internal_rate, nominal_hz=self.PROFILE.shift_hz
            )
            shift = _snap_shift(shift)

        if self._pin_baud is not None:
            baud_candidates = [self._pin_baud]
        elif self._framing == "start_stop":
            baud_candidates = STANDARD_BAUDS
        else:
            baud_candidates = [100.0]
        pol_candidates = [bool(self._pin_invert)] if self._pin_invert is not None else [False, True]

        # The min-frame-gated valid ratio separates the fundamental baud from its 2x
        # harmonic and the right polarity from the wrong one; both wrong choices frame noisily.
        best: tuple[float, float, bool, FSKFrontEnd, Framer, list[DecodedMessage]] | None = None
        for cand_baud in baud_candidates:
            frontend = FSKFrontEnd(self._internal_rate, cand_baud, shift, center_hz=center)
            soft = frontend.process(buf)
            for invert in pol_candidates:
                framer = self._new_framer()
                messages = framer.process(-soft if invert else soft, ts)
                valid, frames = framer.fit
                quality = valid / frames if frames >= _ACQ_MIN_FRAMES else 0.0
                if best is None or quality > best[0]:
                    best = (quality, cand_baud, invert, frontend, framer, messages)

        assert best is not None
        quality, baud, invert, frontend, framer, messages = best
        if quality < _ACQ_MIN_QUALITY and self._acq_attempts < _ACQ_MAX_ATTEMPTS:
            logger.debug(
                "fsk_acquire_retry shift=%.0f quality=%.2f attempt=%d",
                shift,
                quality,
                self._acq_attempts,
            )
            return
        logger.info(
            "fsk_acquired baud=%.2f shift=%.0f reverse=%d quality=%.2f",
            baud,
            shift,
            invert,
            quality,
        )
        self._baud, self._shift, self._invert = baud, shift, invert
        self._frontend, self._framer = frontend, framer
        self._collect(messages)

    def _collect(self, messages: list[DecodedMessage]) -> None:
        for msg in messages:
            self._messages += 1
            self._pending.append(msg)

    def get_messages(self) -> list[DecodedMessage]:
        messages = self._pending
        self._pending = []
        return messages

    def status(self) -> DemodStatus:
        if self._baud is None or self._shift is None:
            description = "acquiring…"
        else:
            description = f"{self._baud:g} Bd / {self._shift:.0f} Hz" + (
                " rev" if self._invert else ""
            )
        return DemodStatus(
            quality_label=f"{self._messages} msg" if self._messages else None,
            description=description,
        )


class NAVTEXDecoder(FSKDecoder):
    LABEL = "NAVTEX"
    MODULATION = "SITOR-B"
    FIXED_CHANNEL_BANDWIDTH = 500
    PROFILE = NAVTEX_PROFILE


class RTTYDecoder(FSKDecoder):
    LABEL = "RTTY"
    MODULATION = "Baudot"
    FIXED_CHANNEL_BANDWIDTH = 500
    PROFILE = RTTY_PROFILE
    AUTO_ACQUIRE = True


class FSKGenericDecoder(FSKDecoder):
    LABEL = "FSK"
    MODULATION = "2-FSK"
    FIXED_CHANNEL_BANDWIDTH = 1000
    PROFILE = GENERIC_PROFILE
    AUTO_ACQUIRE = True
