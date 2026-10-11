"""FT8 / FT4 (WSJT-X family) demodulator.

    IQ (any device rate) -> decimating anti-alias FIR -> 12 kHz IQ
                         -> real projection (USB-style: keeps positive sideband)
                         -> per-slot ring buffer
                         -> on slot end: decode_slot() -> DecodedMessage[]

12 kHz is the canonical WSJT-X audio rate, so matching it means no extra
resampler inside the decoder. USB projection assumes the receiver is tuned
to the suppressed carrier with audio at +200..+3000 Hz of baseband.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from math import ceil
from typing import ClassVar, Literal

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.radio.decoders.wsjt.decode import analyze_slot, decode_candidates
from tsdr.radio.decoders.wsjt.tables import FT4_SLOT_TIME, FT8_SLOT_TIME
from tsdr.radio.demodulators import NYQUIST_MARGIN, Demodulator
from tsdr.radio.dsp._kernels import (
    StreamingFilter,
    StreamingPolyphaseResampler,
    make_rational_resampler,
)
from tsdr.radio.dsp.decimator import MultistageDecimator, smooth_decimation
from tsdr.radio.dsp.filters import firwin

logger = logging.getLogger(__name__)

TARGET_RATE = 12_000  # canonical WSJT-X audio rate (Hz)

# Upper-sideband selection before taking the real part (the lower sideband
# would fold onto the same audio): shift the IQ down by a quarter of its rate,
# low-pass to that quarter, shift back. 129 Kaiser taps: ~60 dB, ~340 Hz
# transition at 12 kS/s, so audio from ~170 Hz up keeps its LSB rejection.
_USB_TAPS = 129
_USB_KAISER_BETA = 5.65
_QUARTER_TURNS = np.array([1, -1j, -1, 1j], dtype=np.complex64)  # (-j)^n
WSJTMode = Literal["FT8", "FT4"]
SUPPORTED_MODES: tuple[WSJTMode, ...] = ("FT8", "FT4")


class WSJTDemodulator(Demodulator):
    """Slot-based FT8 / FT4 demodulator + decoder."""

    HAS_AUDIO: ClassVar[bool] = True
    MODULATION = "FSK"
    MESSAGE_TYPE = "text"
    HAS_TEXT = True
    SIDEBAND = "upper"

    DEFAULT_CHANNEL_BANDWIDTH = 3_000.0
    MAX_CHANNEL_BANDWIDTH = TARGET_RATE * NYQUIST_MARGIN

    def __init__(
        self,
        mode: WSJTMode | str,
        sample_rate: float,
        channel_bandwidth: float | None = None,
    ):
        super().__init__()
        mode_upper = mode.upper()
        if mode_upper not in SUPPORTED_MODES:
            raise ValueError(f"Invalid WSJT mode: {mode}; expected one of {SUPPORTED_MODES}")
        self.mode: WSJTMode = mode_upper  # type: ignore[assignment]
        self.sample_rate = float(sample_rate)
        self.channel_bandwidth = float(channel_bandwidth or self.DEFAULT_CHANNEL_BANDWIDTH)
        self.slot_seconds = FT4_SLOT_TIME if self.mode == "FT4" else FT8_SLOT_TIME
        self._slot_samples = int(self.slot_seconds * TARGET_RATE)
        self._decim: MultistageDecimator | None = None
        self._resampler: StreamingPolyphaseResampler | None = None
        self._setup_decim()

        # Slot ring buffer: capacity 2× slot so we can take a slice without
        # reallocating, then shift the tail down once per slot.
        self._buffer = np.zeros(self._slot_samples * 2, dtype=np.float32)
        self._buffer_fill = 0
        self._messages: list[DecodedMessage] = []
        self._align_samples_remaining = 0
        self._slot0_utc_s: float | None = None
        self._slot_index = 0

    @property
    def is_ft4(self) -> bool:
        return self.mode == "FT4"

    def _setup_decim(self) -> None:
        """Configure the device-rate -> 12 kHz IQ path: a multistage decimator to
        the nearest rate at or above 12 kHz that small factors reach (exactly
        12 kHz from the usual HF SDR rates 48k/96k/192k/240k/2400k), then a
        rational resampler for the small remainder when there is one.
        """
        sr = self.sample_rate
        if sr < TARGET_RATE:
            raise ValueError(f"WSJT demod requires sample_rate >= {TARGET_RATE}; got {sr}")
        decimation = smooth_decimation(sr / TARGET_RATE)
        self._decim = (
            MultistageDecimator(sr, decimation, TARGET_RATE * 0.45, dtype=np.complex64)
            if decimation > 1
            else None
        )
        mid_rate = sr / decimation
        self._resampler = (
            None
            if abs(mid_rate - TARGET_RATE) < 1e-6
            else make_rational_resampler(TARGET_RATE, mid_rate)
        )
        usb_taps = firwin(_USB_TAPS, mid_rate / 4, fs=mid_rate, window=("kaiser", _USB_KAISER_BETA))
        self._usb_filter = StreamingFilter(usb_taps, [1.0], dtype=np.complex64)
        self._usb_phase = 0

    def _usb_audio(self, iq: np.ndarray) -> np.ndarray:
        """Real audio of the upper sideband alone."""
        turns = _QUARTER_TURNS[(self._usb_phase + np.arange(len(iq))) % 4]
        self._usb_phase = (self._usb_phase + len(iq)) % 4
        shifted = self._usb_filter.process(iq * turns) * np.conj(turns)
        return np.ascontiguousarray(shifted.real, dtype=np.float32)

    def set_channel_bandwidth(self, bandwidth: float) -> None:
        self.channel_bandwidth = min(float(bandwidth), TARGET_RATE * NYQUIST_MARGIN)

    def set_sample_rate(self, rate: float) -> None:
        self.sample_rate = float(rate)
        self._setup_decim()
        self._reset_slot_state()

    def _reset_slot_state(self) -> None:
        """Clear the slot ring buffer and UTC alignment state.

        The slot anchor (`_slot0_utc_s`) is re-derived from the first chunk's
        `capture_utc_s` on the next call to `demodulate()`.
        """
        self._buffer_fill = 0
        self._align_samples_remaining = 0
        self._slot0_utc_s = None
        self._slot_index = 0

    @classmethod
    def _label_for(cls, mode: str) -> str:
        return mode.upper()

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        if len(iq_samples) == 0:
            return
        # `MultistageDecimator` and the no-resample fallback return views into
        # buffers we don't own past this call. We hand `audio` to `_emit_audio`
        # (consumed asynchronously by the audio worker) and to the slot ring
        # buffer, so it must own its memory.
        iq_lo = self._decim.process(iq_samples) if self._decim is not None else iq_samples
        if len(iq_lo) == 0:
            return
        audio = self._usb_audio(iq_lo)
        if self._resampler is not None:
            out = self._resampler.process(audio.reshape(-1, 1))
            if out.shape[0] == 0:
                return
            audio = np.ascontiguousarray(out[:, 0])

        self._emit_audio(audio, TARGET_RATE)

        slot0_utc_s = self._slot0_utc_s
        if slot0_utc_s is None:
            # Align slot 0 to the next UTC slot boundary. Residual TX ramp-up
            # and clock drift are absorbed by the candidate search's symbol-
            # block time-offset window (see find_candidates).
            slot0_utc_s = ceil(capture_utc_s / self.slot_seconds) * self.slot_seconds
            self._align_samples_remaining = int(round((slot0_utc_s - capture_utc_s) * TARGET_RATE))
            self._slot0_utc_s = slot0_utc_s

        if self._align_samples_remaining > 0:
            drop = min(self._align_samples_remaining, audio.shape[0])
            audio = audio[drop:]
            self._align_samples_remaining -= drop
            if audio.shape[0] == 0:
                return

        self._append_to_buffer(audio)

        while self._buffer_fill >= self._slot_samples:
            slot = self._buffer[: self._slot_samples].copy()
            slot_utc_s = slot0_utc_s + self._slot_index * self.slot_seconds
            # Shift the unconsumed tail down so the next chunk appends after it.
            tail = self._buffer_fill - self._slot_samples
            if tail > 0:
                self._buffer[:tail] = self._buffer[self._slot_samples : self._buffer_fill]
            self._buffer_fill = tail
            self._slot_index += 1
            self._decode_one_slot(slot, slot_utc_s)

    def _append_to_buffer(self, audio: np.ndarray) -> None:
        n = audio.shape[0]
        needed = self._buffer_fill + n
        if needed > self._buffer.shape[0]:
            new_cap = max(needed, self._buffer.shape[0] * 2)
            larger = np.empty(new_cap, dtype=np.float32)
            larger[: self._buffer_fill] = self._buffer[: self._buffer_fill]
            self._buffer = larger
        self._buffer[self._buffer_fill : self._buffer_fill + n] = audio
        self._buffer_fill = needed

    def _decode_one_slot(self, slot_audio: np.ndarray, slot_utc_s: float) -> None:
        wf, candidates = analyze_slot(slot_audio, is_ft4=self.is_ft4, sample_rate=TARGET_RATE)
        decodes, stats = decode_candidates(wf, candidates, is_ft4=self.is_ft4)

        slot_utc_iso = datetime.fromtimestamp(slot_utc_s, tz=UTC).isoformat(timespec="milliseconds")
        logger.info(
            "wsjt_slot_summary mode=%s utc=%s candidates=%d top_score=%.1f "
            "ldpc_pass=%d crc_pass=%d decodes=%d",
            self.mode,
            slot_utc_iso,
            stats.num_candidates,
            stats.top_score,
            stats.ldpc_pass,
            stats.crc_pass,
            stats.unique_decodes,
        )

        for d in decodes:
            text = f"{d.freq_hz:6.1f}Hz  {d.text}"
            logger.info(
                "wsjt_decode mode=%s freq=%.1f score=%.1f text=%r",
                self.mode,
                d.freq_hz,
                d.score,
                d.text,
            )
            self._messages.append(DecodedMessage(text=text, timestamp=slot_utc_s, data=d))

    def get_messages(self) -> list[DecodedMessage]:
        out = self._messages
        self._messages = []
        return out

    def reset(self) -> None:
        super().reset()
        if self._decim is not None:
            self._decim.reset()
        if self._resampler is not None:
            self._resampler.reset()
        self._usb_filter.reset()
        self._usb_phase = 0
        self._reset_slot_state()
        self._messages = []
