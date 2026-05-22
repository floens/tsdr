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
from math import ceil, gcd
from typing import ClassVar, Literal

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import SignalInfo
from tsdr.radio.decoders.wsjt.decode import analyze_slot, decode_candidates
from tsdr.radio.decoders.wsjt.tables import FT4_SLOT_TIME, FT8_SLOT_TIME
from tsdr.radio.demodulators import NYQUIST_MARGIN, Demodulator
from tsdr.radio.dsp import StreamingDecimFilter, firwin
from tsdr.radio.dsp._kernels import StreamingPolyphaseResampler

logger = logging.getLogger(__name__)

TARGET_RATE = 12_000  # canonical WSJT-X audio rate (Hz)
WSJTMode = Literal["FT8", "FT4"]
SUPPORTED_MODES: tuple[WSJTMode, ...] = ("FT8", "FT4")


class WSJTDemodulator(Demodulator):
    """Slot-based FT8 / FT4 demodulator + decoder."""

    has_audio: ClassVar[bool] = True

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
        self._decim: StreamingDecimFilter | None = None
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
        """Configure the device-rate -> 12 kHz IQ path.

        Integer-ratio (typical for HF SDRs running 12k/24k/48k/96k/192k/240k/2400k):
        decimating FIR — cheap, no resampling artifacts. Otherwise fall back to
        a rational polyphase resampler that runs I and Q as two real channels.
        """
        sr = int(round(self.sample_rate))
        if sr < TARGET_RATE:
            raise ValueError(f"WSJT demod requires sample_rate >= {TARGET_RATE}; got {sr}")
        if sr % TARGET_RATE == 0:
            decimation = sr // TARGET_RATE
            if decimation > 1:
                aa = firwin(64, TARGET_RATE * 0.45, fs=self.sample_rate, window=("kaiser", 6.0))
                self._decim = StreamingDecimFilter(
                    aa,
                    decimation=decimation,
                    dtype=np.complex64,
                    expected_input_size=200_000,
                )
            else:
                self._decim = None
            self._resampler = None
            return

        self._decim = None
        g = gcd(sr, TARGET_RATE)
        up = TARGET_RATE // g
        down = sr // g
        # 20 taps per polyphase phase gives a clean transition band at any
        # rational ratio HF SDRs throw at us.
        n_taps = 20 * max(up, down) + 1
        self._resampler = StreamingPolyphaseResampler(up=up, down=down, n_taps=n_taps)

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

    def info(self) -> SignalInfo:
        return SignalInfo(
            label=self.mode,
            channel_bandwidth=self.channel_bandwidth,
            modulation="FSK",
            sample_rate=self.sample_rate,
            has_audio=True,
            has_text=True,
            message_type="text",
            # USB: audio band is +200..+3000 Hz of the suppressed carrier.
            sideband="upper",
        )

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        if len(iq_samples) == 0:
            return
        # `StreamingDecimFilter` and the no-resample fallback return views into
        # buffers we don't own past this call. We hand `audio` to `_emit_audio`
        # (consumed asynchronously by the audio worker) and to the slot ring
        # buffer, so it must own its memory.
        if self._decim is not None:
            iq_lo = self._decim.process(iq_samples)
            if len(iq_lo) == 0:
                return
            audio = np.ascontiguousarray(iq_lo.real, dtype=np.float32)
        elif self._resampler is not None:
            real_in = np.ascontiguousarray(iq_samples.real, dtype=np.float32).reshape(-1, 1)
            out = self._resampler.process(real_in)
            if out.shape[0] == 0:
                return
            audio = np.ascontiguousarray(out[:, 0])
        else:
            audio = np.ascontiguousarray(iq_samples.real, dtype=np.float32)

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
        self._reset_slot_state()
        self._messages = []
