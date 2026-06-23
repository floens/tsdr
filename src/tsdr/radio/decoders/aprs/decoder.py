"""Streaming APRS decoder: AFSK1200 / AX.25 over narrowband FM.

A shared `FMChannelizer` feeds three AFSK receiver profiles (Dire Wolf-style
diversity in tone-LPF bandwidth and timing-loop gain) that each run
AFSK1200Demod -> MuellerMuller -> NRZI -> HDLC. Frames from all profiles are
deduplicated by content over a short window, validated (with a single-bit-fix
retry) and parsed into `APRSPacket`s.
"""

from __future__ import annotations

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import DemodStatus
from tsdr.radio.decoders.aprs import payload
from tsdr.radio.decoders.aprs.ax25 import recover
from tsdr.radio.decoders.aprs.hdlc import HDLCDeframer, NRZIDecoder
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.dsp import AFSK1200Demod, DCBlocker, DPLLBitSync, FMChannelizer

_DEVIATION = 3000.0  # Hz, typical APRS NBFM deviation
_CHANNEL_CUTOFF = 7000.0  # ~14 kHz APRS channel; tolerates ~1-2 kHz PPM tuning offset
_AUDIO_RATE = 48_000.0  # channelizer target (degrades to the device rate if lower)
_BAUD = 1200.0
_DEDUP_WINDOW_S = 0.5  # parallel profiles decode a packet within ms; repeats are seconds apart

# (tone-LPF cutoff Hz, DPLL loop gain): baseline / wide / narrow.
_PROFILES = ((1200.0, 0.1), (1500.0, 0.15), (900.0, 0.06))


class _Receiver:
    """One AFSK demod profile + its bit-timing, NRZI and HDLC state."""

    def __init__(self, audio_rate: float, lpf_cutoff: float, dpll_k: float) -> None:
        self._afsk = AFSK1200Demod(audio_rate, lpf_cutoff=lpf_cutoff)
        self._sync = DPLLBitSync(audio_rate / _BAUD, k=dpll_k)
        self._nrzi = NRZIDecoder()
        self._hdlc = HDLCDeframer()

    def process(self, audio: np.ndarray) -> list[bytes]:
        levels = self._sync.process(self._afsk.process(audio))
        if len(levels) == 0:
            return []
        bits = self._nrzi.process(levels)
        return self._hdlc.process(bits)

    def reset(self) -> None:
        self._afsk.reset()
        self._sync.reset()
        self._nrzi.reset()
        self._hdlc.reset()


class APRSDecoder(Demodulator):
    LABEL = "APRS"
    MODULATION = "AFSK1200"
    MESSAGE_TYPE = "text"
    HAS_TEXT = True
    FIXED_CHANNEL_BANDWIDTH = 12_500

    def __init__(self, sample_rate: float = 250_000) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self._channelizer = FMChannelizer(
            sample_rate, _DEVIATION, target_rate=_AUDIO_RATE, channel_cutoff=_CHANNEL_CUTOFF
        )
        audio_rate = self._channelizer.audio_rate
        self._dc = DCBlocker(audio_rate)  # strip carrier-offset DC from the discriminator
        self._profiles = [_Receiver(audio_rate, lpf, gain) for lpf, gain in _PROFILES]
        self._recent: dict[int, float] = {}
        self._pending: list[DecodedMessage] = []
        self._frames_ok = 0
        self._last_callsign = ""

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        audio = self._dc.process(self._channelizer.process(iq_samples))
        for rx in self._profiles:
            for raw in rx.process(audio):
                self._handle_frame(raw, capture_utc_s)

    def _handle_frame(self, raw: bytes, ts: float) -> None:
        frame = recover(raw)
        if frame is None:
            return
        key = hash((str(frame.src), str(frame.dest), frame.info))
        self._recent = {k: t for k, t in self._recent.items() if ts - t < _DEDUP_WINDOW_S}
        if key in self._recent:
            self._recent[key] = ts
            return
        self._recent[key] = ts
        self._frames_ok += 1
        self._last_callsign = str(frame.src)
        packet = payload.parse(frame)
        self._pending.append(
            DecodedMessage(text=_format_text(packet), timestamp=ts, data=packet, markup=True)
        )

    def get_messages(self) -> list[DecodedMessage]:
        messages = self._pending
        self._pending = []
        return messages

    def status(self) -> DemodStatus:
        return DemodStatus(
            quality_label=f"{self._frames_ok} pkt" if self._frames_ok else None,
            description=self._last_callsign or None,
        )

    def reset(self) -> None:
        super().reset()
        self._channelizer.reset()
        self._dc.reset()
        for rx in self._profiles:
            rx.reset()
        self._recent = {}
        self._pending = []
        self._frames_ok = 0
        self._last_callsign = ""


def _e(s: str) -> str:
    """Escape Rich markup in decoder-supplied text ('[' is the only tag opener)."""
    return s.replace("[", r"\[")


def _format_text(p: payload.APRSPacket) -> str:
    head = f"[bold cyan]{_e(p.source)}[/][dim]>{_e(p.dest)}"
    if p.digis:
        head += "," + ",".join(_e(d) for d in p.digis)
    return f"{head}[/] {_format_body(p)}"


def _format_body(p: payload.APRSPacket) -> str:
    if p.info_type in ("position", "mic_e") and p.latitude is not None:
        body = f"[green]{p.latitude:.4f},{p.longitude:.4f}[/]"
        if p.symbol:
            body += f" [yellow]{_e(p.symbol)}[/]"
        if p.course is not None and p.speed is not None:
            body += f" [magenta]cse{p.course} spd{p.speed:.0f}kt[/]"
        if p.altitude is not None:
            body += f" [magenta]alt{p.altitude:.0f}ft[/]"
        if p.comment:
            body += f" {_e(p.comment)}"
        return (r"[blue]\[Mic-E][/] " if p.info_type == "mic_e" else "") + body
    if p.info_type == "message" and p.message_text is not None:
        return f"[bold yellow]>{_e(p.addressee or '')}[/]: {_e(p.message_text)}"
    if p.info_type == "status" and p.status is not None:
        return f"[italic yellow]>{_e(p.status)}[/]"
    if p.info_type == "third_party" and p.third_party is not None:
        return "[dim]}[/] " + _format_text(p.third_party)
    return _e(p.raw_info)


__all__ = ["APRSDecoder"]
