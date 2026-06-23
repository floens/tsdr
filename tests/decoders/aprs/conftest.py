"""Synthetic APRS test vectors: build AX.25 frames (and later AFSK/FM IQ).

Exposed via the ``synth`` fixture so test files share one encoder without a
cross-test import (the repo runs pytest in importlib mode with no test packages).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tsdr.radio.decoders.aprs.fec import crc16_x25


def _parse_call(spec: str) -> tuple[str, int, bool]:
    spec = spec.strip()
    repeated = spec.endswith("*")
    if repeated:
        spec = spec[:-1]
    call, _, ssid = spec.partition("-")
    return call.upper(), int(ssid or 0), repeated


def _encode_call(call: str, ssid: int, *, last: bool, repeated: bool) -> bytes:
    padded = call.upper().ljust(6)[:6]
    out = bytearray((ord(c) << 1) & 0xFF for c in padded)
    out.append(0x60 | ((ssid & 0x0F) << 1) | (0x80 if repeated else 0) | (1 if last else 0))
    return bytes(out)


def build_ax25(
    src: str,
    dest: str,
    digis: tuple[str, ...] = (),
    info: str | bytes = "",
    *,
    control: int = 0x03,
    pid: int = 0xF0,
) -> bytes:
    """Build a complete AX.25 UI frame (including FCS) from SRC>DEST,digis:info."""
    if isinstance(info, str):
        info = info.encode("latin1")
    addrs = [_parse_call(dest), _parse_call(src), *(_parse_call(d) for d in digis)]
    out = bytearray()
    for idx, (call, ssid, repeated) in enumerate(addrs):
        out += _encode_call(call, ssid, last=(idx == len(addrs) - 1), repeated=repeated)
    out += bytes([control, pid]) + info
    fcs = crc16_x25(bytes(out))
    out += bytes([fcs & 0xFF, (fcs >> 8) & 0xFF])
    return bytes(out)


def _bytes_to_bits_lsb(data: bytes) -> np.ndarray:
    return np.unpackbits(
        np.frombuffer(data, np.uint8).reshape(-1, 1), axis=1, bitorder="little"
    ).ravel()


def frame_to_hdlc_bits(frame: bytes, nflags: int = 20) -> np.ndarray:
    """Frame bytes -> bit-stuffed bitstream wrapped in flag preamble/postamble."""
    flag = _bytes_to_bits_lsb(bytes([0x7E]))
    stuffed: list[int] = []
    ones = 0
    for b in _bytes_to_bits_lsb(frame):
        stuffed.append(int(b))
        if b == 1:
            ones += 1
            if ones == 5:
                stuffed.append(0)
                ones = 0
        else:
            ones = 0
    return np.concatenate(
        [np.tile(flag, nflags), np.array(stuffed, np.uint8), np.tile(flag, nflags)]
    )


def nrzi_encode(bits: np.ndarray) -> np.ndarray:
    """Data bits -> tone-level bits (0 = transition, 1 = no change)."""
    level = 1
    out = np.empty(len(bits), np.uint8)
    for i, b in enumerate(bits):
        if b == 0:
            level ^= 1
        out[i] = level
    return out


def afsk_modulate(levels: np.ndarray, fs: float) -> np.ndarray:
    """Tone-level bits -> Bell 202 AFSK audio at ``fs`` (mark 1200, space 2200 Hz)."""
    spb = int(round(fs / 1200.0))
    freqs = np.where(np.repeat(levels, spb) == 1, 1200.0, 2200.0).astype(np.float64)
    return np.sin(2 * np.pi * np.cumsum(freqs) / fs).astype(np.float32)


def ax25_to_iq(
    frame: bytes,
    *,
    fs: float = 30_000.0,
    deviation: float = 3000.0,
    snr_db: float | None = None,
    offset_hz: float = 0.0,
    pad_s: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Full synthetic chain: AX.25 frame -> HDLC/NRZI -> AFSK -> FM -> IQ (+ noise)."""
    levels = nrzi_encode(frame_to_hdlc_bits(frame))
    audio = afsk_modulate(levels, fs)
    iq = np.exp(1j * 2 * np.pi * deviation * np.cumsum(audio) / fs).astype(np.complex64)
    pad = int(pad_s * fs)
    iq = np.concatenate([np.zeros(pad, np.complex64), iq, np.zeros(pad, np.complex64)])
    if offset_hz:
        t = np.arange(len(iq)) / fs
        iq = (iq * np.exp(1j * 2 * np.pi * offset_hz * t)).astype(np.complex64)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        sig_p = float(np.mean(np.abs(iq[pad:-pad]) ** 2))
        npow = sig_p / (10 ** (snr_db / 10))
        noise = np.sqrt(npow / 2) * (
            rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
        )
        iq = (iq + noise).astype(np.complex64)
    return iq


@pytest.fixture
def synth() -> SimpleNamespace:
    return SimpleNamespace(
        build_ax25=build_ax25,
        frame_to_hdlc_bits=frame_to_hdlc_bits,
        nrzi_encode=nrzi_encode,
        afsk_modulate=afsk_modulate,
        ax25_to_iq=ax25_to_iq,
    )
