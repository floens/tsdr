from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tsdr.radio.decoders.aprs import APRSDecoder
from tsdr.radio.decoders.aprs.payload import APRSPacket

_FS = 30_000.0


def _decode(iq: np.ndarray, chunk: int = 8192) -> list[APRSPacket]:
    dec = APRSDecoder(sample_rate=_FS)
    out: list[APRSPacket] = []
    for i in range(0, len(iq), chunk):
        dec.demodulate(iq[i : i + chunk], 0.0)
        out.extend(m.data for m in dec.get_messages())
    return out


def test_decode_clean_packet(synth: SimpleNamespace) -> None:
    frame = synth.build_ax25("DL1ABC-9", "APRS", ("WIDE1-1",), "!4807.00N/01133.00E-Hello APRS")
    packets = _decode(synth.ax25_to_iq(frame, fs=_FS))
    matches = [p for p in packets if p.source == "DL1ABC-9"]
    assert matches, "expected DL1ABC-9 packet"
    p = matches[0]
    assert p.info_type == "position"
    assert p.latitude == pytest.approx(48.1167, abs=1e-3)
    assert p.longitude == pytest.approx(11.55, abs=1e-3)


def test_dedup_emits_once(synth: SimpleNamespace) -> None:
    # All 3 profiles decode the same clean packet; it must surface exactly once.
    frame = synth.build_ax25("DL1ABC", "APRS", (), "!4807.00N/01133.00E-x")
    packets = _decode(synth.ax25_to_iq(frame, fs=_FS))
    assert sum(p.source == "DL1ABC" for p in packets) == 1


def test_decode_with_frequency_offset(synth: SimpleNamespace) -> None:
    frame = synth.build_ax25("N0CALL-9", "APRS", (), "!4807.00N/01133.00E-")
    iq = synth.ax25_to_iq(frame, fs=_FS, offset_hz=1500.0)  # ~10 ppm at 144.8 MHz
    assert any(p.source == "N0CALL-9" for p in _decode(iq))


@pytest.mark.parametrize("snr_db", [25, 18, 12])
def test_decode_under_noise(synth: SimpleNamespace, snr_db: int) -> None:
    frame = synth.build_ax25("DL1ABC", "APRS", (), "!4807.00N/01133.00E-noise test")
    iq = synth.ax25_to_iq(frame, fs=_FS, snr_db=snr_db, seed=snr_db)
    assert any(p.source == "DL1ABC" for p in _decode(iq))


def test_chunk_invariance(synth: SimpleNamespace) -> None:
    frame = synth.build_ax25("DL1ABC-9", "APRS", ("WIDE1-1",), "!4807.00N/01133.00E-chunk")
    iq = synth.ax25_to_iq(frame, fs=_FS)
    big = {(p.source, p.raw_info) for p in _decode(iq, chunk=131072)}
    small = {(p.source, p.raw_info) for p in _decode(iq, chunk=997)}
    assert big == small
    assert ("DL1ABC-9", "!4807.00N/01133.00E-chunk") in big
