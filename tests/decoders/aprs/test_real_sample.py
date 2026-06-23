"""End-to-end decode of a real off-air capture.

A 22 s window of 144.800 MHz: iGates gating APRS-IS beacons back to RF -- a mobile
station and a positionless telemetry beacon. The bursts span the bit-phase range,
so this guards the DPLL timing recovery (Mueller-Muller dropped the off-phase
ones, including the telemetry beacon asserted below).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr.io import load_iq
from tsdr.radio.decoders.aprs import APRSDecoder
from tsdr.radio.decoders.aprs.payload import APRSPacket

SAMPLE = Path(__file__).parents[2] / "samples" / "freq=144.8M_sr=30k_dur=22s_gain=0_aprs.cf32.zst"
FS = 30_000.0


@pytest.fixture(scope="module")
def aprs_iq() -> np.ndarray:
    if not SAMPLE.exists():
        pytest.skip(f"Sample not found: {SAMPLE}")
    return load_iq(SAMPLE).astype(np.complex64)


def _decode(iq: np.ndarray, chunk: int) -> list[APRSPacket]:
    dec = APRSDecoder(sample_rate=FS)
    out: list[APRSPacket] = []
    for i in range(0, len(iq), chunk):
        dec.demodulate(iq[i : i + chunk], i / FS)
        out.extend(m.data for m in dec.get_messages())
    return out


def test_decodes_packets(aprs_iq: np.ndarray) -> None:
    packets = _decode(aprs_iq, 8192)
    # M&M recovered only the on-phase bursts (~2); the DPLL gets them all.
    assert len(packets) >= 3
    assert all(p.info_type == "third_party" for p in packets)

    # Anchor on a positionless telemetry beacon that sits at a bit phase M&M
    # could not lock: the decoder must unwrap the third-party header and surface
    # the raw "T#..." payload.
    telem = [
        p.third_party for p in packets if p.third_party and p.third_party.raw_info.startswith("T#")
    ]
    assert telem, "expected the telemetry beacon"
    assert telem[0].latitude is None


def test_real_sample_chunk_invariance(aprs_iq: np.ndarray) -> None:
    big = {(p.source, p.raw_info) for p in _decode(aprs_iq, 131072)}
    small = {(p.source, p.raw_info) for p in _decode(aprs_iq, 4096)}
    assert big == small
    assert len(big) >= 3
