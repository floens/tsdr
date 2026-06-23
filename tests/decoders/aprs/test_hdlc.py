from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tsdr.radio.decoders.aprs import ax25
from tsdr.radio.decoders.aprs.hdlc import HDLCDeframer, NRZIDecoder


def _levels(synth: SimpleNamespace, frame: bytes) -> np.ndarray:
    return synth.nrzi_encode(synth.frame_to_hdlc_bits(frame))


def _deframe(levels: np.ndarray) -> list[bytes]:
    bits = NRZIDecoder().process(levels)
    return HDLCDeframer().process(bits)


def test_nrzi_roundtrip(synth: SimpleNamespace) -> None:
    data = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1], np.uint8)
    levels = synth.nrzi_encode(data)
    recovered = NRZIDecoder().process(levels)
    # NRZI decode recovers data after the first bit (no prior transition reference).
    assert np.array_equal(recovered[1:], data[1:])


def test_deframe_roundtrip(synth: SimpleNamespace) -> None:
    frame = synth.build_ax25("DL1ABC-9", "APRS", ("WIDE1-1",), "!4807.00N/01133.00E-test")
    frames = _deframe(_levels(synth, frame))
    assert frame in frames


def test_closing_flag_with_five_ones_in_payload(synth: SimpleNamespace) -> None:
    # Info chosen to contain bytes with 5+ consecutive 1s, exercising bit-stuffing
    # and the closing-flag handling that was the prototype's one bug.
    frame = synth.build_ax25("DL1ABC", "APRS", (), "!" + "\xff\xfe\x1f" * 4)
    frames = _deframe(_levels(synth, frame))
    assert frame in frames
    assert ax25.parse_frame(frame) is not None


def test_chunked_equals_whole(synth: SimpleNamespace) -> None:
    frame = synth.build_ax25("DL1IGW-10", "APRS", ("WIDE1-1",), "}DL2XYZ-8>APRS::pos")
    levels = _levels(synth, frame)
    bits = NRZIDecoder().process(levels)

    whole = HDLCDeframer().process(bits)
    streamed: list[bytes] = []
    deframer = NRZIDecoder()
    hdlc = HDLCDeframer()
    for start in range(0, len(levels), 7):  # odd chunk size to stress state carry
        streamed.extend(hdlc.process(deframer.process(levels[start : start + 7])))
    assert frame in whole
    assert streamed == whole


def test_single_bit_fix_recovers(synth: SimpleNamespace) -> None:
    frame = synth.build_ax25("DL1ABC", "APRS", (), "!4807.00N/01133.00E-x")
    corrupt = bytearray(frame)
    corrupt[20] ^= 0x08  # flip one bit in the info field
    assert ax25.parse_frame(bytes(corrupt)) is None  # FCS now fails
    recovered = ax25.recover(bytes(corrupt))
    assert recovered is not None
    assert str(recovered.src) == "DL1ABC"


def test_bit_fix_rejects_unrecoverable(synth: SimpleNamespace) -> None:
    frame = synth.build_ax25("DL1ABC", "APRS", (), "!pos")
    corrupt = bytearray(frame)
    corrupt[0] ^= 0x55  # multi-bit smash in the dest address
    corrupt[8] ^= 0x33
    assert ax25.recover(bytes(corrupt)) is None
