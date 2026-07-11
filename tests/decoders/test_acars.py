"""Tests for the ACARS decoder (2400-baud MSK on an AM VHF channel).

Synthetic tests cover the CRC and syndrome FEC; a staged group decodes the real
131.824 MHz capture end to end through `ACARSDecoder` (IQ -> AM envelope -> MSK
-> frame), including the higher-rate decimation path.
"""

from pathlib import Path

import numpy as np
import pytest
from rich.text import Text

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.io import load_iq as _load_iq
from tsdr.radio.decoders.acars import ACARSDecoder
from tsdr.radio.decoders.acars.constants import ETX, PLUS, SOH, STAR, SYN
from tsdr.radio.decoders.acars.crc import crc16, update, valid
from tsdr.radio.decoders.acars.fec import fix_double_error, fix_parity_errors
from tsdr.radio.decoders.acars.frame import AcarsFramer
from tsdr.radio.decoders.acars.labels import describe_label
from tsdr.radio.decoders.acars.oooi import decode_oooi
from tsdr.radio.dsp.filters import resample_poly
from tsdr.radio.registry import DEMODULATORS, make_demodulator

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "samples" / "acars_131824k_sr12k.cf32.zst"
SAMPLE_RATE = 12000.0


def _odd_parity_block(text: bytes) -> bytearray:
    """7-bit ASCII bytes with the ACARS odd-parity MSb set."""
    block = bytearray()
    for c in text:
        b = c & 0x7F
        if b.bit_count() % 2 == 0:
            b |= 0x80
        block.append(b)
    return block


def _crc_bytes(block: bytes) -> tuple[int, int]:
    r = crc16(block)
    return r & 0xFF, (r >> 8) & 0xFF


def _residual(block: bytes, crc0: int, crc1: int) -> int:
    return update(update(crc16(block), crc0), crc1)


# mode '2', reg '.A6-EGZ', ack 'J', label 'B6', bid 'Q' (non-digit -> no seq/flight
# split), STX, then free text. 13 header bytes + text, all odd-parity.
_BURST_CORE = _odd_parity_block(b"2.A6-EGZJB6Q") + bytes([0x02]) + _odd_parity_block(b"HELLO WORLD")


def _burst(core: bytes) -> np.ndarray:
    """Soft bits for one burst: sync anchor + core + ETX + 2 CRC bytes, LSb-first, +-1."""
    body = bytes(core) + bytes([ETX])
    r = crc16(body)
    seq = [PLUS, STAR, SYN, SYN, SOH, *body, r & 0xFF, (r >> 8) & 0xFF]
    bits = [(byte >> i) & 1 for byte in seq for i in range(8)]
    return np.array([1.0 if b else -1.0 for b in bits], dtype=np.float32)


def _flip(soft: np.ndarray, block_index: int, bit: int) -> None:
    soft[40 + block_index * 8 + bit] *= -1.0  # 40 sync bits precede collected block byte 0


def _decode(decoder: ACARSDecoder, iq: np.ndarray, chunk: int = 8192) -> list:
    out = []
    for i in range(0, len(iq), chunk):
        decoder.demodulate(iq[i : i + chunk], 0.0)
        out.extend(decoder.get_messages())
    return out


def _plain(m) -> str:
    """Rendered text with Rich markup stripped (rows now carry color markup)."""
    return Text.from_markup(m.text).plain


class TestCRC:
    def test_valid_codeword(self):
        block = _odd_parity_block(b"2Q0EXAMPLE.BLK")
        assert valid(block, *_crc_bytes(block))

    def test_flipped_bit_fails(self):
        block = _odd_parity_block(b"2Q0EXAMPLE.BLK")
        crc0, crc1 = _crc_bytes(block)
        block[4] ^= 0x08
        assert not valid(block, crc0, crc1)


class TestFEC:
    def _block(self):
        block = _odd_parity_block(b"2.A6-EGZ.B6.STXHELLO")
        return block, *_crc_bytes(block)

    def test_single_parity_error(self):
        block, crc0, crc1 = self._block()
        corrupt = bytearray(block)
        corrupt[3] ^= 0x04
        pr = [i for i, b in enumerate(corrupt) if b.bit_count() % 2 == 0]
        assert pr and fix_parity_errors(corrupt, _residual(corrupt, crc0, crc1), pr)
        assert corrupt == block

    def test_double_bit_in_byte(self):
        block, crc0, crc1 = self._block()
        corrupt = bytearray(block)
        corrupt[5] ^= 0x03  # two bits in one byte -> parity intact, CRC breaks
        assert all(b.bit_count() % 2 for b in corrupt)
        residual = _residual(corrupt, crc0, crc1)
        assert residual != 0 and fix_double_error(corrupt, residual)
        assert corrupt == block


class TestRegistry:
    def test_acars_registered(self):
        assert "ACARS" in DEMODULATORS

    def test_make_demodulator(self):
        d = make_demodulator(DemodSpec(mode="ACARS"), SAMPLE_RATE)
        assert isinstance(d, ACARSDecoder)


@pytest.fixture(scope="module")
def acars_iq():
    if not SAMPLE_FILE.exists():
        pytest.skip(f"Sample file not found: {SAMPLE_FILE}")
    return _load_iq(SAMPLE_FILE)


@pytest.fixture(scope="module")
def decoded(acars_iq):
    return _decode(ACARSDecoder(sample_rate=SAMPLE_RATE), acars_iq)


class TestRealSample:
    def test_decode(self, decoded):
        assert len(decoded) >= 6
        joined = "\n".join(_plain(m) for m in decoded)
        assert "A6-EGZ" in joined and "EK001J" in joined and ".ADS." in joined
        assert "EK0150" in joined
        assert "N477MC" in joined and "GTI604" in joined
        assert "ARINC" in joined

    def test_descriptions_rendered(self, decoded):
        joined = "\n".join(_plain(m) for m in decoded)
        assert "message to/from terminal" in joined
        assert all(m.markup for m in decoded)

    def test_fec_recovers_extra(self, decoded):
        assert any(m.data.verified and m.data.errors > 0 for m in decoded)

    def test_all_verified(self, decoded):
        assert all(m.data.verified for m in decoded)

    def test_chunk_size_invariance(self, acars_iq):
        a = [m.text for m in _decode(ACARSDecoder(sample_rate=SAMPLE_RATE), acars_iq, chunk=1024)]
        b = [m.text for m in _decode(ACARSDecoder(sample_rate=SAMPLE_RATE), acars_iq, chunk=131072)]
        assert a == b

    def test_higher_rate_decimation(self, acars_iq):
        # upsample 12 kHz -> 48 kHz to exercise the device->intermediate->12 kHz path
        up = resample_poly(acars_iq, 4, 1).astype(np.complex64)
        msgs = _decode(ACARSDecoder(sample_rate=48000.0), up)
        joined = "\n".join(_plain(m) for m in msgs)
        assert len(msgs) >= 3 and "A6-EG" in joined


class TestPartial:
    """Emitting unverified blocks (parity/CRC/FEC failure) via synthetic bursts."""

    def test_valid_burst_verified(self):
        f = AcarsFramer()
        f.process(_burst(_BURST_CORE), 0.0)
        msgs = f.drain()
        assert len(msgs) == 1
        m = msgs[0]
        assert m.data.verified and m.data.errors == 0
        assert m.data.text == "HELLO WORLD" and "A6-EGZ" in _plain(m)
        assert m.data.label_desc == "ADS report" and "ADS report" in _plain(m)
        assert "[unverified" not in _plain(m)

    def test_corrupt_burst_unverified_masked(self):
        soft = _burst(_BURST_CORE)
        for bi in (13, 15, 17, 19):  # 4 text bytes -> exceeds the 3-byte FEC budget
            _flip(soft, bi, 0)
        f = AcarsFramer()
        f.process(soft, 0.0)
        msgs = f.drain()
        assert len(msgs) == 1
        m = msgs[0]
        assert not m.data.verified and m.data.errors == 4
        assert "[unverified 4 bad]" in _plain(m)
        assert "A6-EGZ" in _plain(m)
        assert all(m.data.text[i] == "." for i in (0, 2, 4, 6))
        assert m.data.label_desc == "" and m.data.oooi is None
        assert "ADS report" not in _plain(m)

    def test_partial_disabled_drops_corrupt(self):
        soft = _burst(_BURST_CORE)
        for bi in (13, 15, 17, 19):
            _flip(soft, bi, 0)
        f = AcarsFramer(emit_partial=False)
        f.process(soft, 0.0)
        assert f.drain() == []


class TestLabels:
    def test_known(self):
        assert describe_label("B6") == "ADS report"
        assert describe_label("H1") == "message to/from terminal"

    def test_oooi_class(self):
        assert describe_label("Q1") == "OOOI / movement report"

    def test_tech_ack_variant(self):
        assert describe_label("_d") == "no-op / technical ack"

    def test_unknown_fallback(self):
        assert describe_label("ZZ") == ""


class TestOooi:
    def test_q2_dep_eta(self):
        o = decode_oooi("Q2", "KJFK1830")
        assert o is not None and o.dep == "KJFK" and o.eta == "1830"

    def test_q1_full_block(self):
        o = decode_oooi("Q1", "KLAX1200121517301745XXXXKJFK")
        assert o is not None
        assert o.dep == "KLAX" and o.gate_out == "1200" and o.wheels_off == "1215"
        assert o.wheels_on == "1730" and o.gate_in == "1745" and o.dest == "KJFK"

    def test_qn_dest_eta(self):
        o = decode_oooi("QN", "0000EGLL1930")
        assert o is not None and o.dest == "EGLL" and o.eta == "1930"

    def test_21_guarded_commas(self):
        o = decode_oooi("21", "POS011,KLAX,KSFO")  # commas at [6] and [11]
        assert o is not None and o.dep == "KLAX" and o.dest == "KSFO"

    def test_guard_fails_returns_none(self):
        assert decode_oooi("21", "no commas here at all") is None

    def test_non_oooi_label(self):
        assert decode_oooi("H1", "whatever text") is None
