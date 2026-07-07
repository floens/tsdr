"""Tests for the FSK teleprinter framework (NAVTEX / RTTY / generic FSK).

Synthetic round-trip tests (encode -> 2-FSK IQ -> decode) run without hardware;
staged groups decode the real NAVTEX and DWD-RTTY captures end to end, including
the no-args auto-acquisition of baud / shift / polarity.
"""

from pathlib import Path

import numpy as np
import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.io import load_iq as _load_iq
from tsdr.radio.decoders.fsk import FSKGenericDecoder, NAVTEXDecoder, RTTYDecoder
from tsdr.radio.decoders.fsk.fec import check_bits, resolve_char
from tsdr.radio.decoders.fsk.profile import NAVTEX_PROFILE, PROFILES, RTTY_PROFILE
from tsdr.radio.decoders.fsk.tables import (
    ALPHABETS,
    CCIR_ALPHA,
    CCIR_FIGS_CODE,
    CCIR_LTRS_CODE,
    CCIR_REP,
    ITA2_FIGS_CODE,
    ITA2_LTRS_CODE,
)
from tsdr.radio.dsp import estimate_fsk_shift
from tsdr.radio.dsp.fsk import FSKFrontEnd
from tsdr.radio.registry import make_demodulator

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "samples" / "navtex_518k_sr5k.cf32.zst"
SAMPLE_RATE = 5000.0
EXPECTED_HEADERS = ("ZCZC VA95", "ZCZC VA88")
EXPECTED_FRAGMENTS = ("DOVER", "THAMES")
MIN_MESSAGES = 2

DWD_SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "rtty_dwd_11039k.cf32.zst"
DWD_RATE = 2000.0


# --------------------------- synthetic encoders ---------------------------
def _reverse(table: list[str | None]) -> dict[str, int]:
    out: dict[str, int] = {}
    for code, char in enumerate(table):
        if char is not None and char not in out:
            out[char] = code
    return out


def _encode_chars(text: str, alphabet: str) -> list[int]:
    ltrs, figs, ltrs_code, figs_code = ALPHABETS[alphabet]
    to_ltrs, to_figs = _reverse(ltrs), _reverse(figs)
    codes: list[int] = []
    figs_shift = False
    for ch in text.upper():
        if ch in to_ltrs and not (figs_shift and ch in to_figs):
            if figs_shift:
                codes.append(ltrs_code)
                figs_shift = False
            codes.append(to_ltrs[ch])
        elif ch in to_figs:
            if not figs_shift:
                codes.append(figs_code)
                figs_shift = True
            codes.append(to_figs[ch])
    return codes


def _codes_to_bits(codes: list[int], nbits: int) -> list[int]:
    bits: list[int] = []
    for code in codes:
        for i in range(nbits):
            bits.append((code >> i) & 1)
    return bits


def _rtty_bits(text: str, *, stop_bits: int = 2) -> np.ndarray:
    codes = _encode_chars(text, "ita2")
    bits = [1] * 30  # idle mark
    for code in codes:
        bits.append(0)  # start = space
        bits.extend((code >> i) & 1 for i in range(5))
        bits.extend([1] * stop_bits)
    bits += [1] * 30
    return np.array(bits, dtype=np.int8)


def _sitor_b_bits(text: str) -> np.ndarray:
    """CCIR-476 FEC-B: DX chars at even slots, rep 5 slots later at odd slots."""
    info = _encode_chars(text, "ccir476")
    total = 2 * len(info) + 6
    slots: list[int | None] = [None] * total
    for k, code in enumerate(info):
        slots[2 * k] = code
        slots[2 * k + 5] = code
    for j in range(total):
        if slots[j] is None:
            slots[j] = CCIR_REP if j % 2 == 0 else CCIR_ALPHA
    phasing = [CCIR_REP if j % 2 == 0 else CCIR_ALPHA for j in range(40)]
    return np.array(_codes_to_bits(phasing + slots, 7), dtype=np.int8)  # type: ignore[arg-type]


def _modulate(bits: np.ndarray, sps: int, fs: float, *, shift: float = 170.0, snr_db=None, seed=0):
    freqs = np.where(bits > 0, shift / 2.0, -shift / 2.0)
    inst = np.repeat(freqs, sps).astype(np.float64)
    iq = np.exp(1j * 2 * np.pi * np.cumsum(inst) / fs).astype(np.complex64)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
        npow = np.mean(np.abs(iq) ** 2) / (10 ** (snr_db / 10))
        iq = (iq + noise.astype(np.complex64) * np.sqrt(npow / 2)).astype(np.complex64)
    return iq


def _decode(decoder, iq: np.ndarray, chunk: int = 8192) -> list[str]:
    texts: list[str] = []
    for i in range(0, len(iq), chunk):
        decoder.demodulate(iq[i : i + chunk], 0.0)
        texts.extend(m.text for m in decoder.get_messages())
    return texts


# --------------------------- tables & FEC ---------------------------
class TestTables:
    def test_ita2_round_trip(self):
        ltrs, figs, _, _ = ALPHABETS["ita2"]
        for table in (ltrs, figs):
            for code, char in enumerate(table):
                if char is None or ord(char) < 32:
                    continue
                codes = _encode_chars(char, "ita2")
                assert codes[-1] == code, f"{char!r} did not round-trip"

    def test_ita2_shift_codes_distinct(self):
        assert ITA2_LTRS_CODE != ITA2_FIGS_CODE

    def test_ccir_all_mapped_codes_valid(self):
        ltrs, figs, _, _ = ALPHABETS["ccir476"]
        for code, char in enumerate(ltrs):
            if char is not None:
                assert check_bits(code), f"CCIR letter code {code:#x} not 4-of-7"
        for code, char in enumerate(figs):
            if char is not None:
                assert check_bits(code), f"CCIR figure code {code:#x} not 4-of-7"
        assert check_bits(CCIR_LTRS_CODE) and check_bits(CCIR_FIGS_CODE)

    def test_check_bits(self):
        assert check_bits(0x0F)  # 4 bits
        assert not check_bits(0x07)  # 3 bits
        assert not check_bits(0x1F)  # 5 bits


class TestFEC:
    def _soft(self, code: int, rep: int) -> np.ndarray:
        bits = np.zeros(42, dtype=np.float64)
        for i in range(7):
            bits[35 + i] = 1.0 if (code >> i) & 1 else -1.0
            bits[i] = 1.0 if (rep >> i) & 1 else -1.0
        return bits

    def test_alpha_valid(self):
        bits = self._soft(0x47, 0x00)  # 'A'
        code, status = resolve_char(bits, 35, 0)
        assert code == 0x47 and status == 1

    def test_rep_fallback(self):
        bits = self._soft(0x00, 0x47)  # alpha invalid, rep = 'A'
        code, status = resolve_char(bits, 35, 0)
        assert code == 0x47 and status == 0

    def test_average_recovery(self):
        # alpha 0x47 with one bit flipped, rep 0x47 with a different bit flipped;
        # bitwise-summed average recovers 0x47.
        a = 0x47 ^ 0x01
        r = 0x47 ^ 0x40
        bits = self._soft(a, r)
        code, status = resolve_char(bits, 35, 0)
        assert code == 0x47 and status == -1

    def test_hard_fail(self):
        bits = self._soft(0x00, 0x00)
        code, status = resolve_char(bits, 35, 0)
        assert code is None and status == -2


# --------------------------- front-end ---------------------------
class TestFrontEnd:
    def test_symbol_rate_and_sign(self):
        fs, baud = 6000.0, 100.0
        sps = int(fs / baud)
        # alternating mark/space, 200 symbols
        bits = np.tile([1, 0], 200).astype(np.int8)
        iq = _modulate(bits, sps, fs)
        fe = FSKFrontEnd(fs, baud, 170.0)
        soft = fe.process(iq)
        # ~one soft bit per symbol
        assert abs(len(soft) - len(bits)) <= 3
        # after settling the signs alternate and split cleanly
        settled = soft[20:]
        assert np.mean(settled > 0) == pytest.approx(0.5, abs=0.1)


# --------------------------- round trips ---------------------------
class TestRoundTripRTTY:
    def _rtty(self, fs):
        return RTTYDecoder(sample_rate=fs, baud=45.45, shift_hz=170.0)

    def test_clean(self):
        text = "RYRY DE PA0FSK THE QUICK BROWN FOX 1234567890\r\n"
        fs = RTTY_PROFILE.internal_rate
        sps = int(round(fs / RTTY_PROFILE.baud))
        iq = _modulate(_rtty_bits(text), sps, fs)
        out = " ".join(_decode(self._rtty(fs), iq))
        assert "QUICK BROWN FOX" in out
        assert "1234567890" in out

    def test_noisy(self):
        text = "CQ CQ DE TEST RTTY DECODER 599\r\n"
        fs = RTTY_PROFILE.internal_rate
        sps = int(round(fs / RTTY_PROFILE.baud))
        iq = _modulate(_rtty_bits(text), sps, fs, snr_db=6, seed=3)
        out = " ".join(_decode(self._rtty(fs), iq))
        assert "RTTY DECODER" in out

    def test_reverse_polarity(self):
        # A tone-swapped (LSB) signal decodes with reverse=True, not reverse=False.
        text = "REVERSE POLARITY RTTY\r\n"
        fs = RTTY_PROFILE.internal_rate
        sps = int(round(fs / RTTY_PROFILE.baud))
        iq = _modulate(1 - _rtty_bits(text), sps, fs)
        rev = _decode(RTTYDecoder(sample_rate=fs, baud=45.45, shift_hz=170.0, reverse=True), iq)
        norm = _decode(RTTYDecoder(sample_rate=fs, baud=45.45, shift_hz=170.0, reverse=False), iq)
        assert "REVERSE POLARITY" in " ".join(rev)
        assert "REVERSE POLARITY" not in " ".join(norm)


class TestRTTYStreaming:
    """RTTY (start/stop framing) streams the in-progress line as `partial=True`
    redraws, then seals the finished line on CR/LF. SitorB (NAVTEX) does not."""

    def _split(self, decoder, iq, chunk):
        partial: list[str] = []
        sealed: list[str] = []
        for i in range(0, len(iq), chunk):
            decoder.demodulate(iq[i : i + chunk], 0.0)
            for m in decoder.get_messages():
                (partial if m.partial else sealed).append(m.text)
        return partial, sealed

    def test_partial_lines_grow_then_seal(self):
        text = "RYRY DE STREAM TEST 123\r\n"
        fs = RTTY_PROFILE.internal_rate
        sps = int(round(fs / RTTY_PROFILE.baud))
        iq = _modulate(_rtty_bits(text), sps, fs)
        d = RTTYDecoder(sample_rate=fs, baud=45.45, shift_hz=170.0)
        partial, sealed = self._split(d, iq, chunk=512)

        full = next((t for t in sealed if "STREAM TEST" in t), None)
        assert full is not None, "line never sealed on CR"
        assert partial, "expected streaming partials"
        # some partial is a proper, growing prefix of the eventual sealed line
        assert any(p != full and full.startswith(p) for p in partial)

    def test_navtex_emits_no_partials(self):
        text = "ZCZC AB12 NO PARTIALS HERE NNNN"
        fs = NAVTEX_PROFILE.internal_rate
        sps = int(round(fs / NAVTEX_PROFILE.baud))
        iq = _modulate(_sitor_b_bits(text), sps, fs)
        d = NAVTEXDecoder(sample_rate=fs)
        partial, sealed = self._split(d, iq, chunk=8192)
        assert sealed and not partial


class TestRoundTripNAVTEX:
    def test_clean(self):
        text = "ZCZC AB12 THE QUICK BROWN FOX 12345 NNNN"
        fs = NAVTEX_PROFILE.internal_rate
        sps = int(round(fs / NAVTEX_PROFILE.baud))
        iq = _modulate(_sitor_b_bits(text), sps, fs)
        out = _decode(NAVTEXDecoder(sample_rate=fs), iq)
        joined = " ".join(out)
        assert "ZCZC AB12" in joined
        assert "THE QUICK BROWN FOX" in joined

    def test_noisy_fec(self):
        text = "ZCZC EA47 GALE WARNING NNNN"
        fs = NAVTEX_PROFILE.internal_rate
        sps = int(round(fs / NAVTEX_PROFILE.baud))
        iq = _modulate(_sitor_b_bits(text), sps, fs, snr_db=3, seed=7)
        out = " ".join(_decode(NAVTEXDecoder(sample_rate=fs), iq))
        assert "GALE WARNING" in out


class TestStreamingConsistency:
    def test_chunk_size_invariance(self):
        text = "ZCZC AB12 STREAMING CONSISTENCY CHECK NNNN"
        fs = NAVTEX_PROFILE.internal_rate
        sps = int(round(fs / NAVTEX_PROFILE.baud))
        iq = _modulate(_sitor_b_bits(text), sps, fs)
        a = _decode(NAVTEXDecoder(sample_rate=fs), iq, chunk=512)
        b = _decode(NAVTEXDecoder(sample_rate=fs), iq, chunk=131072)
        assert a == b


# --------------------------- real sample ---------------------------
@pytest.fixture(scope="module")
def navtex_iq():
    if not SAMPLE_FILE.exists():
        pytest.skip(f"Sample file not found: {SAMPLE_FILE}")
    return _load_iq(SAMPLE_FILE)


class TestRealSample:
    def test_frontend_bimodal(self, navtex_iq):
        iq = np.ascontiguousarray(navtex_iq) / (np.mean(np.abs(navtex_iq)) + 1e-12)
        fe = FSKFrontEnd(SAMPLE_RATE, 100.0, 170.0)
        soft = fe.process(iq.astype(np.complex64))
        # a real NAVTEX signal drives soft bits well away from zero on both rails
        assert np.mean(soft > 0) == pytest.approx(0.5, abs=0.15)
        assert np.median(np.abs(soft)) > 0.3

    def test_full_decode(self, navtex_iq):
        decoder = NAVTEXDecoder(sample_rate=SAMPLE_RATE)
        texts = _decode(decoder, navtex_iq)
        assert len(texts) >= MIN_MESSAGES
        joined = "\n".join(texts)
        for header in EXPECTED_HEADERS:
            assert header in joined, f"missing {header!r}"
        for fragment in EXPECTED_FRAGMENTS:
            assert fragment in joined, f"missing {fragment!r}"


# --------------------------- auto-acquisition ---------------------------
class TestAutoAcquire:
    def test_baud_and_polarity_synthetic(self):
        text = "RYRY DE AUTO ACQUIRE TEST 12345\r\n" * 3
        fs = RTTY_PROFILE.internal_rate
        iq = _modulate(_rtty_bits(text), int(round(fs / 45.45)), fs)
        d = RTTYDecoder(sample_rate=fs)  # nothing pinned
        out = " ".join(_decode(d, iq))
        assert d.acquired and d._baud == 45.45 and not d._invert
        assert "AUTO ACQUIRE TEST" in out

    def test_reverse_auto(self):
        text = "INVERTED SIGNAL AUTO DETECT\r\n" * 3
        fs = RTTY_PROFILE.internal_rate
        iq = _modulate(1 - _rtty_bits(text), int(round(fs / 45.45)), fs)
        d = RTTYDecoder(sample_rate=fs)
        out = " ".join(_decode(d, iq))
        assert d._invert
        assert "INVERTED SIGNAL" in out

    def test_baud_75_from_standard_set(self):
        # 75 is well-separated from its neighbours (45.45 vs 50 differ by only 10%,
        # within async-framing tolerance, so those two are not reliably distinguishable).
        text = "SEVENTY FIVE BAUD TEST\r\n" * 3
        fs = 6000.0
        iq = _modulate(_rtty_bits(text), int(round(fs / 75.0)), fs)
        d = RTTYDecoder(sample_rate=fs)
        out = " ".join(_decode(d, iq))
        assert d._baud == 75.0
        assert "SEVENTY FIVE BAUD" in out

    def test_retry_until_confident_not_stuck_on_noise(self):
        # A noisy start must NOT permanently lock; once the real signal arrives it acquires.
        fs = RTTY_PROFILE.internal_rate
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(int(3 * fs)) + 1j * rng.standard_normal(int(3 * fs))) * 0.3
        sig = _modulate(_rtty_bits("RETRY LOCK TEST 42\r\n" * 3), int(round(fs / 45.45)), fs)
        iq = np.concatenate([noise.astype(np.complex64), sig])
        d = RTTYDecoder(sample_rate=fs)
        out = " ".join(_decode(d, iq))
        assert d.acquired
        assert "RETRY LOCK TEST" in out


class TestShiftEstimate:
    def test_synthetic(self):
        fs = 6000.0
        iq = _modulate(_rtty_bits("SHIFT TEST\r\n" * 4), int(round(fs / 50.0)), fs, shift=450.0)
        center, shift = estimate_fsk_shift(iq, fs)
        assert shift == pytest.approx(450, abs=25)
        assert abs(center) < 25


class TestGenericMode:
    def test_start_stop(self):
        text = "GENERIC FSK START STOP\r\n" * 2
        fs = RTTY_PROFILE.internal_rate
        iq = _modulate(_rtty_bits(text), int(round(fs / 45.45)), fs)
        d = FSKGenericDecoder(sample_rate=fs, baud=45.45, shift_hz=170.0)
        assert "GENERIC FSK" in " ".join(_decode(d, iq))

    def test_sitor_b_selects_ccir476(self):
        text = "ZCZC AB12 GENERIC SITOR NNNN"
        fs = NAVTEX_PROFILE.internal_rate
        iq = _modulate(_sitor_b_bits(text), int(round(fs / 100.0)), fs)
        d = FSKGenericDecoder(sample_rate=fs, framing="sitor_b", baud=100.0, shift_hz=170.0)
        assert d._alphabet == "ccir476"
        assert "GENERIC SITOR" in " ".join(_decode(d, iq))

    def test_make_demodulator_forwards_params(self):
        spec = DemodSpec(
            mode="FSK",
            fsk_baud=50.0,
            fsk_shift_hz=450.0,
            fsk_reverse=True,
            fsk_alphabet="ita2",
            fsk_framing="start_stop",
        )
        d = make_demodulator(spec, 6000.0, 1000)
        assert isinstance(d, FSKGenericDecoder)
        assert d._pin_baud == 50.0 and d._pin_shift == 450.0 and d._pin_invert is True


class TestPresets:
    def test_dwd(self):
        p = PROFILES["dwd"]
        assert (p.baud, p.shift_hz, p.polarity) == (50.0, 450.0, "reverse")

    def test_all_well_formed(self):
        for p in PROFILES.values():
            assert p.framing in ("start_stop", "sitor_b")
            assert p.alphabet in ("ita2", "ccir476")


# --------------------------- real DWD RTTY sample ---------------------------
@pytest.fixture(scope="module")
def dwd_iq():
    if not DWD_SAMPLE.exists():
        pytest.skip(f"Sample file not found: {DWD_SAMPLE}")
    return _load_iq(DWD_SAMPLE)


class TestRealSampleDWD:
    def test_shift_estimate(self, dwd_iq):
        _, shift = estimate_fsk_shift(dwd_iq, DWD_RATE)
        assert shift == pytest.approx(450, abs=30)

    def test_explicit_decode(self, dwd_iq):
        d = RTTYDecoder(sample_rate=DWD_RATE, baud=50.0, shift_hz=450.0, reverse=True)
        assert "00Z" in " ".join(_decode(d, dwd_iq))

    def test_auto_acquire(self, dwd_iq):
        d = RTTYDecoder(sample_rate=DWD_RATE)  # bare: self-configures baud/shift/polarity
        out = " ".join(_decode(d, dwd_iq))
        assert d.acquired and d._baud == 50.0 and d._invert
        assert d._shift == pytest.approx(450, abs=30)
        assert "00Z" in out
