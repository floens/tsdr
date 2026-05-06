"""FLEX protocol decoder for paging.

Decodes FLEX paging protocol (1600 baud, 2-FSK, ±4.8 kHz deviation)
from IQ samples. Used by paging systems such as Dutch P2000 on 169.650 MHz.

IQ -> decimate -> FM discriminator -> M&M timing -> bits -> frame sync -> BCH -> messages

Reference: ARIB STD-T68, Motorola FLEX specification
"""

from __future__ import annotations

import logging
import time

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import SignalInfo
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.dsp import FMDiscriminator, MuellerMuller, StreamingFilter, firwin

logger = logging.getLogger(__name__)

# FLEX constants
FLEX_BAUD = 1600  # symbols/second
FLEX_DEVIATION = 4800.0  # Hz, ±4.8 kHz for 2-FSK
FLEX_FRAME_DURATION = 1.875  # seconds
FLEX_BLOCKS_PER_FRAME = 11
FLEX_CODEWORDS_PER_BLOCK = 8
FLEX_CODEWORD_BITS = 32
FLEX_DATA_BITS = FLEX_BLOCKS_PER_FRAME * FLEX_CODEWORDS_PER_BLOCK * FLEX_CODEWORD_BITS  # 2816

# Minimum bits to collect after sync1 detection + data.
# After sync1 marker (32-bit shift register match):
#   A1_low (16) + tail (8) + padding (~16) + FIW (32) + sync2 (~32) + data (2816)
# We collect enough to handle the worst-case offset (~104 bits before data).
FLEX_POST_SYNC_BITS = 110 + FLEX_DATA_BITS  # collect 110 + 2816 = 2926

# Sync words (1600/2-FSK). A' = bitwise NOT of A.
# Polarity flag indicates whether data bits need inversion AFTER matching.
# Our FM discriminator outputs with inverted polarity relative to the FLEX
# convention, so matching the complement sync (~A) means data is already
# correct (no inversion needed).
FLEX_SYNC1_WORDS = {
    0xA6C6AAAA: ("1600/2", True),  # A in our FM polarity -> invert data
    0x59395555: ("1600/2", False),  # ~A in our FM polarity -> data correct
}

# FLEX idle codeword (21 info bits all 1s)
FLEX_IDLE_CODEWORD = 0x1FFFFF

# Precompute GF(2^5) tables for BCH(31,21)
# Primitive polynomial: M1(x) = x^5 + x^2 + 1
_GF_EXP = [0] * 32
_GF_LOG = [0] * 32
_val = 1
for _i in range(31):
    _GF_EXP[_i] = _val
    _GF_LOG[_val] = _i
    _val <<= 1
    if _val & 0x20:
        _val ^= 0x25
_GF_EXP[31] = _GF_EXP[0]


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] + _GF_LOG[b]) % 31]


def _gf_inv(a: int) -> int:
    if a == 0:
        return 0
    return _GF_EXP[(31 - _GF_LOG[a]) % 31]


def _gf_pow(a: int, n: int) -> int:
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] * n) % 31]


def bch_syndrome(codeword: int) -> tuple[int, int]:
    """Compute BCH(31,21) syndromes S1, S3.

    Evaluates the 31-bit codeword (bits 31..1, ignoring bit 0 parity)
    at alpha and alpha^3 where alpha is a root of M1.
    """
    s1 = 0
    s3 = 0
    for i in range(31, 0, -1):
        bit = (codeword >> i) & 1
        if bit:
            s1 ^= _GF_EXP[i - 1]
            s3 ^= _GF_EXP[((i - 1) * 3) % 31]
    return s1, s3


def bch_correct(codeword: int) -> tuple[int, bool]:
    """Correct up to 2 errors in a BCH(31,21) codeword.

    Returns (corrected_codeword, success).
    Layout: [bit31..bit11: 21 info] [bit10..bit1: 10 parity] [bit0: even parity]
    """
    s1, s3 = bch_syndrome(codeword)

    if s1 == 0 and s3 == 0:
        parity = bin(codeword).count("1") & 1
        return codeword, parity == 0

    if s1 == 0 or s3 == 0:
        return codeword, False

    # Single error: S3 = S1^3
    if s3 == _gf_pow(s1, 3):
        pos = _GF_LOG[s1]
        codeword ^= 1 << (pos + 1)
        parity = bin(codeword).count("1") & 1
        return codeword, parity == 0

    # Two errors: solve x^2 + S1*x + (S3/S1 + S1^2) = 0
    c = _gf_mul(s3, _gf_inv(s1)) ^ _gf_pow(s1, 2)
    if c == 0:
        return codeword, False

    roots = [x for x in range(1, 32) if _gf_pow(x, 2) ^ _gf_mul(s1, x) ^ c == 0]
    if len(roots) != 2:
        return codeword, False

    for r in roots:
        codeword ^= 1 << (_GF_LOG[r] + 1)

    parity = bin(codeword).count("1") & 1
    return codeword, parity == 0


def _hamming_distance_32(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _reverse_bits_21(v: int) -> int:
    """Reverse 21-bit info field.

    BCH works on MSB-first codewords, but FLEX info bits are transmitted
    LSB-first. Reverse after BCH correction to get correct info values.
    """
    r = 0
    for _ in range(21):
        r = (r << 1) | (v & 1)
        v >>= 1
    return r


class FLEXDecoder(Demodulator):
    """FLEX protocol decoder for paging.

    Processes raw IQ samples through the full chain:
    decimate -> FM discriminator -> Mueller-Muller timing -> bit sync -> frame decode -> BCH -> messages
    """

    def __init__(self, sample_rate: float = 250_000):
        super().__init__()
        self.sample_rate = sample_rate

        # Decimation to ~32 kHz (~20 samples/symbol at 1600 baud)
        self._decimation = max(1, round(sample_rate / 32000.0))
        self._decimated_rate = sample_rate / self._decimation
        self._sps = self._decimated_rate / FLEX_BAUD

        # Anti-alias filter before decimation
        cutoff = min(self._decimated_rate * 0.45, FLEX_DEVIATION * 2)
        self._antialias = StreamingFilter(
            firwin(101, cutoff, fs=sample_rate),
            [1.0],
            dtype=np.complex64,
        )
        self._decim_phase = 0

        # FM discriminator
        self._fm = FMDiscriminator(self._decimated_rate, FLEX_DEVIATION)

        # Mueller-Muller symbol timing recovery
        self._mm = MuellerMuller(self._sps)

        # Sync detection and frame collection
        self._shift_reg = 0
        self._inverted = False
        self._state = "IDLE"
        self._frame_bits: list[int] = []
        self._frame_bits_needed = 0

        # Statistics
        self._syncs_found = 0
        self._frames_decoded = 0
        self._codewords_ok = 0
        self._codewords_fail = 0

        # Accumulated output
        self._pending_messages: list[DecodedMessage] = []

    def demodulate(self, iq_samples: np.ndarray, timestamp: float) -> None:
        """Process IQ samples and accumulate decoded messages."""
        # Anti-alias filter + decimate
        filtered = self._antialias.process(iq_samples)
        n_before = len(filtered)
        decimated = filtered[self._decim_phase :: self._decimation]
        n_used = n_before - self._decim_phase
        self._decim_phase = (self._decimation - n_used % self._decimation) % self._decimation

        # FM discriminator
        fm = self._fm.process(decimated)

        # Mueller-Muller symbol timing recovery
        symbols = self._mm.process(fm)
        if len(symbols) < 2:
            return

        # Bit decisions + frame sync/decode
        bits = (symbols > 0).astype(np.uint8)
        self._pending_messages.extend(self._process_bits(bits))

    def get_messages(self) -> list[DecodedMessage]:
        messages = self._pending_messages
        self._pending_messages = []
        return messages

    def _process_bits(self, bits: np.ndarray) -> list[DecodedMessage]:
        """Process recovered bits through sync detection and frame collection."""
        messages: list[DecodedMessage] = []

        for i in range(len(bits)):
            bit = int(bits[i])
            self._shift_reg = ((self._shift_reg << 1) | bit) & 0xFFFFFFFF

            if self._state == "IDLE":
                for sync_word, (_mode, inverted) in FLEX_SYNC1_WORDS.items():
                    if _hamming_distance_32(self._shift_reg, sync_word) <= 2:
                        self._inverted = inverted
                        self._syncs_found += 1
                        self._state = "SYNC1"
                        self._frame_bits = []
                        self._frame_bits_needed = FLEX_POST_SYNC_BITS
                        logger.debug(
                            "FLEX sync found (inverted=%s, total=%d)",
                            inverted,
                            self._syncs_found,
                        )
                        break

            elif self._state == "SYNC1":
                self._frame_bits.append(bit)
                if len(self._frame_bits) >= self._frame_bits_needed:
                    msgs = self._decode_frame(self._frame_bits)
                    messages.extend(msgs)
                    self._state = "IDLE"
                    self._frame_bits = []

        return messages

    def _decode_frame(self, frame_bits: list[int]) -> list[DecodedMessage]:
        """Decode a complete FLEX frame from collected bits.

        After sync1 marker detection, the bit stream contains:
        A1_low (16) + tail (8) + FIW (32) + sync2 (~48) + data (2816).
        Data starts at fixed offset 104 from the sync1 marker.
        """
        if self._inverted:
            frame_bits = [1 - b for b in frame_bits]

        # Data offset after sync1 marker detection:
        # A1_low(16) + tail(8) + FIW(32) + sync2(~48) = 104 bits before data.
        # This is fixed by the FLEX 1600/2 frame structure.
        data_offset = 104

        data_bits = frame_bits[data_offset:]
        if len(data_bits) < FLEX_DATA_BITS:
            return []

        # De-interleave and BCH-correct all 88 codewords (11 blocks × 8 words)
        codewords: list[tuple[int, bool]] = []
        for blk in range(FLEX_BLOCKS_PER_FRAME):
            block = data_bits[blk * 256 : (blk + 1) * 256]
            for row in range(FLEX_CODEWORDS_PER_BLOCK):
                cw = 0
                for col in range(FLEX_CODEWORD_BITS):
                    cw = (cw << 1) | block[col * FLEX_CODEWORDS_PER_BLOCK + row]
                corrected, ok = bch_correct(cw)
                if ok:
                    self._codewords_ok += 1
                    # We read codeword columns MSB-first (cw <<= 1), which
                    # gives the best BCH correction rate (99% vs 92% LSB-first).
                    # BCH(31,21) is bit-order invariant so error correction
                    # works either way, but the resulting 21-bit info field
                    # ends up reversed relative to FLEX protocol field layouts,
                    # so we reverse it back here.
                    info = _reverse_bits_21((corrected >> 11) & 0x1FFFFF)
                    corrected = (info << 11) | (corrected & 0x7FF)
                else:
                    self._codewords_fail += 1
                codewords.append((corrected, ok))

        msgs = self._parse_frame(codewords)
        if msgs:
            self._frames_decoded += 1
        return msgs

    def _parse_frame(self, codewords: list[tuple[int, bool]]) -> list[DecodedMessage]:
        """Parse decoded codewords into messages.

        BIW layout (21-bit info):
          bits 15-10: vector field start word (6 bits)
          bits 9-8:   address offset (2 bits), +1 for actual start
          bits 6-4:   (vector type field positions within vectors)

        Vector layout (21-bit info):
          bits 20-14: message length in words (7 bits)
          bits 13-7:  message start word (7 bits, absolute index)
          bits 6-4:   message type (3 bits)
        """
        messages: list[DecodedMessage] = []

        if len(codewords) < 2:
            return messages

        biw_cw, biw_ok = codewords[0]
        if not biw_ok:
            return messages

        biw = (biw_cw >> 11) & 0x1FFFFF
        aoffset = ((biw >> 8) & 0x03) + 1
        voffset = (biw >> 10) & 0x3F

        if voffset == 0 or aoffset >= voffset or voffset > 40:
            return messages

        # Extract addresses (short address: capcode = info - 0x8000)
        addresses: list[int] = []
        i = aoffset
        while i < voffset and i < len(codewords):
            cw, ok = codewords[i]
            info = (cw >> 11) & 0x1FFFFF
            if not ok or info == FLEX_IDLE_CODEWORD or info == 0:
                addresses.append(-1)
                i += 1
                continue

            # Long address check: two consecutive words form one address
            long_addr = info < 0x8001 or (0x1E0000 < info < 0x1F0001) or info > 0x1F7FFE
            if long_addr and i + 1 < voffset and i + 1 < len(codewords):
                cw2, ok2 = codewords[i + 1]
                info2 = (cw2 >> 11) & 0x1FFFFF
                if ok2:
                    capcode = ((info2 ^ 0x1FFFFF) << 15) + 2068480 + info
                    addresses.append(capcode)
                else:
                    addresses.append(-1)
                i += 2
            else:
                addresses.append(info - 0x8000)
                i += 1

        n_addrs = len(addresses)

        # Process vectors (one per address, starting at voffset)
        for addr_idx in range(n_addrs):
            vec_idx = voffset + addr_idx
            if vec_idx >= len(codewords):
                break

            cw, ok = codewords[vec_idx]
            if not ok:
                continue

            info = (cw >> 11) & 0x1FFFFF
            if info == FLEX_IDLE_CODEWORD:
                continue

            msg_type = (info >> 4) & 0x07
            msg_start = (info >> 7) & 0x7F  # absolute word index
            msg_len = (info >> 14) & 0x7F

            if addresses[addr_idx] < 0:
                continue
            capcode = addresses[addr_idx]

            if msg_type == 0:
                messages.append(
                    DecodedMessage(
                        text=f"[{capcode:07d}] <tone>",
                        timestamp=time.time(),
                    )
                )

            elif msg_type == 1:
                messages.append(
                    DecodedMessage(
                        text=f"[{capcode:07d}] <short>",
                        timestamp=time.time(),
                    )
                )

            elif msg_type == 2:
                messages.append(
                    DecodedMessage(
                        text=f"[{capcode:07d}] <secure>",
                        timestamp=time.time(),
                    )
                )

            elif msg_type == 3:
                # Standard numeric
                text = self._decode_numeric(codewords, msg_start, msg_len)
                if text:
                    messages.append(
                        DecodedMessage(
                            text=f"[{capcode:07d}] {text}",
                            timestamp=time.time(),
                        )
                    )

            elif msg_type == 5:
                # Alphanumeric
                text = self._decode_alpha(codewords, msg_start, msg_len)
                if text:
                    messages.append(
                        DecodedMessage(
                            text=f"[{capcode:07d}] {text}",
                            timestamp=time.time(),
                        )
                    )

            elif msg_type == 7:
                # Numbered numeric
                text = self._decode_numeric(codewords, msg_start, msg_len, numbered=True)
                if text:
                    messages.append(
                        DecodedMessage(
                            text=f"[{capcode:07d}] {text}",
                            timestamp=time.time(),
                        )
                    )

            else:
                # Binary or other - report as tone
                messages.append(
                    DecodedMessage(
                        text=f"[{capcode:07d}] <type{msg_type}>",
                        timestamp=time.time(),
                    )
                )

        return messages

    def _decode_alpha(self, codewords: list[tuple[int, bool]], start: int, n_words: int) -> str:
        """Decode alphanumeric message (7-bit chars, 3 per 21-bit info word).

        First word contains fragment/continuation metadata (skipped).
        Character packing: c1=bits 6-0, c2=bits 13-7, c3=bits 20-14
        """
        if n_words < 2:
            return ""

        # First word: fragment info (bits 12-11) and continuation flag (bit 10)
        first_cw, first_ok = codewords[start] if start < len(codewords) else (0, False)
        first_info = (first_cw >> 11) & 0x1FFFFF if first_ok else 0
        frag = (first_info >> 11) & 0x03
        skip_first_char = frag == 0x03  # Complete message: skip c1 of first data word

        chars: list[str] = []
        for i in range(1, n_words):  # Skip first word (fragment header)
            idx = start + i
            if idx >= len(codewords):
                break
            cw, ok = codewords[idx]
            if not ok:
                chars.append("?")
                continue
            info = (cw >> 11) & 0x1FFFFF
            if info == FLEX_IDLE_CODEWORD or info == 0:
                continue

            c1 = info & 0x7F
            c2 = (info >> 7) & 0x7F
            c3 = (info >> 14) & 0x7F

            first_word = i == 1
            for j, c in enumerate([c1, c2, c3]):
                if first_word and j == 0 and skip_first_char:
                    continue
                if c == 0x03:  # ETX
                    return "".join(chars).strip()
                if 0x20 <= c <= 0x7E:
                    chars.append(chr(c))
                elif c in (0x0D, 0x0A):
                    chars.append(" ")
                elif c != 0:
                    chars.append("")

        return "".join(chars).strip()

    def _decode_numeric(
        self,
        codewords: list[tuple[int, bool]],
        start: int,
        n_words: int,
        *,
        numbered: bool = False,
    ) -> str:
        """Decode numeric message using LSB-first bitstream extraction.

        Builds a contiguous bitstream from all message words, skips header
        bits, then extracts 4-bit BCD digits LSB-first.
        """
        bcd_map = "0123456789 U -]["
        fill_char = 0x0C

        # Build contiguous bitstream from all message words (LSB-first per word)
        bitstream: list[int] = []
        for i in range(n_words):
            idx = start + i
            if idx >= len(codewords):
                break
            cw, ok = codewords[idx]
            if not ok:
                # Pad with zeros to maintain alignment
                bitstream.extend([0] * 21)
                continue
            info = (cw >> 11) & 0x1FFFFF
            if info == FLEX_IDLE_CODEWORD or info == 0:
                # Pad with fill chars to maintain alignment
                bitstream.extend([0] * 21)
                continue
            for bit_pos in range(21):
                bitstream.append((info >> bit_pos) & 1)

        # Skip header bits: 10 for numbered numeric, 2 for standard/special
        skip = 10 if numbered else 2
        pos = skip

        digits: list[str] = []
        while pos + 4 <= len(bitstream):
            d = 0
            for b in range(4):
                d |= bitstream[pos + b] << b
            pos += 4
            if d == fill_char:
                continue
            if d < len(bcd_map):
                digits.append(bcd_map[d])

        result = "".join(digits).strip()
        # Filter out messages that are just repeated zeros (empty/padding)
        if result and len(set(result)) <= 1:
            return ""
        return result

    def info(self) -> SignalInfo:
        """Thread-safe: callable from any thread. Reads scalar counters only."""
        quality = None
        quality_label = None
        total_cw = self._codewords_ok + self._codewords_fail
        if total_cw > 0:
            quality = self._codewords_ok / total_cw
            quality_label = f"BCH {quality * 100:.0f}%"
        return SignalInfo(
            label="FLEX 1600/2",
            channel_bandwidth=25_000,
            modulation="2-FSK",
            has_text=True,
            message_type="text",
            quality_label=quality_label,
            quality=quality,
        )

    def reset(self) -> None:
        """Reset decoder state."""
        self._antialias.reset()
        self._decim_phase = 0
        self._fm.reset()
        self._mm.reset()
        self._shift_reg = 0
        self._inverted = False
        self._state = "IDLE"
        self._frame_bits = []
        self._frame_bits_needed = 0
        self._syncs_found = 0
        self._frames_decoded = 0
        self._codewords_ok = 0
        self._codewords_fail = 0
        self._pending_messages.clear()
