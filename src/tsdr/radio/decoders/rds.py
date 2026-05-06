"""RDS (Radio Data System) decoder.

Decodes RDS data from FM-demodulated audio. The RDS subcarrier is at
57 kHz (3× the 19 kHz pilot), BPSK modulated at 1187.5 bps.

Reference: IEC 62106 (RDS specification)
"""

import logging
from dataclasses import dataclass

import numba as nb
import numpy as np

from tsdr.radio.dsp import CostasLoop, MuellerMuller, StreamingFilter, firwin
from tsdr.radio.dsp._kernels import _freq_shift_f32_to_c64

# RDS constants
RDS_SUBCARRIER_FREQ = 57000.0  # Hz
RDS_BITRATE = 1187.5  # bps
RDS_SYMBOL_RATE = RDS_BITRATE  # symbols/s (same as bitrate for BPSK)
RDS_BLOCK_SIZE = 26  # bits per block

# CRC polynomial: x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1
RDS_CRC_POLY = 0x1B9

# Syndrome values for each block type (A, B, C, C', D)
# These equal the standard RDS offset words (IEC 62106) since
# syndrome(valid_block) = offset_word for this CRC implementation.
RDS_SYNDROMES = [0x0FC, 0x198, 0x168, 0x350, 0x1B4]
RDS_OFFSET_POS = [0, 1, 2, 2, 3]  # Block position (C and C' both at pos 2)


@nb.njit(cache=True)
def _calc_syndrome_jit(block: int, poly: int) -> int:
    """Calculate CRC syndrome for a 26-bit RDS block (numba-jit'd)."""
    reg = 0
    for i in range(25, -1, -1):
        bit = (block >> i) & 1
        msb = (reg >> 9) & 1
        reg = ((reg << 1) | bit) & 0x3FF
        if msb:
            reg ^= poly
    return reg


def _build_error_lookup(max_burst: int = 5, include_2bit_data: bool = False) -> dict[int, int]:
    """Build syndrome -> error-mask lookup for burst error correction.

    Covers all single-bit errors in the 26-bit block, plus burst errors
    up to `max_burst` bits long within the 10-bit checkword (bits 0-9).
    If include_2bit_data, also covers 2-bit errors in the data portion (bits 10-25).
    Checkword-only errors don't corrupt data so they're safe to "correct".
    """
    lookup: dict[int, int] = {}

    for bit in range(26):
        s = int(_calc_syndrome_jit(1 << bit, RDS_CRC_POLY))
        lookup[s] = 1 << bit

    if include_2bit_data:
        for i in range(10, 26):
            for j in range(i + 1, 26):
                mask = (1 << i) | (1 << j)
                s = int(_calc_syndrome_jit(mask, RDS_CRC_POLY))
                if s not in lookup:
                    lookup[s] = mask

    for burst_len in range(2, max_burst + 1):
        for start in range(10 - burst_len + 1):
            mask = ((1 << burst_len) - 1) << start
            s = int(_calc_syndrome_jit(mask, RDS_CRC_POLY))
            if s not in lookup:
                lookup[s] = mask

    return lookup


_ERROR_SYNDROMES = _build_error_lookup()
_ERROR_SYNDROMES_AGGRESSIVE = _build_error_lookup(include_2bit_data=True)

# Program Type names (USA RBDS)
PTY_NAMES = [
    "None",
    "News",
    "Information",
    "Sports",
    "Talk",
    "Rock",
    "Classic Rock",
    "Adult Hits",
    "Soft Rock",
    "Top 40",
    "Country",
    "Oldies",
    "Soft",
    "Nostalgia",
    "Jazz",
    "Classical",
    "R&B",
    "Soft R&B",
    "Language",
    "Religious Music",
    "Religious Talk",
    "Personality",
    "Public",
    "College",
    "Spanish Talk",
    "Spanish Music",
    "Hip Hop",
    "Unassigned",
    "Unassigned",
    "Weather",
    "Emergency Test",
    "Emergency",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RDSGroup:
    """Single decoded RDS group."""

    group_type: int  # 0-15
    version: int  # 0=A, 1=B
    pi_code: int
    pty: int
    summary: str  # Human-readable one-line summary


@dataclass(frozen=True)
class RDSData:
    """Immutable snapshot of decoded RDS data."""

    pi_code: int | None = None
    ps_name: str = ""
    radio_text: str = ""
    pty: int = 0
    pty_name: str = ""

    sync_locked: bool = False
    groups_received: int = 0
    block_error_rate: float = 0.0
    uncorrectable_blocks: int = 0
    sync_confidence: float = 0.0
    baseband_offset_hz: float = 0.0  # Costas loop residual frequency offset

    recent_groups: tuple[RDSGroup, ...] = ()  # Groups decoded since last snapshot


class RDSDecoder:
    """RDS decoder with pilot-locked carrier recovery.

    Pipeline:
    1. Frequency shift 57 kHz subcarrier to baseband
    2. Anti-alias LPF (4 kHz) + decimate to ~10 kHz
    3. Sharp matched filter (2.4 kHz, Nuttall window) at decimated rate
    4. AGC
    5. Mueller-Muller symbol timing recovery
    6. Costas loop (phase/frequency correction)
    7. Differential decode
    8. Block sync + CRC + group decode
    """

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

        # Stage 1: Decimate to ~10 kHz (~8 sps). Lower sps makes M&M
        # converge reliably regardless of initial timing phase (like SDR++).
        self._decimation = max(1, round(sample_rate / 10000))
        self._decimated_rate = sample_rate / self._decimation
        decimated_rate = self._decimated_rate

        # Fractional samples-per-symbol: M&M handles non-integer sps natively
        # via its linear interpolation + mu accumulator.
        self._sps = decimated_rate / RDS_SYMBOL_RATE

        # Anti-alias LPF before decimation: wide cutoff (4 kHz), just enough
        # to prevent aliasing at the decimated rate (~10 kHz Nyquist = 5 kHz).
        self._antialias = StreamingFilter(
            firwin(51, 4000, fs=sample_rate),
            [1.0],
            dtype=np.complex64,
        )

        # Sharp matched filter after decimation: tight cutoff at RDS bandwidth
        # with Nuttall window for high stopband attenuation (~-98 dB).
        # SDR++ uses 3.8 * fs / trans_width taps; we use 200 Hz transition.
        n_sharp = int(3.8 * decimated_rate / 200) | 1  # ensure odd
        self._sharp = StreamingFilter(
            firwin(n_sharp, 2400, fs=decimated_rate, window="nuttall"),
            [1.0],
            dtype=np.complex64,
        )

        # 57 kHz carrier state (phase-continuous across chunks)
        self._carrier_freq = 2 * np.pi * RDS_SUBCARRIER_FREQ / sample_rate  # rad/sample
        self._carrier_phase = 0.0
        # Pre-allocated output buffer for freq shift (grows if needed)
        self._freq_shift_buf = np.empty(200_000, dtype=np.complex64)

        # Decimation phase tracking (streaming continuity)
        self._decim_phase = 0

        # AGC: IIR power tracker for envelope compression
        # Fast tc compresses BPSK envelope, improving M&M timing recovery.
        agc_alpha = 2.0 / (100 + 1)
        self._agc = StreamingFilter(
            np.array([agc_alpha]),
            np.array([1.0, -(1.0 - agc_alpha)]),
            dtype=np.float64,
        )

        # Mueller-Muller symbol timing recovery
        self._mm = MuellerMuller(self._sps, gain=0.008)

        # Costas loop (phase/frequency correction)
        self._costas = CostasLoop()

        # Differential decode state
        self._prev_bit = 0
        self._diff_buf = np.empty(2048, dtype=np.uint8)

        # Block sync state
        self._shift_reg = 0
        self._synced = False
        self._presync = False
        self._lastseen_offset = 0
        self._lastseen_offset_counter = 0
        self._bit_counter = 0
        self._block_bit_counter = 0
        self._block_number = 0
        self._group_blocks = [0, 0, 0, 0]
        # Which blocks in the current group carry fresh data (vs stale from the
        # previous group). Gates _decode_group so PS/RT/ODA never read mixed blocks.
        self._block_valid = [False, False, False, False]
        self._wrong_blocks_counter = 0
        self._blocks_counter = 0
        self._consecutive_wrong = 0

        # Sync watchdog: nudge M&M timing phase if no sync after N bits.
        # Tries 4 timing offsets (0, sps/4, sps/2, 3*sps/4) before giving up.
        self._sync_watchdog_bits = 0
        self._sync_nudge_count = 0

        # Decoded data
        self._pi_code: int | None = None
        self._ps_chars = [""] * 8
        self._ps_votes: list[dict[str, int]] = [{} for _ in range(8)]  # majority voting
        self._rt_chars = [""] * 64
        self._rt_votes: list[dict[str, int]] = [{} for _ in range(64)]
        self._rt_ab: int | None = None
        self._pty_votes: dict[int, int] = {}
        self._pi_candidate: int | None = None
        self._pty = 0

        # ODA: group type -> AID mapping (populated by 3A groups)
        self._oda_map: dict[int, int] = {}

        # Statistics
        self._groups_received = 0
        self._blocks_total = 0
        self._uncorrectable = 0

        # Recent group buffer (flushed on each snapshot)
        self._recent_groups: list[RDSGroup] = []

        # Constellation buffer: ~1 second of symbols at RDS baud rate
        self._constellation_size = int(RDS_SYMBOL_RATE)
        self._constellation_buf = np.zeros(self._constellation_size, dtype=np.complex64)
        self._constellation_pos = 0
        self._constellation_points: np.ndarray | None = None

    def process(self, audio: np.ndarray) -> RDSData:
        """Process FM-demodulated audio and decode RDS."""
        # 1. Frequency shift 57 kHz to baseband
        iq = self._freq_shift(audio)

        # 2. Anti-alias filter + decimate to ~10 kHz
        iq = self._antialias.process(iq)
        n_before = len(iq)
        iq = iq[self._decim_phase :: self._decimation]
        n_used = n_before - self._decim_phase
        self._decim_phase = (self._decimation - n_used % self._decimation) % self._decimation

        # 3. Sharp matched filter at decimated rate
        iq = self._sharp.process(iq)

        # 4. AGC via IIR-smoothed power envelope
        power = np.abs(iq) ** 2
        smooth_power = self._agc.process(power)
        gain = 1.0 / np.maximum(np.sqrt(smooth_power), 1e-10)
        iq = iq * gain

        # 5. Mueller-Muller symbol timing recovery
        symbols = self._mm.process(iq)

        if len(symbols) < 10:
            return self._snapshot()

        # 6. Costas loop
        symbols = self._costas.process(symbols)

        # Collect post-Costas symbols for constellation display
        n = len(symbols)
        buf = self._constellation_buf
        pos = self._constellation_pos
        if n >= len(buf):
            buf[:] = symbols[-len(buf) :]
            self._constellation_pos = 0
        else:
            space = len(buf) - pos
            if n <= space:
                buf[pos : pos + n] = symbols
                self._constellation_pos = pos + n
            else:
                buf[pos:] = symbols[:space]
                buf[: n - space] = symbols[space:]
                self._constellation_pos = n - space
        self._constellation_points = buf.copy()

        # 7. Differential decode
        bits = self._demod_bits(symbols)

        # 8. Block sync
        self._process_bits(bits)

        # 9. Sync watchdog: nudge M&M timing if stuck
        self._check_sync_watchdog(len(bits))

        return self._snapshot()

    def get_constellation(self) -> np.ndarray | None:
        points = self._constellation_points
        self._constellation_points = None
        return points

    # Nudge timing after this many bits without sync (~2.5s at 1187.5 bps)
    _SYNC_WATCHDOG_LIMIT = 3000

    def _nudge_timing(self) -> None:
        """Nudge M&M timing phase by sps/4. Preserves Costas lock."""
        self._sync_nudge_count = (self._sync_nudge_count + 1) % 4
        self._mm.nudge(0.25)
        logger.debug(
            "RDS timing nudge #%d, mu=%.2f",
            self._sync_nudge_count,
            self._mm._mu,
        )

    def _sync_drop(self) -> None:
        """Drop sync, reset counters, and nudge for re-acquisition.

        PS/RT/PI are preserved: a re-sync on the same station (detected via PI
        match in _decode_group) keeps accumulating votes. A real station change
        clears them in the PI-mismatch branch there.
        """
        self._synced = False
        self._presync = False
        self._groups_received = 0
        self._blocks_total = 0
        self._uncorrectable = 0
        self._wrong_blocks_counter = 0
        self._blocks_counter = 0
        self._consecutive_wrong = 0
        self._sync_nudge_count = 0
        self._nudge_timing()

    def _check_sync_watchdog(self, n_bits: int) -> None:
        """Nudge M&M timing phase if no sync achieved within the bit limit.

        First few nudges use a shorter limit so we sweep phases quickly, then
        relax to avoid thrashing once we've explored the initial options.
        """
        if self._synced:
            self._sync_watchdog_bits = 0
            return

        self._sync_watchdog_bits += n_bits
        # Aggressive phase sweep for the first 4 nudges (one full cycle of
        # sps/4 offsets), then fall back to the longer limit.
        limit = 750 if self._sync_nudge_count < 4 else self._SYNC_WATCHDOG_LIMIT
        if self._sync_watchdog_bits < limit:
            return

        self._nudge_timing()
        self._presync = False
        self._sync_watchdog_bits = 0

    def _freq_shift(self, audio: np.ndarray) -> np.ndarray:
        """Shift 57 kHz RDS subcarrier to baseband (numba kernel)."""
        n = len(audio)
        if n > len(self._freq_shift_buf):
            self._freq_shift_buf = np.empty(n, dtype=np.complex64)
        audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)
        self._carrier_phase = _freq_shift_f32_to_c64(
            audio_f32, self._carrier_freq, self._carrier_phase, self._freq_shift_buf
        )
        return self._freq_shift_buf[:n]

    def _demod_bits(self, symbols: np.ndarray) -> np.ndarray:
        """BPSK demodulation with differential decoding."""
        if len(symbols) < 1:
            return np.array([], dtype=np.uint8)

        hard_bits = (np.real(symbols) > 0).astype(np.uint8)

        # Differential decode against previous-chunk last bit, reusing a streaming buffer.
        n = len(hard_bits)
        if n + 1 > len(self._diff_buf):
            self._diff_buf = np.empty(n + 1, dtype=np.uint8)
        self._diff_buf[0] = self._prev_bit
        self._diff_buf[1 : n + 1] = hard_bits
        diff_bits = np.bitwise_xor(self._diff_buf[1 : n + 1], self._diff_buf[0:n])

        self._prev_bit = hard_bits[-1]
        return diff_bits

    def _process_bits(self, bits: np.ndarray) -> None:
        """Process bits through block sync state machine."""
        for i in range(len(bits)):
            self._process_bit(int(bits[i]))

    def _process_bit(self, bit: int) -> None:
        """Process bit through block sync state machine."""
        self._bit_counter += 1
        self._shift_reg = ((self._shift_reg << 1) | bit) & 0x3FFFFFF

        if not self._synced:
            syndrome = self._calc_syndrome(self._shift_reg)

            for j in range(5):
                if syndrome == RDS_SYNDROMES[j]:
                    if not self._presync:
                        self._lastseen_offset = j
                        self._lastseen_offset_counter = self._bit_counter
                        self._presync = True
                    else:
                        last_pos = RDS_OFFSET_POS[self._lastseen_offset]
                        curr_pos = RDS_OFFSET_POS[j]

                        if last_pos >= curr_pos:
                            block_distance = curr_pos + 4 - last_pos
                        else:
                            block_distance = curr_pos - last_pos

                        expected_bits = block_distance * RDS_BLOCK_SIZE
                        actual_bits = self._bit_counter - self._lastseen_offset_counter

                        if abs(expected_bits - actual_bits) <= 2:
                            self._synced = True
                            self._block_bit_counter = 0
                            self._block_number = (j + 1) % 4
                            self._wrong_blocks_counter = 0
                            self._blocks_counter = 0
                            self._block_valid = [False, False, False, False]
                        else:
                            self._presync = False
                    break
        else:
            self._block_bit_counter += 1

            if self._block_bit_counter == RDS_BLOCK_SIZE:
                self._block_bit_counter = 0
                self._blocks_total += 1
                self._blocks_counter += 1

                data, valid = self._verify_block(self._shift_reg, self._block_number)

                if valid:
                    self._group_blocks[self._block_number] = data
                    self._block_valid[self._block_number] = True
                    self._consecutive_wrong = 0
                else:
                    self._block_valid[self._block_number] = False
                    self._wrong_blocks_counter += 1
                    self._uncorrectable += 1
                    self._consecutive_wrong += 1

                if self._block_number == 3:
                    self._decode_group(self._group_blocks, self._block_valid)
                    for i in range(4):
                        self._block_valid[i] = False

                self._block_number = (self._block_number + 1) % 4

                if self._consecutive_wrong >= 10:
                    self._sync_drop()
                elif self._blocks_counter >= 40:
                    if self._wrong_blocks_counter / self._blocks_counter > 0.40:
                        self._sync_drop()
                    else:
                        self._wrong_blocks_counter = 0
                        self._blocks_counter = 0

    def _calc_syndrome(self, block: int) -> int:
        """Calculate syndrome for a 26-bit block."""
        return int(_calc_syndrome_jit(block, RDS_CRC_POLY))

    def _verify_block(self, block: int, block_num: int) -> tuple[int, bool]:
        """Verify block CRC with single-bit error correction.

        Uses the aggressive (2-bit data) correction table only when a PI is
        already established, so noise can't false-sync via inflated error space.
        """
        syndrome = self._calc_syndrome(block)
        ecc_table = _ERROR_SYNDROMES_AGGRESSIVE if self._pi_code is not None else _ERROR_SYNDROMES

        offsets = {
            0: (RDS_SYNDROMES[0],),
            1: (RDS_SYNDROMES[1],),
            2: (RDS_SYNDROMES[2], RDS_SYNDROMES[3]),
            3: (RDS_SYNDROMES[4],),
        }

        for offset in offsets.get(block_num, ()):
            error_syndrome = syndrome ^ offset
            if error_syndrome == 0:
                # No errors
                return (block >> 10) & 0xFFFF, True
            if error_syndrome in ecc_table:
                error_mask = ecc_table[error_syndrome]
                corrected = block ^ error_mask
                return (corrected >> 10) & 0xFFFF, True

        return (block >> 10) & 0xFFFF, False

    def _decode_group(self, blocks: list[int], valid: list[bool]) -> None:
        """Decode RDS group. `valid[i]` gates reads of blocks[i] to avoid stale data."""
        # Without a fresh B we can't know group_type/version; everything below keys
        # off block B, so skip the group entirely rather than dispatch on stale bits.
        if not valid[1]:
            return

        a, b, c, d = blocks
        if valid[0]:
            if self._pi_code is None:
                self._pi_code = a
            elif self._pi_code != a:
                # During acquisition a changing PI almost always means noise
                # lock: drop and re-search. After we're established, require
                # the new PI to repeat before accepting a station change;
                # a single wrong A-block is usually a bit error.
                if self._groups_received < 4:
                    self._sync_drop()
                    self._clear_decoded_data()
                    return
                if self._pi_candidate == a:
                    self._clear_decoded_data()
                    self._pi_code = a
                else:
                    self._pi_candidate = a
            else:
                self._pi_candidate = None
        pty_vote = (b >> 5) & 0x1F
        self._pty_votes[pty_vote] = self._pty_votes.get(pty_vote, 0) + 1
        self._pty = max(self._pty_votes.items(), key=lambda kv: kv[1])[0]

        group_type = (b >> 12) & 0x0F
        version = (b >> 11) & 0x01

        if group_type == 0 and valid[0] and valid[3]:
            self._decode_ps(b, d)
        elif group_type == 2:
            if version == 0 and (valid[2] or valid[3]):
                self._decode_rt(b, c, d, version, valid[2], valid[3])
            elif version == 1 and valid[3]:
                self._decode_rt(b, c, d, version, False, True)
        elif group_type == 3 and version == 0 and valid[3]:
            oda_gt = (b >> 1) & 0x0F
            aid = d
            self._oda_map[oda_gt] = aid

        self._groups_received += 1

        summary = self._summarize_group(group_type, version, a, b, c, d)
        self._recent_groups.append(
            RDSGroup(
                group_type=group_type,
                version=version,
                pi_code=a,
                pty=self._pty,
                summary=summary,
            )
        )

        oda_aid = self._oda_map.get(group_type)
        if oda_aid == 0xCD46 and valid[2] and valid[3]:
            ver_str = "A" if version == 0 else "B"
            label = _ODA_NAMES.get(oda_aid, "TMC")
            prefix = f"{group_type}{ver_str} {label}"
            _raw, derived = self._summarize_tmc(prefix, b, c, d)
            logger.info("TMC [%04X %04X %04X %04X] %s", a, b, c, d, _raw)
            self._recent_groups.append(
                RDSGroup(
                    group_type=group_type,
                    version=version,
                    pi_code=a,
                    pty=self._pty,
                    summary=derived,
                )
            )

    def _summarize_group(self, gt: int, ver: int, a: int, b: int, c: int, d: int) -> str:
        """Create a one-line summary for a decoded group."""
        ver_str = "A" if ver == 0 else "B"
        label = _GROUP_TYPE_NAMES.get(gt, "?")

        # Resolve ODA label from 3A mapping
        aid = self._oda_map.get(gt)
        if aid is not None and label == "ODA":
            label = _ODA_NAMES.get(aid, f"ODA:{aid:04X}")
        prefix = f"{gt}{ver_str} {label}"

        if gt == 0:
            return self._summarize_0(prefix, ver, b, c, d)
        if gt == 1:
            return self._summarize_1(prefix, ver, b, c, d)
        if gt == 2:
            return self._summarize_2(prefix, ver, b, c, d)
        if gt == 3 and ver == 0:
            return self._summarize_3a(prefix, b, c, d)
        if gt == 4 and ver == 0:
            return self._summarize_4a(prefix, b, c, d)
        if gt == 10 and ver == 0:
            return self._summarize_10a(prefix, b, c, d)
        if gt == 14:
            return self._summarize_14(prefix, ver, b, c, d)
        if gt == 15 and ver == 1:
            ta = (b >> 4) & 0x01
            tp = (b >> 10) & 0x01
            return f"{prefix} TA={ta} TP={tp}"

        # ODA groups: decode based on AID from 3A
        if aid == 0xCD46:
            raw, _derived = self._summarize_tmc(prefix, b, c, d)
            return raw
        if aid == 0x4BD7:
            return self._summarize_rtplus(prefix, b, c, d)

        # Generic: show raw blocks
        return f"{prefix} [{a:04X} {b:04X} {c:04X} {d:04X}]"

    def _summarize_0(self, prefix: str, ver: int, b: int, c: int, d: int) -> str:
        """0A/0B: Basic tuning, PS name, AF list."""
        pos = b & 0x03
        ta = (b >> 4) & 0x01
        ms = (b >> 3) & 0x01
        c1, c2 = (d >> 8) & 0xFF, d & 0xFF
        ch1 = chr(c1) if 32 <= c1 <= 126 else "."
        ch2 = chr(c2) if 32 <= c2 <= 126 else "."
        flags = "T" if ta else ""
        flags += "M" if ms else "S"
        return f'{prefix} "{ch1}{ch2}"@{pos * 2} {flags}'

    def _summarize_1(self, prefix: str, ver: int, b: int, c: int, d: int) -> str:
        """1A/1B: Program Item Number and slow labeling codes."""
        pin = d
        day = (pin >> 11) & 0x1F
        hour = (pin >> 6) & 0x1F
        minute = pin & 0x3F
        pin_str = f"PIN={day:02d}d {hour:02d}:{minute:02d}"

        if ver == 0:
            # 1A: block C has slow labeling data, variant in bits 14-12 of B
            variant = b & 0x07
            if variant == 0:
                ecc = c & 0xFF
                return f"{prefix} {pin_str} ECC={ecc:02X}"
            if variant == 3:
                lang = c & 0xFF
                return f"{prefix} {pin_str} Lang={lang}"
            return f"{prefix} {pin_str} var={variant} data={c:04X}"

        return f"{prefix} {pin_str}"

    def _summarize_2(self, prefix: str, ver: int, b: int, c: int, d: int) -> str:
        """2A/2B: Radio Text."""
        pos = b & 0x0F
        ab = (b >> 4) & 0x01
        if ver == 0:
            idx = pos * 4
            chars = [(c >> 8) & 0xFF, c & 0xFF, (d >> 8) & 0xFF, d & 0xFF]
        else:
            idx = pos * 2
            chars = [(d >> 8) & 0xFF, d & 0xFF]
        txt = "".join(chr(ch) if 32 <= ch <= 126 else "." for ch in chars)
        return f'{prefix} RT[{idx}]="{txt}" A/B={ab}'

    def _summarize_3a(self, prefix: str, b: int, c: int, d: int) -> str:
        """3A: ODA Application Identification."""
        oda_gt = (b >> 1) & 0x0F
        oda_ver = "A" if (b & 0x01) == 0 else "B"
        aid = d
        app_name = _ODA_NAMES.get(aid, f"{aid:04X}")
        return f"{prefix} {oda_gt}{oda_ver}={app_name} msg={c:04X}"

    def _summarize_4a(self, prefix: str, b: int, c: int, d: int) -> str:
        """4A: Clock-time and date."""
        mjd = ((b & 0x03) << 15) | (c >> 1)
        hour = ((c & 0x01) << 4) | (d >> 12)
        minute = (d >> 6) & 0x3F
        offset_val = d & 0x1F
        offset_sign = "-" if (d >> 5) & 0x01 else "+"
        # Convert MJD to calendar date
        y_ = int((mjd - 15078.2) / 365.25)
        m_ = int((mjd - 14956.1 - int(y_ * 365.25)) / 30.6001)
        day = mjd - 14956 - int(y_ * 365.25) - int(m_ * 30.6001)
        k = 1 if (m_ == 14 or m_ == 15) else 0
        year = y_ + k + 1900
        month = m_ - 1 - k * 12
        return (
            f"{prefix} {year}-{month:02d}-{day:02d}"
            f" {hour:02d}:{minute:02d} UTC{offset_sign}{offset_val // 2}h{(offset_val % 2) * 30:02d}"
        )

    def _summarize_10a(self, prefix: str, b: int, c: int, d: int) -> str:
        """10A: Program Type Name."""
        pos = b & 0x01
        c1, c2 = (c >> 8) & 0xFF, c & 0xFF
        c3, c4 = (d >> 8) & 0xFF, d & 0xFF
        txt = "".join(chr(ch) if 32 <= ch <= 126 else "." for ch in [c1, c2, c3, c4])
        return f'{prefix} PTYN[{pos * 4}]="{txt}"'

    def _summarize_14(self, prefix: str, ver: int, b: int, c: int, d: int) -> str:
        """14A/14B: Enhanced Other Networks."""
        tp_on = (b >> 4) & 0x01
        variant = b & 0x0F
        if ver == 0:
            other_pi = d
            # Variant determines what block C carries
            if variant <= 3:
                # PS name chars for other network
                idx = variant * 2
                c1, c2 = (c >> 8) & 0xFF, c & 0xFF
                ch1 = chr(c1) if 32 <= c1 <= 126 else "."
                ch2 = chr(c2) if 32 <= c2 <= 126 else "."
                return f'{prefix} PI={other_pi:04X} PS[{idx}]="{ch1}{ch2}" TP={tp_on}'
            if variant == 4:
                # AF for other network
                af1 = (c >> 8) & 0xFF
                af2 = c & 0xFF
                freqs = []
                for af in [af1, af2]:
                    if 1 <= af <= 204:
                        freqs.append(f"{87.5 + af * 0.1:.1f}")
                af_str = ",".join(freqs) if freqs else f"{c:04X}"
                return f"{prefix} PI={other_pi:04X} AF={af_str} TP={tp_on}"
            if variant == 5:
                return f"{prefix} PI={other_pi:04X} TA-freq={c:04X} TP={tp_on}"
            if variant == 8:
                return f"{prefix} PI={other_pi:04X} PTY={c & 0x1F} TP={tp_on}"
            if variant == 9:
                return f"{prefix} PI={other_pi:04X} TA-time={c:04X} TP={tp_on}"
            if variant in (12, 13, 14):
                return f"{prefix} PI={other_pi:04X} linkage={c:04X} TP={tp_on}"
            return f"{prefix} PI={other_pi:04X} v{variant}={c:04X} TP={tp_on}"
        # 14B
        ta_on = (b >> 3) & 0x01
        return f"{prefix} PI={d:04X} TA={ta_on} TP={tp_on}"

    def _summarize_tmc(self, prefix: str, b: int, c: int, d: int) -> tuple[str, str]:
        """Decode TMC. Returns (raw summary, derived human-readable summary)."""
        x4 = (b >> 4) & 0x01  # T bit: 0=single, 1=multi-group
        x3 = (b >> 3) & 0x01  # F bit: 1=follow diversion
        dp = b & 0x07  # duration & persistence

        event_code = d & 0x7FF
        location = c
        extent = (d >> 11) & 0x07
        direction = (d >> 14) & 0x01

        desc = _tmc_event_text(event_code)
        dur = _TMC_DURATION[dp]
        divert = " div" if x3 else ""
        dir_str = "-" if direction else "+"
        multi = " multi" if x4 else ""

        raw = f"{prefix} {desc} e{event_code} L{location} x{extent}{dir_str} {dur}{divert}{multi}"

        # Derived: human-friendly summary
        parts = [desc]
        if extent > 0:
            parts.append(f"{extent}seg {dir_str}")
        parts.append(dur)
        if x3:
            parts.append("DIVERT")
        if x4:
            parts.append("(cont)")
        derived = f"TMC» {' | '.join(parts)}"

        return raw, derived

    def _summarize_rtplus(self, prefix: str, b: int, c: int, d: int) -> str:
        """Decode RT+ (Radio Text Plus) ODA tags."""
        # RT+ carries content type tags that annotate the Radio Text
        toggle = (b >> 4) & 0x01
        running = (b >> 3) & 0x01

        # Tag 1
        typ1 = ((b & 0x07) << 3) | (c >> 13)
        start1 = (c >> 7) & 0x3F
        len1 = (c >> 1) & 0x3F

        # Tag 2
        typ2 = ((c & 0x01) << 5) | (d >> 11)
        start2 = (d >> 5) & 0x3F
        len2 = d & 0x1F

        tag1 = f"t{typ1}@{start1}+{len1}"
        tag2 = f"t{typ2}@{start2}+{len2}"
        run_str = " run" if running else ""
        return f"{prefix} [{tag1}] [{tag2}] A/B={toggle}{run_str}"

    def _decode_ps(self, b: int, d: int) -> None:
        """Decode Program Service name with majority voting."""
        pos = b & 0x03
        idx = pos * 2
        c1, c2 = (d >> 8) & 0xFF, d & 0xFF

        for i, ch in [(idx, c1), (idx + 1, c2)]:
            if 32 <= ch <= 126:
                votes = self._ps_votes[i]
                c = chr(ch)
                votes[c] = votes.get(c, 0) + 1
                self._ps_chars[i] = max(votes.items(), key=lambda kv: kv[1])[0]

    def _decode_rt(
        self, b: int, c: int, d: int, version: int, c_valid: bool, d_valid: bool
    ) -> None:
        """Decode Radio Text with A/B toggle reset and per-position voting.

        For 2A (version 0), chars 0-1 come from block C, chars 2-3 from D; the
        corresponding validity flag gates writing so a partial group still
        contributes the blocks that did pass CRC.
        """
        ab = (b >> 4) & 0x01
        if self._rt_ab is not None and ab != self._rt_ab:
            for k in range(64):
                self._rt_chars[k] = ""
                self._rt_votes[k].clear()
        self._rt_ab = ab

        pos = b & 0x0F
        if version == 0:
            idx = pos * 4
            chars = [(c >> 8) & 0xFF, c & 0xFF, (d >> 8) & 0xFF, d & 0xFF]
            valid_mask = [c_valid, c_valid, d_valid, d_valid]
        else:
            idx = pos * 2
            chars = [(d >> 8) & 0xFF, d & 0xFF]
            valid_mask = [d_valid, d_valid]

        for i, ch in enumerate(chars):
            if not valid_mask[i]:
                continue
            k = idx + i
            if k >= 64:
                break
            if ch == 0x0D:
                for m in range(k, 64):
                    self._rt_chars[m] = ""
                    self._rt_votes[m].clear()
                break
            if 32 <= ch <= 126:
                votes = self._rt_votes[k]
                cs = chr(ch)
                votes[cs] = votes.get(cs, 0) + 1
                self._rt_chars[k] = max(votes.items(), key=lambda kv: kv[1])[0]

    def _snapshot(self) -> RDSData:
        """Create immutable snapshot."""
        ber = self._uncorrectable / max(self._blocks_total, 1)
        pty_name = PTY_NAMES[self._pty] if 0 <= self._pty < len(PTY_NAMES) else "Unknown"

        if self._synced:
            sync_conf = min(0.5 + self._groups_received / 20.0, 1.0)
        elif self._presync:
            sync_conf = 0.25
        else:
            sync_conf = 0.0

        # Costas loop frequency -> Hz offset from 57 kHz subcarrier
        # freq is rad/symbol; convert via symbol rate to Hz
        baseband_offset = self._costas.freq * self._decimated_rate / (2 * np.pi * self._sps)

        # Flush recent groups into snapshot
        groups = tuple(self._recent_groups)
        self._recent_groups.clear()

        return RDSData(
            pi_code=self._pi_code,
            ps_name="".join(self._ps_chars).strip(),
            radio_text="".join(self._rt_chars).strip(),
            pty=self._pty,
            pty_name=pty_name,
            sync_locked=self._synced,
            groups_received=self._groups_received,
            block_error_rate=ber,
            uncorrectable_blocks=self._uncorrectable,
            sync_confidence=sync_conf,
            baseband_offset_hz=baseband_offset,
            recent_groups=groups,
        )

    def reset(self) -> None:
        """Reset decoder state."""
        self._carrier_phase = 0.0
        self._antialias.reset()
        self._sharp.reset()
        self._decim_phase = 0
        self._agc.reset()
        self._sync_watchdog_bits = 0
        self._sync_nudge_count = 0
        self._mm.reset()
        self._costas.reset()
        self._prev_bit = 0

        self._shift_reg = 0
        self._synced = False
        self._presync = False
        self._lastseen_offset = 0
        self._lastseen_offset_counter = 0
        self._bit_counter = 0
        self._block_bit_counter = 0
        self._block_number = 0
        self._group_blocks = [0, 0, 0, 0]
        self._block_valid = [False, False, False, False]
        self._wrong_blocks_counter = 0
        self._blocks_counter = 0
        self._consecutive_wrong = 0

        self._clear_decoded_data()

    def _clear_decoded_data(self) -> None:
        """Clear all decoded station data and statistics.

        Called on sync loss so stale data from a previous station
        doesn't persist when a new station is acquired.
        """
        self._pi_code = None
        self._pi_candidate = None
        self._pty_votes = {}
        self._ps_chars = [""] * 8
        self._ps_votes = [{} for _ in range(8)]
        self._rt_chars = [""] * 64
        self._rt_votes = [{} for _ in range(64)]
        self._rt_ab = None
        self._pty = 0
        self._oda_map = {}
        self._groups_received = 0
        self._blocks_total = 0
        self._uncorrectable = 0


# RDS group / ODA / TMC lookup tables.

# Group type base names (before ODA resolution)
_GROUP_TYPE_NAMES: dict[int, str] = {
    0: "PS",
    1: "PIN/SlowLabel",
    2: "RadioText",
    3: "AppID",
    4: "Clock",
    5: "TDC",
    6: "IH",
    7: "RadioPaging",
    8: "ODA",
    9: "EWS",
    10: "PTYN",
    11: "ODA",
    12: "ODA",
    13: "ODA",
    14: "EON",
    15: "FastSwitch",
}

# Known ODA Application IDs
_ODA_NAMES: dict[int, str] = {
    0xCD46: "TMC",
    0x4BD7: "RT+",
    0x0093: "DAB cross-ref",
    0x6365: "TMC-Alert-C",
    0x4AA1: "ITTS",
    0x6552: "eRT",
}

# TMC duration/persistence labels (3-bit dp field)
_TMC_DURATION = [
    "none",
    "<15m",
    "15-30m",
    "30m-1h",
    "1-2h",
    "2-4h",
    ">4h",
    "long",
]

# Alert-C event code descriptions (EN ISO 14819-2 public subset)
# Codes not in this table fall back to category ranges below.
_TMC_EVENTS: dict[int, str] = {
    # Cancel
    0: "cancel",
    # Traffic (1-99)
    1: "jam",
    2: "queuing traffic",
    3: "slow traffic",
    4: "heavy traffic",
    5: "traffic flowing freely",
    6: "traffic building up",
    7: "traffic heavier than normal",
    8: "traffic lighter than normal",
    9: "long queues",
    10: "stop-and-go traffic",
    11: "stationary traffic",
    12: "stationary traffic 1-2 km",
    13: "stationary traffic 2-4 km",
    14: "stationary traffic 4-10 km",
    15: "stationary traffic >10 km",
    22: "queuing traffic 1-2 km",
    23: "queuing traffic 2-4 km",
    24: "queuing traffic 4-10 km",
    25: "queuing traffic >10 km",
    40: "danger of stationary traffic",
    41: "danger of queuing traffic",
    51: "delay",
    52: "delay 15m",
    53: "delay 30m",
    54: "delay 1h",
    55: "delay 2h",
    56: "delay >2h",
    61: "multi-vehicle accident",
    62: "accident",
    63: "accident heavy traffic",
    64: "accident slow traffic",
    65: "accident queuing traffic",
    66: "accident stationary traffic",
    67: "accident cleared",
    # Forecast (100-199)
    101: "forecast: jam",
    102: "forecast: heavy traffic",
    103: "forecast: slow traffic",
    106: "forecast: normal traffic",
    # Closures (200-399)
    201: "closed",
    202: "blocked",
    203: "closed ahead",
    206: "entry slip road closed",
    207: "exit slip road closed",
    208: "no through traffic",
    211: "carriageway reduced 2->1",
    212: "carriageway reduced 3->2",
    213: "carriageway reduced 3->1",
    218: "contraflow",
    231: "lane closed",
    232: "hard shoulder closed",
    235: "overtaking lane closed",
    237: "road cleared",
    244: "reopened",
    245: "message cancelled",
    250: "closed for cars",
    251: "closed for trucks",
    252: "closed for all vehicles",
    270: "bridge closed",
    271: "tunnel closed",
    # Restrictions (400-499)
    401: "speed limit 20",
    402: "speed limit 30",
    403: "speed limit 40",
    404: "speed limit 50",
    405: "speed limit 60",
    406: "speed limit 70",
    407: "speed limit 80",
    408: "speed limit 90",
    409: "speed limit 100",
    410: "speed limit 110",
    411: "speed limit 120",
    430: "no overtaking",
    436: "width limit",
    437: "height limit",
    438: "weight limit",
    445: "restrictions lifted",
    # Roadworks (500-599)
    501: "roadworks",
    502: "major roadworks",
    503: "roadworks cleared",
    514: "resurfacing",
    515: "lane markings work",
    516: "bridge maintenance",
    534: "slow traffic due to roadworks",
    # Obstructions (600-699)
    601: "obstruction on road",
    603: "shed load",
    605: "broken down vehicle",
    606: "broken down truck",
    608: "animals on road",
    609: "children on road",
    611: "cyclists on road",
    612: "pedestrians on road",
    615: "object on road",
    631: "fallen trees",
    633: "flooding",
    634: "landslide",
    637: "fallen power cables",
    638: "fire",
    641: "bomb alert",
    643: "avalanche risk",
    # Conditions (700-799)
    701: "slippery road",
    702: "icy",
    703: "snow on road",
    704: "packed snow",
    706: "oil on road",
    707: "loose gravel",
    711: "road surface in poor condition",
    713: "subsidence",
    735: "conditions improved",
    # Weather (800-899)
    801: "fog",
    802: "dense fog",
    803: "fog patches",
    804: "freezing fog",
    805: "smoke haze",
    806: "blowing dust",
    811: "rain",
    812: "heavy rain",
    813: "thunderstorm",
    814: "hail",
    815: "drizzle",
    816: "freezing rain",
    821: "snow",
    822: "heavy snow",
    823: "blizzard",
    824: "sleet",
    831: "strong wind",
    832: "gale",
    833: "storm",
    834: "hurricane",
    835: "gusty winds",
    836: "crosswind",
    841: "temperature drop",
    842: "extreme heat",
    843: "extreme cold",
    851: "visibility reduced",
    852: "visibility <100m",
    853: "visibility <50m",
    855: "white-out",
    861: "weather improved",
    # Warning (900-999)
    901: "danger",
    902: "warning lifted",
    903: "risk of aquaplaning",
    905: "risk of ice",
    906: "danger of fire",
    907: "toxic waste alert",
    908: "radiation hazard",
    # Service (1000-1199)
    1001: "fuel station",
    1004: "rest area",
    1012: "police checkpoint",
    1013: "border crossing",
    1015: "toll booth",
    1101: "information service",
    1102: "radio coverage",
    # Parking (1200-1299)
    1201: "car park full",
    1202: "car park almost full",
    1203: "car park spaces available",
    1204: "car park multi-storey full",
    1206: "park-and-ride available",
    # Transport (1300-1399)
    1301: "ferry service suspended",
    1302: "ferry delayed",
    1303: "rail service suspended",
    1304: "rail service delayed",
    1320: "bus service suspended",
    1321: "bus service delayed",
    # Activities/events (1400-1499)
    1401: "sports event",
    1402: "fair",
    1403: "show",
    1404: "festival",
    1405: "exhibition",
    1406: "marathon",
    1410: "demonstration",
    1411: "strike",
    # Special (1500-1599)
    1501: "security alert",
    1502: "terrorist incident",
    1503: "gunfire on road",
    1510: "civil emergency",
    1511: "air raid warning",
    # Info (1600+)
    1601: "traffic information service",
    1602: "travel weather",
    1603: "road condition report",
    1604: "travel time info",
    1610: "level crossing",
    1611: "drawbridge",
    1620: "traffic signal failure",
    1621: "traffic signal working again",
    1625: "no public transport",
    1631: "traffic congestion report",
}

# Fallback: event code ranges -> broad category
_TMC_EVENT_RANGES: list[tuple[int, int, str]] = [
    (1, 99, "traffic"),
    (100, 199, "forecast"),
    (200, 399, "closure"),
    (400, 499, "restriction"),
    (500, 599, "roadworks"),
    (600, 699, "obstruction"),
    (700, 799, "conditions"),
    (800, 899, "weather"),
    (900, 999, "warning"),
    (1000, 1199, "service"),
    (1200, 1299, "parking"),
    (1300, 1399, "transport"),
    (1400, 1499, "activity"),
    (1500, 1599, "special"),
    (1600, 2047, "info"),
]


def _tmc_event_text(event_code: int) -> str:
    """Look up event code description, falling back to category name."""
    if event_code in _TMC_EVENTS:
        return _TMC_EVENTS[event_code]
    for lo, hi, cat in _TMC_EVENT_RANGES:
        if lo <= event_code <= hi:
            return cat
    return "unknown"
