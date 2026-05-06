"""ADS-B 1090ES decoder (Mode S Extended Squitter).

Decodes ADS-B messages from IQ samples using amplitude-based PPM demodulation.
No FM discriminator or symbol timing recovery needed - works directly on
signal magnitude with correlation-based bit extraction.

IQ -> magnitude -> preamble detection -> correlation slicers -> CRC-24 -> message parsing
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import numba as nb
import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import SignalInfo
from tsdr.radio.demodulators import Demodulator

logger = logging.getLogger(__name__)

# CRC-24 lookup table (Mode S generator polynomial 0xFFF409)

_CRC_POLY = 0xFFF409

_CRC_TABLE_LIST = [0] * 256
for _i in range(256):
    _c = _i << 16
    for _j in range(8):
        if _c & 0x800000:
            _c = (_c << 1) ^ _CRC_POLY
        else:
            _c = _c << 1
    _CRC_TABLE_LIST[_i] = _c & 0x00FFFFFF

_CRC_TABLE = np.array(_CRC_TABLE_LIST, dtype=np.int64)


def crc24(msg: bytes | bytearray) -> int:
    """Compute Mode S CRC-24 remainder."""
    rem = 0
    n = len(msg)
    for i in range(n - 3):
        rem = ((rem << 8) ^ _CRC_TABLE_LIST[msg[i] ^ ((rem >> 16) & 0xFF)]) & 0xFFFFFF
    rem ^= (msg[n - 3] << 16) | (msg[n - 2] << 8) | msg[n - 1]
    return rem


# Phase table for bit extraction (numba-friendly flat arrays).
# Per-phase: 8 (slice_phase, ptr_offset) pairs, then next_phase, advance.
# Shape: (5, 8, 2) - [phase][bit][0=slice_phase, 1=offset]
_PHASE_BITS = np.array(
    [
        # phase 0
        [[0, 0], [2, 2], [4, 4], [1, 7], [3, 9], [0, 12], [2, 14], [4, 16]],
        # phase 1
        [[1, 0], [3, 2], [0, 5], [2, 7], [4, 9], [1, 12], [3, 14], [0, 17]],
        # phase 2
        [[2, 0], [4, 2], [1, 5], [3, 7], [0, 10], [2, 12], [4, 14], [1, 17]],
        # phase 3
        [[3, 0], [0, 3], [2, 5], [4, 7], [1, 10], [3, 12], [0, 15], [2, 17]],
        # phase 4
        [[4, 0], [1, 3], [3, 5], [0, 8], [2, 10], [4, 12], [1, 15], [3, 17]],
    ],
    dtype=np.int32,
)

_PHASE_NEXT = np.array([1, 2, 3, 4, 0], dtype=np.int32)
_PHASE_ADVANCE = np.array([19, 19, 19, 19, 20], dtype=np.int32)

# Valid DF bitsets (bit N set = DF N is valid)
_VALID_DF_SHORT_BITS = (1 << 0) | (1 << 4) | (1 << 5) | (1 << 11)
_VALID_DF_LONG_BITS = (1 << 16) | (1 << 17) | (1 << 18) | (1 << 20) | (1 << 21)


# Numba-jitted hot path: preamble validation + extraction + CRC + scoring


@nb.njit(cache=True)
def _slice(m, p, sp):
    """Compute correlation slice value for given slice phase."""
    if sp == 0:
        return 5 * m[p] - 3 * m[p + 1] - 2 * m[p + 2]
    if sp == 1:
        return 4 * m[p] - m[p + 1] - 3 * m[p + 2]
    if sp == 2:
        return 3 * m[p] + m[p + 1] - 4 * m[p + 2]
    if sp == 3:
        return 2 * m[p] + 3 * m[p + 1] - 5 * m[p + 2]
    # sp == 4
    return m[p] + 5 * m[p + 1] - 5 * m[p + 2] - m[p + 3]


@nb.njit(cache=True)
def _extract_msg_jit(mag, data_start, try_phase, n_bytes, phase_bits, phase_next, phase_advance):
    """Extract n_bytes from magnitude using correlation slicers."""
    ptr = data_start + try_phase // 5
    phase = try_phase % 5
    result = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bytes):
        byte_val = 0
        for bit_idx in range(8):
            sp = phase_bits[phase, bit_idx, 0]
            off = phase_bits[phase, bit_idx, 1]
            if _slice(mag, ptr + off, sp) > 0:
                byte_val |= 0x80 >> bit_idx
        result[i] = byte_val
        ptr += phase_advance[phase]
        phase = phase_next[phase]
    return result


@nb.njit(cache=True)
def _crc24_jit(msg, n, crc_table):
    """Compute CRC-24 remainder on a uint8 array of length n."""
    rem = np.int64(0)
    for i in range(n - 3):
        rem = ((rem << 8) ^ crc_table[msg[i] ^ ((rem >> 16) & 0xFF)]) & 0xFFFFFF
    rem ^= (np.int64(msg[n - 3]) << 16) | (np.int64(msg[n - 2]) << 8) | np.int64(msg[n - 1])
    return rem


@nb.njit(cache=True)
def _detect_and_decode_jit(
    mag,
    candidates,
    crc_table,
    phase_bits,
    phase_next,
    phase_advance,
    known_icaos,
    n_known_in,
    pending_icaos,
    n_pending_in,
    valid_short,
    valid_long,
):
    """Process preamble candidates: validate, extract bits, CRC, score.

    Returns (n_messages, offsets[MAX], messages[MAX, 14], lens[MAX],
            n_known_out, n_pending_out).
    """
    max_messages = 8000
    msg_offsets = np.zeros(max_messages, dtype=np.int64)
    msg_data = np.zeros((max_messages, 14), dtype=np.uint8)
    msg_lens = np.zeros(max_messages, dtype=np.int32)
    n_messages = 0

    skip_until = np.int64(0)
    n_known = n_known_in
    n_pending = n_pending_in

    for ci in range(len(candidates)):
        j = candidates[ci]
        if j < skip_until:
            continue

        p1 = mag[j + 1]
        p2 = mag[j + 2]
        p3 = mag[j + 3]
        p4 = mag[j + 4]
        p5 = mag[j + 5]
        p6 = mag[j + 6]
        p7 = mag[j + 7]
        p8 = mag[j + 8]
        p9 = mag[j + 9]
        p10 = mag[j + 10]
        p11 = mag[j + 11]
        p12 = mag[j + 12]

        base_signal = 0
        base_noise = 0
        high = 0
        matched = False

        if p1 > p2 and p2 < p3 and p3 > p4 and p8 < p9 and p9 > p10 and p10 < p11:
            high = (p1 + p3 + p9 + p11 + p12) // 4
            base_signal = p1 + p3 + p9
            base_noise = p5 + p6 + p7
            matched = True
        elif p1 > p2 and p2 < p3 and p3 > p4 and p8 < p9 and p9 > p10 and p11 < p12:
            high = (p1 + p3 + p9 + p12) // 4
            base_signal = p1 + p3 + p9 + p12
            base_noise = p5 + p6 + p7 + p8
            matched = True
        elif p1 > p2 and p2 < p3 and p4 > p5 and p8 < p9 and p10 > p11 and p11 < p12:
            high = (p1 + p3 + p4 + p9 + p10 + p12) // 4
            base_signal = p1 + p12
            base_noise = p6 + p7
            matched = True
        elif p1 > p2 and p3 < p4 and p4 > p5 and p9 < p10 and p10 > p11 and p11 < p12:
            high = (p1 + p4 + p10 + p12) // 4
            base_signal = p1 + p4 + p10 + p12
            base_noise = p5 + p6 + p7 + p8
            matched = True
        elif p2 > p3 and p3 < p4 and p4 > p5 and p9 < p10 and p10 > p11 and p11 < p12:
            high = (p1 + p2 + p4 + p10 + p12) // 4
            base_signal = p4 + p10 + p12
            base_noise = p6 + p7 + p8
            matched = True

        if not matched:
            continue

        if base_signal * 2 < 3 * base_noise:
            continue

        # Quiet-zone check
        if (
            p5 >= high
            or p6 >= high
            or p7 >= high
            or p8 >= high
            or mag[j + 14] >= high
            or mag[j + 15] >= high
            or mag[j + 16] >= high
            or mag[j + 17] >= high
            or mag[j + 18] >= high
        ):
            continue

        best_score = -1
        best_phase = -1

        for try_phase in range(4, 9):
            first = _extract_msg_jit(
                mag, j + 19, try_phase, 1, phase_bits, phase_next, phase_advance
            )
            df = first[0] >> 3

            if df < 32 and (valid_long >> df) & 1:
                n_bytes = 14
            elif df < 32 and (valid_short >> df) & 1:
                n_bytes = 7
            else:
                continue

            msg = _extract_msg_jit(
                mag, j + 19, try_phase, n_bytes, phase_bits, phase_next, phase_advance
            )
            remainder = _crc24_jit(msg, n_bytes, crc_table)

            score = 0
            if df == 17 or df == 18:
                if remainder == 0:
                    icao = (np.int64(msg[1]) << 16) | (np.int64(msg[2]) << 8) | np.int64(msg[3])
                    found = False
                    for ki in range(n_known):
                        if known_icaos[ki] == icao:
                            found = True
                            break
                    score = 1000 if found else 500
                else:
                    continue
            elif df == 11:
                # DF11 All-Call: ICAO is at bytes 1-3 (explicit, like DF17).
                # CRC remainder = Interrogator ID (IID), not ICAO.
                # Valid IID is small (0 for spontaneous, ≤15 for standard).
                if remainder > 80:
                    continue  # CRC error (not a valid IID)
                icao = (np.int64(msg[1]) << 16) | (np.int64(msg[2]) << 8) | np.int64(msg[3])
                found = False
                for ki in range(n_known):
                    if known_icaos[ki] == icao:
                        found = True
                        break
                if found:
                    score = 400
                else:
                    # 2-hit confirmation for unknown ICAOs
                    in_pending = False
                    for pi in range(n_pending):
                        if pending_icaos[pi] == icao:
                            in_pending = True
                            break
                    if in_pending:
                        if n_known < len(known_icaos):
                            known_icaos[n_known] = icao
                            n_known += 1
                        score = 200
                    else:
                        if n_pending < len(pending_icaos):
                            pending_icaos[n_pending] = icao
                            n_pending += 1
                        continue
            else:
                # Other DFs: CRC remainder = ICAO address.
                # Only accept if it matches a known ICAO.
                icao = remainder
                found = False
                for ki in range(n_known):
                    if known_icaos[ki] == icao:
                        found = True
                        break
                if not found:
                    continue
                score = 300

            if score > best_score:
                best_score = score
                best_phase = try_phase

        if best_score >= 50 and best_phase >= 0 and n_messages < max_messages:
            # Re-extract with best phase
            first = _extract_msg_jit(
                mag, j + 19, best_phase, 1, phase_bits, phase_next, phase_advance
            )
            df = first[0] >> 3
            if df < 32 and (valid_long >> df) & 1:
                n_bytes = 14
            else:
                n_bytes = 7
            msg = _extract_msg_jit(
                mag, j + 19, best_phase, n_bytes, phase_bits, phase_next, phase_advance
            )

            msg_offsets[n_messages] = j
            for bi in range(n_bytes):
                msg_data[n_messages, bi] = msg[bi]
            msg_lens[n_messages] = n_bytes

            # Update known ICAOs
            df2 = msg[0] >> 3
            if df2 == 17 or df2 == 18 or df2 == 11:
                # ICAO explicitly at bytes 1-3
                new_icao = (np.int64(msg[1]) << 16) | (np.int64(msg[2]) << 8) | np.int64(msg[3])
            else:
                # CRC remainder = ICAO for other DFs (already known)
                new_icao = np.int64(-1)
            if new_icao >= 0:
                # Add to known_icaos if not already present
                found = False
                for ki in range(n_known):
                    if known_icaos[ki] == new_icao:
                        found = True
                        break
                if not found and n_known < len(known_icaos):
                    known_icaos[n_known] = new_icao
                    n_known += 1

            n_messages += 1

            # Skip past message
            msgbits = 112 if (df & 0x10) else 56
            skip_until = j + (msgbits + 8) * 12 // 5

    return n_messages, msg_offsets, msg_data, msg_lens, n_known, n_pending


# Python wrapper around JIT core


def _find_preamble_candidates(mag: np.ndarray) -> np.ndarray:
    """Vectorized quick edge check to find preamble candidate positions."""
    n = len(mag) - 290
    if n <= 0:
        return np.array([], dtype=np.int64)
    edge_rising = mag[:n] < mag[1 : n + 1]
    edge_falling = mag[12 : n + 12] > mag[13 : n + 13]
    return np.flatnonzero(edge_rising & edge_falling).astype(np.int64)


def _detect_and_decode(
    mag: np.ndarray,
    known_icaos: set[int],
    pending_icaos: set[int] | None = None,
) -> list[tuple[int, bytes]]:
    """Detect preambles and decode messages from magnitude array."""
    candidates = _find_preamble_candidates(mag)
    if len(candidates) == 0:
        return []

    if pending_icaos is None:
        pending_icaos = set()

    # Convert sets to fixed-size numpy arrays for numba
    max_icaos = max(len(known_icaos) + 500, 1000)
    icao_arr = np.full(max_icaos, -1, dtype=np.int64)
    for i, v in enumerate(known_icaos):
        icao_arr[i] = v
    n_known_in = len(known_icaos)

    max_pending = max(len(pending_icaos) + 2000, 4000)
    pending_arr = np.full(max_pending, -1, dtype=np.int64)
    for i, v in enumerate(pending_icaos):
        pending_arr[i] = v
    n_pending_in = len(pending_icaos)

    mag32 = mag.astype(np.int32)

    n_msgs, offsets, data, lens, n_known_out, n_pending_out = _detect_and_decode_jit(
        mag32,
        candidates,
        _CRC_TABLE,
        _PHASE_BITS,
        _PHASE_NEXT,
        _PHASE_ADVANCE,
        icao_arr,
        n_known_in,
        pending_arr,
        n_pending_in,
        _VALID_DF_SHORT_BITS,
        _VALID_DF_LONG_BITS,
    )

    # Update sets from the arrays
    for i in range(n_known_in, n_known_out):
        if icao_arr[i] >= 0:
            known_icaos.add(int(icao_arr[i]))
    for i in range(n_pending_in, n_pending_out):
        if pending_arr[i] >= 0:
            pending_icaos.add(int(pending_arr[i]))

    # Convert results to list of (offset, bytes)
    messages: list[tuple[int, bytes]] = []
    for i in range(n_msgs):
        msg_bytes = bytes(data[i, : lens[i]])
        messages.append((int(offsets[i]), msg_bytes))
    return messages


def _extract_bytes(mag: np.ndarray, start: int, try_phase: int, n_bytes: int) -> bytes:
    """Extract n_bytes from magnitude array (Python wrapper for tests)."""
    mag32 = mag.astype(np.int32) if mag.dtype != np.int32 else mag
    result = _extract_msg_jit(
        mag32, start, try_phase, n_bytes, _PHASE_BITS, _PHASE_NEXT, _PHASE_ADVANCE
    )
    return bytes(result)


# Magnitude computation


def magnitude_uc8(raw: np.ndarray) -> np.ndarray:
    """Convert raw UC8 IQ bytes to uint16 magnitude.

    Center 127.4, scale 65536/128.
    """
    pairs = raw.reshape(-1, 2).astype(np.float32)
    i_val = pairs[:, 0] - 127.4
    q_val = pairs[:, 1] - 127.4
    mag = np.sqrt(i_val * i_val + q_val * q_val) * (65536.0 / 128.0)
    result: np.ndarray = np.round(mag).clip(0, 65535).astype(np.uint16)
    return result


def magnitude_complex(iq: np.ndarray) -> np.ndarray:
    """Convert complex64 IQ to uint16 magnitude (streaming path)."""
    mag = np.abs(iq) * 65536.0
    result: np.ndarray = np.clip(mag, 0, 65535).astype(np.uint16)
    return result


# ADS-B message parsing (DF17)

_ADSB_CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"


def _parse_identification(me: bytes) -> str:
    """Parse TC 1-4: aircraft identification (callsign)."""
    val = int.from_bytes(me, "big")
    chars = []
    for i in range(8):
        idx = (val >> (42 - 6 * i)) & 0x3F
        ch = _ADSB_CHARSET[idx] if idx < len(_ADSB_CHARSET) else "?"
        chars.append(ch)
    return "".join(chars).strip()


def _parse_altitude(me: bytes) -> int | None:
    """Parse altitude from TC 9-18 position message.

    12-bit altitude field at ME bits 8-19. Q-bit at position 4 within the field.
    """
    alt_bits = ((me[1] & 0xFF) << 4) | ((me[2] >> 4) & 0x0F)
    if alt_bits == 0:
        return None
    q_bit = (alt_bits >> 4) & 1
    if q_bit:
        n = ((alt_bits >> 5) << 4) | (alt_bits & 0x0F)
        return 25 * n - 1000
    return None


def _parse_velocity(me: bytes) -> tuple[float | None, float | None, int | None]:
    """Parse TC 19: airborne velocity. Returns (speed_kt, heading_deg, vrate_fpm)."""
    st = (me[0] >> 1) & 0x07

    # Vertical rate
    vr_sign = (me[4] >> 3) & 1
    vr_val = ((me[4] & 0x07) << 6) | ((me[5] >> 2) & 0x3F)
    vrate = None
    if vr_val != 0:
        vrate = (1 if vr_sign == 0 else -1) * 64 * (vr_val - 1)

    if st in (1, 2):
        dew = (me[1] >> 2) & 1
        vew = ((me[1] & 0x03) << 8) | me[2]
        dns = (me[3] >> 7) & 1
        vns = ((me[3] & 0x7F) << 3) | ((me[4] >> 5) & 0x07)
        if vew == 0 or vns == 0:
            return None, None, vrate
        scale = 4 if st == 2 else 1
        vx = scale * (vew - 1) * (1 if dew == 0 else -1)
        vy = scale * (vns - 1) * (1 if dns == 0 else -1)
        speed = math.sqrt(vx * vx + vy * vy)
        heading = math.atan2(vx, vy) * 360.0 / (2 * math.pi)
        heading = heading % 360
        return round(speed, 1), round(heading, 2), vrate

    if st in (3, 4):
        sh = (me[1] >> 2) & 1
        if sh == 0:
            return None, None, vrate
        hdg_raw = ((me[1] & 0x03) << 8) | me[2]
        heading = hdg_raw * 360.0 / 1024.0
        as_val = ((me[3] & 0x7F) << 3) | ((me[4] >> 5) & 0x07)
        if as_val == 0:
            return None, round(heading, 2), vrate
        scale = 4 if st == 4 else 1
        speed = scale * (as_val - 1)
        return float(speed), round(heading, 2), vrate

    return None, None, vrate


def _parse_cpr_position(me: bytes) -> tuple[int, int, int]:
    """Parse TC 9-18: extract CPR lat, CPR lon, F-bit from ME bytes.

    Returns (cprlat, cprlon, fflag) where cprlat/cprlon are 17-bit encoded values.
    ME layout (56 bits = 7 bytes):
      bits 0-4: TC, 5-6: SS, 7: SAF, 8-19: altitude, 20: T, 21: F,
      bits 22-38: CPR latitude (17 bits), bits 39-55: CPR longitude (17 bits)
    """
    fflag = (me[2] >> 2) & 1
    cprlat = ((me[2] & 0x03) << 15) | (me[3] << 7) | (me[4] >> 1)
    cprlon = ((me[4] & 0x01) << 16) | (me[5] << 8) | me[6]
    return cprlat, cprlon, fflag


# CPR (Compact Position Reporting) decoder


def _cpr_nl(lat: float) -> int:
    """NL function: number of longitude zones at a given latitude."""
    lat = abs(lat)
    if lat < 10.47047130:
        return 59
    if lat < 14.82817437:
        return 58
    if lat < 18.18626357:
        return 57
    if lat < 21.02939493:
        return 56
    if lat < 23.54504487:
        return 55
    if lat < 25.82924707:
        return 54
    if lat < 27.93898710:
        return 53
    if lat < 29.91135686:
        return 52
    if lat < 31.77209708:
        return 51
    if lat < 33.53993436:
        return 50
    if lat < 35.22899598:
        return 49
    if lat < 36.85025108:
        return 48
    if lat < 38.41241892:
        return 47
    if lat < 39.92256684:
        return 46
    if lat < 41.38651832:
        return 45
    if lat < 42.80914012:
        return 44
    if lat < 44.19454951:
        return 43
    if lat < 45.54626723:
        return 42
    if lat < 46.86733252:
        return 41
    if lat < 48.16039128:
        return 40
    if lat < 49.42776439:
        return 39
    if lat < 50.67150166:
        return 38
    if lat < 51.89342469:
        return 37
    if lat < 53.09516153:
        return 36
    if lat < 54.27817472:
        return 35
    if lat < 55.44378444:
        return 34
    if lat < 56.59318756:
        return 33
    if lat < 57.72747354:
        return 32
    if lat < 58.84763776:
        return 31
    if lat < 59.95459277:
        return 30
    if lat < 61.04917774:
        return 29
    if lat < 62.13216659:
        return 28
    if lat < 63.20427479:
        return 27
    if lat < 64.26616523:
        return 26
    if lat < 65.31845310:
        return 25
    if lat < 66.36171008:
        return 24
    if lat < 67.39646774:
        return 23
    if lat < 68.42322022:
        return 22
    if lat < 69.44242631:
        return 21
    if lat < 70.45451075:
        return 20
    if lat < 71.45986473:
        return 19
    if lat < 72.45884545:
        return 18
    if lat < 73.45177442:
        return 17
    if lat < 74.43893416:
        return 16
    if lat < 75.42056257:
        return 15
    if lat < 76.39684391:
        return 14
    if lat < 77.36789461:
        return 13
    if lat < 78.33374083:
        return 12
    if lat < 79.29428225:
        return 11
    if lat < 80.24923213:
        return 10
    if lat < 81.19801349:
        return 9
    if lat < 82.13956981:
        return 8
    if lat < 83.07199445:
        return 7
    if lat < 83.99173563:
        return 6
    if lat < 84.89166191:
        return 5
    if lat < 85.75541621:
        return 4
    if lat < 86.53536998:
        return 3
    if lat < 87.00000000:
        return 2
    return 1


def _cpr_n(lat: float, fflag: int) -> int:
    nl = _cpr_nl(lat) - fflag
    return max(nl, 1)


def _cpr_dlon(lat: float, fflag: int) -> float:
    return 360.0 / _cpr_n(lat, fflag)


def decode_cpr_airborne(
    even_cprlat: int,
    even_cprlon: int,
    odd_cprlat: int,
    odd_cprlon: int,
    fflag: int,
) -> tuple[float, float] | None:
    """Global airborne CPR decode from even+odd frame pair.

    fflag: 0 = even message is latest, 1 = odd message is latest.
    Returns (lat, lon) or None on zone crossing / bad data.
    """
    air_dlat0 = 360.0 / 60.0
    air_dlat1 = 360.0 / 59.0

    j = math.floor(((59 * even_cprlat - 60 * odd_cprlat) / 131072) + 0.5)
    rlat0 = air_dlat0 * ((j % 60 + 60) % 60 + even_cprlat / 131072)
    rlat1 = air_dlat1 * ((j % 59 + 59) % 59 + odd_cprlat / 131072)

    if rlat0 >= 270:
        rlat0 -= 360
    if rlat1 >= 270:
        rlat1 -= 360

    if rlat0 < -90 or rlat0 > 90 or rlat1 < -90 or rlat1 > 90:
        return None

    if _cpr_nl(rlat0) != _cpr_nl(rlat1):
        return None  # zone crossing

    if fflag:
        ni = _cpr_n(rlat1, 1)
        m = math.floor(
            ((even_cprlon * (_cpr_nl(rlat1) - 1) - odd_cprlon * _cpr_nl(rlat1)) / 131072.0) + 0.5
        )
        rlon = _cpr_dlon(rlat1, 1) * ((m % ni + ni) % ni + odd_cprlon / 131072)
        rlat = rlat1
    else:
        ni = _cpr_n(rlat0, 0)
        m = math.floor(
            ((even_cprlon * (_cpr_nl(rlat0) - 1) - odd_cprlon * _cpr_nl(rlat0)) / 131072) + 0.5
        )
        rlon = _cpr_dlon(rlat0, 0) * ((m % ni + ni) % ni + even_cprlon / 131072)
        rlat = rlat0

    # Normalize to -180..+180
    rlon -= math.floor((rlon + 180) / 360) * 360

    return rlat, rlon


def _fmod_positive(a: float, b: float) -> float:
    """Always-positive fmod."""
    res = math.fmod(a, b)
    if res < 0:
        res += b
    return res


def decode_cpr_relative(
    reflat: float,
    reflon: float,
    cprlat: int,
    cprlon: int,
    fflag: int,
) -> tuple[float, float] | None:
    """Relative CPR decode using a reference position.

    Returns (lat, lon) or None if result is unreasonable (>1/2 cell from reference).
    """
    air_dlat = 360.0 / (59.0 if fflag else 60.0)
    frac_lat = cprlat / 131072.0
    frac_lon = cprlon / 131072.0

    j = int(
        math.floor(reflat / air_dlat)
        + math.floor(0.5 + _fmod_positive(reflat, air_dlat) / air_dlat - frac_lat)
    )
    rlat = air_dlat * (j + frac_lat)
    if rlat >= 270:
        rlat -= 360

    if rlat < -90 or rlat > 90:
        return None
    if abs(rlat - reflat) > air_dlat / 2:
        return None

    air_dlon = _cpr_dlon(rlat, fflag)
    m = int(
        math.floor(reflon / air_dlon)
        + math.floor(0.5 + _fmod_positive(reflon, air_dlon) / air_dlon - frac_lon)
    )
    rlon = air_dlon * (m + frac_lon)
    if rlon > 180:
        rlon -= 360

    if abs(rlon - reflon) > air_dlon / 2:
        return None

    return rlat, rlon


# Aircraft state tracking

_STALE_TIMEOUT = 300  # Remove aircraft after 300s without messages
_CPR_PAIR_TIMEOUT = 10.0  # Even/odd pair must arrive within 10s
_RATE_WINDOW = 5.0  # Message rate averaging window


@dataclass(frozen=True)
class AircraftState:
    """Immutable snapshot of a single tracked aircraft."""

    icao: str
    callsign: str = ""
    altitude: int | None = None
    speed: float | None = None
    heading: float | None = None
    vertical_rate: int | None = None
    lat: float | None = None
    lon: float | None = None
    messages: int = 0
    last_seen: float = 0.0


@dataclass(frozen=True)
class ADSBData:
    """Immutable snapshot of all ADS-B tracking state for the widget."""

    aircraft: tuple[AircraftState, ...] = ()
    total_messages: int = 0
    tracking_count: int = 0  # Aircraft with known position
    unique_icaos: int = 0
    messages_per_second: float = 0.0


class _MutableAircraft:
    """Internal mutable aircraft state with CPR buffers."""

    __slots__ = (
        "icao",
        "callsign",
        "altitude",
        "speed",
        "heading",
        "vertical_rate",
        "lat",
        "lon",
        "messages",
        "last_seen",
        "even_cprlat",
        "even_cprlon",
        "even_time",
        "odd_cprlat",
        "odd_cprlon",
        "odd_time",
    )

    def __init__(self, icao: str) -> None:
        self.icao = icao
        self.callsign = ""
        self.altitude: int | None = None
        self.speed: float | None = None
        self.heading: float | None = None
        self.vertical_rate: int | None = None
        self.lat: float | None = None
        self.lon: float | None = None
        self.messages = 0
        self.last_seen = 0.0
        self.even_cprlat = 0
        self.even_cprlon = 0
        self.even_time = 0.0
        self.odd_cprlat = 0
        self.odd_cprlon = 0
        self.odd_time = 0.0

    def to_state(self) -> AircraftState:
        return AircraftState(
            icao=self.icao,
            callsign=self.callsign,
            altitude=self.altitude,
            speed=self.speed,
            heading=self.heading,
            vertical_rate=self.vertical_rate,
            lat=self.lat,
            lon=self.lon,
            messages=self.messages,
            last_seen=self.last_seen,
        )


class AircraftTracker:
    """Accumulates ADS-B messages into per-aircraft state."""

    def __init__(self) -> None:
        self._aircraft: dict[str, _MutableAircraft] = {}
        self._total_messages = 0
        self._all_icaos: set[str] = set()
        self._msg_timestamps: deque[float] = deque()

    def update(self, msg: bytes, timestamp: float) -> None:
        """Process a decoded Mode S message and update aircraft state."""
        if len(msg) < 7:
            return
        df = msg[0] >> 3
        if df not in (17, 18):
            return

        self._total_messages += 1
        self._msg_timestamps.append(timestamp)

        icao = f"{msg[1]:02X}{msg[2]:02X}{msg[3]:02X}"
        self._all_icaos.add(icao)

        ac = self._aircraft.get(icao)
        if ac is None:
            ac = _MutableAircraft(icao)
            self._aircraft[icao] = ac
        ac.messages += 1
        ac.last_seen = timestamp

        me = msg[4:11]
        tc = (me[0] >> 3) & 0x1F

        if 1 <= tc <= 4:
            ac.callsign = _parse_identification(me)
        elif 9 <= tc <= 18:
            alt = _parse_altitude(me)
            if alt is not None:
                ac.altitude = alt
            self._update_position(ac, me, timestamp)
        elif tc == 19:
            speed, heading, vrate = _parse_velocity(me)
            if speed is not None:
                ac.speed = speed
            if heading is not None:
                ac.heading = heading
            if vrate is not None:
                ac.vertical_rate = vrate

    def _update_position(self, ac: _MutableAircraft, me: bytes, timestamp: float) -> None:
        cprlat, cprlon, fflag = _parse_cpr_position(me)

        if fflag:
            ac.odd_cprlat = cprlat
            ac.odd_cprlon = cprlon
            ac.odd_time = timestamp
        else:
            ac.even_cprlat = cprlat
            ac.even_cprlon = cprlon
            ac.even_time = timestamp

        # Try relative decode first if we already have a position
        if ac.lat is not None and ac.lon is not None:
            result = decode_cpr_relative(ac.lat, ac.lon, cprlat, cprlon, fflag)
            if result is not None:
                ac.lat, ac.lon = result
                return

        # Global decode: need both even and odd within timeout
        if ac.even_time == 0.0 or ac.odd_time == 0.0:
            return
        if abs(ac.even_time - ac.odd_time) > _CPR_PAIR_TIMEOUT:
            return

        # Use whichever frame arrived most recently
        use_odd = ac.odd_time > ac.even_time
        result = decode_cpr_airborne(
            ac.even_cprlat,
            ac.even_cprlon,
            ac.odd_cprlat,
            ac.odd_cprlon,
            int(use_odd),
        )
        if result is not None:
            ac.lat, ac.lon = result

    def snapshot(self, now: float | None = None) -> ADSBData:
        """Produce an immutable snapshot, pruning stale aircraft."""
        if now is None:
            now = time.time()

        # Prune stale
        stale = [icao for icao, ac in self._aircraft.items() if now - ac.last_seen > _STALE_TIMEOUT]
        for icao in stale:
            del self._aircraft[icao]

        # Prune rate window
        cutoff = now - _RATE_WINDOW
        while self._msg_timestamps and self._msg_timestamps[0] < cutoff:
            self._msg_timestamps.popleft()
        rate = len(self._msg_timestamps) / _RATE_WINDOW if self._msg_timestamps else 0.0

        # Sort: active (seen <60s) first, then by message count descending
        states = sorted(
            (ac.to_state() for ac in self._aircraft.values()),
            key=lambda s: (now - s.last_seen < 60, s.messages),
            reverse=True,
        )

        return ADSBData(
            aircraft=tuple(states),
            total_messages=self._total_messages,
            tracking_count=sum(1 for s in states if s.lat is not None),
            unique_icaos=len(self._all_icaos),
            messages_per_second=round(rate, 1),
        )

    def reset(self) -> None:
        self._aircraft.clear()
        self._total_messages = 0
        self._all_icaos.clear()
        self._msg_timestamps.clear()


def format_message(msg: bytes) -> str:
    """Format a decoded Mode S message as human-readable text."""
    hex_str = msg.hex()
    df = msg[0] >> 3

    if df == 17 or df == 18:
        icao = f"{msg[1]:02X}{msg[2]:02X}{msg[3]:02X}"
        me = msg[4:11]
        tc = (me[0] >> 3) & 0x1F

        if 1 <= tc <= 4:
            callsign = _parse_identification(me)
            return f"*{hex_str};  DF{df} ICAO={icao} [ID] {callsign}"
        if 9 <= tc <= 18:
            alt = _parse_altitude(me)
            f_bit = (me[2] >> 2) & 1
            frame = "odd" if f_bit else "even"
            alt_str = f"Alt={alt}ft" if alt is not None else "Alt=?"
            return f"*{hex_str};  DF{df} ICAO={icao} [Pos] {alt_str} ({frame})"
        if tc == 19:
            speed, heading, vrate = _parse_velocity(me)
            parts = []
            if speed is not None:
                parts.append(f"{speed:.0f}kt")
            if heading is not None:
                parts.append(f"hdg={heading:.0f}")
            if vrate is not None:
                parts.append(f"vr={vrate}fpm")
            return f"*{hex_str};  DF{df} ICAO={icao} [Vel] {' '.join(parts)}"

        return f"*{hex_str};  DF{df} ICAO={icao} TC={tc}"

    if df == 11:
        icao_val = crc24(msg)
        icao = f"{icao_val:06X}"
        return f"*{hex_str};  DF{df} ICAO={icao}"

    return f"*{hex_str};  DF{df}"


_OVERLAP_SAMPLES = 288


class ADSBDecoder(Demodulator):
    """ADS-B 1090ES decoder."""

    def __init__(self, sample_rate: float = 2_400_000):
        super().__init__()
        self.sample_rate = sample_rate
        self._overlap = np.zeros(0, dtype=np.uint16)
        self._known_icaos: set[int] = set()
        self._pending_icaos: set[int] = set()
        self._pending_messages: list[DecodedMessage] = []
        self._messages_decoded = 0
        self._df17_count = 0
        self._tracker = AircraftTracker()

    def demodulate(self, iq_samples: np.ndarray, timestamp: float) -> None:
        mag = magnitude_complex(iq_samples)
        if len(self._overlap) > 0:
            mag = np.concatenate([self._overlap, mag])

        raw_messages = _detect_and_decode(mag, self._known_icaos, self._pending_icaos)

        now = time.time()
        for _offset, msg_bytes in raw_messages:
            self._messages_decoded += 1
            if msg_bytes[0] >> 3 == 17:
                self._df17_count += 1
            text = format_message(msg_bytes)
            self._pending_messages.append(DecodedMessage(text=text, timestamp=now))
            self._tracker.update(msg_bytes, now)

        if len(mag) > _OVERLAP_SAMPLES:
            self._overlap = mag[-_OVERLAP_SAMPLES:].copy()
        else:
            self._overlap = mag.copy()

    def get_messages(self) -> list[DecodedMessage]:
        messages = self._pending_messages
        self._pending_messages = []
        if messages:
            # Attach tracker snapshot to the last message
            snapshot = self._tracker.snapshot()
            last = messages[-1]
            messages[-1] = DecodedMessage(text=last.text, timestamp=last.timestamp, data=snapshot)
        return messages

    def info(self) -> SignalInfo:
        """Thread-safe: callable from any thread. Reads scalar counters only."""
        quality = None
        quality_label = None
        if self._messages_decoded > 0:
            quality = min(1.0, self._df17_count / self._messages_decoded)
            quality_label = f"DF17 {quality * 100:.0f}%"
        return SignalInfo(
            label="ADS-B 1090ES",
            channel_bandwidth=2_000_000,
            modulation="PPM",
            sample_rate=2_400_000,
            has_text=True,
            message_type="adsb",
            quality_label=quality_label,
            quality=quality,
        )

    def reset(self) -> None:
        self._overlap = np.zeros(0, dtype=np.uint16)
        self._known_icaos.clear()
        self._pending_icaos.clear()
        self._pending_messages.clear()
        self._messages_decoded = 0
        self._df17_count = 0
        self._tracker.reset()
