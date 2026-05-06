from dataclasses import dataclass

import numpy as np
from reedsolo import ReedSolomonError, RSCodec

FPAD_LEN = 2


@dataclass(frozen=True)
class SuperframeFormat:
    """Audio format parsed from the DAB+ superframe header."""

    core_sample_rate: int  # Core AAC rate before SBR (e.g. 24000)
    channels: int  # 1=mono, 2=stereo
    sbr: bool  # Spectral Band Replication (doubles output rate)
    ps: bool  # Parametric Stereo
    num_aus: int  # Access Units per superframe


# Fire code CRC-16 polynomial for superframe sync (ETSI EN 300 401)
_FIRE_CODE_POLY = 0x782F


def _fire_code_crc(data: bytes, length: int) -> int:
    """Compute fire code CRC-16 over data[0:length]."""
    crc = 0
    for i in range(length):
        crc ^= data[i] << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ _FIRE_CODE_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


N_LOGICAL_FRAMES_PER_SUPERFRAME = 5

# RS(120,110) over GF(2^8) using reedsolo
_RS_N = 120  # codeword length
_RS_K = 110  # data length

_RS_CODEC = RSCodec(nsym=_RS_N - _RS_K, nsize=_RS_N, fcr=0, prim=0x11D, c_exp=8)


def _rs_decode(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Reed-Solomon RS(120,110) decode over GF(2^8).

    Returns (corrected_data[:110], n_errors). n_errors = -1 if uncorrectable.
    """
    try:
        decoded_data, _, errata_pos = _RS_CODEC.decode(bytes(data.astype(np.uint8)))
        result = np.frombuffer(bytes(decoded_data[:_RS_K]), dtype=np.uint8).astype(np.int32)
        return result, len(errata_pos)
    except ReedSolomonError:
        return data[:_RS_K].copy(), -1


def _extract_pad(au: bytes) -> tuple[bytes, bytes] | None:
    """Extract PAD from an AAC AU by looking for MPEG-4 DSE (Data Stream Element).

    Returns (xpad_data, fpad_data) or None if no DSE present.
    X-PAD bytes are in reversed order (as received), F-PAD is always 2 bytes.
    """
    if len(au) < 3:
        return None
    # DSE element_id = 4 (0b100) in upper 3 bits of first byte
    if (au[0] >> 5) != 4:
        return None
    pad_start = 2
    pad_len = au[1]
    if pad_len == 255:
        if len(au) < 4:
            return None
        pad_len += au[2]
        pad_start = 3
    if pad_len < FPAD_LEN or len(au) < pad_start + pad_len:
        return None
    xpad_data = au[pad_start : pad_start + pad_len - FPAD_LEN]
    fpad_data = au[pad_start + pad_len - FPAD_LEN : pad_start + pad_len]
    return xpad_data, fpad_data


class _DabPlusSuperframe:
    """Accumulates 5 logical frames into a DAB+ superframe and extracts AUs."""

    def __init__(self):
        self._frames: list[bytes] = []
        self._frame_size = 0  # set on first push
        self._synced = False
        self._count = 0

    def push(
        self, logical_frame: bytes
    ) -> tuple[list[bytes], list[tuple[bytes, bytes] | None], SuperframeFormat] | None:
        """Push a logical frame (after Viterbi + PRBS).

        Returns (aus, pad_list, format) or None.
        pad_list has one entry per AU: (xpad_data, fpad_data) or None.

        Uses a sliding window to find initial alignment, then locks to
        5-frame boundaries to avoid misaligned decodes producing garbage.
        """
        if self._frame_size == 0:
            self._frame_size = len(logical_frame)

        if self._synced:
            # Locked: accumulate exactly 5 frames, then decode
            self._frames.append(logical_frame)
            self._count += 1
            if self._count < N_LOGICAL_FRAMES_PER_SUPERFRAME:
                return None
            result = self._decode_superframe(self._frames)
            self._frames = []
            self._count = 0
            if result is not None:
                return result
            # Lost sync - fall back to search
            self._synced = False
            return None

        # Search mode: sliding window to find alignment
        if len(self._frames) == N_LOGICAL_FRAMES_PER_SUPERFRAME:
            self._frames = self._frames[1:]
        self._frames.append(logical_frame)

        if len(self._frames) < N_LOGICAL_FRAMES_PER_SUPERFRAME:
            return None

        result = self._decode_superframe(self._frames)
        if result is not None:
            # Found alignment - lock to this boundary
            self._synced = True
            self._frames = []
            self._count = 0
        return result

    def _decode_superframe(
        self, frames: list[bytes]
    ) -> tuple[list[bytes], list[tuple[bytes, bytes] | None], SuperframeFormat] | None:
        """Decode a complete superframe (5 logical frames) into AAC AUs.

        The superframe is a matrix of 120 rows × subch_index columns.
        RS(120,110) is applied per COLUMN (not per row). After RS correction,
        the data occupies the first 110 rows (110 × subch_index bytes).
        """
        sf = bytearray(b"".join(frames))
        sf_len = len(sf)
        subch_index = sf_len // _RS_N
        if subch_index == 0:
            return None

        # RS correction: extract each column, decode, write back
        for i in range(subch_index):
            rs_packet = bytearray(120)
            for pos in range(120):
                rs_packet[pos] = sf[pos * subch_index + i]

            row_data = np.array(list(rs_packet), dtype=np.int32)
            decoded, nerr = _rs_decode(row_data)
            if nerr < 0:
                pass
            elif nerr > 0:
                # Write corrections back
                corrected_full = list(rs_packet)
                corrected_full[:_RS_K] = decoded.astype(np.uint8).tolist()
                for pos in range(120):
                    sf[pos * subch_index + i] = corrected_full[pos]

        # Data = first 110 rows of the matrix
        data_len = _RS_K * subch_index
        data = bytes(sf[:data_len])

        # Fire code sync check (first 2 bytes = CRC over bytes 2..10)
        if data[3] == 0x00 and data[4] == 0x00:
            return None
        crc_stored = data[0] << 8 | data[1]
        crc_calced = _fire_code_crc(data[2:], 9)
        if crc_stored != crc_calced:
            return None

        # Parse format byte
        dac_rate = (data[2] >> 6) & 1
        sbr_flag = (data[2] >> 5) & 1
        aac_channel_mode = (data[2] >> 4) & 1
        ps_flag = (data[2] >> 3) & 1

        # Number of AUs per superframe
        if dac_rate and sbr_flag:
            num_aus = 3
        elif dac_rate and not sbr_flag:
            num_aus = 6
        elif not dac_rate and sbr_flag:
            num_aus = 2
        else:
            num_aus = 4

        # AU start offsets (12-bit values packed after the format byte)
        au_starts = [0] * (num_aus + 1)
        au_starts[0] = (
            6
            if (dac_rate and sbr_flag)
            else (11 if (dac_rate and not sbr_flag) else (5 if (not dac_rate and sbr_flag) else 8))
        )
        au_starts[num_aus] = data_len  # pseudo end (after RS data)

        au_starts[1] = data[3] << 4 | data[4] >> 4
        if num_aus >= 3:
            au_starts[2] = (data[4] & 0x0F) << 8 | data[5]
        if num_aus >= 4:
            au_starts[3] = data[6] << 4 | data[7] >> 4
        if num_aus == 6:
            au_starts[4] = (data[7] & 0x0F) << 8 | data[8]
            au_starts[5] = data[9] << 4 | data[10] >> 4

        # Validate AU ordering
        for i in range(num_aus):
            if au_starts[i] >= au_starts[i + 1]:
                return None

        # Core AAC sample rate (SBR doubles the output rate)
        if dac_rate:
            core_sr = 24000 if sbr_flag else 48000
        else:
            core_sr = 16000 if sbr_flag else 32000

        # Extract AUs (strip 2-byte CRC suffix) and PAD from each
        aus = []
        pad_list: list[tuple[bytes, bytes] | None] = []
        for i in range(num_aus):
            au = data[au_starts[i] : au_starts[i + 1] - 2]  # -2 for CRC
            if len(au) < 2:
                continue
            aus.append(bytes(au))
            pad_list.append(_extract_pad(au))

        if not aus:
            return None
        fmt = SuperframeFormat(
            core_sample_rate=core_sr,
            channels=2 if aac_channel_mode else 1,
            sbr=bool(sbr_flag),
            ps=bool(ps_flag),
            num_aus=num_aus,
        )
        return aus, pad_list, fmt

    def reset(self):
        self._frames.clear()
        self._synced = False
        self._count = 0
