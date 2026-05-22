"""Per-slot FT8 / FT4 decoder orchestration.

For each Costas-sync candidate the slot decoder:
1. Extracts soft 174-LLR vector from the waterfall.
2. Runs the BP LDPC decoder.
3. CRC-verifies the recovered 91-bit message.
4. XOR-removes the FT4 PRBS mask (FT4 only).
5. Hands the 77-bit payload to the message unpacker.

The dedup pass keeps the highest-score decode per unique CRC.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .crc import verify_crc
from .ldpc import bp_decode
from .message import MessageType, decode_message
from .sync import (
    Candidate,
    Waterfall,
    WaterfallParams,
    compute_waterfall,
    extract_llrs,
    find_candidates,
    ft4_params,
    ft8_params,
)
from .tables import FT4_XOR_SEQUENCE


class _Stage(Enum):
    LDPC_FAIL = "ldpc_fail"
    CRC_FAIL = "crc_fail"
    OK = "ok"


@dataclass(frozen=True)
class SlotDecode:
    """A successful decode of one candidate."""

    text: str
    msg_type: MessageType
    payload: bytes
    score: float
    freq_hz: float
    time_offset_s: float
    crc: int


@dataclass(frozen=True)
class SlotStats:
    """Per-slot diagnostics — what the candidate pipeline produced.

    `num_candidates`: how many sync hits passed `min_score`.
    `top_score`: best candidate score (0.0 if none).
    `ldpc_pass` / `crc_pass`: count of candidates that survived each stage.
    `unique_decodes`: distinct decoded messages after CRC dedup.
    """

    num_candidates: int
    top_score: float
    ldpc_pass: int
    crc_pass: int
    unique_decodes: int


_FT4_XOR_INT = int.from_bytes(FT4_XOR_SEQUENCE[:10], "big")


def _pack_91_bits_into_a91(bits: np.ndarray) -> bytes:
    """Pack the first 91 bits of a 174-bit codeword (MSB-first) into 12 bytes.

    Bits 91..95 are zero — they fall in the slack at the end of byte 11 that the
    CRC verifier zeros out anyway.
    """
    padded = np.zeros(96, dtype=np.uint8)
    padded[:91] = bits[:91]
    return bytes(np.packbits(padded))


def _decode_candidate(
    wf: Waterfall,
    cand: Candidate,
    *,
    is_ft4: bool,
    max_iterations: int,
) -> tuple[SlotDecode | None, _Stage]:
    log174 = extract_llrs(wf, cand, is_ft4=is_ft4)
    plain, errors = bp_decode(log174, max_iters=max_iterations)
    if errors != 0:
        return None, _Stage.LDPC_FAIL
    a91 = _pack_91_bits_into_a91(plain)
    crc_ok, crc = verify_crc(a91)
    if not crc_ok:
        return None, _Stage.CRC_FAIL

    if is_ft4:
        payload = (int.from_bytes(a91[:10], "big") ^ _FT4_XOR_INT).to_bytes(10, "big")
    else:
        payload = bytes(a91[:10])

    decoded = decode_message(payload)
    p = wf.params
    freq_hz = p.f_min + (cand.freq_offset + cand.freq_sub / p.freq_osr) / p.symbol_period
    time_off_s = (cand.time_offset + cand.time_sub / p.time_osr) * p.symbol_period
    return (
        SlotDecode(
            text=decoded.text,
            msg_type=decoded.msg_type,
            payload=payload,
            score=cand.score,
            freq_hz=freq_hz,
            time_offset_s=time_off_s,
            crc=crc,
        ),
        _Stage.OK,
    )


def analyze_slot(
    real_audio: np.ndarray,
    *,
    is_ft4: bool,
    sample_rate: int = 12000,
    f_min: float = 200.0,
    f_max: float = 3000.0,
    num_candidates: int = 120,
    min_score: float = 10.0,
) -> tuple[Waterfall, list[Candidate]]:
    """Compute the slot waterfall and the ranked Costas-sync candidates.

    Split out from ``decode_slot`` so callers that also want the waterfall
    (e.g. the demodulator wiring it into a diagnostic visualization) can do
    so without re-running the STFT + sync scan.
    """
    params: WaterfallParams = (
        ft4_params(sample_rate, f_min, f_max) if is_ft4 else ft8_params(sample_rate, f_min, f_max)
    )
    wf = compute_waterfall(real_audio, params)
    candidates = find_candidates(
        wf,
        is_ft4=is_ft4,
        num_candidates=num_candidates,
        min_score=min_score,
    )
    return wf, candidates


def decode_candidates(
    wf: Waterfall,
    candidates: list[Candidate],
    *,
    is_ft4: bool,
    max_iterations: int = 25,
) -> tuple[list[SlotDecode], SlotStats]:
    """Run LDPC + CRC + message unpack over a pre-ranked candidate list.

    ``candidates`` must be sorted descending by score (as ``find_candidates``
    returns it). Dedup keeps the first-seen — therefore highest-scoring —
    decode per CRC, and ``decodes`` inherits that order.
    """
    seen_crc: dict[int, SlotDecode] = {}
    ldpc_pass = 0
    crc_pass = 0
    for cand in candidates:
        decoded, stage = _decode_candidate(wf, cand, is_ft4=is_ft4, max_iterations=max_iterations)
        if stage in (_Stage.CRC_FAIL, _Stage.OK):
            ldpc_pass += 1
        if stage is _Stage.OK and decoded is not None:
            crc_pass += 1
            if decoded.crc not in seen_crc:
                seen_crc[decoded.crc] = decoded
    decodes = list(seen_crc.values())
    stats = SlotStats(
        num_candidates=len(candidates),
        top_score=float(candidates[0].score) if candidates else 0.0,
        ldpc_pass=ldpc_pass,
        crc_pass=crc_pass,
        unique_decodes=len(decodes),
    )
    return decodes, stats


def decode_slot(
    real_audio: np.ndarray,
    *,
    is_ft4: bool,
    sample_rate: int = 12000,
    f_min: float = 200.0,
    f_max: float = 3000.0,
    num_candidates: int = 120,
    min_score: float = 10.0,
    max_iterations: int = 25,
) -> list[SlotDecode]:
    """Decode every distinct message in one slot of mono audio.

    Args:
        real_audio: float32 mono audio at ``sample_rate`` Hz covering at least one
            full slot (15 s for FT8, 7.5 s for FT4). Anything past the slot is
            ignored; anything shorter loses tail candidates.
        is_ft4: select FT4 (True) or FT8 (False) decoder.
    """
    wf, candidates = analyze_slot(
        real_audio,
        is_ft4=is_ft4,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        num_candidates=num_candidates,
        min_score=min_score,
    )
    decodes, _stats = decode_candidates(
        wf, candidates, is_ft4=is_ft4, max_iterations=max_iterations
    )
    return decodes


__all__ = ["SlotDecode", "SlotStats", "analyze_slot", "decode_candidates", "decode_slot"]
