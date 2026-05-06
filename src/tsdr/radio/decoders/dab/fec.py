import numba as nb
import numpy as np

from .constants import PRBS_INIT


def _generate_prbs(n_bits: int) -> np.ndarray:
    """Generate PRBS sequence for energy dispersal.

    LFSR polynomial x^9 + x^5 + 1, initialized to all 1s.
    Output: feedback bit = reg[8] XOR reg[4].
    """
    reg = PRBS_INIT
    bits = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        fb = ((reg >> 8) ^ (reg >> 4)) & 1
        bits[i] = fb
        reg = ((reg << 1) | fb) & 0x1FF
    return bits


# Pre-generate PRBS for 768 bits (one FIC block = 3 FIBs)
_PRBS_768 = _generate_prbs(768)


@nb.njit(cache=True)
def _crc16_bytes_jit(data: np.ndarray) -> int:
    """JIT-compiled CRC-16 CCITT over uint8 array."""
    crc = nb.uint32(0xFFFF)
    poly = nb.uint32(0x1021)
    for i in range(len(data)):
        crc ^= nb.uint32(data[i]) << nb.uint32(8)
        for _ in range(8):
            if crc & nb.uint32(0x8000):
                crc = ((crc << nb.uint32(1)) ^ poly) & nb.uint32(0xFFFF)
            else:
                crc = (crc << nb.uint32(1)) & nb.uint32(0xFFFF)
    return int(crc)


def _crc16_bytes(data: bytes | np.ndarray) -> int:
    """Compute CRC-16 CCITT over byte array."""
    if isinstance(data, bytes):
        data = np.frombuffer(data, dtype=np.uint8)
    return int(_crc16_bytes_jit(data))


def _check_fib_crc(fib_bytes: bytes | np.ndarray) -> bool:
    """Check CRC of a 32-byte FIB.

    Process all 32 bytes (30 data + 2 CRC). The CRC field contains the
    ones' complement. Correct residual is 0x1D0F.
    """
    data = bytes(fib_bytes[:32])
    # Invert the CRC bytes (last 2) before checking
    data = data[:30] + bytes([data[30] ^ 0xFF, data[31] ^ 0xFF])
    return _crc16_bytes(data) == 0
