"""ACARS (ARINC-618) protocol and MSK physical-layer constants.

VHF ACARS is 2400 baud MSK (modulation index 0.5: mark 2400 Hz, space 1200 Hz,
subcarrier centre 1800 Hz) on an AM voice channel, carrying 7-bit ASCII blocks
with per-character odd parity and a block CRC-16. Reference: ARINC Spec 618.
"""

BAUD = 2400.0  # symbols/second
INTERNAL_RATE = 12_000.0  # MSK demod rate: exactly 5 samples/symbol at 2400 Bd
INTERMEDIATE_RATE = 24_000.0  # pre-envelope rate; headroom for a tuning offset
MSK_CENTER_HZ = 1800.0  # subcarrier centre (mark 2400 / space 1200)
MSK_SPACE_HZ = 1200.0

CHANNEL_BANDWIDTH = 14_000.0  # fixed; wide enough for ~+-3-5 kHz tuning slack

# --- link layer: ARINC-618 block bytes, values include the odd-parity MSb ---
SYN = 0x16
SOH = 0x01
ETX = 0x83  # 0x03 | 0x80
ETB = 0x97  # 0x17 | 0x80
DEL = 0x7F
PLUS = ord("+") | 0x80  # 0xAB, first bit-sync char
STAR = ord("*")  # 0x2A, second bit-sync char

# Block field layout (bytes, parity included): mode addr[7] ack label[2] bid sot text...
# TXTMIN = header without any text; TXTMAX bounds a runaway block.
TXT_MIN_LEN = 13
TXT_MAX_LEN = 240
MAX_PARITY_ERRORS = 3
