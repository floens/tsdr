"""Golden MAC PDU test vectors from real bursts.

Locks in parser behavior so refactors cannot silently break field
extraction.

Vectors are stored as bit strings (one char per bit, MSB first) and
compared against the same parse_* functions that produced them.
"""

import numpy as np
import pytest

from tsdr.radio.decoders.tetra.mac import (
    parse_aach,
    parse_mac_pdu,
    parse_sb1,
    parse_sysinfo,
)


def _bits(s: str) -> np.ndarray:
    return np.array([int(c) for c in s], dtype=np.uint8)


def _pack(parts: list[tuple[int, int]]) -> str:
    """Build a synthetic bit string from (n_bits, value) tuples (MSB-first)."""
    return "".join(f"{val:0{n}b}" for n, val in parts)


# Common MAC-RESOURCE prefix builder for the synthetic MM/SDS vectors below.
# Layout: PDU type=RESOURCE, no flags set, SSI addressing, no power/slot/channel
# allocation flags, BL-UDATA LLC header. Returns the (n, value) tuples so a
# test can append its own MLE body.
def _mac_resource_prefix(ssi: int) -> list[tuple[int, int]]:
    return [
        (2, 0b00),  # MAC PDU type RESOURCE
        (1, 0),  # fill bit indication
        (1, 0),  # position of grant
        (2, 0),  # encryption (clear)
        (1, 0),  # random access flag
        (6, 0),  # length indication
        (3, 0b001),  # address type SSI
        (24, ssi),
        (1, 0),  # power_control_flag
        (1, 0),  # slot_granting_flag
        (1, 0),  # chan_flag
        (4, 0b0010),  # LLC BL-UDATA
    ]


SB1_VECTORS = [
    {
        "bits": "000100001011011111000010000000000110011000000011111010011001",
        "expected": {
            "system_code": 1,
            "colour_code": 2,
            "timeslot": 3,
            "frame_number": 15,
            "multiframe_number": 33,
            "mcc": 204,
            "mnc": 500,
            "scramble_init": 855766027,
        },
    },
    {
        "bits": "000100001001100001000010000000000110011000000011111010011001",
        "expected": {
            "system_code": 1,
            "colour_code": 2,
            "timeslot": 1,
            "frame_number": 16,
            "multiframe_number": 33,
            "mcc": 204,
            "mnc": 500,
            "scramble_init": 855766027,
        },
    },
    {
        "bits": "000100001010100001000010000000000110011000000011111010011001",
        "expected": {
            "system_code": 1,
            "colour_code": 2,
            "timeslot": 2,
            "frame_number": 16,
            "multiframe_number": 33,
            "mcc": 204,
            "mnc": 500,
            "scramble_init": 855766027,
        },
    },
]


@pytest.mark.parametrize("vec", SB1_VECTORS)
def test_parse_sb1(vec):
    result = parse_sb1(_bits(vec["bits"]))
    for key, expected in vec["expected"].items():
        assert getattr(result, key) == expected, key


SYSINFO_VECTORS = [
    {
        "bits": "1000010000011100010001000000100001101001111000001110101110011101000000100000000000000000000000011111111111111111110101110101",
        "expected": {
            "main_carrier": 1052,
            "freq_band": 4,
            "freq_offset": 1,
            "duplex_spacing": 0,
            "reverse_operation": 0,
            "dl_freq_hz": 426306250,
            "ul_freq_hz": 416306250,
            "location_area": 1,
            "bs_service_details": 3445,
        },
    },
]


@pytest.mark.parametrize("vec", SYSINFO_VECTORS)
def test_parse_sysinfo(vec):
    result = parse_sysinfo(_bits(vec["bits"]))
    assert result is not None
    for key, expected in vec["expected"].items():
        assert getattr(result, key) == expected, key


AACH_VECTORS = [
    {
        "bits": "10100010110000",
        "expected": {"header": 2, "field1": 34, "field2": 48},
    },
    {
        "bits": "01100100110110",
        "expected": {"header": 1, "field1": 36, "field2": 54},
    },
    {
        "bits": "10100010110000",
        "expected": {"header": 2, "field1": 34, "field2": 48},
    },
    {
        "bits": "01100100110110",
        "expected": {"header": 1, "field1": 36, "field2": 54},
    },
    {
        "bits": "11100010110000",
        "expected": {"header": 3, "field1": 34, "field2": 48},
    },
    {
        "bits": "11100010110000",
        "expected": {"header": 3, "field1": 34, "field2": 48},
    },
]


@pytest.mark.parametrize("vec", AACH_VECTORS)
def test_parse_aach(vec):
    result = parse_aach(_bits(vec["bits"]))
    for key, expected in vec["expected"].items():
        assert getattr(result, key) == expected, key


CMCE_VECTORS = [
    {
        "bits": "0010000001001001000011111001000000000001000001001001001000000111011110010010000111110001000011110111110111111110000001111000",
        "summary": "RESOURCE clear SSI=1019905 BL-UDATA D-TX-CEASED call=239",
        "cmce": {"summary_contains": "D-TX-CEASED", "call_id": 239, "encryption_type": 0},
    },
    {
        "bits": "0000000001101001000011111001000000000001000001001001011000000111011111100010101000011110111110111111110000000000000100000000",
        "summary": "RESOURCE clear SSI=1019905 BL-UDATA D-TX-GRANTED call=239",
        "cmce": {"summary_contains": "D-TX-GRANTED", "call_id": 239, "encryption_type": 0},
    },
    {
        "bits": "0010000001001001000011111001000000000001000001001001001000000111100000010010000111110001000011110111110101001000000001111000",
        "summary": "RESOURCE clear SSI=1019905 BL-UDATA D-TX-CEASED call=240",
        "cmce": {"summary_contains": "D-TX-CEASED", "call_id": 240, "encryption_type": 0},
    },
    {
        "bits": "0000000001101001000011111001000000000001000001001001011000000111100001100010101000011110111110111111110000000000000100000000",
        "summary": "RESOURCE clear SSI=1019905 BL-UDATA D-TX-GRANTED call=240",
        "cmce": {"summary_contains": "D-TX-GRANTED", "call_id": 240, "encryption_type": 0},
    },
]


@pytest.mark.parametrize("vec", CMCE_VECTORS)
def test_parse_mac_pdu_cmce(vec):
    result = parse_mac_pdu(_bits(vec["bits"]))
    assert result is not None
    assert result.summary == vec["summary"]
    assert result.cmce is not None
    assert result.cmce.msg_type == vec["cmce"]["summary_contains"]
    assert result.cmce.call_id == vec["cmce"]["call_id"]
    assert result.cmce.encryption_type == vec["cmce"]["encryption_type"]


# MM PDU surfacing
#
# Synthetic vectors: BL-UDATA + MLE discriminator=MM + 4-bit MM PDU type.
# Body is opaque; the test only asserts the PDU name appears in the summary.
MM_VECTORS = [
    {
        "mm_type": 0x6,
        "name": "D-LOCATION-UPDATE-ACCEPT",
    },
    {
        "mm_type": 0x7,
        "name": "D-LOCATION-UPDATE-COMMAND",
    },
    {
        "mm_type": 0x1,
        "name": "D-AUTHENTICATION",
    },
    {
        "mm_type": 0x4,
        "name": "D-DISABLE",
    },
]


@pytest.mark.parametrize("vec", MM_VECTORS)
def test_parse_mac_pdu_mm(vec):
    bits = _pack(
        _mac_resource_prefix(ssi=1019905)
        + [
            (3, 0b001),  # MLE discriminator: MM
            (4, vec["mm_type"]),
        ]
    )
    result = parse_mac_pdu(_bits(bits))
    assert result is not None
    assert result.summary == f"RESOURCE clear SSI=1019905 BL-UDATA {vec['name']}"
    # MM PDUs do not produce CMCE call-tracking events.
    assert result.cmce is None


def test_parse_mac_pdu_mm_unknown_type():
    bits = _pack(
        _mac_resource_prefix(ssi=1019905)
        + [
            (3, 0b001),  # MLE discriminator: MM
            (4, 0xE),  # reserved (unmapped) MM PDU type
        ]
    )
    result = parse_mac_pdu(_bits(bits))
    assert result is not None
    assert result.summary == "RESOURCE clear SSI=1019905 BL-UDATA MM-unknown"


# SDS payload formatting
#
# Verify the hex extraction added so non-text SDS payloads are visible in the
# decoder console (previously protocol_id was logged but the payload was
# dropped; SDTI=2 64-bit data was truncated to 32 bits).
def _build_sds_pdu(parts: list[tuple[int, int]]) -> np.ndarray:
    return _bits(
        _pack(
            _mac_resource_prefix(ssi=0x123456)
            + [
                (3, 0b010),  # MLE discriminator: CMCE
                (5, 0x0F),  # CMCE PDU type: D-SDS-DATA
            ]
            + parts
        )
    )


def test_sds_text_payload():
    bits = _build_sds_pdu(
        [
            (2, 0b01),  # CPTI: SSI addressing
            (24, 0x123456),  # from-SSI
            (2, 0b11),  # SDTI=3 (variable-length data with protocol id)
            (8, 0x82),  # text protocol
            (8, ord("H")),
            (8, ord("i")),
        ]
    )
    result = parse_mac_pdu(bits)
    assert result is not None
    assert result.summary == 'RESOURCE clear SSI=1193046 BL-UDATA SDS from=1193046 "Hi"'


def test_sds_var_length_non_text_payload():
    bits = _build_sds_pdu(
        [
            (2, 0b01),
            (24, 0x123456),
            (2, 0b11),  # SDTI=3 var-length
            (8, 0x42),  # non-text protocol
            (16, 0xABCD),  # opaque payload
        ]
    )
    result = parse_mac_pdu(bits)
    assert result is not None
    assert result.summary == (
        "RESOURCE clear SSI=1193046 BL-UDATA SDS from=1193046 proto=0x42 data=0xABCD"
    )


def test_sds_64bit_data_not_truncated():
    bits = _build_sds_pdu(
        [
            (2, 0b01),
            (24, 0x123456),
            (2, 0b10),  # SDTI=2: 64-bit data
            (32, 0x01234567),
            (32, 0x89ABCDEF),
        ]
    )
    result = parse_mac_pdu(bits)
    assert result is not None
    assert result.summary == (
        "RESOURCE clear SSI=1193046 BL-UDATA SDS from=1193046 data=0x0123456789ABCDEF"
    )


def test_sds_16bit_data_preserves_leading_zeros():
    bits = _build_sds_pdu(
        [
            (2, 0b01),
            (24, 0x123456),
            (2, 0b00),  # SDTI=0: 16-bit data
            (16, 0x0042),
        ]
    )
    result = parse_mac_pdu(bits)
    assert result is not None
    assert result.summary == ("RESOURCE clear SSI=1193046 BL-UDATA SDS from=1193046 data=0x0042")
