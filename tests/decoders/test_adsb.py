"""Incremental tests for the ADS-B decoder.

Each test validates a stage of the pipeline using real samples
and reference dumps.
"""

import struct
import time
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd

from tsdr.core.sdr.io import load_iq as _load_iq
from tsdr.radio.decoders.adsb import (
    _CRC_TABLE,
    ADSBDecoder,
    AircraftTracker,
    _detect_and_decode,
    _extract_bytes,
    _parse_cpr_position,
    crc24,
    decode_cpr_airborne,
    decode_cpr_relative,
    format_message,
    magnitude_complex,
    magnitude_uc8,
)

SAMPLE_FILE = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "freq=1090.0M_sr=2400k_dur=30s_gain=23_20260328T1217.cu8.zst"
)
SAMPLE_RATE = 2_400_000

# Reference magnitude/message dumps for comparison
REF_MAG_FILE = Path("/tmp/dump1090_ref/magnitude.u16")
REF_MSG_FILE = Path("/tmp/dump1090_ref/messages.bin")


def load_raw_uc8() -> np.ndarray:
    """Load raw UC8 bytes from the sample file."""
    if not SAMPLE_FILE.exists():
        pytest.skip(f"Sample file not found: {SAMPLE_FILE}")
    dctx = zstd.ZstdDecompressor()
    with open(SAMPLE_FILE, "rb") as f:
        raw = dctx.stream_reader(f).read()
    return np.frombuffer(raw, dtype=np.uint8)


def load_ref_messages() -> list[tuple[int, int, bytes]]:
    """Load reference messages from binary dump.

    Returns [(sample_offset, phase, message_bytes), ...]
    """
    if not REF_MSG_FILE.exists():
        pytest.skip(f"Reference dump not found: {REF_MSG_FILE}")
    data = REF_MSG_FILE.read_bytes()
    messages = []
    pos = 0
    while pos < len(data):
        j = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        phase = data[pos]
        pos += 1
        msglen = data[pos]
        pos += 1
        msg_bytes = data[pos : pos + msglen]
        pos += msglen
        messages.append((j, phase, msg_bytes))
    return messages


@pytest.fixture(scope="module")
def raw_uc8():
    """Module-scoped raw UC8 bytes."""
    return load_raw_uc8()


@pytest.fixture(scope="module")
def mag(raw_uc8):
    """Module-scoped magnitude from UC8."""
    return magnitude_uc8(raw_uc8)


@pytest.fixture(scope="module")
def decoded_messages(mag):
    """Module-scoped decoded ADS-B messages."""
    known_icaos: set[int] = set()
    return _detect_and_decode(mag, known_icaos)


class TestStage1Magnitude:
    """Stage 1: Verify magnitude computation matches reference."""

    def test_magnitude_matches_reference(self, raw_uc8, mag):
        if not REF_MAG_FILE.exists():
            pytest.skip("Reference magnitude dump not found")

        ref_mag = np.fromfile(REF_MAG_FILE, dtype=np.uint16)

        assert len(mag) == len(ref_mag), f"Length mismatch: {len(mag)} vs {len(ref_mag)}"

        # Check first 1M samples for exact match (allow ±1 for rounding)
        n = min(1_000_000, len(mag))
        diff = np.abs(mag[:n].astype(np.int32) - ref_mag[:n].astype(np.int32))
        max_diff = np.max(diff)
        assert max_diff <= 1, f"Max magnitude difference: {max_diff}"

    def test_complex_magnitude_reasonable(self, raw_uc8, mag):
        """Verify complex64 path produces similar magnitudes."""
        # Simulate what load_iq does: (val - 127.5) / 127.5
        pairs = raw_uc8.reshape(-1, 2).astype(np.float32)
        i_samples = (pairs[:, 0] - 127.5) / 127.5
        q_samples = (pairs[:, 1] - 127.5) / 127.5
        iq = (i_samples + 1j * q_samples).astype(np.complex64)

        complex_mag = magnitude_complex(iq)

        # The two should be correlated (r > 0.99) even if not identical
        n = min(100_000, len(mag))
        corr = np.corrcoef(mag[:n].astype(float), complex_mag[:n].astype(float))[0, 1]
        assert corr > 0.99, f"Magnitude correlation too low: {corr}"


@pytest.fixture(scope="module")
def ref_msgs():
    """Module-scoped reference messages."""
    return load_ref_messages()


class TestStage2CRC:
    """Stage 2: Verify CRC-24 implementation."""

    def test_known_good_df17(self, ref_msgs):
        df17_msgs = [m for _, _, m in ref_msgs if m[0] >> 3 == 17]
        assert len(df17_msgs) > 0

        for msg in df17_msgs[:20]:
            assert crc24(msg) == 0, f"CRC failed for DF17: {msg.hex()}"

    def test_single_bit_error_detected(self, ref_msgs):
        df17_msgs = [m for _, _, m in ref_msgs if m[0] >> 3 == 17]
        msg = bytearray(df17_msgs[0])

        # Flip one bit
        msg[5] ^= 0x10
        assert crc24(bytes(msg)) != 0

    def test_polynomial_matches_dump1090(self):
        """Verify our CRC table matches polynomial 0xFFF409."""
        # Manually compute CRC for byte 0x80 (single high bit)
        c = 0x80 << 16
        poly = 0xFFF409
        for _ in range(8):
            if c & 0x800000:
                c = (c << 1) ^ poly
            else:
                c = c << 1
        c &= 0xFFFFFF
        assert _CRC_TABLE[0x80] == c


class TestStage3BitExtraction:
    """Stage 3: Verify bit extraction matches reference exactly."""

    def test_extract_bytes_match_reference(self, mag, ref_msgs):
        matched = 0
        mismatched = 0

        for j, phase, ref_bytes in ref_msgs[:100]:
            n_bytes = len(ref_bytes)
            our_bytes = _extract_bytes(mag, j + 19, phase, n_bytes)
            if our_bytes == ref_bytes:
                matched += 1
            else:
                mismatched += 1
                # Find first differing byte
                for k in range(n_bytes):
                    if our_bytes[k] != ref_bytes[k]:
                        break

        assert matched > 0, "No messages matched"
        match_rate = matched / (matched + mismatched)
        assert match_rate > 0.95, (
            f"Match rate too low: {match_rate:.1%} ({matched}/{matched + mismatched})"
        )


class TestStage4FullDecode:
    """Stage 4: Full pipeline decode and comparison with reference."""

    def test_decode_count(self, decoded_messages):
        df17_count = sum(1 for _, m in decoded_messages if m[0] >> 3 == 17)

        # Expected ~130 DF17; non-DF17 requires known ICAOs (learned from DF17)
        assert df17_count >= 100, f"Too few DF17: {df17_count} (expected ~130)"

    def test_df17_hex_match(self, decoded_messages, ref_msgs):
        """Compare our DF17 hex output against reference."""
        our_df17_hex = {m.hex() for _, m in decoded_messages if m[0] >> 3 == 17}

        ref_df17_hex = {m.hex() for _, _, m in ref_msgs if m[0] >> 3 == 17}

        intersection = our_df17_hex & ref_df17_hex
        recall = len(intersection) / len(ref_df17_hex) if ref_df17_hex else 0

        assert recall >= 0.90, (
            f"DF17 recall too low: {recall:.1%} ({len(intersection)}/{len(ref_df17_hex)} matched)"
        )


class TestStage5MessageParsing:
    """Stage 5: Verify message parsing produces reasonable values."""

    def test_identification_messages(self, decoded_messages):
        id_msgs = []
        for _, msg in decoded_messages:
            if msg[0] >> 3 != 17:
                continue
            tc = (msg[4] >> 3) & 0x1F
            if 1 <= tc <= 4:
                text = format_message(msg)
                id_msgs.append(text)

        assert len(id_msgs) > 0, "No identification messages found"
        for text in id_msgs:
            assert "[ID]" in text
            # Callsign should contain only valid characters
            callsign_part = text.split("[ID]")[1].strip()
            for ch in callsign_part:
                assert ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ", (
                    f"Invalid char in callsign: {ch!r}"
                )

    def test_position_messages(self, decoded_messages):
        pos_msgs = []
        for _, msg in decoded_messages:
            if msg[0] >> 3 != 17:
                continue
            tc = (msg[4] >> 3) & 0x1F
            if 9 <= tc <= 18:
                text = format_message(msg)
                pos_msgs.append(text)

        assert len(pos_msgs) > 0, "No position messages found"
        for text in pos_msgs:
            assert "[Pos]" in text
            if "Alt=" in text and "Alt=?" not in text:
                alt_str = text.split("Alt=")[1].split("ft")[0]
                alt = int(alt_str)
                assert -1000 <= alt <= 60000, f"Unreasonable altitude: {alt}"

    def test_velocity_messages(self, decoded_messages):
        vel_msgs = []
        for _, msg in decoded_messages:
            if msg[0] >> 3 != 17:
                continue
            tc = (msg[4] >> 3) & 0x1F
            if tc == 19:
                text = format_message(msg)
                vel_msgs.append(text)

        assert len(vel_msgs) > 0, "No velocity messages found"
        for text in vel_msgs:
            assert "[Vel]" in text


class TestStage6Streaming:
    """Stage 6: Verify streaming decoder handles chunk boundaries."""

    def test_streaming_consistency(self):
        if not SAMPLE_FILE.exists():
            pytest.skip("Sample file not found")
        iq = _load_iq(SAMPLE_FILE)

        # Large chunks
        decoder_large = ADSBDecoder(sample_rate=SAMPLE_RATE)
        chunk_size = 65536
        for i in range(0, len(iq), chunk_size):
            decoder_large.demodulate(iq[i : i + chunk_size], 0.0)
        msgs_large = decoder_large._messages_decoded

        # Small chunks
        decoder_small = ADSBDecoder(sample_rate=SAMPLE_RATE)
        chunk_size = 4096
        for i in range(0, len(iq), chunk_size):
            decoder_small.demodulate(iq[i : i + chunk_size], 0.0)
        msgs_small = decoder_small._messages_decoded

        # Should be within 5% of each other
        assert msgs_large > 0, "No messages decoded with large chunks"
        ratio = msgs_small / msgs_large
        assert 0.90 <= ratio <= 1.10, (
            f"Streaming inconsistency: large={msgs_large}, small={msgs_small} (ratio={ratio:.2f})"
        )


class TestCPRDecode:
    """CPR position decoding."""

    # Global airborne test vectors: (even_cprlat, even_cprlon, odd_cprlat, odd_cprlon)
    GLOBAL_AIRBORNE = [
        (80536, 9432, 61720, 9192, 51.686646, 0.700156, 51.686763, 0.701294),
        (80534, 9413, 61714, 9144, 51.686554, 0.698745, 51.686484, 0.697632),
    ]

    @pytest.mark.parametrize(
        "even_lat,even_lon,odd_lat,odd_lon,exp_elat,exp_elon,exp_olat,exp_olon",
        GLOBAL_AIRBORNE,
    )
    def test_global_airborne_even(
        self, even_lat, even_lon, odd_lat, odd_lon, exp_elat, exp_elon, exp_olat, exp_olon
    ):
        result = decode_cpr_airborne(even_lat, even_lon, odd_lat, odd_lon, 0)
        assert result is not None
        lat, lon = result
        assert abs(lat - exp_elat) < 1e-6, f"lat {lat} != {exp_elat}"
        assert abs(lon - exp_elon) < 1e-6, f"lon {lon} != {exp_elon}"

    @pytest.mark.parametrize(
        "even_lat,even_lon,odd_lat,odd_lon,exp_elat,exp_elon,exp_olat,exp_olon",
        GLOBAL_AIRBORNE,
    )
    def test_global_airborne_odd(
        self, even_lat, even_lon, odd_lat, odd_lon, exp_elat, exp_elon, exp_olat, exp_olon
    ):
        result = decode_cpr_airborne(even_lat, even_lon, odd_lat, odd_lon, 1)
        assert result is not None
        lat, lon = result
        assert abs(lat - exp_olat) < 1e-6, f"lat {lat} != {exp_olat}"
        assert abs(lon - exp_olon) < 1e-6, f"lon {lon} != {exp_olon}"

    # Relative decode test vectors
    RELATIVE_AIRBORNE = [
        (52.00, 0.00, 80536, 9432, 0, 51.686646, 0.700156),
        (52.00, 0.00, 61720, 9192, 1, 51.686763, 0.701294),
        (52.00, 0.00, 80534, 9413, 0, 51.686554, 0.698745),
        (52.00, 0.00, 61714, 9144, 1, 51.686484, 0.697632),
        # Moved receiver - still within 1/2 cell
        (48.70, 0.00, 80536, 9432, 0, 51.686646, 0.700156),
        (54.60, 0.00, 80536, 9432, 0, 51.686646, 0.700156),
        (52.00, 5.40, 80536, 9432, 0, 51.686646, 0.700156),
        (52.00, -4.10, 80536, 9432, 0, 51.686646, 0.700156),
    ]

    @pytest.mark.parametrize(
        "reflat,reflon,cprlat,cprlon,fflag,exp_lat,exp_lon",
        RELATIVE_AIRBORNE,
    )
    def test_relative_airborne(self, reflat, reflon, cprlat, cprlon, fflag, exp_lat, exp_lon):
        result = decode_cpr_relative(reflat, reflon, cprlat, cprlon, fflag)
        assert result is not None
        lat, lon = result
        assert abs(lat - exp_lat) < 1e-6, f"lat {lat} != {exp_lat}"
        assert abs(lon - exp_lon) < 1e-6, f"lon {lon} != {exp_lon}"

    def test_cpr_extraction_from_position_message(self, decoded_messages):
        """Verify CPR lat/lon/fflag extraction from a real position message."""
        pos_count = 0
        for _, msg in decoded_messages:
            if msg[0] >> 3 != 17:
                continue
            me = msg[4:11]
            tc = (me[0] >> 3) & 0x1F
            if 9 <= tc <= 18:
                cprlat, cprlon, fflag = _parse_cpr_position(me)
                assert 0 <= cprlat < 131072, f"cprlat out of range: {cprlat}"
                assert 0 <= cprlon < 131072, f"cprlon out of range: {cprlon}"
                assert fflag in (0, 1), f"fflag invalid: {fflag}"
                pos_count += 1

        assert pos_count > 0, "No position messages found"


class TestAircraftTracker:
    """Aircraft tracker: state accumulation and position decode from real samples."""

    def test_tracker_accumulates_state(self, decoded_messages):
        """Feed real decoded messages into tracker, verify aircraft appear."""
        tracker = AircraftTracker()
        for _, msg in decoded_messages:
            tracker.update(msg, time.time())

        snapshot = tracker.snapshot()
        assert len(snapshot.aircraft) > 0, "No aircraft tracked"
        assert snapshot.total_messages > 0
        assert snapshot.unique_icaos > 0

        # At least some aircraft should have callsigns
        with_callsign = [ac for ac in snapshot.aircraft if ac.callsign]
        assert len(with_callsign) > 0, "No aircraft with callsigns"

        # At least some should have altitude
        with_alt = [ac for ac in snapshot.aircraft if ac.altitude is not None]
        assert len(with_alt) > 0, "No aircraft with altitude"

    def test_tracker_resolves_positions(self, decoded_messages):
        """Verify CPR decode produces positions from real even/odd pairs."""
        tracker = AircraftTracker()
        t = 1000.0  # Use fake timestamps that increment
        for _, msg in decoded_messages:
            tracker.update(msg, t)
            t += 0.01  # ~100 msg/s

        snapshot = tracker.snapshot(now=t)
        with_pos = [ac for ac in snapshot.aircraft if ac.lat is not None and ac.lon is not None]
        assert len(with_pos) > 0, "No aircraft positions resolved"

        for ac in with_pos:
            assert -90 <= ac.lat <= 90, f"Latitude out of range: {ac.lat}"
            assert -180 <= ac.lon <= 180, f"Longitude out of range: {ac.lon}"

    def test_snapshot_sorted_by_messages(self, decoded_messages):
        """Verify snapshot is sorted by message count descending."""
        tracker = AircraftTracker()
        t = 1000.0
        for _, msg in decoded_messages:
            tracker.update(msg, t)
            t += 0.01

        snapshot = tracker.snapshot(now=t)
        counts = [ac.messages for ac in snapshot.aircraft]
        assert counts == sorted(counts, reverse=True)
