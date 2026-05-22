"""Drive the WSJT FT8 demodulator end-to-end on a real 12 kHz IQ recording."""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr.io import load_iq
from tsdr.radio.demodulators.wsjt import WSJTDemodulator

_FT8_SAMPLE = (
    Path(__file__).parents[1]
    / "samples"
    / "freq=14.074M_sr=12k_dur=60s_gain=0_20260521T1949.cu8.zst"
)
_FT4_SAMPLE = (
    Path(__file__).parents[1]
    / "samples"
    / "freq=7.0475M_sr=12k_dur=60s_gain=0_20260522T2110.cu8.zst"
)

# Recording timestamps for each sample; feeding them with these UTC anchors
# keeps slot windows aligned to the embedded transmissions.
_FT8_RECORDING_UTC = datetime(2026, 5, 21, 19, 49, 0, tzinfo=UTC).timestamp()
_FT4_RECORDING_UTC = datetime(2026, 5, 22, 19, 10, 30, tzinfo=UTC).timestamp()


def test_wsjt_ft8_demod_decodes_real_iq(caplog: pytest.LogCaptureFixture) -> None:
    iq = load_iq(str(_FT8_SAMPLE)).astype(np.complex64, copy=False)
    demod = WSJTDemodulator(mode="FT8", sample_rate=12000)

    # Feed in 1-second chunks to exercise the streaming path.
    chunk = 12000
    t = _FT8_RECORDING_UTC
    with caplog.at_level(logging.INFO, logger="tsdr.radio.demodulators.wsjt"):
        for start in range(0, iq.size, chunk):
            demod.demodulate(iq[start : start + chunk], capture_utc_s=t)
            t += chunk / 12000.0

    msgs = demod.get_messages()
    assert msgs, "WSJTDemodulator returned no FT8 decodes on real recording"
    for m in msgs:
        assert m.text.strip()

    pattern = re.compile(
        r"^wsjt_slot_summary mode=\w+ utc=\S+ candidates=\d+ "
        r"top_score=-?\d+\.\d+ ldpc_pass=\d+ crc_pass=\d+ decodes=\d+$"
    )
    summaries = [r.message for r in caplog.records if r.message.startswith("wsjt_slot_summary")]
    assert summaries, "expected at least one wsjt_slot_summary log line"
    for line in summaries:
        assert pattern.match(line), f"malformed slot summary: {line!r}"


def test_wsjt_ft4_demod_decodes_real_iq() -> None:
    """The candidate search must cover FT4's full ≈+2.4 s head slack to find
    UTC-anchored TXs that arrive late within the 7.5 s slot."""
    iq = load_iq(str(_FT4_SAMPLE)).astype(np.complex64, copy=False)
    demod = WSJTDemodulator(mode="FT4", sample_rate=12000)

    chunk = 12000
    t = _FT4_RECORDING_UTC
    for start in range(0, iq.size, chunk):
        demod.demodulate(iq[start : start + chunk], capture_utc_s=t)
        t += chunk / 12000.0

    msgs = demod.get_messages()
    assert len(msgs) >= 5, f"expected ≥5 FT4 decodes on real recording, got {len(msgs)}"
    for m in msgs:
        assert m.text.strip()


def test_wsjt_demod_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Invalid WSJT mode"):
        WSJTDemodulator(mode="FT9", sample_rate=12000)


def test_wsjt_demod_accepts_non_integer_rate_ratio() -> None:
    """Non-multiple-of-12k device rates (e.g. RTL-SDR 250 kHz, 2.4 MHz) must work
    via the polyphase fallback. Decimation here is a hot path for live HF use."""
    demod = WSJTDemodulator(mode="FT8", sample_rate=250000)
    # Feed a few seconds of dummy IQ and confirm the resampler produces output.
    iq = np.zeros(50_000, dtype=np.complex64)
    iq[::50] = 0.1  # impulse train, doesn't matter what, just non-empty
    demod.demodulate(iq, capture_utc_s=0.0)
    audio_batches = demod.get_audio()
    n_out = sum(b.samples.size for b in audio_batches)
    expected_out = int(50_000 * 12000 / 250000)
    # polyphase startup eats some samples; require we got at least 80%.
    assert n_out >= int(0.8 * expected_out), f"got {n_out}, expected ~{expected_out}"


def test_wsjt_demod_rejects_below_target_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        WSJTDemodulator(mode="FT8", sample_rate=8000)
