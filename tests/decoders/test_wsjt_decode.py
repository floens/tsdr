"""End-to-end FT8 slot-decode test on a real 14.074 MHz recording.

The 60-second IQ sample at 12 kHz captures roughly four FT8 slots; at least
one of them must produce a recoverable message.
"""

from pathlib import Path

import numpy as np

from tsdr.core.sdr.io import load_iq
from tsdr.radio.decoders.wsjt.decode import decode_slot

_SAMPLE = (
    Path(__file__).parents[1]
    / "samples"
    / "freq=14.074M_sr=12k_dur=60s_gain=0_20260521T1949.cu8.zst"
)


def test_decode_real_ft8_slot_recovers_some_message() -> None:
    iq = load_iq(str(_SAMPLE))
    # USB demod path: keep the positive sideband by taking the real part.
    audio = iq.real.astype(np.float32, copy=False)
    slot_len = 12000 * 15

    all_decodes = []
    for start in range(0, len(audio) - slot_len + 1, slot_len):
        all_decodes.extend(decode_slot(audio[start : start + slot_len], is_ft4=False))

    assert all_decodes, "no FT8 messages decoded in 60 s of band-active recording"
    for d in all_decodes:
        assert d.text.strip()
        assert 200.0 <= d.freq_hz <= 3000.0
