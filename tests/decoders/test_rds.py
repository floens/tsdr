import numpy as np

from tsdr.core.sdr.io import load_iq
from tsdr.radio.decoders.rds import RDSDecoder


def _fm_demod(iq_path: str, sample_rate: int) -> np.ndarray:
    """Load IQ file and FM-demodulate (normalized by 75 kHz)."""
    raw = load_iq(iq_path)
    phase = np.unwrap(np.angle(raw))
    return np.diff(phase) * sample_rate / (2 * np.pi) / 75000.0


def test_rds_decoder_98_9():
    """RDS decode on 5s 240k sample of NPO Radio 1."""
    sample_rate = 240_000
    audio = _fm_demod(
        "tests/samples/freq=98.9M_sr=240k_dur=5s_gain=28_20260423T1733.cu8.zst",
        sample_rate,
    )
    dec = RDSDecoder(sample_rate)
    best_pi = None
    best_ps = ""
    best_score = (-1, -1)
    for i in range(0, len(audio), 25_000):
        dec.process(audio[i : i + 25_000])
        s = dec._snapshot()
        if s.sync_locked and s.ps_name and s.pi_code:
            rt_len = sum(1 for c in dec._rt_chars if c)
            score = (len(s.ps_name), rt_len)
            if score > best_score:
                best_score = score
                best_pi = s.pi_code
                best_ps = s.ps_name
    assert best_pi is not None, "never synced"
    assert best_pi == 0x8201, f"PI wrong: {best_pi:#06x}"
    assert best_ps.rstrip() == "RADIO 1", f"PS wrong: {best_ps!r}"
