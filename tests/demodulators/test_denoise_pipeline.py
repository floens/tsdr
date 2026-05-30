"""End-to-end smoke test: the audio pipeline runs with RNNoise denoise enabled.

Exercises the full wiring (global ``SDRConfig.denoise`` drives the ``DenoiserStage``,
which denoises the audio ``DemodulatorStage`` attaches to the batch) under live
threading, on both a rate that needs resampling to 48 kHz (250k → 50000) and one
that does not (240k → 48000).
"""

import pytest

from tsdr.core.events.events import StatsUpdateEvent
from tsdr.radio.dsp.rnnoise import rnnoise_available

pytestmark = pytest.mark.skipif(
    not rnnoise_available(), reason="librnnoise not installed (denoise extra)"
)

SAMPLE_250K = "tests/samples/freq=438.35M_sr=250k_dur=9s_gain=24_20260412T1050.cu8.zst"
SAMPLE_240K = "tests/samples/freq=169.65M_sr=240k_dur=0s_gain=5_20260419T1353.cu8.zst"


@pytest.mark.parametrize(
    "iq_path,sample_rate",
    [(SAMPLE_250K, 250_000.0), (SAMPLE_240K, 240_000.0)],
)
def test_nfm_pipeline_runs_with_denoise(run_pipeline, iq_path, sample_rate):
    events = run_pipeline(
        iq_path,
        "nfm",
        sample_rate,
        StatsUpdateEvent,
        min_events=3,
        timeout=15.0,
        denoise=True,
    )
    assert len(events) >= 3
