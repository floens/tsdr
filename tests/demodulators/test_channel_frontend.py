"""The audio demods share one IQ front-end (design_channel_frontend). Its
anti-alias cutoff must stay inside (0, Nyquist) even when the input rate is at
or below audio_rate, as with a 12 kHz KiwiSDR channel — the regression that
crashed AM/NFM/CW/SSB (only SSTV had patched around it locally).
"""

import pytest

from tsdr.radio.demodulators import NYQUIST_MARGIN, design_channel_frontend
from tsdr.radio.demodulators.am import AMDemodulator
from tsdr.radio.demodulators.cw import CWDemodulator
from tsdr.radio.demodulators.nfm import NarrowbandFMDemodulator
from tsdr.radio.demodulators.ssb import SSBDemodulator
from tsdr.radio.demodulators.sstv import SSTVDemodulator


def test_frontend_low_rate_no_decimation():
    fe = design_channel_frontend(12_000.0, 48_000.0, 3_000.0)
    assert fe.decimation == 1
    assert fe.decimated_rate == 12_000.0
    assert fe.channel_bandwidth == 3_000.0


def test_frontend_clamps_bandwidth_to_decimated_nyquist():
    fe = design_channel_frontend(12_000.0, 48_000.0, 100_000.0)
    assert fe.channel_bandwidth == pytest.approx(12_000.0 * NYQUIST_MARGIN)


def test_frontend_high_rate_decimates_to_audio():
    fe = design_channel_frontend(2_400_000.0, 48_000.0, 10_000.0)
    assert fe.decimation == 50
    assert fe.decimated_rate == 48_000.0


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda sr: AMDemodulator(sample_rate=sr), id="am"),
        pytest.param(lambda sr: SSBDemodulator(mode="LSB", sample_rate=sr), id="ssb"),
        pytest.param(lambda sr: NarrowbandFMDemodulator(sample_rate=sr), id="nfm"),
        pytest.param(lambda sr: CWDemodulator(sample_rate=sr), id="cw"),
        pytest.param(lambda sr: SSTVDemodulator(sample_rate=sr), id="sstv"),
    ],
)
def test_hf_demod_constructs_at_kiwisdr_rate(make):
    demod = make(12_000.0)
    assert demod.decimated_rate == 12_000.0
