"""The audio demods share one IQ front-end (design_channel_frontend). Its
anti-alias cutoff must stay inside (0, Nyquist) even when the input rate is at
or below audio_rate, as with a 12 kHz KiwiSDR channel — the regression that
crashed AM/NFM/CW/SSB (only SSTV had patched around it locally).
"""

import numpy as np
import pytest

from tsdr.radio.demodulators import NYQUIST_MARGIN, design_channel_frontend
from tsdr.radio.demodulators.am import AMDemodulator
from tsdr.radio.demodulators.cw import CWDemodulator
from tsdr.radio.demodulators.nfm import NarrowbandFMDemodulator
from tsdr.radio.demodulators.ssb import SSBDemodulator
from tsdr.radio.demodulators.sstv import SSTVDemodulator
from tsdr.radio.demodulators.wfm import WidebandFMDemodulator


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


def _tone_power(stage, fs: float, offset: float, seconds: float = 0.05) -> float:
    t = np.arange(int(fs * seconds)) / fs
    y = stage(np.exp(2j * np.pi * offset * t).astype(np.complex64))
    return float(np.mean(np.abs(y[len(y) // 2 :]) ** 2))


@pytest.mark.parametrize("fs", [2e6, 10e6])
def test_nfm_rejects_a_station_one_decimated_rate_away(fs: float) -> None:
    # A station 45 kHz away used to fold onto the channel (0.5 dB rejection at
    # 10 MS/s, 11 dB at 2 MS/s): a single 64-tap filter decimating ~200:1.
    def channel(x):
        d = NarrowbandFMDemodulator(sample_rate=fs, channel_bandwidth=16e3)
        return d._channel.process(d._decim.process(x))

    on = _tone_power(channel, fs, 0.0)
    off = _tone_power(channel, fs, 45e3)
    assert 10 * np.log10(off / on) < -80.0


def test_wfm_rejects_a_station_beside_the_intermediate_rate() -> None:
    # At 10 MS/s the old single filter decimated 40:1 to 250 kHz and let a
    # station 300 kHz away fold into the channel.
    def channel(x):
        d = WidebandFMDemodulator(sample_rate=10e6)
        return d._isolate_channel(x)

    on = _tone_power(channel, 10e6, 0.0)
    off = _tone_power(channel, 10e6, 300e3)
    assert 10 * np.log10(off / on) < -80.0


def _wfm_rejection_db(fs: float, bandwidth: float, offset: float) -> float:
    def channel(x):
        return WidebandFMDemodulator(sample_rate=fs, channel_bandwidth=bandwidth)._isolate_channel(
            x
        )

    return 10 * np.log10(_tone_power(channel, fs, offset) / _tone_power(channel, fs, 10e3))


@pytest.mark.parametrize("fs", [240e3, 2.4e6])
def test_wfm_channel_width_rejects_a_neighbor_inside_the_intermediate_band(fs: float) -> None:
    # The decimator alone passes everything up to the intermediate Nyquist
    # (0.7 dB rejection here at 2.4 MS/s); the channel FIR must cut at the
    # channel width, also at decimation 1, where the old code had no filter.
    assert _wfm_rejection_db(fs, 50e3, 75e3) < -60.0


def test_wfm_narrowing_the_bandwidth_narrows_the_channel() -> None:
    assert _wfm_rejection_db(2.4e6, 200e3, 75e3) > -1.0
    assert _wfm_rejection_db(2.4e6, 50e3, 75e3) < -60.0
