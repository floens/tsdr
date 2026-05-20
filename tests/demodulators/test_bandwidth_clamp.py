"""Post-decimation demods clamp inherited channel_bandwidth to Nyquist.

Regression: switching WFM (200 kHz default) to AM/NFM/SSB/CW carried the
200 kHz value into the new demodulator, which crashed `firwin` because
its channel filter runs at decimated_rate ≈ audio_rate (Nyquist 24 kHz).
"""

from __future__ import annotations

import pytest

from tsdr.radio.demodulators.am import AMDemodulator
from tsdr.radio.demodulators.cw import CWDemodulator
from tsdr.radio.demodulators.nfm import NarrowbandFMDemodulator
from tsdr.radio.demodulators.ssb import SSBDemodulator
from tsdr.radio.demodulators.wfm import WidebandFMDemodulator

SAMPLE_RATE = 2_400_000.0
WIDE_BW = 200_000.0
EXPECTED_CLAMP = 48_000.0 * 0.95


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AMDemodulator(sample_rate=SAMPLE_RATE, channel_bandwidth=WIDE_BW),
        lambda: NarrowbandFMDemodulator(sample_rate=SAMPLE_RATE, channel_bandwidth=WIDE_BW),
        lambda: SSBDemodulator(mode="USB", sample_rate=SAMPLE_RATE, channel_bandwidth=WIDE_BW),
        lambda: SSBDemodulator(mode="LSB", sample_rate=SAMPLE_RATE, channel_bandwidth=WIDE_BW),
        lambda: CWDemodulator(sample_rate=SAMPLE_RATE, channel_bandwidth=WIDE_BW),
    ],
    ids=["AM", "NFM", "USB", "LSB", "CW"],
)
def test_post_decim_demods_clamp_inherited_wide_bandwidth(factory):
    demod = factory()
    assert demod.channel_bandwidth == pytest.approx(EXPECTED_CLAMP)
    assert demod.info().channel_bandwidth == pytest.approx(EXPECTED_CLAMP)


def test_set_channel_bandwidth_clamps_too_wide_value():
    demod = AMDemodulator(sample_rate=SAMPLE_RATE, channel_bandwidth=10_000)
    demod.set_channel_bandwidth(100_000)
    assert demod.channel_bandwidth == pytest.approx(EXPECTED_CLAMP)


def test_bandwidth_override_wfm_to_am():
    # Carryover from WFM (200 kHz) exceeds AM's Nyquist cap -> reset to AM default.
    assert AMDemodulator.bandwidth_override_on_mode_switch(200_000) == pytest.approx(10_000)


def test_bandwidth_override_lsb_to_usb_keeps_value():
    # User-tuned 2.4 kHz fits USB's cap -> no reset (keep user's value).
    assert SSBDemodulator.bandwidth_override_on_mode_switch(2_400) is None


def test_bandwidth_override_nfm_to_am_keeps_value():
    # 12.5 kHz fits AM's cap -> no reset.
    assert AMDemodulator.bandwidth_override_on_mode_switch(12_500) is None


def test_bandwidth_override_anything_to_wfm_keeps_value():
    # WFM's MAX_CHANNEL_BANDWIDTH is inf -> never reset.
    assert WidebandFMDemodulator.bandwidth_override_on_mode_switch(3_000) is None
    assert WidebandFMDemodulator.bandwidth_override_on_mode_switch(200_000) is None


def test_bandwidth_override_none_current_keeps_none():
    assert AMDemodulator.bandwidth_override_on_mode_switch(None) is None


def test_wfm_unaffected_by_wide_bandwidth():
    # WFM filters at the native sample rate -- 200 kHz must work without clamping.
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, channel_bandwidth=200_000)
    assert demod.channel_bandwidth == pytest.approx(200_000)
