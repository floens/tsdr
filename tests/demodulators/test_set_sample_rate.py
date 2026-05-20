from __future__ import annotations

import numpy as np
import pytest

from tsdr.radio.demodulators.am import AMDemodulator
from tsdr.radio.demodulators.cw import CWDemodulator
from tsdr.radio.demodulators.nfm import NarrowbandFMDemodulator
from tsdr.radio.demodulators.ssb import SSBDemodulator
from tsdr.radio.demodulators.wfm import WidebandFMDemodulator


def test_am_set_sample_rate_updates_decimation():
    d = AMDemodulator(sample_rate=384_000, audio_rate=48_000)
    assert d.channel_decimation == 8
    d.set_sample_rate(192_000)
    assert d.sample_rate == 192_000
    assert d.channel_decimation == 4
    assert d.decimated_rate == 48_000


def test_ssb_set_sample_rate_updates_decimation():
    d = SSBDemodulator(mode="LSB", sample_rate=384_000, audio_rate=48_000)
    assert d.channel_decimation == 8
    d.set_sample_rate(192_000)
    assert d.sample_rate == 192_000
    assert d.channel_decimation == 4
    assert d.decimated_rate == 48_000


def test_nfm_set_sample_rate_updates_decimation():
    d = NarrowbandFMDemodulator(sample_rate=240_000, audio_rate=48_000)
    assert d.channel_decimation == 5
    d.set_sample_rate(96_000)
    assert d.sample_rate == 96_000
    assert d.channel_decimation == 2
    assert d.decimated_rate == 48_000


def test_cw_set_sample_rate_updates_decimation():
    d = CWDemodulator(sample_rate=384_000, audio_rate=48_000)
    assert d.channel_decimation == 8
    d.set_sample_rate(192_000)
    assert d.sample_rate == 192_000
    assert d.channel_decimation == 4
    assert d.decimated_rate == 48_000


def test_wfm_set_sample_rate_updates_decimation():
    d = WidebandFMDemodulator(sample_rate=2_400_000, audio_rate=48_000)
    decim_a = d.channel_decimation
    d.set_sample_rate(1_200_000)
    assert d.sample_rate == 1_200_000
    assert d.channel_decimation == decim_a // 2
    assert d.intermediate_rate == 1_200_000 / d.channel_decimation
    assert d.output_sample_rate == d.intermediate_rate / d.audio_decimation_factor
    assert d._rds_decoder is None
    assert d.stereo_detected is False


def test_ssb_emits_audio_at_new_decimated_rate():
    d = SSBDemodulator(mode="LSB", sample_rate=384_000, audio_rate=48_000)
    d.demodulate(np.ones(8192, dtype=np.complex64), 0.0)
    d.get_audio()
    d.set_sample_rate(192_000)
    d.demodulate(np.ones(4096, dtype=np.complex64), 1.0)
    batches = d.get_audio()
    assert batches
    assert batches[0].sample_rate == d.decimated_rate


def test_am_emits_audio_at_new_decimated_rate():
    d = AMDemodulator(sample_rate=384_000, audio_rate=48_000)
    d.demodulate(np.ones(8192, dtype=np.complex64), 0.0)
    d.get_audio()
    d.set_sample_rate(192_000)
    d.demodulate(np.ones(4096, dtype=np.complex64), 1.0)
    batches = d.get_audio()
    assert batches
    assert batches[0].sample_rate == d.decimated_rate


def test_wfm_emits_audio_at_new_output_rate():
    d = WidebandFMDemodulator(sample_rate=2_400_000, audio_rate=48_000, rds_enabled=False)
    d.demodulate(np.ones(8192, dtype=np.complex64), 0.0)
    d.get_audio()
    d.set_sample_rate(1_200_000)
    d.demodulate(np.ones(4096, dtype=np.complex64), 1.0)
    batches = d.get_audio()
    assert batches
    assert batches[0].sample_rate == d.output_sample_rate


def test_nfm_set_deviation_updates_scaling():
    d = NarrowbandFMDemodulator(sample_rate=240_000, deviation=5_000)
    assert d.deviation == 5_000
    scale_before = d._fm_discrim._scale
    d.set_deviation(2_500)
    assert d.deviation == 2_500
    assert d._fm_discrim._scale == pytest.approx(2.0 * scale_before)


def test_am_set_sample_rate_clamps_bandwidth_above_nyquist():
    d = AMDemodulator(sample_rate=384_000, audio_rate=48_000, channel_bandwidth=40_000)
    d.channel_bandwidth = 60_000
    d.set_sample_rate(48_000)
    assert d.channel_bandwidth == pytest.approx(48_000 * 0.95)
