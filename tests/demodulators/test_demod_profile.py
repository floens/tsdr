"""DemodProfile (structural, spec-derived) and DemodStatus (dynamic) tests.

`demod_profile()` is the single source of a demod's structural fields, derived
from the mode + config without building an instance. `status()` carries only
the runtime fields.
"""

import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.radio.registry import DEMODULATORS, demod_profile, make_demodulator


def test_sideband_from_mode() -> None:
    assert demod_profile("USB").sideband == "upper"
    assert demod_profile("LSB").sideband == "lower"
    assert demod_profile("SSTV").sideband == "upper"
    assert demod_profile("FT8").sideband == "upper"
    assert demod_profile("NFM").sideband is None
    assert demod_profile("AM").sideband is None


def test_message_type_from_mode() -> None:
    assert demod_profile("WFM").message_type == "rds"
    assert demod_profile("CW").message_type == "text"
    assert demod_profile("ADSB").message_type == "adsb"
    assert demod_profile("DAB").message_type == "dab"
    assert demod_profile("DMR").message_type == "dmr"
    assert demod_profile("TETRA").message_type == "tetra"
    assert demod_profile("SSTV").message_type == "sstv"
    assert demod_profile("NFM").message_type is None


def test_has_audio_from_mode() -> None:
    assert demod_profile("NFM").has_audio is True
    assert demod_profile("WFM").has_audio is True
    assert demod_profile("ADSB").has_audio is False
    assert demod_profile("FLEX").has_audio is False


def test_channel_bandwidth_fixed_vs_config() -> None:
    # Protocol decoders are spec-locked and ignore the config bandwidth.
    assert demod_profile("ADSB", channel_bandwidth=99_999).channel_bandwidth == 2_000_000
    assert demod_profile("DAB").channel_bandwidth == 1_536_000
    # Audio demods take the config bandwidth, falling back to the mode default.
    assert demod_profile("NFM", channel_bandwidth=15_000).channel_bandwidth == 15_000
    assert demod_profile("NFM").channel_bandwidth == 12_500


def test_sample_rate_fixed_for_spec_locked_decoders() -> None:
    # Only spec-locked decoders report a required rate (drives the tuner's
    # "Needs X" warning); rate-flexible demods leave it unset.
    assert demod_profile("ADSB").sample_rate == 2_400_000
    assert demod_profile("DAB").sample_rate == 2_048_000
    assert demod_profile("FT8").sample_rate is None
    assert demod_profile("NFM").sample_rate is None


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown demodulator mode"):
        demod_profile("BOGUS")


def test_every_registered_mode_has_a_profile() -> None:
    for mode in DEMODULATORS:
        profile = demod_profile(mode)
        assert profile.label, f"{mode} has empty label"
        assert profile.modulation, f"{mode} has empty modulation"
        assert profile.channel_bandwidth > 0


def test_nfm_status_reports_squelch() -> None:
    demod = make_demodulator(DemodSpec(mode="NFM"), 240_000)
    demod.set_squelch(enabled=True, threshold_db=-42.0, hang_ms=100.0)
    status = demod.status()
    assert status.squelch_open is not None
    assert status.squelch_threshold_db == pytest.approx(-42.0)


def test_wfm_status_reports_pilot_quality() -> None:
    demod = make_demodulator(DemodSpec(mode="WFM"), 2_400_000)
    status = demod.status()
    assert status.quality_label is not None
    assert status.quality_label.startswith("Pilot")
