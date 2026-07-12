import pytest

from tsdr.core.devices import PersistedDevice
from tsdr.core.preferences import _build_params, _persist_params
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices.factory import create_device
from tsdr.devices.iq_file import IQFileParams
from tsdr.devices.registry import DEVICE_TYPE_NAMES, DEVICE_TYPES, DeviceType
from tsdr.tui.commands.base import CommandParser
from tsdr.tui.commands.sdr.add import SDRAddCommand


def _sample_params(dt: DeviceType):
    """A constructible Params for the type. iq-file needs a real path + format so
    the factory doesn't try to auto-detect; everything else uses defaults (network
    defaults ports are valid)."""
    if dt.params_cls is IQFileParams:
        return IQFileParams(path="rec.cu8", sample_format=SampleFormat.UINT8_IQ)
    return dt.params_cls()


@pytest.mark.parametrize("dt", DEVICE_TYPES, ids=lambda d: d.name)
def test_device_type_round_trips_through_persistence(dt: DeviceType):
    """Every registered device must survive persist -> restore. A forgotten
    registry entry (the class of bug that dropped KiwiSDR on restart) fails here."""
    params = _sample_params(dt)
    persisted = PersistedDevice(id="d", type=dt.name, **_persist_params(params))
    assert _build_params(persisted) == params


@pytest.mark.parametrize("dt", DEVICE_TYPES, ids=lambda d: d.name)
def test_factory_builds_every_registry_type(dt: DeviceType):
    """create_device must have a case for every registry params class (no `case _`)."""
    assert create_device(_sample_params(dt)) is not None


def test_add_type_choices_match_registry():
    parser = CommandParser(add_help=False)
    SDRAddCommand().configure(parser)
    type_action = next(a for a in parser._actions if a.dest == "device_type")
    assert tuple(type_action.choices) == DEVICE_TYPE_NAMES
