from unittest.mock import MagicMock, patch

from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.core.sdr.workers.io_worker import IOWorker
from tsdr.devices import RTLTCPDevice
from tsdr.tui.commands.sdr.config import SDRConfigCommand


class FakeDevice(RTLTCPDevice):
    """RTLTCPDevice that captures commands instead of sending them."""

    def __init__(self):
        super().__init__("fake", 1234)
        self.commands: list[tuple[int, int]] = []

    def _send_command(self, command: int, parameter: int) -> None:
        self.commands.append((command, parameter))


def test_gain_maps_to_correct_index():
    """Verify rf_gain=40.2 maps to R820T gain index 24."""
    gain = 40.2
    gain_tenths = int(gain * 10)  # 402
    best_idx = 0
    best_diff = abs(RTLTCPDevice.R820T_GAINS[0] - gain_tenths)
    for idx, g in enumerate(RTLTCPDevice.R820T_GAINS):
        diff = abs(g - gain_tenths)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx

    assert best_idx == 22  # index 22 = 402 tenths = 40.2 dB
    assert RTLTCPDevice.R820T_GAINS[best_idx] == 402


def test_config_command_gain_maps_to_rf_gain():
    """The /config --gain flag should map to rf_gain in the changes dict."""
    cmd = SDRConfigCommand()
    mock_engine = MagicMock()

    captured_changes = {}

    def capture(**kwargs):
        captured_changes.update(kwargs)

    mock_engine.update_device_config = lambda dev_id, **kw: capture(**kw)
    mock_engine.update_global_config = lambda **kw: capture(**kw)

    mock_engine.devices = {}
    with patch("tsdr.tui.commands.sdr.config.get_engine", return_value=mock_engine):
        cmd.execute(["--device", "rtl0", "--gain", "40"])

    assert "rf_gain" in captured_changes, f"Expected 'rf_gain' in changes, got: {captured_changes}"
    assert captured_changes["rf_gain"] == 40.0
    assert "gain" not in captured_changes, "'gain' should not be in changes dict"


def test_config_with_changes_rf_gain():
    """with_changes(rf_gain=...) should work."""
    config = DeviceConfig()
    new = config.with_changes(rf_gain=20.0)
    assert new.rf_gain == 20.0


def test_config_with_changes_auto_gain():
    """with_changes(auto_gain=...) should work."""
    config = DeviceConfig()
    new = config.with_changes(auto_gain=True)
    assert new.auto_gain is True


def test_set_gain_sends_manual_mode_then_index():
    """RTLTCPDevice.set_gain() should send gain_mode=manual then gain_index."""
    dev = FakeDevice()
    dev.set_gain(40.2)

    # Should send: SET_GAIN_MODE(1=manual), then SET_GAIN_INDEX(22)
    assert len(dev.commands) == 2
    assert dev.commands[0] == (0x03, 1)  # CMD_SET_GAIN_MODE = manual
    assert dev.commands[1] == (0x0D, 22)  # CMD_SET_GAIN_INDEX = index 22 (40.2 dB)


def test_set_auto_gain_sends_mode_auto():
    """RTLTCPDevice.set_auto_gain(True) should send gain_mode=0 (auto)."""
    dev = FakeDevice()
    dev.set_auto_gain(True)

    assert len(dev.commands) == 1
    assert dev.commands[0] == (0x03, 0)  # CMD_SET_GAIN_MODE = auto


def test_io_worker_sets_gain_on_startup():
    """IO worker setup should call set_gain with config.rf_gain."""
    device = MagicMock()
    device.get_sample_format.return_value = SampleFormat.UINT8_IQ
    device_context = MagicMock()
    device_context.device = device
    device_context.device_id = "test"

    config = DeviceConfig(rf_gain=40.2, auto_gain=False)
    device_context.config = config

    worker = IOWorker(device_context)
    worker_context = MagicMock()

    worker.setup(worker_context)

    device.set_gain.assert_called_once_with(40.2)
    device.set_auto_gain.assert_not_called()


def test_io_worker_sets_agc_on_startup():
    """IO worker setup should call set_auto_gain when auto_gain=True."""
    device = MagicMock()
    device.get_sample_format.return_value = SampleFormat.UINT8_IQ
    device_context = MagicMock()
    device_context.device = device
    device_context.device_id = "test"

    config = DeviceConfig(auto_gain=True)
    device_context.config = config

    worker = IOWorker(device_context)
    worker_context = MagicMock()

    worker.setup(worker_context)

    device.set_auto_gain.assert_called_once_with(True)
    device.set_gain.assert_not_called()


def test_r820t_gain_table_sorted():
    """R820T gain table should be monotonically increasing."""
    gains = RTLTCPDevice.R820T_GAINS
    for i in range(1, len(gains)):
        assert gains[i] >= gains[i - 1], f"Gain table not sorted at index {i}"


def test_r820t_gain_table_length():
    """R820T has 29 gain values (index 0-28)."""
    assert len(RTLTCPDevice.R820T_GAINS) == 29


def test_gain_boundary_values():
    """set_gain should handle boundary values correctly."""
    dev = FakeDevice()

    # Minimum gain (0 dB)
    dev.commands.clear()
    dev.set_gain(0.0)
    assert dev.commands[-1] == (0x0D, 0)  # index 0 = 0 dB

    # Maximum gain (49.6 dB)
    dev.commands.clear()
    dev.set_gain(49.6)
    assert dev.commands[-1] == (0x0D, 28)  # index 28 = 49.6 dB

    # Negative gain should clamp to index 0
    dev.commands.clear()
    dev.set_gain(-10.0)
    assert dev.commands[-1] == (0x0D, 0)

    # Excessive gain should clamp to index 28
    dev.commands.clear()
    dev.set_gain(100.0)
    assert dev.commands[-1] == (0x0D, 28)
