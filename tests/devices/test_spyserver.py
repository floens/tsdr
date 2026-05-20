import socket

import numpy as np
import pytest

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.devices.spyserver import (
    IqFormat,
    MsgType,
    SpyServerDevice,
    _ClientSync,
    _DeviceInfo,
    _SpyServerCodec,
)


def _make_device_info(
    *,
    device_type: int = 3,  # RTLSDR
    serial: int = 0xDEADBEEF,
    max_sample_rate: int = 2_400_000,
    decimation_stage_count: int = 3,
    min_iq_decimation: int = 0,
    max_gain_index: int = 28,
    min_freq: int = 24_000_000,
    max_freq: int = 1_766_000_000,
    forced_iq_format: int = 0,
) -> _DeviceInfo:
    return _DeviceInfo(
        device_type=device_type,
        serial=serial,
        max_sample_rate=max_sample_rate,
        max_bandwidth=0,
        decimation_stage_count=decimation_stage_count,
        gain_stage_count=1,
        max_gain_index=max_gain_index,
        min_freq=min_freq,
        max_freq=max_freq,
        resolution=1,
        min_iq_decimation=min_iq_decimation,
        forced_iq_format=forced_iq_format,
    )


def _make_client_sync(
    *,
    can_control: bool = True,
    min_iq_freq: int = 0,
    max_iq_freq: int = 0,
) -> _ClientSync:
    return _ClientSync(
        can_control=can_control,
        current_gain=0,
        device_center_freq=100_000_000,
        iq_center_freq=100_000_000,
        fft_center_freq=100_000_000,
        min_iq_freq=min_iq_freq,
        max_iq_freq=max_iq_freq,
        min_fft_freq=0,
        max_fft_freq=0,
    )


def _device() -> SpyServerDevice:
    return SpyServerDevice(host="localhost", port=5555)


def test_device_info_populates_identity_and_capabilities():
    dev = _device()
    dev._apply_device_info(_make_device_info())
    dev._apply_client_sync(_make_client_sync(can_control=True))

    assert dev.identity.type_label == "RTLSDR"
    assert dev.identity.serial == "0xdeadbeef"

    caps = dev.capabilities
    assert caps.frequency_range == (24_000_000.0, 1_766_000_000.0)
    assert caps.sample_rates == (2_400_000.0, 1_200_000.0, 600_000.0, 300_000.0)
    assert caps.gain_supported is True
    assert caps.gain_range == (0.0, 28.0)
    assert caps.gain_step == 1.0
    assert caps.gain_unit == "index"
    assert caps.bias_tee_supported is False


def test_client_sync_flips_gain_supported():
    dev = _device()
    dev._apply_device_info(_make_device_info())
    dev._apply_client_sync(_make_client_sync(can_control=True))
    assert dev.capabilities.gain_supported is True

    dev._apply_client_sync(_make_client_sync(can_control=False))
    assert dev.capabilities.gain_supported is False


def test_client_sync_keeps_device_range_when_controllable():
    dev = _device()
    dev._apply_device_info(_make_device_info())
    dev._apply_client_sync(
        _make_client_sync(can_control=True, min_iq_freq=88_000_000, max_iq_freq=108_000_000)
    )
    assert dev.capabilities.frequency_range == (24_000_000.0, 1_766_000_000.0)
    assert dev.capabilities.frequency_controllable is True


def test_client_sync_narrows_to_iq_window_when_locked():
    dev = _device()
    dev._apply_device_info(_make_device_info())
    dev._apply_client_sync(
        _make_client_sync(can_control=False, min_iq_freq=88_000_000, max_iq_freq=108_000_000)
    )
    assert dev.capabilities.frequency_range == (88_000_000.0, 108_000_000.0)
    assert dev.capabilities.frequency_controllable is False


def test_freq_range_re_widens_when_control_returns():
    dev = _device()
    dev._apply_device_info(_make_device_info())
    dev._apply_client_sync(
        _make_client_sync(can_control=False, min_iq_freq=88_000_000, max_iq_freq=108_000_000)
    )
    assert dev.capabilities.frequency_range == (88_000_000.0, 108_000_000.0)

    dev._apply_client_sync(
        _make_client_sync(can_control=True, min_iq_freq=88_000_000, max_iq_freq=108_000_000)
    )
    assert dev.capabilities.frequency_range == (24_000_000.0, 1_766_000_000.0)
    assert dev.capabilities.frequency_controllable is True


def test_min_iq_decimation_truncates_sample_rate_list():
    dev = _device()
    dev._apply_device_info(_make_device_info(min_iq_decimation=1, decimation_stage_count=3))
    dev._apply_client_sync(_make_client_sync(can_control=True))
    assert dev.capabilities.sample_rates == (1_200_000.0, 600_000.0, 300_000.0)


def test_capabilities_swap_yields_new_object_identity():
    dev = _device()
    dev._apply_device_info(_make_device_info())
    dev._apply_client_sync(_make_client_sync(can_control=True))
    caps_before = dev.capabilities

    dev._apply_client_sync(_make_client_sync(can_control=False))
    caps_after = dev.capabilities

    assert caps_before is not caps_after
    assert caps_before != caps_after


def _encode_int24_le(values: list[int]) -> bytes:
    out = bytearray()
    for v in values:
        u = v & 0xFFFFFF
        out.append(u & 0xFF)
        out.append((u >> 8) & 0xFF)
        out.append((u >> 16) & 0xFF)
    return bytes(out)


def test_decode_int24_iq_roundtrip():
    a, b = socket.socketpair()
    try:
        codec = _SpyServerCodec(a)
        values = [0, 1, -1, 2, -2, 0x7FFFFF, -0x800000, 12345]
        body = _encode_int24_le(values)
        out = codec.decode_iq(MsgType.INT24_IQ, mflags=0, body=body)
        floats = np.frombuffer(out, dtype=np.complex64).view(np.float32)
        expected = np.array(values, dtype=np.float32) / np.float32(8388608.0)
        np.testing.assert_allclose(floats, expected, rtol=0, atol=1e-9)
    finally:
        a.close()
        b.close()


def test_decode_int24_iq_drops_trailing_partial_pair():
    a, b = socket.socketpair()
    try:
        codec = _SpyServerCodec(a)
        body = _encode_int24_le([10, 20, 30])
        out = codec.decode_iq(MsgType.INT24_IQ, mflags=0, body=body)
        assert len(out) == 8
        floats = np.frombuffer(out, dtype=np.complex64).view(np.float32)
        expected = np.array([10, 20], dtype=np.float32) / np.float32(8388608.0)
        np.testing.assert_allclose(floats, expected, rtol=0, atol=1e-9)
    finally:
        a.close()
        b.close()


def test_forced_iq_format_int24_accepted():
    dev = _device()
    dev._apply_device_info(_make_device_info(forced_iq_format=IqFormat.INT24))
    assert dev._iq_format == IqFormat.INT24


def test_forced_iq_format_dint4_rejected():
    dev = _device()
    with pytest.raises(DeviceError, match="DINT4"):
        dev._apply_device_info(_make_device_info(forced_iq_format=IqFormat.DINT4))
