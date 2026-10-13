"""The native SoapySDR reader (tsdr.devices.soapy_reader) against a fake
driver: a ctypes callback stands in for `SoapySDRDevice_readStream` and
writes each sample's running index, so continuity is exact."""

import ctypes
import time

import numpy as np
import pytest

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.devices import soapy
from tsdr.devices import soapy_reader as sr
from tsdr.devices.soapy_reader import NativeStreamReader, StreamError

SAMPLE_BYTES = 8  # one int64 index per sample


class FakeDriver:
    """Paced like a radio at `rate`; returns `script` codes before data."""

    def __init__(self, rate: float = 2e6, per_call: int = 4096, script: tuple[int, ...] = ()):
        self.rate = rate
        self.per_call = per_call
        self.script = list(script)
        self.after_script: int | None = None  # code to return forever once the script ends
        self.next_index = 0
        self.calls = 0
        self.block_s = 0.0  # the next call stays in the "driver" this long
        self.blocking = False

    def read(self, buffs: int, n: int) -> int:
        self.calls += 1
        if self.block_s:
            self.blocking = True
            time.sleep(self.block_s)
            self.block_s = 0.0
            self.blocking = False
        if self.script:
            code = self.script.pop(0)
            if code <= 0:
                time.sleep(0.001)
                return code
        if self.after_script is not None:
            time.sleep(0.001)
            return self.after_script
        n = min(n, self.per_call)
        if self.rate:
            time.sleep(n / self.rate)
        addr = ctypes.c_uint64.from_address(buffs).value
        data = np.arange(self.next_index, self.next_index + n, dtype=np.int64)
        ctypes.memmove(addr, data.ctypes.data, n * SAMPLE_BYTES)
        self.next_index += n
        return n


_driver = FakeDriver()


@ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_size_t,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_long,
)
def _fake_read_stream(device, stream, buffs, n, flags, time_ns, timeout_us):  # type: ignore[no-untyped-def]
    return _driver.read(buffs, n)


@pytest.fixture
def driver() -> FakeDriver:
    global _driver
    _driver = FakeDriver()
    return _driver


def _reader(rate: float = 2e6) -> NativeStreamReader:
    return NativeStreamReader(
        _fake_read_stream, 1, 2, sample_bytes=SAMPLE_BYTES, sample_rate=rate, ring_seconds=0.1
    )


def _read(reader: NativeStreamReader, n: int, **kw) -> np.ndarray:
    kw.setdefault("idle_timeout_s", 2.0)
    kw.setdefault("interrupted", lambda: False)
    return np.frombuffer(reader.read(n, **kw), dtype=np.int64)


def test_samples_arrive_in_order_across_the_ring_wrap(driver: FakeDriver) -> None:
    reader = _reader()
    try:
        got = np.concatenate([_read(reader, 100_000) for _ in range(8)])
    finally:
        assert reader.stop()
    assert len(got) > 2 * reader.capacity
    assert np.array_equal(got, np.arange(got[0], got[0] + len(got)))
    assert reader.dropped == 0


def test_a_stalled_consumer_loses_the_oldest_samples(driver: FakeDriver) -> None:
    driver.rate = 20e6
    reader = _reader(rate=2e6)
    try:
        first = _read(reader, 1000)
        time.sleep(0.3)  # the ring holds ~0.13 s at the driver's pace
        after = np.concatenate([_read(reader, 50_000) for _ in range(3)])
    finally:
        assert reader.stop()
    assert reader.dropped > 0
    assert after[0] > first[-1] + 1
    assert np.array_equal(after, np.arange(after[0], after[0] + len(after)))


def test_a_driver_overflow_is_counted_and_reading_goes_on(driver: FakeDriver) -> None:
    driver.script = [sr._SOAPY_SDR_OVERFLOW]
    reader = _reader()
    try:
        assert len(_read(reader, 10_000)) == 10_000
    finally:
        assert reader.stop()
    assert reader.overflows == 1


def test_a_driver_error_ends_the_loop_and_the_read(driver: FakeDriver) -> None:
    driver.script = [-3]
    reader = _reader()
    with pytest.raises(StreamError) as e:
        _read(reader, 10_000)
    assert e.value.code == -3
    assert reader.stop(timeout_s=0.5)


def test_timeouts_keep_the_loop_alive_until_the_read_gives_up(driver: FakeDriver) -> None:
    driver.after_script = sr._SOAPY_SDR_TIMEOUT
    reader = _reader()
    try:
        with pytest.raises(DeviceError, match="timed out"):
            _read(reader, 10_000, idle_timeout_s=0.2)
        assert reader.error is None
    finally:
        assert reader.stop(timeout_s=0.5)


def test_interrupt_ends_a_waiting_read(driver: FakeDriver) -> None:
    driver.after_script = sr._SOAPY_SDR_TIMEOUT
    reader = _reader()
    deadline = time.monotonic() + 0.1
    try:
        with pytest.raises(DeviceError, match="interrupted"):
            _read(reader, 10_000, interrupted=lambda: time.monotonic() > deadline)
    finally:
        assert reader.stop(timeout_s=0.5)


class _Stream(int):
    """A stream handle that, like a SWIG pointer, converts to an address."""


class FakeSoapyDevice:
    def __init__(self) -> None:
        self.this = 1
        self.rate = 1e6
        self.calls: list[str] = []

    def getSampleRate(self, direction, channel):  # noqa: N802 (SoapySDR API name)
        return self.rate

    def setSampleRate(self, direction, channel, rate):  # noqa: N802 (SoapySDR API name)
        self.calls.append("setSampleRate")
        self.rate = rate

    def setFrequency(self, direction, channel, freq):  # noqa: N802 (SoapySDR API name)
        self.calls.append("setFrequency")

    def deactivateStream(self, stream):  # noqa: N802 (SoapySDR API name)
        self.calls.append("deactivateStream")

    def closeStream(self, stream):  # noqa: N802 (SoapySDR API name)
        self.calls.append("closeStream")


def test_soapy_device_reads_through_the_native_reader(
    driver: FakeDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(soapy, "_native_read_stream", lambda: _fake_read_stream)
    dev = soapy.SoapySDRDevice(driver="x", serial="", antenna="", device_args="")
    fake = FakeSoapyDevice()
    dev._device = fake
    dev._stream = _Stream(2)
    raw = dev.read_samples(1000 * dev.get_sample_format().bytes_per_sample)
    assert len(raw) == 1000 * 8
    reader = dev._reader
    assert reader is not None

    dev.set_sample_rate(2e6)
    assert dev._reader is None
    assert not reader._thread.is_alive()

    dev.read_samples(1000 * 8)
    reader = dev._reader
    assert reader is not None
    dev.close()
    assert not reader._thread.is_alive()
    assert fake.calls == ["setSampleRate", "deactivateStream", "closeStream"]


def _native_device(monkeypatch: pytest.MonkeyPatch) -> tuple[soapy.SoapySDRDevice, FakeSoapyDevice]:
    monkeypatch.setattr(soapy, "_native_read_stream", lambda: _fake_read_stream)
    dev = soapy.SoapySDRDevice(driver="x", serial="", antenna="", device_args="")
    fake = FakeSoapyDevice()
    dev._device = fake
    dev._stream = _Stream(2)
    dev.read_samples(1000 * 8)
    return dev, fake


def _wait_until(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_a_reader_stuck_in_the_driver_blocks_rate_changes_until_it_leaves(
    driver: FakeDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, fake = _native_device(monkeypatch)
    monkeypatch.setattr(soapy, "_READER_STOP_TIMEOUT_S", 0.05)
    driver.block_s = 0.5
    _wait_until(lambda: driver.blocking)

    with pytest.raises(DeviceError, match="stuck"):
        dev.set_sample_rate(2e6)
    assert "setSampleRate" not in fake.calls
    with pytest.raises(DeviceError, match="still inside the driver"):
        dev.read_samples(1000 * 8)

    stuck = dev._reader
    assert stuck is not None
    _wait_until(lambda: not stuck.alive)
    assert len(dev.read_samples(1000 * 8)) == 1000 * 8  # a fresh reader
    assert dev._reader is not stuck
    dev.close()
    assert fake.calls == ["deactivateStream", "closeStream"]


def test_closing_with_a_stuck_reader_abandons_the_stream(
    driver: FakeDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, fake = _native_device(monkeypatch)
    monkeypatch.setattr(soapy, "_READER_STOP_TIMEOUT_S", 0.05)
    driver.block_s = 0.3
    _wait_until(lambda: driver.blocking)
    dev.close()
    assert fake.calls == []  # neither deactivated nor closed under the loop
    abandoned = soapy._abandoned_streams.pop()
    assert abandoned[0] is fake
    _wait_until(lambda: not abandoned[2].alive)


def test_a_read_stops_where_the_tag_changes(driver: FakeDriver) -> None:
    reader = _reader()
    try:
        reader.mark(1.0)
        first = _read(reader, 5000)
        time.sleep(0.05)  # a backlog builds up
        reader.mark(2.0)
        boundary = reader._marks[-1][0]
        before = _read(reader, 10 * reader.capacity)
        assert reader.tag == 1.0
        after = _read(reader, 5000)
        assert reader.tag == 2.0
    finally:
        assert reader.stop()
    got = np.concatenate([first, before, after])
    assert np.array_equal(got, np.arange(got[0], got[0] + len(got)))  # nothing dropped
    assert before[-1] == boundary - 1 and after[0] == boundary


def test_a_retune_labels_samples_with_the_frequency_they_were_taken_at(
    driver: FakeDriver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(soapy, "_native_read_stream", lambda: _fake_read_stream)
    dev = soapy.SoapySDRDevice(driver="x", serial="", antenna="", device_args="")
    dev._device = FakeSoapyDevice()
    dev._stream = _Stream(2)
    dev.set_frequency(100e6)
    batches = [(dev.read_center_frequency, dev.read_samples(1000 * 8))]
    time.sleep(0.1)  # a stall: the ring holds samples taken at 100 MHz
    assert dev.read_backlog_samples > 0
    boundary = driver.next_index
    dev.set_frequency(101e6)
    for _ in range(40):
        raw = dev.read_samples(20_000 * 8)
        batches.append((dev.read_center_frequency, raw))
    dev.close()
    labels = [c for c, _ in batches[1:]]
    values = [np.frombuffer(raw, dtype=np.int64) for _, raw in batches]
    got = np.concatenate(values)
    assert np.array_equal(got, np.arange(got[0], got[0] + len(got)))  # nothing dropped
    switch = labels.index(101e6)
    assert set(labels[:switch]) == {100e6} and set(labels[switch:]) == {101e6}
    assert values[switch + 1][0] >= boundary  # nothing old labeled new
