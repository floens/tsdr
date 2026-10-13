"""Read a SoapySDR stream on a native thread, into a ring buffer.

The I/O worker is a Python thread, so a garbage collection (which stops every
thread) or a long GIL hold elsewhere stalls it. Drivers that refill in the
caller's thread, like SoapyPlutoSDR, then lose whatever arrives meanwhile:
the Pluto+ buffers ~100 ms at 10 MS/s, and a collection in the TUI can take
longer. Here a numba `nogil` loop calls the SoapySDR C API
(`SoapySDRDevice_readStream`) and never enters the interpreter, so neither
can stop it; the I/O worker drains the ring instead.

The loop publishes a running sample count after each read. A fence orders
the samples before the count (the reader fences after loading it), which
matters on weakly ordered CPUs such as Apple silicon.
"""

import ctypes
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, cast

import numba as nb
import numpy as np
from numba import types
from numba.extending import intrinsic

from tsdr.core.sdr.exceptions import DeviceError

logger = logging.getLogger(__name__)

_SOAPY_SDR_TIMEOUT = -1
_SOAPY_SDR_OVERFLOW = -4

# Most samples asked of the driver per call; the ring is a whole number of these.
CHUNK = 65536
# Driver timeout per call: how long a stop request can wait.
_CALL_TIMEOUT_US = 100_000

# Slots of the shared state array.
_WRITTEN = 0  # samples published so far
_STOP = 1  # set by Python to end the loop
_OVERFLOWS = 2  # SOAPY_SDR_OVERFLOW returns
_FAILED = 3  # set by the loop when the driver returns an error
_ERROR = 4  # that error code

_READ_STREAM_ARGTYPES = [
    ctypes.c_uint64,  # SoapySDRDevice *device
    ctypes.c_uint64,  # SoapySDRStream *stream
    ctypes.c_uint64,  # void *const *buffs
    ctypes.c_size_t,  # size_t numElems
    ctypes.c_uint64,  # int *flags
    ctypes.c_uint64,  # long long *timeNs
    ctypes.c_long,  # long timeoutUs
]


@intrinsic
def _fence_intrinsic(typingctx):  # type: ignore[no-untyped-def]
    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        builder.fence("seq_cst")
        return context.get_dummy_value()

    return types.void(), codegen


# A memory fence, called with no arguments from compiled code (numba supplies
# the intrinsic's typing context).
_fence = cast(Callable[[], None], _fence_intrinsic)


@nb.njit(cache=True)
def _published(state: np.ndarray) -> int:
    written = state[_WRITTEN]
    _fence()
    return int(written)


_loops: dict[int | None, Any] = {}


def _read_loop(read_stream: Any) -> Any:
    """The native loop, bound to a `SoapySDRDevice_readStream` function.

    Compiled once per function address. Not cached on disk: numba bakes the
    address into the code.
    """
    key = ctypes.cast(read_stream, ctypes.c_void_p).value
    if key not in _loops:
        _loops[key] = _compile_read_loop(read_stream)
    return _loops[key]


def _compile_read_loop(read_stream: Any) -> Any:
    @nb.njit(nogil=True)
    def loop(device, stream, ring, sample_bytes, state, scratch):  # type: ignore[no-untyped-def]
        capacity = ring.size // sample_bytes
        # scratch: [0] buffer pointer, [1] flags, [2] timeNs
        while True:
            _fence()
            if state[_STOP]:
                return
            written = state[_WRITTEN]
            pos = written % capacity
            scratch[0] = ring.ctypes.data + pos * sample_bytes
            ret = read_stream(
                device,
                stream,
                scratch.ctypes.data,
                min(CHUNK, capacity - pos),
                scratch.ctypes.data + 8,
                scratch.ctypes.data + 16,
                _CALL_TIMEOUT_US,
            )
            if ret > 0:
                _fence()
                state[_WRITTEN] = written + ret
            elif ret == _SOAPY_SDR_OVERFLOW:
                state[_OVERFLOWS] += 1
            elif ret != _SOAPY_SDR_TIMEOUT:
                state[_ERROR] = ret
                state[_FAILED] = 1
                return

    return loop


def compile_read_loop(read_stream: Any) -> None:
    """Compile the loop for `read_stream` now rather than on the first read.

    Some drivers stall if the first read after setup comes late: SoapyAirspy
    stops streaming for good when nothing reads for ~0.2 s after the rate is
    set, and compiling takes ~0.5 s.
    """
    state = np.zeros(8, dtype=np.int64)
    state[_STOP] = 1
    ring = np.zeros(8, dtype=np.uint8)
    _read_loop(read_stream)(0, 0, ring, 8, state, np.zeros(4, dtype=np.uint64))


class StreamError(DeviceError):
    """The driver returned an error; `code` is the SoapySDR error code."""

    def __init__(self, code: int):
        super().__init__(f"SoapySDR readStream error {code}")
        self.code = code


def bind_read_stream(library_path: str) -> Any:
    """`SoapySDRDevice_readStream` from the library at `library_path` (or one
    it links, such as the SoapySDR Python extension)."""
    fn = ctypes.CDLL(library_path).SoapySDRDevice_readStream
    fn.argtypes = _READ_STREAM_ARGTYPES
    fn.restype = ctypes.c_int
    return fn


class NativeStreamReader:
    """A native thread that reads one activated stream into a ring.

    `device` and `stream` are the C handles (`SoapySDRDevice *`,
    `SoapySDRStream *`). The ring holds `ring_seconds` at `sample_rate`; a
    consumer that falls further behind than that loses the oldest samples,
    counted in `dropped`.
    """

    def __init__(
        self,
        read_stream: Any,
        device: int,
        stream: int,
        sample_bytes: int,
        sample_rate: float,
        ring_seconds: float = 0.5,
    ):
        self._sample_bytes = sample_bytes
        self._sample_rate = sample_rate
        chunks = max(4, math.ceil(ring_seconds * sample_rate / CHUNK))
        self.capacity = chunks * CHUNK
        self._ring = np.zeros(self.capacity * sample_bytes, dtype=np.uint8)
        self._state = np.zeros(8, dtype=np.int64)
        self._scratch = np.zeros(4, dtype=np.uint64)
        self._consumed = 0
        self.dropped = 0
        # Upcoming tag changes as (first sample, tag), and the current tag.
        self._marks: deque[tuple[int, float]] = deque()
        self._tag: float | None = None
        self._thread = threading.Thread(
            target=_read_loop(read_stream),
            args=(device, stream, self._ring, sample_bytes, self._state, self._scratch),
            name="soapy-native-reader",
            daemon=True,
        )
        self._thread.start()

    @property
    def overflows(self) -> int:
        return int(self._state[_OVERFLOWS])

    @property
    def stopping(self) -> bool:
        return bool(self._state[_STOP])

    @property
    def alive(self) -> bool:
        """The loop is running, or still inside the driver after a stop."""
        return self._thread.is_alive()

    @property
    def backlog(self) -> int:
        """Samples captured but not yet read."""
        return max(0, int(_published(self._state)) - self._consumed)

    @property
    def tag(self) -> float | None:
        """The tag of the samples the last read returned (see mark)."""
        return self._tag

    def mark(self, tag: float) -> None:
        """Samples captured from now on carry `tag` (a new center frequency).

        A read never spans two tags: it returns early at the change. A driver
        call already under way may still deliver up to CHUNK samples taken
        before the change under the new tag.
        """
        self._marks.append((int(_published(self._state)), tag))

    def _enter_marked(self) -> None:
        while self._marks and self._marks[0][0] <= self._consumed:
            self._tag = self._marks.popleft()[1]

    @property
    def error(self) -> int | None:
        """The driver error that ended the loop, or None while it runs."""
        return int(self._state[_ERROR]) if self._state[_FAILED] else None

    def read(self, count: int, *, idle_timeout_s: float, interrupted: Callable[[], bool]) -> bytes:
        """The next `count` samples, as raw bytes in the stream format; fewer
        when a tag change (see mark) comes first.

        Raises DeviceError when the loop stopped on a driver error, when
        nothing arrives for `idle_timeout_s`, or when `interrupted()`.
        """
        sb = self._sample_bytes
        out = np.empty(count * sb, dtype=np.uint8)
        filled = 0
        last_written = -1
        idle_since = time.monotonic()
        while filled < count:
            written = _published(self._state)
            self._skip_overwritten(written)
            self._enter_marked()
            end = min(written, self._marks[0][0]) if self._marks else written
            take = min(end - self._consumed, count - filled)
            if take > 0:
                self._copy(self._consumed, take, out[filled * sb : (filled + take) * sb])
                # The loop kept writing while we copied: if it reached the
                # copied span, the copy may be torn.
                if _published(self._state) + CHUNK - self.capacity > self._consumed:
                    continue
                self._consumed += take
                filled += take
                if self._marks and self._consumed >= self._marks[0][0]:
                    break  # the tag changes here: return this tag's samples
                continue
            if self.error is not None:
                raise StreamError(self.error)
            if interrupted():
                raise DeviceError("SoapySDR read interrupted")
            now = time.monotonic()
            if written != last_written:
                last_written = written
                idle_since = now
            elif now - idle_since > idle_timeout_s:
                raise DeviceError("SoapySDR readStream timed out")
            missing = count - filled
            time.sleep(min(0.02, max(0.001, missing / self._sample_rate)))
        return out[: filled * sb].tobytes()

    def _skip_overwritten(self, written: int) -> None:
        # The loop may be writing anywhere in [written, written + CHUNK).
        lag = written - self._consumed
        if lag > self.capacity - CHUNK:
            keep = self.capacity - 2 * CHUNK
            self.dropped += lag - keep
            self._consumed = written - keep

    def _copy(self, start: int, n: int, dest: np.ndarray) -> None:
        sb = self._sample_bytes
        pos = start % self.capacity
        first = min(n, self.capacity - pos)
        dest[: first * sb] = self._ring[pos * sb : (pos + first) * sb]
        if first < n:
            dest[first * sb :] = self._ring[: (n - first) * sb]

    def stop(self, timeout_s: float = 5.0) -> bool:
        """End the loop; False if it is still inside the driver."""
        self._state[_STOP] = 1
        self._thread.join(timeout_s)
        return not self._thread.is_alive()
