import numpy as np


class CircularBuffer:
    """Circular buffer for accumulating IQ samples.

    Used for FFT window accumulation. Samples are appended and the buffer
    wraps around when full.
    """

    def __init__(self, size: int, dtype=np.complex64):
        self._buffer = np.zeros(size, dtype=dtype)
        self._size = size
        self._write_pos = 0
        self._available = 0
        self._dtype = dtype

    def append(self, samples: np.ndarray) -> None:
        n_samples = len(samples)

        if n_samples >= self._size:
            # Input larger than buffer: keep only the last _size samples.
            self._buffer[:] = samples[-self._size :]
            self._write_pos = 0
            self._available = self._size
            return

        space_before_wrap = self._size - self._write_pos

        if n_samples <= space_before_wrap:
            self._buffer[self._write_pos : self._write_pos + n_samples] = samples
            self._write_pos = (self._write_pos + n_samples) % self._size
        else:
            first_chunk = space_before_wrap
            second_chunk = n_samples - first_chunk
            self._buffer[self._write_pos :] = samples[:first_chunk]
            self._buffer[:second_chunk] = samples[first_chunk:]
            self._write_pos = second_chunk

        self._available = min(self._available + n_samples, self._size)

    def get_window(self, window_size: int) -> np.ndarray:
        """Return the most recent `window_size` samples, zero-padded if short."""
        if window_size > self._size:
            raise ValueError(f"Window size {window_size} exceeds buffer size {self._size}")

        if self._available < window_size:
            result = np.zeros(window_size, dtype=self._dtype)
            if self._available > 0:
                result[-self._available :] = self._get_recent(self._available)
            return result

        return self._get_recent(window_size)

    def _get_recent(self, count: int) -> np.ndarray:
        if count > self._available:
            count = self._available

        if self._available < self._size:
            start_pos = 0
        else:
            start_pos = (self._write_pos - count + self._size) % self._size

        if start_pos + count <= self._size:
            # No wrap -- return view (caller must not hold across append() calls)
            return self._buffer[start_pos : start_pos + count]
        else:
            first_chunk = self._size - start_pos
            second_chunk = count - first_chunk
            result = np.empty(count, dtype=self._dtype)
            result[:first_chunk] = self._buffer[start_pos:]
            result[first_chunk:] = self._buffer[:second_chunk]
            return result

    @property
    def size(self) -> int:
        return self._available

    @property
    def capacity(self) -> int:
        return self._size

    def clear(self) -> None:
        self._write_pos = 0
        self._available = 0
        self._buffer.fill(0)

    def __len__(self) -> int:
        return self._available

    def __str__(self) -> str:
        return f"CircularBuffer(size={self._available}/{self._size})"
