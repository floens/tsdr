import sys
import types

import numpy as np

# pyfftw.interfaces probes scipy at import time -- block it since we only
# need the core FFTW class and don't use the numpy/scipy wrappers.
sys.modules.setdefault("pyfftw.interfaces", types.ModuleType("pyfftw.interfaces"))
import pyfftw  # noqa: E402


class FFTPlan:
    """Pre-configured FFTW plan for forward complex FFT.

    Create once per FFT size, call ``execute()`` on every frame.
    FFTW plans encode the optimal algorithm for a given size and
    hardware -- reusing the plan avoids repeated planning overhead.
    """

    def __init__(self, size: int):
        self.size = size
        self._input = pyfftw.empty_aligned(size, dtype="complex64")
        self._output = pyfftw.empty_aligned(size, dtype="complex64")
        self._plan = pyfftw.FFTW(
            self._input,
            self._output,
            direction="FFTW_FORWARD",
            flags=["FFTW_MEASURE"],
        )

    def execute(self, windowed_samples: np.ndarray) -> np.ndarray:
        """Run FFT on pre-windowed samples. Returns complex64 result.

        The returned array is owned by this plan -- do not hold references
        across calls.
        """
        self._input[:] = windowed_samples
        self._plan()
        result: np.ndarray = self._output
        return result
