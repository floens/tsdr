"""Core Qt infrastructure for debug visualization.

Uses a separate process for Qt since QApplication must run on the main thread.
Uses subprocess + pickle over stdin to avoid multiprocessing fd issues.
"""

import atexit
import pickle
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VizType(Enum):
    """Visualization types."""

    FFT = auto()
    CONSTELLATION = auto()
    HEX = auto()
    PLOT = auto()
    SCATTER = auto()
    CLOSE = auto()


@dataclass
class VizRequest:
    """Request to update or create a visualization."""

    name: str
    viz_type: VizType
    data: NDArray | bytes | None = None
    kwargs: dict[str, Any] | None = None


# Global state
_gui_process: subprocess.Popen | None = None
_initialized = False


def _ensure_initialized() -> None:
    """Ensure the GUI process is running."""
    global _gui_process, _initialized

    if _initialized:
        return

    # Launch subprocess as a module so relative imports work
    _gui_process = subprocess.Popen(
        [sys.executable, "-m", "tsdr.debug._gui_process"],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    _initialized = True


def _cleanup() -> None:
    """Cleanup on exit."""
    global _gui_process

    if _gui_process is not None:
        try:
            if _gui_process.stdin:
                _gui_process.stdin.close()
            _gui_process.terminate()
            _gui_process.wait(timeout=0.5)
        except Exception:  # noqa: BLE001
            pass


atexit.register(_cleanup)


def submit_request(request: VizRequest) -> None:
    """Submit a visualization request (thread-safe)."""
    _ensure_initialized()
    if _gui_process is not None and _gui_process.stdin is not None:
        try:
            data = pickle.dumps(request)
            # Length-prefixed message
            _gui_process.stdin.write(len(data).to_bytes(4, "little"))
            _gui_process.stdin.write(data)
            _gui_process.stdin.flush()
        except BrokenPipeError, OSError:
            pass
