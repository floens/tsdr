"""Standalone GUI process for debug visualization.

Reads pickled VizRequest objects from stdin (length-prefixed) and displays them.
"""

import os
import pickle
import sys
from typing import Any

from PyQt6.QtCore import QSocketNotifier
from PyQt6.QtWidgets import QApplication

from ._core import VizRequest, VizType
from ._visualizations import (
    ConstellationWindow,
    DebugWindow,
    FFTWindow,
    HexWindow,
    ScatterWindow,
    TimeSeriesWindow,
)

windows: dict[str, DebugWindow] = {}
_window_count = 0  # For cascade positioning


def position_window(window: DebugWindow) -> None:
    """Position window in a 4x3 grid."""
    global _window_count

    screen = QApplication.primaryScreen()
    if screen is None:
        return

    cols, rows = 4, 3
    title_bar_height = 36
    geometry = screen.availableGeometry()

    cell_width = geometry.width() // cols
    cell_height = geometry.height() // rows

    slot = _window_count % (cols * rows)
    col = slot % cols
    row = slot // cols

    x = geometry.x() + col * cell_width
    y = geometry.y() + row * cell_height

    window.move(x, y)
    window.resize(cell_width, cell_height - title_bar_height)
    _window_count += 1


def create_window(name: str, viz_type: VizType, kwargs: dict[str, Any] | None) -> DebugWindow:
    kwargs = kwargs or {}
    match viz_type:
        case VizType.FFT:
            return FFTWindow(name)
        case VizType.CONSTELLATION:
            return ConstellationWindow(name)
        case VizType.HEX:
            return HexWindow(name, bytes_per_row=kwargs.get("bytes_per_row", 16))
        case VizType.PLOT:
            return TimeSeriesWindow(name)
        case VizType.SCATTER:
            return ScatterWindow(name)
        case _:
            raise ValueError(f"Unknown visualization type: {viz_type}")


def handle_request(request: VizRequest) -> None:
    if request.viz_type == VizType.CLOSE:
        if request.name == "__all__":
            for window in list(windows.values()):
                window.close()
            windows.clear()
        elif request.name in windows:
            windows[request.name].close()
            del windows[request.name]
        return

    # Get or create window
    if request.name not in windows:
        window = create_window(request.name, request.viz_type, request.kwargs)
        windows[request.name] = window
        position_window(window)
        window.show()

    window = windows[request.name]

    # Update data
    if request.data is not None:
        kwargs = request.kwargs or {}
        match request.viz_type:
            case VizType.FFT:
                window.update_data(
                    request.data,
                    sample_rate=kwargs.get("sample_rate", 1.0),
                    center_freq=kwargs.get("center_freq", 0.0),
                )
            case VizType.CONSTELLATION | VizType.HEX | VizType.PLOT | VizType.SCATTER:
                window.update_data(request.data)


def main() -> None:
    app = QApplication([])

    # Set stdin to non-blocking mode to prevent read() from blocking
    # when QSocketNotifier fires but less than requested bytes available
    os.set_blocking(sys.stdin.fileno(), False)

    # Buffer for reading from stdin
    buffer = bytearray()
    expected_len = 0

    def read_stdin() -> None:
        nonlocal buffer, expected_len

        try:
            data = sys.stdin.buffer.read(4096)
            if data is None:
                # No data available (non-blocking mode returns None)
                return
            if len(data) == 0:
                # EOF - stdin closed
                app.quit()
                return
            buffer.extend(data)
        except BlockingIOError:
            # No data available yet
            return
        except OSError:
            return

        # Process complete messages
        while True:
            if expected_len == 0:
                if len(buffer) >= 4:
                    expected_len = int.from_bytes(buffer[:4], "little")
                    buffer = buffer[4:]
                else:
                    break

            if len(buffer) >= expected_len:
                msg_data = bytes(buffer[:expected_len])
                buffer = buffer[expected_len:]
                expected_len = 0

                try:
                    request = pickle.loads(msg_data)
                    handle_request(request)
                except pickle.UnpicklingError, ValueError, KeyError:
                    pass
            else:
                break

    # Use QSocketNotifier to watch stdin
    notifier = QSocketNotifier(sys.stdin.fileno(), QSocketNotifier.Type.Read)
    notifier.activated.connect(read_stdin)

    app.exec()


if __name__ == "__main__":
    main()
