from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QMainWindow, QPlainTextEdit

from ._stats import StatsOverlay

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DebugWindow(QMainWindow):
    """Base class for debug visualization windows."""

    def __init__(self, key: str):
        super().__init__()
        self.key = key
        self.stats = StatsOverlay()
        self.setWindowTitle(key)
        self.resize(800, 600)

    def update_title(self) -> None:
        """Update window title with stats."""
        stats_text = self.stats.get_stats_text()
        self.setWindowTitle(f"{self.key} | {stats_text}")


class FFTWindow(DebugWindow):
    """Window for FFT/frequency domain visualization."""

    def __init__(self, key: str):
        super().__init__(key)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("k")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Power", units="dB")
        self.plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self.plot_widget.disableAutoRange()

        self.curve = self.plot_widget.plot(pen=pg.mkPen("y", width=1))

        # Tracked axis limits (only expand, never shrink)
        self.x_min: float | None = None
        self.x_max: float | None = None
        self.y_min: float | None = None
        self.y_max: float | None = None

        self.setCentralWidget(self.plot_widget)

    def _update_range(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        """Update axis range, only expanding."""
        if self.x_min is None or self.x_max is None or self.y_min is None or self.y_max is None:
            self.x_min, self.x_max = x_min, x_max
            self.y_min, self.y_max = y_min, y_max
        else:
            self.x_min = min(self.x_min, x_min)
            self.x_max = max(self.x_max, x_max)
            self.y_min = min(self.y_min, y_min)
            self.y_max = max(self.y_max, y_max)

        self.plot_widget.setXRange(self.x_min, self.x_max, padding=0)
        self.plot_widget.setYRange(self.y_min, self.y_max, padding=0.02)

    def update_data(
        self,
        data: NDArray[np.floating],
        sample_rate: float = 1.0,
        center_freq: float = 0.0,
    ) -> None:
        """Update the FFT plot with new data."""
        n = len(data)
        freqs = np.fft.fftshift(np.fft.fftfreq(n, 1 / sample_rate)) + center_freq
        self.curve.setData(freqs, data)

        self._update_range(freqs.min(), freqs.max(), data.min(), data.max())

        self.stats.record_update(n)
        self.update_title()


class ConstellationWindow(DebugWindow):
    """Window for IQ constellation diagram."""

    MAX_POINTS = 10000

    def __init__(self, key: str):
        super().__init__(key)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("k")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Q (Imaginary)")
        self.plot_widget.setLabel("bottom", "I (Real)")
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.disableAutoRange()

        self.scatter = pg.ScatterPlotItem(
            pen=None,
            brush=pg.mkBrush(100, 200, 255, 120),
            size=3,
        )
        self.plot_widget.addItem(self.scatter)

        # Tracked axis limit (symmetric for constellation)
        self.axis_limit: float | None = None

        self.setCentralWidget(self.plot_widget)

    def _update_range(self, max_val: float) -> None:
        """Update axis range symmetrically, only expanding."""
        if self.axis_limit is None:
            self.axis_limit = max_val
        else:
            self.axis_limit = max(self.axis_limit, max_val)

        self.plot_widget.setXRange(-self.axis_limit, self.axis_limit, padding=0.02)
        self.plot_widget.setYRange(-self.axis_limit, self.axis_limit, padding=0.02)

    def update_data(self, iq_data: NDArray[np.complexfloating]) -> None:
        """Update the constellation plot with new IQ data."""
        n = len(iq_data)

        # Downsample for performance
        if n > self.MAX_POINTS:
            step = n // self.MAX_POINTS
            iq_data = iq_data[::step]

        self.scatter.setData(iq_data.real, iq_data.imag)

        max_val = max(np.abs(iq_data.real).max(), np.abs(iq_data.imag).max())
        self._update_range(max_val)

        self.stats.record_update(n)
        self.update_title()


class HexWindow(DebugWindow):
    """Window for hex dump visualization of bytes."""

    def __init__(self, key: str, bytes_per_row: int = 16):
        super().__init__(key)
        self.bytes_per_row = bytes_per_row

        self.text_widget = QPlainTextEdit()
        self.text_widget.setFont(QFont("Courier", 10))
        self.text_widget.setReadOnly(True)
        self.text_widget.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        # Dark theme
        self.text_widget.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; }"
        )

        self.setCentralWidget(self.text_widget)

    def update_data(self, data: bytes | NDArray) -> None:
        """Update the hex display with new data."""
        if isinstance(data, np.ndarray):
            data = data.tobytes()

        n = len(data)
        lines = []

        for offset in range(0, n, self.bytes_per_row):
            chunk = data[offset : offset + self.bytes_per_row]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            # Pad hex part for alignment
            hex_width = self.bytes_per_row * 3 - 1
            lines.append(f"{offset:08x}  {hex_part:<{hex_width}}  {ascii_part}")

        self.text_widget.setPlainText("\n".join(lines))
        self.stats.record_update(n)
        self.update_title()


class TimeSeriesWindow(DebugWindow):
    """Window for generic time series / line plot visualization."""

    def __init__(self, key: str):
        super().__init__(key)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("k")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.setLabel("bottom", "Sample")
        self.plot_widget.disableAutoRange()

        self.curve = self.plot_widget.plot(pen=pg.mkPen("c", width=1))

        # Tracked axis limits (only expand, never shrink)
        self.x_max: int = 0
        self.y_min: float | None = None
        self.y_max: float | None = None

        self.setCentralWidget(self.plot_widget)

    def _update_range(self, x_max: int, y_min: float, y_max: float) -> None:
        """Update axis range, only expanding."""
        self.x_max = max(self.x_max, x_max)

        if self.y_min is None or self.y_max is None:
            self.y_min, self.y_max = y_min, y_max
        else:
            self.y_min = min(self.y_min, y_min)
            self.y_max = max(self.y_max, y_max)

        self.plot_widget.setXRange(0, self.x_max, padding=0)
        self.plot_widget.setYRange(self.y_min, self.y_max, padding=0.02)

    def update_data(self, data: NDArray[np.floating]) -> None:
        """Update the plot with new data."""
        self.curve.setData(data)

        self._update_range(len(data), data.min(), data.max())

        self.stats.record_update(len(data))
        self.update_title()


class ScatterWindow(DebugWindow):
    """Window for scatter plot visualization (dots, no lines)."""

    MAX_POINTS = 10000

    def __init__(self, key: str):
        super().__init__(key)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("k")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.setLabel("bottom", "Index")
        self.plot_widget.disableAutoRange()

        self.scatter = pg.ScatterPlotItem(
            pen=None,
            brush=pg.mkBrush(100, 200, 255, 120),
            size=3,
        )
        self.plot_widget.addItem(self.scatter)

        self.x_max: int = 0
        self.y_min: float | None = None
        self.y_max: float | None = None

        self.setCentralWidget(self.plot_widget)

    def _update_range(self, x_max: int, y_min: float, y_max: float) -> None:
        # Reset range each update - each burst is independent
        self.plot_widget.setXRange(0, x_max, padding=0.02)
        self.plot_widget.setYRange(y_min, y_max, padding=0.05)

    def update_data(self, data: NDArray[np.floating]) -> None:
        """Update scatter plot with new data. X = index, Y = value."""
        n = len(data)

        if n > self.MAX_POINTS:
            step = n // self.MAX_POINTS
            data = data[::step]
            n = len(data)

        x = np.arange(n, dtype=np.float32)
        self.scatter.setData(x, data.astype(np.float32))

        self._update_range(n, float(data.min()), float(data.max()))

        self.stats.record_update(n)
        self.update_title()
