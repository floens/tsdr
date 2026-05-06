"""Debug visualization module for real-time numpy array inspection.

This module provides a simple API for visualizing numpy arrays during development.
Windows are persistent and update in-place. All functions are thread-safe.

Usage:
    from tsdr.debug import viz

    # FFT / frequency domain
    viz.fft("spectrum", fft_data, sample_rate=2.4e6, center_freq=100e6)

    # Constellation diagram (IQ phase)
    viz.constellation("iq", iq_samples)

    # Hex dump
    viz.hex("bytes", raw_bytes)

    # Generic line plot
    viz.plot("signal", data)

    # Close windows
    viz.close("spectrum")
    viz.close_all()
"""

from typing import TYPE_CHECKING

from ._core import VizRequest, VizType, submit_request

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def fft(
    name: str,
    data: NDArray[np.floating],
    *,
    sample_rate: float = 1.0,
    center_freq: float = 0.0,
) -> None:
    """Display FFT / frequency domain visualization.

    Args:
        name: Window identifier (creates new window or updates existing)
        data: Power spectrum data in dB
        sample_rate: Sample rate in Hz (for frequency axis)
        center_freq: Center frequency in Hz (for frequency axis offset)
    """
    submit_request(
        VizRequest(
            name=name,
            viz_type=VizType.FFT,
            data=data,
            kwargs={"sample_rate": sample_rate, "center_freq": center_freq},
        )
    )


def constellation(name: str, iq_data: NDArray[np.complexfloating]) -> None:
    """Display IQ constellation diagram.

    Args:
        name: Window identifier
        iq_data: Complex IQ samples
    """
    submit_request(
        VizRequest(
            name=name,
            viz_type=VizType.CONSTELLATION,
            data=iq_data,
        )
    )


def hex(
    name: str,
    data: bytes | NDArray,
    *,
    bytes_per_row: int = 16,
) -> None:
    """Display hex dump visualization.

    Args:
        name: Window identifier
        data: Bytes or numpy array to display
        bytes_per_row: Number of bytes per row (default 16)
    """
    submit_request(
        VizRequest(
            name=name,
            viz_type=VizType.HEX,
            data=data,
            kwargs={"bytes_per_row": bytes_per_row},
        )
    )


def plot(name: str, data: NDArray[np.floating]) -> None:
    """Display generic line plot.

    Args:
        name: Window identifier
        data: 1D array of values to plot
    """
    submit_request(
        VizRequest(
            name=name,
            viz_type=VizType.PLOT,
            data=data,
        )
    )


def scatter(name: str, data: NDArray[np.floating]) -> None:
    """Display scatter plot (dots, no lines). X = index, Y = value.

    Args:
        name: Window identifier
        data: 1D array of values to plot as dots
    """
    submit_request(
        VizRequest(
            name=name,
            viz_type=VizType.SCATTER,
            data=data,
        )
    )


def close(name: str) -> None:
    """Close a visualization window.

    Args:
        name: Window identifier to close
    """
    submit_request(VizRequest(name=name, viz_type=VizType.CLOSE))


def close_all() -> None:
    """Close all visualization windows."""
    submit_request(VizRequest(name="__all__", viz_type=VizType.CLOSE))


# Namespace object for viz.fft() style calls
class _VizNamespace:
    """Namespace for viz.fft() style API."""

    fft = staticmethod(fft)
    constellation = staticmethod(constellation)
    hex = staticmethod(hex)
    plot = staticmethod(plot)
    scatter = staticmethod(scatter)
    close = staticmethod(close)
    close_all = staticmethod(close_all)


viz = _VizNamespace()

__all__ = ["viz", "fft", "constellation", "hex", "plot", "scatter", "close", "close_all"]
