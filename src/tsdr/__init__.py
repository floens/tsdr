import sys

print("Starting TSDR (this may take a moment on first run)...", file=sys.stderr, flush=True)

from tsdr.tui import TSDRApp  # noqa: E402

__version__ = "0.1.0"
__all__ = ["__version__", "TSDRApp"]
