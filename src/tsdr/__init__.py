import platform
import sys
import types

print("Starting TSDR (this may take a moment on first run)...", file=sys.stderr, flush=True)


def _install_windows_stubs() -> None:
    """Stub POSIX-only stdlib modules the TUI imports for terminal features.

    core.platform (TTY pixel size) and tui.tty (Textual's Linux driver) import
    fcntl/termios/tty unconditionally. None of it runs under Textual's Windows
    driver, so inert stubs let the imports succeed and the features no-op.
    """

    def _ioctl_unavailable(*_args, **_kwargs) -> None:
        raise OSError("fcntl.ioctl is unavailable on Windows")

    fcntl_stub = types.ModuleType("fcntl")
    fcntl_stub.ioctl = _ioctl_unavailable  # type: ignore[attr-defined]
    sys.modules.setdefault("fcntl", fcntl_stub)

    termios_stub = types.ModuleType("termios")
    termios_stub.TIOCGWINSZ = 0  # type: ignore[attr-defined]
    sys.modules.setdefault("termios", termios_stub)

    sys.modules.setdefault("tty", types.ModuleType("tty"))


if platform.system() == "Windows":
    _install_windows_stubs()

try:
    # Preflight: a missing native runtime (e.g. the MSVC redistributable on a
    # fresh Windows install) makes numba's compiled extensions fail to load.
    # Surface a clear, actionable message instead of a raw DLL-load traceback.
    import numba  # noqa: F401, E402
except ImportError as exc:
    _lines = [f"tsdr: numba failed to import ({exc})."]
    if platform.system() == "Windows":
        _lines += [
            "",
            "This usually means the Microsoft Visual C++ runtime is missing.",
            "Install it, then restart your terminal:",
            "",
            "    winget install --id Microsoft.VCRedist.2015+.x64 -e",
        ]
    print("\n".join(_lines), file=sys.stderr)
    sys.exit(1)

from tsdr.tui import TSDRApp  # noqa: E402

__version__ = "0.1.0"
__all__ = ["__version__", "TSDRApp"]
