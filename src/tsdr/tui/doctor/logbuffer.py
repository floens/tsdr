"""In-memory log capture for the doctor's Logs tab.

The ``tsdr doctor`` command configures no file logging, and the capability
probe runs (and logs) before the Textual app starts. So we attach a buffering
handler to the root logger early - from the entrypoint, ahead of
``probe_capabilities()`` - and the Logs tab drains it live. This is the only log
sink for the doctor; nothing is written to disk.
"""

import logging

_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s - %(message)s"
_DATEFMT = "%H:%M:%S"
# Loggers whose DEBUG output is noise rather than "things the doctor is doing".
_QUIET = ("numba", "PIL", "asyncio", "markdown_it")


class _BufferHandler(logging.Handler):
    """Appends formatted records to an in-memory list for later display."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001 - a logging handler must never propagate
            self.handleError(record)


_handler: _BufferHandler | None = None


def install_log_buffer() -> None:
    """Attach the in-memory buffer to the root logger at DEBUG (idempotent)."""
    global _handler
    if _handler is not None:
        return
    _handler = _BufferHandler()
    _handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(_handler)
    for name in _QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)


def log_lines() -> list[str]:
    """A snapshot of all captured log lines, oldest first."""
    return list(_handler.lines) if _handler is not None else []
