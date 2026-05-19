import argparse
import logging
import platform
import signal
import socket
import sys

from tsdr import __version__
from tsdr.core.storage import config_dir
from tsdr.core.tracing import log_stats, span
from tsdr.headless import run_headless
from tsdr.tui.app import TSDRApp

_LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _configure_logging(verbose: bool) -> None:
    log_path = config_dir() / "tsdr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path),
        filemode="w",
        level=logging.DEBUG if verbose else logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
    )
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def _enable_stdout_logging(verbose: bool) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    handler.addFilter(lambda r: not r.name.startswith("tsdr.core.tracing"))
    logging.getLogger().addHandler(handler)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TSDR - Terminal Software Defined Radio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
# Start normally
uv run tsdr

# Execute commands on startup
uv run tsdr -e "add rtl0 --type rtltcp --host localhost --port 1234 --frequency 101.1"
uv run tsdr -e "add rtl0 --type rtltcp --host localhost --port 1234 --frequency 101.1" \\
          -e "start rtl0" \\
          -e "demod rtl0 wfm"
    """,
    )
    parser.add_argument(
        "-e",
        "--exec",
        action="append",
        dest="startup_commands",
        metavar="COMMAND",
        help="Execute command on startup (can be specified multiple times)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without TUI; stream logs to stdout and read commands from stdin",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log at DEBUG level (default: INFO). Affects both tsdr.log and --headless stdout.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    _configure_logging(verbose=args.verbose)
    if args.headless:
        _enable_stdout_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    mode = "headless" if args.headless else "tui"
    logger.info(
        "tsdr_starting version=%s mode=%s python=%s platform=%s hostname=%s",
        __version__,
        mode,
        platform.python_version(),
        platform.platform(terse=True),
        socket.gethostname(),
    )

    with span("startup"):
        if args.headless:
            sys.exit(run_headless(args.startup_commands or []))

        with span("create_app"):
            app = TSDRApp(startup_commands=args.startup_commands or [])

    log_stats(phase="startup")

    # Textual handles SIGINT itself via on_unmount(); we only wire SIGTERM.
    def signal_handler(signum, frame):
        logger.info("signal_received signum=%s action=exit", signum)
        try:
            app.exit()
        except Exception as e:  # noqa: BLE001 - last-resort cleanup in signal handler
            logger.error("signal_exit_failed error=%r", e, exc_info=True)
            sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("app_run_starting")
    try:
        app.run()
    except Exception as e:
        logger.error("app_run_unhandled_exception error=%r", e, exc_info=True)
        raise
    finally:
        log_stats(phase="exit")
        logger.info("tsdr_exited")


if __name__ == "__main__":
    main()
