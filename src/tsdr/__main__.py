import argparse
import logging
import signal
import sys

from tsdr.core.storage import config_dir
from tsdr.core.tracing import log_stats, span
from tsdr.tui.app import TSDRApp


def _configure_logging() -> None:
    log_path = config_dir() / "tsdr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path),
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    with span("startup"):
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
        args = parser.parse_args()

        with span("create_app"):
            app = TSDRApp(startup_commands=args.startup_commands or [])

    log_stats(phase="startup")

    # Textual handles SIGINT itself via on_unmount(); we only wire SIGTERM.
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, requesting app exit")
        try:
            app.exit()
        except Exception as e:  # noqa: BLE001 - last-resort cleanup in signal handler
            logger.error(f"Error requesting exit: {e}", exc_info=True)
            sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Calling app.run()")
    try:
        app.run()
    except Exception as e:
        logger.error(f"Unhandled exception in app.run(): {e}", exc_info=True)
        raise
    finally:
        log_stats(phase="exit")
        logger.info("Application exited")


if __name__ == "__main__":
    main()
