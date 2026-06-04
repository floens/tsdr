"""`tsdr doctor` - terminal & environment capability diagnostic."""

from tsdr.tui.doctor.app import DoctorApp
from tsdr.tui.doctor.checks import run_all
from tsdr.tui.doctor.export import to_json
from tsdr.tui.doctor.report import exit_code
from tsdr.tui.doctor.report import run as run_report


def run_doctor(check: bool, json_out: bool = False) -> int:
    """Entry point. Returns the process exit code (0 = required caps OK)."""
    if json_out:
        print(to_json(run_all()))
        return 0
    if check:
        return run_report()

    results = run_all()  # raw-tty probes must run before Textual grabs the tty
    DoctorApp(results).run()
    return exit_code(results)
