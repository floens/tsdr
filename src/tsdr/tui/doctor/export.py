"""Serialize doctor diagnostics to JSON (data domain - no display deps).

Turns the capability data from `checks` into a single uploadable JSON document.
Usable headlessly (`tsdr doctor --json`) and from the interactive export hotkey.
"""

import json
from dataclasses import asdict
from pathlib import Path

from tsdr.core.clock_sync import now
from tsdr.core.storage import config_dir
from tsdr.tui.doctor.checks import (
    CheckResult,
    installed_packages,
    os_details,
)
from tsdr.tui.widgets.kitty_image import KITTY_TRANSPORT_DESC

_REPORT_NAME = "doctor-report.json"


def diagnostics(results: list[CheckResult]) -> dict:
    return {
        "schema": "tsdr-doctor/1",
        "generated_at": now().isoformat(),
        "image_transport": KITTY_TRANSPORT_DESC,
        "os": os_details(),
        "checks": [{**asdict(r), "status": r.status.value} for r in results],
        "packages": installed_packages(),
    }


def to_json(results: list[CheckResult]) -> str:
    return json.dumps(diagnostics(results), indent=2)


def write_report(results: list[CheckResult]) -> Path:
    """Write the JSON diagnostics to the config dir; return the path."""
    path = config_dir() / _REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(results))
    return path
