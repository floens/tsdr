"""Tests for the JSON diagnostics export (data domain, no display)."""

from __future__ import annotations

import json

from tsdr.tui.doctor import export
from tsdr.tui.doctor.checks import CheckResult, Status


def _results() -> list[CheckResult]:
    return [
        CheckResult("truecolor", Status.OK, "24-bit", True, "render", {"COLORTERM": "truecolor"}),
        CheckResult("kitty_graphics", Status.FAIL, "no response", True, "render"),
    ]


def test_to_json_is_valid_and_complete() -> None:
    data = json.loads(export.to_json(_results()))
    assert data["schema"] == "tsdr-doctor/1"
    assert "generated_at" in data
    assert "os" in data and "system" in data["os"]
    assert isinstance(data["packages"], dict) and data["packages"]  # numpy etc. present
    names = {c["name"]: c for c in data["checks"]}
    assert names["truecolor"]["status"] == "ok"  # enum serialized to its value
    assert names["truecolor"]["detail"]["COLORTERM"] == "truecolor"
    assert names["kitty_graphics"]["status"] == "fail"


def test_write_report_writes_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(export, "config_dir", lambda: tmp_path)
    path = export.write_report(_results())
    assert path.exists()
    assert json.loads(path.read_text())["schema"] == "tsdr-doctor/1"
