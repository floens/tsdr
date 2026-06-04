"""Unit tests for the doctor's --check text report and exit-code logic."""

from __future__ import annotations

from tsdr.tui.doctor.checks import CheckResult, Status
from tsdr.tui.doctor.report import exit_code, format_report


def _r(name: str, status: Status, required: bool, group: str = "render") -> CheckResult:
    return CheckResult(name, status, status.value, required, group)


def test_exit_code_ok() -> None:
    results = [_r("a", Status.OK, True), _r("b", Status.WARN, False)]
    assert exit_code(results) == 0


def test_exit_code_required_fail() -> None:
    results = [_r("a", Status.OK, True), _r("b", Status.FAIL, True)]
    assert exit_code(results) == 1


def test_exit_code_optional_fail_is_ok() -> None:
    results = [_r("a", Status.FAIL, False)]
    assert exit_code(results) == 0


def test_exit_code_required_unknown_is_ok() -> None:
    # kitty graphics through a pipe is UNKNOWN, not a hard failure.
    results = [_r("kitty_graphics", Status.UNKNOWN, True)]
    assert exit_code(results) == 0


def test_format_report_contains_sections() -> None:
    results = [
        _r("truecolor", Status.OK, True),
        _r("pixel_size", Status.WARN, False),
    ]
    text = format_report(results)
    assert "Required:" in text
    assert "Recommended / info:" in text
    assert "[ OK ]" in text
    assert "[WARN]" in text
    assert "Result:" in text
    assert "exit 0" in text
