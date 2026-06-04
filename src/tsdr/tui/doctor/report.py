"""Non-interactive (`--check`) capability report.

``format_report`` is pure (returns text) so it can be unit-tested; ``run``
executes the checks, prints, and returns the process exit code.
"""

from tsdr.tui.doctor.checks import CheckResult, Status, run_all
from tsdr.tui.widgets.kitty_image import KITTY_TRANSPORT_DESC

_MARKER = {
    Status.OK: "[ OK ]",
    Status.WARN: "[WARN]",
    Status.FAIL: "[FAIL]",
    Status.UNKNOWN: "[????]",
}

_VISUAL_CHECKS = "background, bold, gradient, unicode glyphs, kitty image, audio tone"


def _row(r: CheckResult) -> str:
    return f"  {_MARKER[r.status]} {r.name:<20} {r.summary}"


def _by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.name: r for r in results}


def _env_line(results: list[CheckResult]) -> str:
    by = _by_name(results)
    parts = []
    if "terminal_identity" in by:
        d = by["terminal_identity"].detail
        parts.append(f"TERM={d.get('TERM', '?')}")
        if d.get("TERM_PROGRAM", "(unset)") != "(unset)":
            parts.append(f"TERM_PROGRAM={d['TERM_PROGRAM']}")
    if "truecolor" in by:
        parts.append(f"COLORTERM={by['truecolor'].detail.get('COLORTERM', '?')}")
    if "python_version" in by:
        d = by["python_version"].detail
        gil = f" (GIL {d['gil']})" if "gil" in d else ""
        parts.append(f"python {d.get('version', '?')}{gil}")
    if "os_platform" in by:
        parts.append(by["os_platform"].detail.get("platform", "?"))
    if "unicode_locale" in by:
        parts.append(by["unicode_locale"].detail.get("stdout_encoding", "?"))
    return "  " + "  ".join(parts)


def exit_code(results: list[CheckResult]) -> int:
    return 1 if any(r.required and r.status is Status.FAIL for r in results) else 0


def format_report(results: list[CheckResult]) -> str:
    required = [r for r in results if r.required]
    optional = [r for r in results if not r.required]

    lines = ["TSDR doctor - capability report", "", "Environment:", _env_line(results), ""]
    lines.append("Required:")
    lines.extend(_row(r) for r in required)
    lines.append("Recommended / info:")
    lines.extend(_row(r) for r in optional)
    lines += [
        "",
        f"Image transport: {KITTY_TRANSPORT_DESC}",
        f"Visual checks (interactive only): {_VISUAL_CHECKS}.",
        "  Run `tsdr doctor` (no --check) to verify by eye/ear.",
        "",
    ]
    code = exit_code(results)
    verdict = (
        "all required capabilities OK" if code == 0 else "one or more required capabilities FAILED"
    )
    lines.append(f"Result: {verdict}.   (exit {code})")
    return "\n".join(lines)


def run() -> int:
    results = run_all()
    print(format_report(results))
    return exit_code(results)
