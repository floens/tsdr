"""CLI dispatch: `doctor` subcommand must not break the bare/legacy invocation."""

from __future__ import annotations

from tsdr.__main__ import _build_parser


def test_doctor_subcommand() -> None:
    args = _build_parser().parse_args(["doctor", "--check"])
    assert args.command == "doctor"
    assert args.check is True


def test_doctor_json_flag() -> None:
    args = _build_parser().parse_args(["doctor", "--json"])
    assert args.command == "doctor"
    assert args.json is True


def test_doctor_interactive_default() -> None:
    args = _build_parser().parse_args(["doctor"])
    assert args.command == "doctor"
    assert args.check is False


def test_bare_invocation_has_no_command() -> None:
    args = _build_parser().parse_args([])
    assert args.command is None


def test_legacy_exec_flag_still_works() -> None:
    args = _build_parser().parse_args(["-e", "start rtl0"])
    assert args.command is None
    assert args.startup_commands == ["start rtl0"]
