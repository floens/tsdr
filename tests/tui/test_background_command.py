"""The opt-in background-execution seam: commands that declare `runs_in_background`
run in a worker thread so their blocking I/O never freezes the console UI thread."""

from __future__ import annotations

import asyncio
import threading
from argparse import Namespace

from tsdr.tui.app import TSDRApp
from tsdr.tui.commands import registry
from tsdr.tui.commands.base import Command
from tsdr.tui.commands.builtin.echo import EchoCommand
from tsdr.tui.commands.sdr.directory import DirectoryCommand
from tsdr.tui.commands.sdr.soapy import SoapyCommand
from tsdr.tui.console.widget import ConsoleWidget

# --- registry.resolve -------------------------------------------------------


def test_resolve_known_command() -> None:
    command, argv = registry.resolve("directory refresh")
    assert command is registry.COMMANDS["directory"]
    assert argv == ["refresh"]


def test_resolve_empty_and_blank() -> None:
    assert registry.resolve("") == (None, [])
    assert registry.resolve("   ") == (None, [])


def test_resolve_unknown_command() -> None:
    command, _argv = registry.resolve("nope arg")
    assert command is None


def test_resolve_parse_error() -> None:
    assert registry.resolve('echo "unbalanced') == (None, [])


# --- runs_in_background predicates ------------------------------------------


def test_directory_backgrounds_only_network_subcommands() -> None:
    cmd = DirectoryCommand()
    assert cmd.runs_in_background(["refresh"]) is True
    assert cmd.runs_in_background(["ping", "host"]) is True
    assert cmd.runs_in_background(["list"]) is False
    assert cmd.runs_in_background(["add", "x"]) is False
    assert cmd.runs_in_background([]) is False


def test_soapy_backgrounds_probe() -> None:
    cmd = SoapyCommand()
    assert cmd.runs_in_background(["probe"]) is True
    assert cmd.runs_in_background([]) is False


def test_base_default_is_foreground() -> None:
    assert EchoCommand().runs_in_background(["hi"]) is False


# --- framework behaviour (worker + async delivery) --------------------------


class _BlockingCommand(Command):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    @property
    def description(self) -> str:
        return "blocking test command"

    def runs_in_background(self, argv: list[str]) -> bool:
        return True

    def run(self, args: Namespace) -> str:
        self.started.set()
        self.release.wait(timeout=5)
        return "blocking-done"


class _RaisingCommand(Command):
    @property
    def description(self) -> str:
        return "raising test command"

    def runs_in_background(self, argv: list[str]) -> bool:
        return True

    def run(self, args: Namespace) -> str:
        raise RuntimeError("boom")


def _spy_console(app: TSDRApp) -> list[str]:
    """Capture every write_info line; the reconciler re-queries the same instance,
    so patching it once catches both the 'working…' hint and the delivered result."""
    console = app.query_one(ConsoleWidget)
    lines: list[str] = []
    console.write_info = lines.append  # type: ignore[method-assign]
    return lines


def test_background_command_runs_off_thread_and_delivers(monkeypatch) -> None:
    cmd = _BlockingCommand()
    monkeypatch.setitem(registry.COMMANDS, "blocktest", cmd)

    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            lines = _spy_console(app)

            app._execute_submitted("blocktest")  # returns immediately; run() is off-thread
            assert lines == ["[dim]working…[/]"]

            for _ in range(100):  # poll without blocking the loop, so the worker can launch
                if cmd.started.is_set():
                    break
                await pilot.pause()
            assert cmd.started.is_set()  # run() executed on the worker thread
            assert not cmd.release.is_set()  # delivered nothing yet; run() still parked
            assert "blocking-done" not in lines

            cmd.release.set()
            for _ in range(100):
                await pilot.pause()
                if "blocking-done" in lines:
                    break
            assert "blocking-done" in lines

    asyncio.run(go())


def test_background_command_error_writes_red_line(monkeypatch) -> None:
    monkeypatch.setitem(registry.COMMANDS, "raisetest", _RaisingCommand())

    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            lines = _spy_console(app)

            app._execute_submitted("raisetest")
            for _ in range(100):
                await pilot.pause()
                if any("boom" in line for line in lines):
                    break
            assert any(line == "[red]boom[/]" for line in lines)

    asyncio.run(go())
