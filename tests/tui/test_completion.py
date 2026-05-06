from argparse import Namespace

from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.registry import MenuItem, _fuzzy_match_indices, get_filtered_commands


class FlagsOnlyCommand(Command):
    """Like `config` - only flags, no positionals."""

    @property
    def description(self) -> str:
        return "flags only"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("--device", help="Device ID")
        parser.add_argument("--frequency", type=float, help="Center frequency")
        parser.add_argument("--gain", type=float, help="RF gain")
        parser.add_argument("--agc", choices=["on", "off"], help="AGC mode")

    def run(self, args: Namespace) -> str:
        return ""

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        if flag == "--device":
            return [Completion("rtl0"), Completion("rtl1")]
        return []


class PositionalAndFlagsCommand(Command):
    """Like `demod` - one positional with choices, then flags."""

    @property
    def description(self) -> str:
        return "positional + flags"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("mode", choices=["wfm", "nfm", "am", "off"])
        parser.add_argument("--device", help="Device ID")
        parser.add_argument("--offset", type=float, help="Frequency offset")

    def run(self, args: Namespace) -> str:
        return ""

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        if flag == "--device":
            return [Completion("rtl0")]
        return []


class SubparserCommand(Command):
    """Like `audio` - subparsers with nested subparsers."""

    @property
    def description(self) -> str:
        return "subparsers"

    def configure(self, parser: CommandParser) -> None:
        subs = parser.add_subparsers()
        device_parser = subs.add_parser("device")
        device_subs = device_parser.add_subparsers()
        set_parser = device_subs.add_parser("set")
        set_parser.add_argument("--name", help="Device name")
        device_subs.add_parser("list")
        subs.add_parser("volume")

    def run(self, args: Namespace) -> str:
        return ""


class DynamicPositionalCommand(Command):
    """One positional without choices - uses complete() for it."""

    @property
    def description(self) -> str:
        return "dynamic positional"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("device")
        parser.add_argument("--verbose", action="store_true", help="Verbose output")

    def run(self, args: Namespace) -> str:
        return ""

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        if flag is not None:
            return []
        devices = ["rtl0", "rtl1", "hackrf0"]
        return [Completion(d) for d in devices if d.startswith(prefix)]


# Tests


class TestFlagsOnlyCommand:
    def setup_method(self):
        self.cmd = FlagsOnlyCommand()

    def test_empty_suggests_flags_not_device_ids(self):
        completions = self.cmd.get_completions([], "")
        values = {c.value for c in completions}
        assert "--device" in values
        assert "--frequency" in values
        assert "--gain" in values
        assert "--agc" in values
        assert "rtl0" not in values

    def test_after_flag_value_suggests_remaining_flags(self):
        completions = self.cmd.get_completions(["--frequency", "100"], "")
        values = {c.value for c in completions}
        assert "--gain" in values
        assert "--agc" in values
        assert "--frequency" not in values

    def test_prefix_filters_flags(self):
        completions = self.cmd.get_completions([], "--f")
        values = {c.value for c in completions}
        assert values == {"--frequency"}

    def test_flag_value_choices(self):
        completions = self.cmd.get_completions(["--agc"], "")
        values = {c.value for c in completions}
        assert values == {"on", "off"}

    def test_device_flag_completes_device_ids(self):
        completions = self.cmd.get_completions(["--device"], "")
        values = {c.value for c in completions}
        assert values == {"rtl0", "rtl1"}

    def test_frequency_flag_does_not_complete_device_ids(self):
        completions = self.cmd.get_completions(["--frequency"], "")
        values = {c.value for c in completions}
        assert "rtl0" not in values


class TestPositionalAndFlagsCommand:
    def setup_method(self):
        self.cmd = PositionalAndFlagsCommand()

    def test_empty_suggests_positional_choices(self):
        completions = self.cmd.get_completions([], "")
        values = {c.value for c in completions}
        assert "wfm" in values
        assert "nfm" in values
        assert "rtl0" not in values

    def test_positional_prefix(self):
        completions = self.cmd.get_completions([], "w")
        values = {c.value for c in completions}
        assert values == {"wfm"}

    def test_after_positional_suggests_flags(self):
        completions = self.cmd.get_completions(["wfm"], "")
        values = {c.value for c in completions}
        assert "--device" in values
        assert "--offset" in values
        assert "rtl0" not in values
        assert "wfm" not in values

    def test_after_positional_and_flag_suggests_remaining(self):
        completions = self.cmd.get_completions(["wfm", "--device", "rtl0"], "")
        values = {c.value for c in completions}
        assert "--offset" in values
        assert "--device" not in values

    def test_device_flag_completes_device_ids(self):
        completions = self.cmd.get_completions(["wfm", "--device"], "")
        values = {c.value for c in completions}
        assert values == {"rtl0"}

    def test_offset_flag_does_not_complete_device_ids(self):
        completions = self.cmd.get_completions(["wfm", "--offset"], "")
        assert completions == []


class TestSubparserCommand:
    def setup_method(self):
        self.cmd = SubparserCommand()

    def test_empty_suggests_subparser_names(self):
        completions = self.cmd.get_completions([], "")
        values = {c.value for c in completions}
        assert "device" in values
        assert "volume" in values

    def test_recurse_into_subparser(self):
        completions = self.cmd.get_completions(["device"], "")
        values = {c.value for c in completions}
        assert "set" in values
        assert "list" in values

    def test_nested_subparser_flags(self):
        completions = self.cmd.get_completions(["device", "set"], "")
        values = {c.value for c in completions}
        assert "--name" in values


class TestDynamicPositionalCommand:
    def setup_method(self):
        self.cmd = DynamicPositionalCommand()

    def test_empty_suggests_dynamic_completions(self):
        completions = self.cmd.get_completions([], "")
        values = {c.value for c in completions}
        assert "rtl0" in values
        assert "hackrf0" in values

    def test_dynamic_prefix_filtering(self):
        completions = self.cmd.get_completions([], "rtl")
        values = {c.value for c in completions}
        assert values == {"rtl0", "rtl1"}

    def test_after_positional_suggests_flags(self):
        completions = self.cmd.get_completions(["rtl0"], "")
        values = {c.value for c in completions}
        assert "--verbose" in values
        assert "rtl0" not in values

    def test_after_positional_and_flag_no_more(self):
        completions = self.cmd.get_completions(["rtl0", "--verbose"], "")
        assert completions == []


# Fuzzy match indices & MenuItem shape


def test_fuzzy_match_indices_exact() -> None:
    assert _fuzzy_match_indices("echo", "echo") == (0, 1, 2, 3)


def test_fuzzy_match_indices_prefix() -> None:
    assert _fuzzy_match_indices("ec", "echo") == (0, 1)


def test_fuzzy_match_indices_non_contiguous() -> None:
    assert _fuzzy_match_indices("fcus", "focus") == (0, 2, 3, 4)


def test_fuzzy_match_indices_no_match() -> None:
    assert _fuzzy_match_indices("xyz", "echo") is None


def test_get_filtered_commands_returns_menu_items() -> None:
    items = get_filtered_commands("ec")
    assert all(isinstance(it, MenuItem) for it in items)
    echo_match = next((it for it in items if it.value == "echo"), None)
    assert echo_match is not None
    assert echo_match.match_indices == (0, 1)


def test_get_filtered_commands_empty_query() -> None:
    items = get_filtered_commands("")
    assert items
    assert all(it.match_indices == () for it in items)


# Empty-argv help fallback


class RequiredPositionalCommand(Command):
    @property
    def description(self) -> str:
        return "needs a target"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("target", help="what to do it to")

    def run(self, args: Namespace) -> str:
        return f"target={args.target}"


class OptionalOnlyCommand(Command):
    @property
    def description(self) -> str:
        return "all optional"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("--x", help="an x")

    def run(self, args: Namespace) -> str:
        return self.help_text() if args.x is None else f"x={args.x}"


def test_empty_argv_with_required_arg_shows_help() -> None:
    cmd = RequiredPositionalCommand()
    cmd._registered_name = "needy"
    result = cmd.execute([])
    # Full help has the description, not just the one-line usage
    assert "needs a target" in result or "what to do it to" in result
    assert "usage:" in result.lower()


def test_help_text_has_all_args() -> None:
    cmd = RequiredPositionalCommand()
    cmd._registered_name = "needy"
    help_out = cmd.help_text()
    assert "target" in help_out
    assert "what to do it to" in help_out


def test_help_text_uses_registered_name() -> None:
    cmd = RequiredPositionalCommand()
    cmd._registered_name = "needy"
    assert "needy" in cmd.help_text()


def test_optional_only_runs_with_empty_argv() -> None:
    cmd = OptionalOnlyCommand()
    cmd._registered_name = "opt"
    # Empty argv should parse successfully; the command's own no-op branch
    # calls self.help_text() which includes the prog name.
    result = cmd.execute([])
    assert "opt" in result
