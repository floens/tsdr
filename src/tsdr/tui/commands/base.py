from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, _SubParsersAction
from dataclasses import dataclass
from io import StringIO

from tsdr.core.sdr.exceptions import SDRException
from tsdr.tui.commands._format import error, safe


@dataclass(frozen=True)
class Completion:
    """A completion suggestion with optional description."""

    value: str
    description: str = ""


class CommandExit(Exception):
    """Raised by CommandParser instead of sys.exit."""

    def __init__(self, message: str, status: int = 0) -> None:
        self.message = message
        self.status = status


class CommandParser(ArgumentParser):
    """ArgumentParser that raises CommandExit instead of calling sys.exit."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise CommandExit(f"Error: {message}", status=2)

    def exit(self, status: int = 0, message: str | None = None) -> None:  # type: ignore[override]
        raise CommandExit(message or "", status=status)

    def print_help(self, file=None) -> None:  # type: ignore[override]
        buf = StringIO()
        super().print_help(buf)
        raise CommandExit(buf.getvalue().strip())


class Command(ABC):
    """Base class for all commands."""

    _registered_name: str | None = None

    @property
    @abstractmethod
    def description(self) -> str: ...

    def configure(self, parser: CommandParser) -> None:  # noqa: B027
        """Override to add arguments to the parser."""

    @abstractmethod
    def run(self, args: Namespace) -> str: ...

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        """Override for dynamic completions.

        Args:
            flag: The flag being completed (e.g. "--device"), or None for positionals.
            subcommand: The active subparser name when inside a subcommand
                (e.g. "recall" for `memory recall`), or None otherwise.
        """
        return []

    def execute(self, argv: list[str]) -> str:
        parser = self._build_parser()
        try:
            args = parser.parse_args(argv)
        except CommandExit as e:
            # Called with no args and argparse complained: show the full help
            # instead of the one-line usage error (readable + discoverable).
            if not argv and e.status == 2:
                return parser.format_help().strip()
            return e.message
        try:
            return self.run(args)
        except SDRException as e:
            return error(safe(str(e)))
        except ValueError as e:
            return error(safe(str(e)))

    def help_text(self) -> str:
        """Full argparse help for this command, for `run()` to return on no-op input."""
        return self._build_parser().format_help().strip()

    def _build_parser(self) -> CommandParser:
        prog = self._registered_name or self.__class__.__name__
        parser = CommandParser(prog=prog, add_help=True)
        self.configure(parser)
        return parser

    def get_completions(self, tokens: list[str], prefix: str) -> list[Completion]:
        parser = CommandParser(add_help=False)
        self.configure(parser)
        return self._complete_parser(parser, tokens, prefix)

    def _complete_parser(
        self,
        parser: ArgumentParser,
        tokens: list[str],
        prefix: str,
        subcommand: str | None = None,
    ) -> list[Completion]:
        # Check if last token is a flag needing a value
        if tokens and tokens[-1].startswith("-"):
            flag = tokens[-1]
            for action in parser._actions:
                if flag in action.option_strings and action.nargs != 0:
                    if action.choices:
                        return [
                            Completion(str(c)) for c in action.choices if str(c).startswith(prefix)
                        ]
                    return self.complete(tokens, prefix, flag=flag, subcommand=subcommand)

        # Complete flag names when prefix starts with -
        if prefix.startswith("-"):
            completions: list[Completion] = []
            for action in parser._actions:
                for opt in action.option_strings:
                    if opt.startswith(prefix):
                        completions.append(Completion(opt, action.help or ""))
            return completions

        # Recurse into subparsers
        for action in parser._actions:
            if isinstance(action, _SubParsersAction) and action.choices:
                if tokens and tokens[0] in action.choices:
                    sub_parser = action.choices[tokens[0]]
                    return self._complete_parser(
                        sub_parser, tokens[1:], prefix, subcommand=tokens[0]
                    )
                # Suggest subparser names as completions
                return [Completion(name) for name in action.choices if name.startswith(prefix)]

        positional_actions = [a for a in parser._actions if not a.option_strings]
        consumed = self._count_consumed_positionals(parser, tokens)

        # Complete current positional if not all consumed
        if consumed < len(positional_actions):
            current = positional_actions[consumed]
            if current.choices:
                return [Completion(str(c)) for c in current.choices if str(c).startswith(prefix)]
            return self.complete(tokens, prefix, subcommand=subcommand)

        # All positionals consumed - suggest unused flags (fish-style)
        flag_completions: list[Completion] = []
        used_flags = {t for t in tokens if t.startswith("-")}
        for action in parser._actions:
            if not action.option_strings:
                continue
            if used_flags & set(action.option_strings):
                continue
            for opt in action.option_strings:
                if opt.startswith("--") and opt.startswith(prefix):
                    flag_completions.append(Completion(opt, action.help or ""))
                    break
        return flag_completions

    @staticmethod
    def _count_consumed_positionals(parser: ArgumentParser, tokens: list[str]) -> int:
        """Count how many positional arguments have been consumed by tokens."""
        # Build flag->nargs map
        flag_nargs: dict[str, int] = {}
        for action in parser._actions:
            for opt in action.option_strings:
                flag_nargs[opt] = 0 if action.nargs == 0 else 1

        consumed = 0
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith("-") and tok in flag_nargs:
                i += 1 + flag_nargs[tok]  # skip flag + its value
            else:
                consumed += 1
                i += 1
        return consumed
