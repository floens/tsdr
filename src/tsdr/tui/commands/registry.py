import logging
import shlex
from dataclasses import dataclass

from tsdr.tui.commands.base import Command, Completion

logger = logging.getLogger(__name__)

COMMANDS: dict[str, Command] = {}
last_command_output: str = ""


@dataclass(frozen=True)
class MenuItem:
    """A rendered autocomplete menu entry with the chars that matched the query."""

    value: str
    description: str
    match_indices: tuple[int, ...]


def register(name: str, command: Command) -> None:
    command._registered_name = name
    COMMANDS[name] = command


def execute(input_line: str) -> str:
    global last_command_output

    input_line = input_line.strip()
    if not input_line:
        return "Empty command. Type help for available commands."

    logger.info("command line=%r", input_line)

    try:
        parts = shlex.split(input_line)
    except ValueError as e:
        return f"Error parsing command: {e}"

    if not parts:
        return "Empty command. Type help for available commands."

    command_name = parts[0]
    argv = parts[1:]

    command = COMMANDS.get(command_name)
    if command is None:
        return f"Unknown command: {command_name}. Type help for available commands."

    result = command.execute(argv)
    last_command_output = result
    return result


def get_completions(input_line: str) -> list[Completion]:
    parts = input_line.split()

    if not parts:
        return []

    command_name = parts[0]
    command = COMMANDS.get(command_name)

    if command is None:
        return []

    tokens = parts[1:]
    prefix = tokens[-1] if tokens else ""

    # If line ends with space, prefix is empty and tokens are complete
    if input_line.endswith(" "):
        prefix = ""
    elif tokens:
        tokens = tokens[:-1]

    return command.get_completions(tokens, prefix)


def get_command_info() -> list[tuple[str, str]]:
    return [(name, cmd.description) for name, cmd in COMMANDS.items()]


def get_filtered_commands(query: str) -> list[MenuItem]:
    command_info = get_command_info()

    if not query:
        return [MenuItem(name, desc, ()) for name, desc in sorted(command_info)]

    query_lower = query.lower()
    matches: list[tuple[int, MenuItem]] = []

    for name, desc in command_info:
        name_lower = name.lower()
        indices = _fuzzy_match_indices(query_lower, name_lower)
        if indices is None:
            continue
        if name_lower == query_lower:
            priority = 0
            hl = tuple(range(len(name)))
        elif name_lower.startswith(query_lower):
            priority = 1
            hl = tuple(range(len(query)))
        else:
            priority = 2
            hl = indices
        matches.append((priority, MenuItem(name, desc, hl)))

    matches.sort(key=lambda x: (x[0], x[1].value))
    return [item for _, item in matches]


def _fuzzy_match_indices(query: str, target: str) -> tuple[int, ...] | None:
    """Return the matched char indices in target, or None if query isn't a subsequence."""
    indices: list[int] = []
    query_idx = 0
    for i, char in enumerate(target):
        if query_idx < len(query) and char == query[query_idx]:
            indices.append(i)
            query_idx += 1
    return tuple(indices) if query_idx == len(query) else None


# Command registration (imports at bottom to avoid circular deps)

from tsdr.tui.commands.audio.audio import AudioCommand  # noqa: E402
from tsdr.tui.commands.builtin.echo import EchoCommand  # noqa: E402
from tsdr.tui.commands.builtin.exit import ExitCommand  # noqa: E402
from tsdr.tui.commands.builtin.help import HelpCommand  # noqa: E402
from tsdr.tui.commands.builtin.paths import PathsCommand  # noqa: E402
from tsdr.tui.commands.builtin.time import TimeCommand  # noqa: E402
from tsdr.tui.commands.builtin.trace import TraceCommand  # noqa: E402
from tsdr.tui.commands.sdr.add import SDRAddCommand  # noqa: E402
from tsdr.tui.commands.sdr.bandplan import BandplanCommand  # noqa: E402
from tsdr.tui.commands.sdr.config import SDRConfigCommand  # noqa: E402
from tsdr.tui.commands.sdr.dab import DABCommand  # noqa: E402
from tsdr.tui.commands.sdr.demod import SDRDemodCommand  # noqa: E402
from tsdr.tui.commands.sdr.focus import SDRFocusCommand  # noqa: E402
from tsdr.tui.commands.sdr.frequency import FrequencyCommand  # noqa: E402
from tsdr.tui.commands.sdr.list import SDRListCommand  # noqa: E402
from tsdr.tui.commands.sdr.memory import MemoryCommand  # noqa: E402
from tsdr.tui.commands.sdr.pipeline import SDRPipelineCommand  # noqa: E402
from tsdr.tui.commands.sdr.record import SDRRecordCommand  # noqa: E402
from tsdr.tui.commands.sdr.remove import SDRRemoveCommand  # noqa: E402
from tsdr.tui.commands.sdr.scan import ScanCommand  # noqa: E402
from tsdr.tui.commands.sdr.squelch import SDRSquelchCommand  # noqa: E402
from tsdr.tui.commands.sdr.start import SDRStartCommand  # noqa: E402
from tsdr.tui.commands.sdr.stop import SDRStopCommand  # noqa: E402

register("echo", EchoCommand())
register("exit", ExitCommand())
register("quit", ExitCommand())
register("help", HelpCommand())
register("paths", PathsCommand())
register("time", TimeCommand())
register("trace", TraceCommand())
register("add", SDRAddCommand())
register("remove", SDRRemoveCommand())
register("start", SDRStartCommand())
register("stop", SDRStopCommand())
register("list", SDRListCommand())
register("focus", SDRFocusCommand())
register("config", SDRConfigCommand())
register("dab", DABCommand())
register("demod", SDRDemodCommand())
register("pipeline", SDRPipelineCommand())
register("audio", AudioCommand())
register("scan", ScanCommand())
register("squelch", SDRSquelchCommand())
register("f", FrequencyCommand())
register("memory", MemoryCommand())
register("bandplan", BandplanCommand())
register("record", SDRRecordCommand())
