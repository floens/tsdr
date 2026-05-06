from argparse import Namespace

from tsdr.core.tracing import clear_stats, get_stats
from tsdr.tui.commands.base import Command, CommandParser, Completion


class TraceCommand(Command):
    @property
    def description(self) -> str:
        return "Display tracing statistics (use 'clear' to reset)"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("action", nargs="?", choices=["show", "clear"])

    def run(self, args: Namespace) -> str:
        if args.action == "clear":
            clear_stats()
            return "Tracing statistics cleared"

        stats = get_stats()
        if not stats:
            return "No tracing statistics collected"

        lines = ["Tracing Statistics:"]
        for name in sorted(stats.keys()):
            s = stats[name]
            lines.append(
                f"  {name}: n={s.count}, "
                f"min={s.min_ms:.2f}, max={s.max_ms:.2f}, "
                f"avg={s.avg_ms:.2f}, p99={s.p99_ms:.2f}ms"
            )
        return "\n".join(lines)

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        return [Completion(c) for c in ("show", "clear") if c.startswith(prefix)]
