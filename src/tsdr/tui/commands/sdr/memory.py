from argparse import Namespace
from collections import Counter

from tsdr.core.audio_spec import AudioDemodSpec
from tsdr.core.events.events import MemoriesChangedEvent
from tsdr.core.memories import get_memory_store, recall_memory
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import ConfigurationError, SDRException
from tsdr.core.units import parse_hz
from tsdr.radio.registry import DEMODULATORS
from tsdr.tui.commands._format import freq_mhz, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id


class MemoryCommand(Command):
    @property
    def description(self) -> str:
        return "Manage frequency memories (bookmarks)"

    def configure(self, parser: CommandParser) -> None:
        sub = parser.add_subparsers(dest="action")

        add_p = sub.add_parser("add")
        add_p.add_argument("frequency", help="Frequency with optional SI suffix (k/M/G/Hz)")
        add_p.add_argument("name", help="Memory name")
        add_p.add_argument("--mode", default="NFM", help="Demod mode")
        add_p.add_argument(
            "--bw", default="12.5k", help="Bandwidth with SI suffix (e.g. 12.5k, 200k)"
        )
        add_p.add_argument("--tags", default="", help="Comma-separated tags")
        add_p.add_argument("--color", default=None, help="Color hex (e.g. #ff0000)")

        list_p = sub.add_parser("list")
        list_p.add_argument("--tag", help="Filter by tag")

        rm_p = sub.add_parser("remove")
        rm_p.add_argument("id", help="Memory ID (prefix match)")

        recall_p = sub.add_parser("recall")
        recall_p.add_argument("id", help="Memory ID (prefix match)")
        recall_p.add_argument("--device", dest="device_id")

        sub.add_parser("tags")

    def run(self, args: Namespace) -> str:
        if args.action == "add":
            return self._add(args)
        if args.action == "list":
            return self._list(args)
        if args.action == "remove":
            return self._remove(args)
        if args.action == "recall":
            return self._recall(args)
        if args.action == "tags":
            return self._tags()
        return self.help_text()

    def _add(self, args: Namespace) -> str:
        store = get_memory_store()
        frequency = parse_hz(args.frequency)

        mode = args.mode.upper()
        if mode not in DEMODULATORS:
            available = ", ".join(sorted(DEMODULATORS))
            raise ConfigurationError(f"Unknown mode '{mode}'. Available: {available}")

        tags = tuple(t.strip() for t in args.tags.split(",") if t.strip()) if args.tags else ()

        memory = store.add(
            frequency=frequency,
            name=args.name,
            spec=AudioDemodSpec(mode=mode),
            bandwidth=parse_hz(args.bw),
            tags=tags,
            color=args.color,
        )
        self._publish_changed()
        return success(
            f"Added memory '[bold]{memory.name}[/]' "
            f"[dim][[/][bold cyan]{memory.id}[/][dim]][/] at {freq_mhz(frequency)}"
        )

    def _list(self, args: Namespace) -> str:
        store = get_memory_store()
        if args.tag:
            memories = store.find_by_tag(args.tag)
        else:
            memories = store.all()

        if not memories:
            return "No memories found"

        head = f"[bold]{'ID':<10} {'Freq (MHz)':>12} {'Name':<20} {'Mode':<5} {'BW':>8} {'Tags'}[/]"
        lines = [head]
        for m in memories:
            bw_str = f"{m.bandwidth / 1000:.1f}k" if m.bandwidth >= 1000 else str(m.bandwidth)
            tags_str = ", ".join(m.tags) if m.tags else ""
            lines.append(
                f"[dim]{m.id:<10}[/] [cyan]{m.frequency / 1e6:>12.3f}[/] "
                f"[bold]{m.name:<20}[/] [green]{m.mode:<5}[/] "
                f"[yellow]{bw_str:>8}[/] [dim]{tags_str}[/]"
            )
        return "\n".join(lines)

    def _remove(self, args: Namespace) -> str:
        store = get_memory_store()
        matches = store.find_by_prefix(args.id)
        if not matches:
            raise SDRException(f"No memory found matching '{args.id}'")
        if len(matches) > 1:
            ids = ", ".join(m.id for m in matches)
            raise SDRException(f"Ambiguous ID '{args.id}' matches: {ids}")

        memory = matches[0]
        store.remove(memory.id)
        self._publish_changed()
        return success(
            f"Removed memory '[bold]{memory.name}[/]' [dim][[/][bold cyan]{memory.id}[/][dim]][/]"
        )

    def _recall(self, args: Namespace) -> str:
        store = get_memory_store()
        matches = store.find_by_prefix(args.id)
        if not matches:
            raise SDRException(f"No memory found matching '{args.id}'")
        if len(matches) > 1:
            ids = ", ".join(m.id for m in matches)
            raise SDRException(f"Ambiguous ID '{args.id}' matches: {ids}")

        memory = matches[0]
        did = args.device_id or get_focused_device_id()
        recall_memory(memory, did)
        return success(
            f"Recalled '[bold]{memory.name}[/]': "
            f"{freq_mhz(memory.frequency)} [green]{memory.mode}[/]"
        )

    def _tags(self) -> str:
        store = get_memory_store()
        all_memories = store.all()
        if not all_memories:
            return "No memories"

        tag_counts: Counter[str] = Counter()
        for m in all_memories:
            for t in m.tags:
                tag_counts[t] += 1

        if not tag_counts:
            return "No tags"

        lines = [f"[bold]{'Tag':<20} {'Count':>5}[/]"]
        for tag, count in sorted(tag_counts.items()):
            lines.append(f"[dim]{tag:<20}[/] [cyan]{count:>5}[/]")
        return "\n".join(lines)

    def _publish_changed(self) -> None:
        store = get_memory_store()
        engine = get_engine()
        engine.event_bus.publish(MemoriesChangedEvent(memories=tuple(store.all())))

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        if flag == "--mode":
            return [Completion(m) for m in sorted(DEMODULATORS) if m.startswith(prefix.upper())]
        if flag == "--tag":
            store = get_memory_store()
            return [Completion(t) for t in store.tags() if t.startswith(prefix)]
        if flag == "--device":
            return device_id_completions(prefix)

        if subcommand in ("remove", "recall"):
            store = get_memory_store()
            matches = store.find_by_prefix(prefix)
            return [
                Completion(m.id, f"{m.name} ({m.frequency / 1e6:.3f} MHz) {m.mode}")
                for m in matches
            ]

        return []
