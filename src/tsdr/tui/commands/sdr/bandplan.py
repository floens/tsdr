from argparse import Namespace

from tsdr.core.bandplans import get_bandplan_store
from tsdr.core.events.events import BandplanChangedEvent
from tsdr.core.preferences import save_bandplan
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import ConfigurationError
from tsdr.tui.commands._format import field, header, success
from tsdr.tui.commands.base import Command, CommandParser, Completion


class BandplanCommand(Command):
    @property
    def description(self) -> str:
        return "Load, clear, or inspect bandplan overlays"

    def configure(self, parser: CommandParser) -> None:
        sub = parser.add_subparsers(dest="action")

        sub.add_parser("list")

        load_p = sub.add_parser("load")
        load_p.add_argument("filename", help="Bandplan filename (e.g. usa.json)")

        sub.add_parser("off")
        sub.add_parser("show")

    def run(self, args: Namespace) -> str:
        if args.action == "list":
            return self._list()
        if args.action == "load":
            return self._load(args)
        if args.action == "off":
            return self._off()
        if args.action == "show":
            return self._show()
        return self.help_text()

    def _list(self) -> str:
        store = get_bandplan_store()
        plans = store.plans()
        if not plans:
            return "No bandplans found in the bandplans/ config directory"
        active = store.active
        head = f"[bold]{'Filename':<32} {'Name':<24} {'Country':<12} Bands[/]"
        lines = [head]
        for plan in plans:
            is_active = active is not None and active.filename == plan.filename
            marker = "[bold yellow]*[/]" if is_active else " "
            file_display = f"{plan.filename}.json"
            name_md = f"[bold]{plan.name}[/]" if is_active else plan.name
            lines.append(
                f"{marker} {file_display:<30} {name_md:<24} "
                f"[dim]{plan.country_code:<12}[/] [cyan]{len(plan.bands)}[/]"
            )
        return "\n".join(lines)

    def _load(self, args: Namespace) -> str:
        store = get_bandplan_store()
        filename = args.filename
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        plan = store.set_active(filename)
        if plan is None:
            raise ConfigurationError(f"bandplan '{filename}' not found. Try `bandplan list`.")
        save_bandplan(filename)
        self._publish_changed(plan)
        return success(
            f"Loaded bandplan '[bold]{plan.name}[/]' [dim]([cyan]{len(plan.bands)}[/] bands)[/]"
        )

    def _off(self) -> str:
        store = get_bandplan_store()
        store.clear()
        save_bandplan(None)
        self._publish_changed(None)
        return success("Bandplan cleared")

    def _show(self) -> str:
        store = get_bandplan_store()
        plan = store.active
        if plan is None:
            return "No bandplan loaded"
        return "\n".join(
            [
                header(f"{plan.name} ({plan.country_name}, {plan.country_code})"),
                field("author", f"{plan.author_name} <{plan.author_url}>"),
                field("bands", f"[cyan]{len(plan.bands)}[/]"),
            ]
        )

    def _publish_changed(self, plan: object) -> None:
        engine = get_engine()
        engine.event_bus.publish(BandplanChangedEvent(bandplan=plan))

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        if tokens and tokens[0] == "load":
            store = get_bandplan_store()
            completions: list[Completion] = []
            for name in store.filenames():
                if not name.startswith(prefix):
                    continue
                plan = store.get(name)
                completions.append(Completion(name, plan.name if plan else ""))
            return completions
        return []
