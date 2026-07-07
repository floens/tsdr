from argparse import Namespace

from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.model import Edge, UILayout
from tsdr.tui.model.store import get_ui_store
from tsdr.tui.view.panels import PANELS


class PanelCommand(Command):
    @property
    def description(self) -> str:
        return "Show, hide, or list dockable panels"

    def configure(self, parser: CommandParser) -> None:
        sub = parser.add_subparsers(dest="action", required=False)

        sub.add_parser("list", help="List docked panels per edge")

        show_p = sub.add_parser("show", help="Activate a panel on its docked edge")
        show_p.add_argument("id", help="Panel id (rds, dab, stats, …)")

        hide_p = sub.add_parser("hide", help="Hide the active panel on the named edge")
        hide_p.add_argument("id", help="Panel id (rds, dab, stats, …)")

        move_p = sub.add_parser("move", help="Move a panel to another edge and show it there")
        move_p.add_argument("id", help="Panel id (stats, performance, demod, decoder-output)")
        move_p.add_argument("edge", choices=["left", "right", "bottom"], help="Target edge")
        move_p.add_argument(
            "--index", type=int, default=None, help="Insert position within the target edge"
        )

        strips_p = sub.add_parser(
            "strips", help="Show or hide the bottom panel bar (default: toggle)"
        )
        strips_p.add_argument(
            "state",
            nargs="?",
            choices=["on", "off", "toggle"],
            default="toggle",
            help="on=show the bar, off=hide it (hotkeys still work), toggle (default)",
        )

    def run(self, args: Namespace) -> str:
        if args.action is None or args.action == "list":
            return _format_list(get_ui_store().model.layout)
        if args.action == "show":
            return self._show(args.id)
        if args.action == "hide":
            return self._hide(args.id)
        if args.action == "move":
            return self._move(args.id, args.edge, args.index)
        if args.action == "strips":
            return self._strips(args.state)
        return self.help_text()

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        if subcommand in ("show", "hide", "move"):
            return [Completion(pid) for pid in PANELS if pid.startswith(prefix)]
        return []

    def _show(self, panel_id: str) -> str:
        if panel_id not in PANELS:
            return f"Unknown panel: {panel_id}"
        store = get_ui_store()
        layout = store.model.layout
        edge = _find_edge(layout, panel_id)
        if edge is None:
            return f"Panel '{panel_id}' is not docked on any edge"
        store.set_panel_active(edge, panel_id)
        return f"shown panel={panel_id} edge={edge}"

    def _hide(self, panel_id: str) -> str:
        if panel_id not in PANELS:
            return f"Unknown panel: {panel_id}"
        store = get_ui_store()
        layout = store.model.layout
        edge = _find_edge(layout, panel_id)
        if edge is None:
            return f"Panel '{panel_id}' is not docked on any edge"
        edge_panels = getattr(layout, edge)
        if edge_panels.active != panel_id:
            return f"Panel '{panel_id}' is not currently active on edge '{edge}'"
        store.set_panel_active(edge, None)
        return f"hidden panel={panel_id} edge={edge}"

    def _move(self, panel_id: str, edge: Edge, index: int | None) -> str:
        if panel_id not in PANELS:
            return f"Unknown panel: {panel_id}"
        store = get_ui_store()
        store.move_panel(panel_id, edge, index=index)
        store.set_panel_active(edge, panel_id)
        return f"moved panel={panel_id} edge={edge}"

    def _strips(self, state: str) -> str:
        store = get_ui_store()
        current = store.model.layout.strips_visible
        target = (not current) if state == "toggle" else (state == "on")
        store.set_strips_visible(target)
        return f"panel bar {'shown' if target else 'hidden'}"


def _find_edge(layout: UILayout, panel_id: str) -> Edge | None:
    if panel_id in layout.left.panels:
        return "left"
    if panel_id in layout.right.panels:
        return "right"
    if panel_id in layout.bottom.panels:
        return "bottom"
    return None


def _format_list(layout: UILayout) -> str:
    lines: list[str] = []
    for edge in ("left", "right", "bottom"):
        edge_panels = getattr(layout, edge)
        panels = ", ".join(edge_panels.panels) if edge_panels.panels else "(empty)"
        active = edge_panels.active or "(none)"
        lines.append(f"{edge:6}  panels={panels}  active={active}")
    if layout.hotkeys:
        keys = ", ".join(f"{d}={p}" for d, p in layout.hotkeys)
        lines.append(f"hotkeys: {keys}")
    return "\n".join(lines)
