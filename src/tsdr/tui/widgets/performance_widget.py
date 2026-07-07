from __future__ import annotations

from tsdr.core.events.events import StatsUpdateEvent
from tsdr.tui.widgets.section_panel import Section, SectionPanel


def _duration_color(ms: float) -> str:
    """Traffic-light color based on duration magnitude."""
    if ms < 1.0:
        return "green"
    if ms < 5.0:
        return "yellow"
    return "red"


class PerformanceWidget(SectionPanel):
    """Display performance metrics as an indented tree.

    Each top-level metric group is a Section: stacked on the tall sidebars,
    side-by-side columns on the wide bottom bar.
    """

    def __init__(self):
        super().__init__()
        self.current_event: StatsUpdateEvent | None = None

    def update_stats(self, event: StatsUpdateEvent) -> None:
        self.current_event = event
        self.refresh()

    def build_sections(self) -> list[Section]:
        if not self.current_event or not self.current_event.performance_stats:
            return [Section("[dim]Performance: No data[/dim]")]

        tree = self._build_tree(self.current_event.performance_stats)
        sections: list[Section] = []
        for name, data in sorted(tree.items()):
            lines: list[str] = []
            self._render_tree({name: data}, lines, indent=0, parent_ms=0.0)
            sections.append(Section("\n".join(lines)))
        return sections

    @staticmethod
    def _build_tree(stats: dict[str, float]) -> dict:
        """Build a nested tree from dot-separated metric names."""
        tree: dict = {}
        for name, duration in stats.items():
            parts = name.split(".")
            node = tree
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {"_children": {}}
                node = node[part]["_children"]
            leaf = parts[-1]
            if leaf not in node:
                node[leaf] = {"_children": {}}
            node[leaf]["_duration"] = duration
        return tree

    def _render_tree(
        self, node: dict, lines: list[str], indent: int, parent_ms: float, is_root: bool = True
    ) -> None:
        """Recursively render tree nodes."""
        if is_root:
            items = sorted(node.items())
        else:
            items = sorted(node.items(), key=lambda kv: kv[1].get("_duration", 0), reverse=True)

        for name, data in items:
            duration: float | None = data.get("_duration")
            children = data.get("_children", {})
            is_other = name == "other"
            prefix = "  " * indent

            if duration is not None:
                color = _duration_color(duration)
                if is_other:
                    lines.append(f"{prefix}[dim]{name:<16} {duration:>6.2f} ms[/dim]")
                else:
                    lines.append(
                        f"{prefix}[bold]{name:<16}[/bold] [{color}]{duration:>6.2f}[/{color}] ms"
                    )
            else:
                lines.append(f"{prefix}[white]{name}[/white]")

            if children:
                self._render_tree(
                    children, lines, indent + 1, parent_ms=duration or 0.0, is_root=False
                )
