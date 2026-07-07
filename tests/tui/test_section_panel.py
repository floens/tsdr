"""SectionPanel transposes by dock edge: a vertical Group on the side docks, a
horizontal Table.grid on the bottom bar, with per-section width hints honored
only in the horizontal layout."""

from __future__ import annotations

import asyncio

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult

from tsdr.tui.widgets.rds_widget import RDSWidget
from tsdr.tui.widgets.section_panel import Section, SectionPanel


class _Demo(SectionPanel):
    def build_sections(self) -> list[Section]:
        return [
            Section("a1\na2", width=10, min_width=8),
            Section("b1", min_width=6),
            Section("c1\nc2\nc3"),
        ]


class _Harness(App):
    def compose(self) -> ComposeResult:
        yield _Demo()


def _mount_with_edge(edge: str) -> tuple[object, _Demo]:
    captured: dict[str, object] = {}

    async def go() -> None:
        app = _Harness()
        async with app.run_test(size=(120, 20)) as pilot:
            w = app.query_one(_Demo)
            w.dock_edge = edge
            await pilot.pause()
            captured["render"] = w._render_sections()
            captured["widget"] = w

    asyncio.run(go())
    return captured["render"], captured["widget"]  # type: ignore[return-value]


def test_bottom_edge_renders_horizontal_grid() -> None:
    render, w = _mount_with_edge("bottom")
    assert isinstance(render, Table)
    assert len(render.columns) == 3
    assert w.has_class("panel-wide")
    assert not w.has_class("panel-tall")


def test_side_edge_renders_vertical_group() -> None:
    render, w = _mount_with_edge("left")
    assert isinstance(render, Group)
    assert w.has_class("panel-tall")
    assert not w.has_class("panel-wide")


def test_horizontal_columns_carry_width_hints() -> None:
    render, _ = _mount_with_edge("bottom")
    cols = render.columns  # type: ignore[union-attr]
    assert cols[0].width == 10
    assert cols[0].ratio is None
    # A widthless section sizes to its content and is packed left (no ratio,
    # no expand), rather than sharing leftover space.
    assert cols[2].width is None
    assert cols[2].ratio is None
    assert render.expand is False  # type: ignore[union-attr]


def test_vertical_group_separates_sections_with_blank_line() -> None:
    render, _ = _mount_with_edge("left")
    # Three title-less sections → body, blank, body, blank, body.
    parts = render.renderables  # type: ignore[union-attr]
    assert len(parts) == 5
    assert isinstance(parts[1], Text) and str(parts[1]) == ""
    assert isinstance(parts[3], Text) and str(parts[3]) == ""


def test_rds_group_columns_stack_into_one_section_when_tall() -> None:
    # dock_edge defaults to None (a side dock), so groups stack vertically.
    w = RDSWidget()
    w._group_grid = {"0A": "0A foo", "2A": "2A bar"}
    sections = w._group_sections()
    assert len(sections) == 1
    assert "foo" in sections[0].body
    assert "bar" in sections[0].body


def test_rds_no_groups_yields_no_group_sections() -> None:
    w = RDSWidget()
    assert w._group_sections() == []
