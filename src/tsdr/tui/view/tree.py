"""derive_tree — pure function from UIModel to the desired WidgetSpec tree.

The only place that decides which widgets exist on screen and what structural
props they have. Must be deterministic and produce children in a stable order
(use explicit lists/tuples, not dict/set iteration) so the reconciler can
preserve widget identity across renders.
"""

from __future__ import annotations

from typing import Any

from tsdr.tui.model import Edge, UILayout, UIModel, focused_device
from tsdr.tui.view.panels import PANELS, PanelDef
from tsdr.tui.view.spec import WidgetSpec


def derive_tree(model: UIModel) -> WidgetSpec:
    # Keys double as CSS ids (after _safe_id replaces colons with `--`), so
    # dashed form is required for app.tcss selectors like #main-container.
    return WidgetSpec(
        "root",
        "root",
        {},
        (
            WidgetSpec("tuner", "tuner", {}),
            WidgetSpec("main_container", "main-container", {}, _main_children(model)),
            WidgetSpec("status_bar", "status-bar", {}),
            WidgetSpec("console", "console", _console_props(model)),
        ),
    )


def _main_children(m: UIModel) -> tuple[WidgetSpec, ...]:
    left_children = _edge_children(m, "left")
    right_children = _edge_children(m, "right")
    docks_row = WidgetSpec(
        "docks_row",
        "docks-row",
        {},
        (
            WidgetSpec("dock_left", "dock:left", _dock_props(left_children), left_children),
            WidgetSpec("center_column", "center", {}, _center_children(m)),
            WidgetSpec("dock_right", "dock:right", _dock_props(right_children), right_children),
        ),
    )
    if not m.layout.strips_visible:
        return (docks_row,)
    return (
        docks_row,
        WidgetSpec("edge_strip", "panel-bar", {"glyphs": _panel_bar_glyphs(m)}),
    )


def _dock_props(children: tuple[WidgetSpec, ...]) -> dict[str, str]:
    """Tag the dock with `.empty` when it has no content, so CSS can drop the
    side border (the 1-col vertical line would otherwise still be visible)."""
    return {"classes": "empty" if not children else ""}


def _edge_children(m: UIModel, edge: Edge) -> tuple[WidgetSpec, ...]:
    """Left/right docks hold only the active panel's content wrapper (or nothing).

    The panel launcher lives in the single bottom bar, not on the side edges, so
    a dock with no active panel has zero children — `width: auto` collapses it to
    0 cols and `_dock_props` drops its border.
    """
    return _panel_content_wrapper(m, edge, _edge(m.layout, edge).active)


def _center_children(m: UIModel) -> tuple[WidgetSpec, ...]:
    return (
        WidgetSpec("viz_container", "viz-container", {}, _viz_children(m)),
        *_panel_content_wrapper(m, "bottom", m.layout.bottom.active),
    )


def _viz_children(m: UIModel) -> tuple[WidgetSpec, ...]:
    return (
        WidgetSpec(
            "spectrum",
            "spectrum",
            {
                "zoom": m.zoom,
                "db_min": m.db_min,
                "db_max": m.db_max,
                "image_mode": m.image_mode,
            },
        ),
        WidgetSpec(
            "waterfall",
            "waterfall",
            {
                "zoom": m.zoom,
                "db_min": m.db_min,
                "db_max": m.db_max,
                "image_mode": m.image_mode,
            },
        ),
    )


def _panel_content_wrapper(m: UIModel, edge: Edge, active: str | None) -> tuple[WidgetSpec, ...]:
    """Wrap the active panel's content widgets in a per-edge container.

    Returning a 1-tuple lets callers splat it: empty when no active panel,
    one wrapper when active. The wrapper's id is `panel-content:<edge>` so CSS
    can target left/right (fixed column) vs bottom (full-width auto row)
    distinctly.
    """
    if active is None:
        return ()
    panel_def = PANELS.get(active)
    if panel_def is None:
        return ()
    children = _build_panel_children(m, panel_def, edge)
    if not children:
        return ()
    return (WidgetSpec("panel_content", f"panel-content:{edge}", {}, children),)


_DEMOD_FACTORY_KINDS: dict[str, str] = {
    "rds": "decoder_rds",
    "dab": "decoder_dab",
    "adsb": "decoder_adsb",
    "tetra": "decoder_tetra",
    "dmr": "decoder_dmr",
    "sstv": "decoder_sstv",
}


def _build_panel_children(m: UIModel, panel_def: PanelDef, edge: Edge) -> tuple[WidgetSpec, ...]:
    """Inner widget(s) for a panel.

    Every panel widget receives `dock_edge` (the edge it's docked on) per the
    PanelWidget contract, so it can adapt its layout later.

    - `demod` multiplexes to the decoder widget matching the focused device's
      active kind (RDS/DAB/ADSB/TETRA/DMR); no inner widget when there is no
      active decoder (or kind is `text`, which has its own panel).
    - `stats` additionally includes a ConstellationWidget when image_mode is on.
    """
    if panel_def.panel_id == "demod":
        return _build_demod_children(m, edge)
    key = f"panel:{panel_def.panel_id}"
    if panel_def.panel_id == "stats":
        children: list[WidgetSpec] = [
            WidgetSpec(
                panel_def.kind,
                key,
                {"focused_device_id": m.focused_device_id, "dock_edge": edge},
            ),
        ]
        if m.image_mode:
            children.append(WidgetSpec("constellation", "constellation", {"image_mode": True}))
        return tuple(children)
    return (WidgetSpec(panel_def.kind, key, {"dock_edge": edge}),)


def _build_demod_children(m: UIModel, edge: Edge) -> tuple[WidgetSpec, ...]:
    """Pick the decoder widget for the focused device's active kind.

    The key is `panel:demod:<kind>` so the reconciler unmounts the old widget
    and mounts a fresh one when the user switches demods (each decoder widget
    is a different class with different reactive props — sharing a key would
    let the reconciler reuse a mismatched instance).
    """
    focused = focused_device(m)
    if focused is None:
        return ()
    kind = focused.active_decoder_kind
    factory_kind = _DEMOD_FACTORY_KINDS.get(kind) if kind is not None else None
    if factory_kind is None:
        return ()
    props: dict[str, Any] = {"dock_edge": edge}
    if kind in ("dab", "sstv"):
        props["image_mode"] = m.image_mode
    return (WidgetSpec(factory_kind, f"panel:demod:{kind}", props),)


def _panel_bar_glyphs(m: UIModel) -> tuple[tuple[str, str, bool], ...]:
    """Every docked panel as one row of (digit, label, is_active).

    Walked left → bottom → right so the bar order mirrors the screen (left-edge
    panels first, then bottom, then right); panels keep their docked order within
    an edge. `is_active` is true iff the panel is the active one on its own edge.
    The label is resolved from the panel's `title_of(m)` when it has one (e.g.
    demod → the active decoder name), else its static title.
    """
    layout = m.layout
    digit_by_panel: dict[str, int] = {pid: d for d, pid in layout.hotkeys}
    edges: tuple[Edge, ...] = ("left", "bottom", "right")
    out: list[tuple[str, str, bool]] = []
    for edge in edges:
        edge_panels = _edge(layout, edge)
        for panel_id in edge_panels.panels:
            digit = digit_by_panel.get(panel_id)
            out.append(
                (
                    str(digit) if digit is not None else "",
                    _panel_label(panel_id, m),
                    panel_id == edge_panels.active,
                )
            )
    return tuple(out)


def _panel_label(panel_id: str, m: UIModel) -> str:
    panel_def = PANELS.get(panel_id)
    if panel_def is None:
        return panel_id or "?"
    if panel_def.title_of is not None:
        return panel_def.title_of(m) or panel_def.title
    return panel_def.title


def _edge(layout: UILayout, edge: Edge) -> Any:
    if edge == "left":
        return layout.left
    if edge == "right":
        return layout.right
    return layout.bottom


def _console_props(m: UIModel) -> dict[str, Any]:
    # Single state object so the reactive watcher fires once with consistent
    # values, instead of three watchers reading stale companion attrs mid-batch.
    return {"console_state": m.console}
