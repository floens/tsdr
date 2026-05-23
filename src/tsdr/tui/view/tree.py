"""derive_tree — pure function from UIModel to the desired WidgetSpec tree.

The only place that decides which widgets exist on screen and what structural
props they have. Must be deterministic and produce children in a stable order
(use explicit lists/tuples, not dict/set iteration) so the reconciler can
preserve widget identity across renders.
"""

from __future__ import annotations

from typing import Any

from tsdr.tui.model import DeviceUIState, UIModel
from tsdr.tui.view.spec import WidgetSpec


def derive_tree(model: UIModel) -> WidgetSpec:
    # Keys double as CSS ids (after _safe_id replaces colons), so dashed form
    # is required for app.tcss selectors like #main-container to match.
    return WidgetSpec(
        "root",
        "root",
        {},
        (
            WidgetSpec("tuner", "tuner", {}),
            WidgetSpec(
                "main_container",
                "main-container",
                {},
                (
                    WidgetSpec("viz_container", "viz-container", {}, _viz_children(model)),
                    *_sidebar_node(model),
                ),
            ),
            WidgetSpec("status_bar", "status-bar", {}),
            WidgetSpec("console", "console", _console_props(model)),
        ),
    )


def _viz_children(m: UIModel) -> tuple[WidgetSpec, ...]:
    children: list[WidgetSpec] = [
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
    ]
    focused = _focused_device(m)
    if focused and focused.active_decoder_kind:
        kind = focused.active_decoder_kind
        decoder_props: dict[str, Any] = {"image_mode": m.image_mode} if kind == "dab" else {}
        children.append(
            WidgetSpec(
                f"decoder_{kind}",
                f"decoder:{focused.device_id}:{kind}",
                decoder_props,
            )
        )
    return tuple(children)


def _sidebar_node(m: UIModel) -> tuple[WidgetSpec, ...]:
    if m.active_panel is None:
        return ()
    children: list[WidgetSpec] = []
    if m.active_panel == "stats":
        children.append(WidgetSpec("stats", "stats", {"focused_device_id": m.focused_device_id}))
        if m.image_mode:
            children.append(WidgetSpec("constellation", "constellation", {"image_mode": True}))
    elif m.active_panel == "performance":
        children.append(WidgetSpec("performance", "performance", {}))
    return (WidgetSpec("sidebar", "sidebar", {}, tuple(children)),)


def _console_props(m: UIModel) -> dict[str, Any]:
    # Single state object so the reactive watcher fires once with consistent
    # values, instead of three watchers reading stale companion attrs mid-batch.
    return {"console_state": m.console}


def _focused_device(m: UIModel) -> DeviceUIState | None:
    if m.focused_device_id is None:
        return None
    for d in m.devices:
        if d.device_id == m.focused_device_id:
            return d
    return None
