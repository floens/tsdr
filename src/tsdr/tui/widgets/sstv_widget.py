"""SSTV decoder widget.

Surfaces the streaming decoder's state to the operator: detected VIS code,
mode name, line counter, state badge, errors — plus a Kitty-rendered
preview of the running image.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Static

from tsdr.core.events.events import DecoderOutputEvent
from tsdr.radio.decoders.sstv import SSTVData, StreamerState
from tsdr.tui.markup import escape_forced
from tsdr.tui.widgets.kitty_image import KittyImageWidget

logger = logging.getLogger(__name__)

_IMAGE_KEY = "sstv-frame"

_STATE_BADGES = {
    StreamerState.LOOKING: "[dim]LOOKING[/dim]",
    StreamerState.DECODING: "[bold yellow]DECODING[/bold yellow]",
    StreamerState.DONE: "[bold green]DONE[/bold green]",
}


def _rgb_to_rgba(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 → (H, W, 4) uint8 with opaque alpha."""
    rgba = np.empty((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = img
    rgba[:, :, 3] = 255
    return rgba


class SSTVWidget(Horizontal):
    """Mode/VIS/line status plus a Kitty preview of the running SSTV image.

    Reactive props:
      image_mode: bool — when True the Kitty image is shown; when False, an
        alt-text Static stands in.
    """

    image_mode = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._latest: SSTVData | None = None
        self._last_rendered_id: int = -1  # id() of the last numpy image rendered
        self._status = Static("", id="sstv-status")
        self._progress = Static("", id="sstv-progress")
        self._alt = Static("[dim][image mode off][/dim]", id="sstv-image-alt")
        self._kitty = KittyImageWidget(id="sstv-image")

    def compose(self):
        yield self._status
        yield self._progress
        yield self._alt
        yield self._kitty

    def on_mount(self) -> None:
        self.border_title = "SSTV"
        self._refresh_display()
        self._apply_image_mode(self.image_mode)

    def on_unmount(self) -> None:
        self._kitty.remove_image(_IMAGE_KEY)

    def watch_image_mode(self, image_mode: bool) -> None:
        self._apply_image_mode(image_mode)

    def _apply_image_mode(self, enabled: bool) -> None:
        self._kitty.display = enabled
        self._alt.display = not enabled
        if enabled and self._latest is not None and self._latest.image is not None:
            # Defer until layout is established so the Kitty widget has a region.
            self.call_after_refresh(self._render_image, self._latest.image)
        else:
            self._kitty.remove_image(_IMAGE_KEY)
            self._last_rendered_id = -1

    def update_messages(self, event: DecoderOutputEvent) -> None:
        latest: SSTVData | None = None
        for msg in event.messages:
            if isinstance(msg.data, SSTVData):
                latest = msg.data
        if latest is None:
            return
        self._latest = latest
        self._refresh_display()
        # A new transmission's on_mode message arrives as state=decoding with
        # line_index=-1 and no image; drop the previous frame so it doesn't
        # linger under the incoming image's first line snapshot.
        if (
            latest.state == StreamerState.DECODING
            and latest.line_index < 0
            and latest.image is None
        ):
            self._kitty.remove_image(_IMAGE_KEY)
            self._last_rendered_id = -1
        if (
            self.image_mode
            and latest.image is not None
            and id(latest.image) != self._last_rendered_id
        ):
            self._render_image(latest.image)
            self._last_rendered_id = id(latest.image)

    def _refresh_display(self) -> None:
        data = self._latest
        if data is None:
            self._status.update("[dim]Waiting for VIS...[/dim]")
            self._progress.update("")
            return

        badge = _STATE_BADGES.get(data.state, data.state.value)
        lines: list[str] = [badge]
        if data.mode_name is not None:
            lines.append(f"[bold]{escape_forced(data.mode_name)}[/bold]")
        if data.vis_code is not None:
            vis_label = f"VIS {data.vis_code} ({data.vis_code:#04x})"
            if data.forced_mode:
                vis_label += " [dim](forced)[/dim]"
            lines.append(vis_label)
        if data.error:
            lines.append(f"[red]{escape_forced(data.error)}[/red]")
        self._status.update("\n".join(lines))

        if data.total_lines > 0 and data.line_index >= 0:
            shown = data.line_index + 1
            pct = 100 * shown / data.total_lines
            progress_lines = [
                f"Line {shown:>4d} / {data.total_lines}",
                f"[dim]{pct:5.1f}%[/dim]",
            ]
            if data.image_width and data.image_height:
                progress_lines.append(f"[dim]{data.image_width}×{data.image_height}[/dim]")
            self._progress.update("\n".join(progress_lines))
        else:
            self._progress.update("")

    def _render_image(self, img: np.ndarray) -> None:
        w_px, h_px = self._kitty.available_pixel_size
        if w_px <= 0 or h_px <= 0:
            return
        pil = Image.fromarray(img)
        pil.thumbnail((w_px, h_px), Image.Resampling.NEAREST)
        rgba = _rgb_to_rgba(np.asarray(pil, dtype=np.uint8))
        self._kitty.update_image(_IMAGE_KEY, rgba)
