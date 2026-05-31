import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from rich.segment import Segment
from rich.style import Style
from textual.reactive import reactive
from textual.strip import Strip
from textual.widget import Widget

from tsdr.core.events.events import FFTUpdateEvent
from tsdr.core.tracing import traced
from tsdr.tui.widgets.dsp_utils import (
    decimate_spectrum,
    normalize_spectrum,
    status_strip,
    zoom_spectrum,
)
from tsdr.tui.widgets.image_mode_mixin import ImageModeMixin

logger = logging.getLogger(__name__)

# SDR# Classic colormap: dark navy -> blue -> white -> yellow -> orange -> red -> dark red
_GRADIENT_STOPS = (
    (0x00, 0x00, 0x20),
    (0x00, 0x00, 0x30),
    (0x00, 0x00, 0x50),
    (0x00, 0x00, 0x91),
    (0x1E, 0x90, 0xFF),
    (0xFF, 0xFF, 0xFF),
    (0xFF, 0xFF, 0x00),
    (0xFE, 0x6D, 0x16),
    (0xFE, 0x6D, 0x16),
    (0xFF, 0x00, 0x00),
    (0xFF, 0x00, 0x00),
    (0xC6, 0x00, 0x00),
    (0x9F, 0x00, 0x00),
    (0x75, 0x00, 0x00),
    (0x4A, 0x00, 0x00),
)


def _build_luts() -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Build 256-entry RGB and RGBA LUTs by interpolating gradient stops."""
    n_stops = len(_GRADIENT_STOPS)
    stops = np.array(_GRADIENT_STOPS, dtype=np.float64)
    rgb = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0 * (n_stops - 1)
        idx = min(int(t), n_stops - 2)
        frac = t - idx
        color = stops[idx] * (1 - frac) + stops[idx + 1] * frac
        rgb[i] = np.clip(color + 0.5, 0, 255).astype(np.uint8)
    rgba = np.empty((256, 4), dtype=np.uint8)
    rgba[:, :3] = rgb
    rgba[:, 3] = 255
    return rgb, rgba


_RGB_LUT, _RGBA_LUT = _build_luts()

_STYLE_NONE = Style()

# Pre-computed color strings and style cache for text-mode waterfall
_COLOR_STRINGS = tuple(f"rgb({r},{g},{b})" for r, g, b in _RGB_LUT)
_CELL_CACHE: dict[int, tuple[str, Style]] = {}


def _get_cell(top: int, bottom: int) -> tuple[str, Style]:
    """Pick the half-block char + style for a (top, bottom) color index pair.

    Index 0 is treated as "no signal": that half renders as terminal bg
    (no bgcolor emitted) so the widget doesn't paint a dark band over the
    user's terminal background.
    """
    key = top << 8 | bottom
    cell = _CELL_CACHE.get(key)
    if cell is None:
        if top == 0 and bottom == 0:
            cell = (" ", _STYLE_NONE)
        elif bottom == 0:
            cell = ("▀", Style(color=_COLOR_STRINGS[top]))
        elif top == 0:
            cell = ("▄", Style(color=_COLOR_STRINGS[bottom]))
        else:
            cell = ("▀", Style(color=_COLOR_STRINGS[top], bgcolor=_COLOR_STRINGS[bottom]))
        _CELL_CACHE[key] = cell
    return cell


_STRIP_HEIGHT = 64


@dataclass
class _ImageStrip:
    key: str
    buffer: NDArray[np.uint8]  # RGBA (STRIP_HEIGHT, width, 4)
    fill: int = 0
    frozen: bool = False
    transmitted_fill: int = 0  # fill level at last update_image


@dataclass
class _StripPlacement:
    key: str
    y: int
    crop_h: int  # 0 = no crop
    transmit_data: NDArray[np.uint8] | None  # non-None = active strip, needs update_image
    remove: bool = False  # off-screen, needs remove_image
    hide: bool = False  # occluded, hide but keep data
    x: int = 0  # horizontal pixel offset


class WaterfallWidget(ImageModeMixin, Widget):
    """Display waterfall visualization with SDR#-style color gradient.

    Supports two rendering modes via ImageModeMixin:

    - **Text mode**: Half-block characters (▀) with fg/bg colors for 2x vertical
      resolution. Uses a circular buffer of color indices that is rebuilt into
      Rich Strips on each update.

    - **Image mode** (Kitty graphics protocol): Renders RGBA pixel strips placed
      via the Kitty image protocol for full-resolution output. Strips are fixed-
      height (_STRIP_HEIGHT rows); the active strip scrolls new rows in at the
      top and freezes when full, creating a linked list of immutable strips that
      are cropped/removed as they scroll off-screen.

    Both modes share zoom, dB range controls, and the SDR# Classic colormap LUT.

    Reactive props:
      zoom, db_min, db_max: float — invalidates text-mode buffer when any change.
      image_mode: bool — toggles between text and kitty image rendering.
    """

    zoom = reactive(1.0)
    db_min = reactive(-100.0)
    db_max = reactive(-30.0)
    image_mode = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self.current_event: FFTUpdateEvent | None = None
        self._buffer: NDArray[np.uint8] | None = None  # Rolling buffer of color indices
        self._write_pos: int = 0  # Circular write position
        self._strips: list[Strip] = []
        self._strip_cache: dict[tuple[int, int], Strip] = {}
        # Image mode strip state
        self._pixel_scale: int = 2
        self._image_strips: list[_ImageStrip] = []
        self._image_scroll: int = 0
        self._strip_counter: int = 0

    def on_mount(self) -> None:
        self._mount_kitty()

    def watch_zoom(self, _zoom: float) -> None:
        self.invalidate_text_buffer()

    def watch_db_min(self, _db_min: float) -> None:
        self.invalidate_text_buffer()

    def watch_db_max(self, _db_max: float) -> None:
        self.invalidate_text_buffer()

    def watch_image_mode(self, enabled: bool) -> None:
        if enabled:
            self._on_image_mode_enabled()
        else:
            self._on_image_mode_disabled()

    @traced("waterfall")
    def update_waterfall(self, event: FFTUpdateEvent) -> None:
        """Update waterfall display with new FFT line."""
        self.current_event = event
        if self.image_mode:
            self._render_waterfall_image(event)
        else:
            self._add_line(event.spectrum)
            self._rebuild_strips()
            self.refresh()

    def on_resize(self) -> None:
        """Handle resize by clearing buffer (will be recreated on next update)."""
        self._buffer = None
        self._write_pos = 0
        self._strip_cache = {}
        if self.image_mode:
            self._clear_image_strips()
            if self.current_event is not None:
                self._render_waterfall_image(self.current_event)
        elif self.current_event is not None:
            self._rebuild_strips()

    def render_line(self, y: int) -> Strip:
        if self.image_mode:
            if y == 0:
                return self._status_strip()
            # Image-area rows: transparent cells so the Kitty image (drawn at
            # z=-1073741825, under text) shows through. Opaque $app-bg cells
            # would hide the image.
            return Strip.blank(self.size.width)
        if y < len(self._strips):
            return self._strips[y]
        return Strip.blank(self.size.width, self.rich_style)

    def invalidate_text_buffer(self) -> None:
        """Invalidate text-mode buffer after zoom/dB change. Image strips are kept."""
        self._buffer = None
        self._write_pos = 0
        self._strip_cache = {}
        self.refresh()

    def _status_strip(self) -> Strip:
        return status_strip(self.size.width, self.zoom, self.db_min, self.db_max, self.rich_style)

    def _render_waterfall_image(self, event: FFTUpdateEvent) -> None:
        full_w, full_h = self._kitty.full_pixel_size
        occ = self._kitty.occlusion_insets
        w = full_w - occ.left - occ.right
        visible_h = full_h - occ.top - occ.bottom
        # logger.debug(
        #     "waterfall_image: full=(%d,%d) occ=%s visible=(%d,%d)",
        #     full_w,
        #     full_h,
        #     occ,
        #     w,
        #     visible_h,
        # )
        if w <= 0 or visible_h <= 0:
            return

        # Phase 1: Mutate strip state (no kitty commands)
        self._fill_active_strip(event, w)

        # Phase 2: Compute layout from final state (pure reads)
        cell_h = self._kitty.cell_height_px
        layout = self._compute_strip_layout(cell_h, occ.top, visible_h, full_h, occ.left)

        # Phase 3: Emit kitty commands from layout (no state mutations)
        self._emit_strip_commands(layout)

    @traced("image_fill_strip")
    def _fill_active_strip(self, event: FFTUpdateEvent, w: int) -> None:
        """Fill active strip buffer with new spectrum row. Freeze and rotate if full."""
        scale = self._pixel_scale
        w = w - (w % scale)

        zoomed = zoom_spectrum(event.spectrum, self.zoom)
        line = decimate_spectrum(zoomed, w // scale)
        normalized = normalize_spectrum(line, self.db_min, self.db_max)
        indices = (normalized * 255).astype(np.intp)
        row = np.repeat(_RGBA_LUT[indices], scale, axis=0)

        # Ensure active strip exists with correct width
        if not self._image_strips or self._image_strips[0].buffer.shape[1] != w:
            self._clear_image_strips()
            buf = np.zeros((_STRIP_HEIGHT, w, 4), dtype=np.uint8)
            key = f"strip_{self._strip_counter}"
            self._strip_counter += 1
            self._image_strips.insert(0, _ImageStrip(key=key, buffer=buf))
            self._image_scroll = 0

        active = self._image_strips[0]

        # Insert scaled rows at top, shift existing rows down
        space = _STRIP_HEIGHT - active.fill
        n_rows = min(scale, space)
        if active.fill > 0 and n_rows > 0:
            limit = min(active.fill, _STRIP_HEIGHT - n_rows)
            active.buffer[n_rows : n_rows + limit] = active.buffer[:limit]
        for r in range(n_rows):
            active.buffer[r] = row
        active.fill = min(active.fill + n_rows, _STRIP_HEIGHT)
        self._image_scroll += n_rows

        # Freeze and create new strip when full
        if active.fill >= _STRIP_HEIGHT:
            active.frozen = True
            buf = np.zeros((_STRIP_HEIGHT, w, 4), dtype=np.uint8)
            key = f"strip_{self._strip_counter}"
            self._strip_counter += 1
            self._image_strips.insert(0, _ImageStrip(key=key, buffer=buf))
            self._image_scroll = 0

    @traced("image_compute_layout")
    def _compute_strip_layout(
        self, cell_h: int, y_origin: int, visible_h: int, full_h: int, x_offset: int
    ) -> list[_StripPlacement]:
        """Compute placement for all strips from current state.

        Three zones based on y position relative to the widget:
        - Visible: y_origin to y_origin + visible_h -> display normally
        - Occluded: past visible_h but within full_h -> hide (keep data)
        - Off-screen: past full_h -> remove (clean up)
        """
        layout: list[_StripPlacement] = []
        visible_end = y_origin + visible_h

        # Active strip at index 0
        active = self._image_strips[0]
        if active.fill > 0:
            layout.append(
                _StripPlacement(
                    key=active.key,
                    y=y_origin + cell_h,
                    crop_h=0,
                    transmit_data=active.buffer[: active.fill],
                    x=x_offset,
                )
            )

        # Remaining strips (frozen)
        y_offset = y_origin + cell_h + active.fill
        for i in range(1, len(self._image_strips)):
            strip = self._image_strips[i]

            if y_offset >= full_h:
                # Off-screen: remove entirely
                layout.append(
                    _StripPlacement(
                        key=strip.key,
                        y=0,
                        crop_h=0,
                        transmit_data=None,
                        remove=True,
                    )
                )
                continue

            if y_offset >= visible_end:
                # Occluded: hide but keep data
                layout.append(
                    _StripPlacement(
                        key=strip.key,
                        y=y_offset,
                        crop_h=0,
                        transmit_data=None,
                        hide=True,
                    )
                )
                y_offset += strip.fill
                continue

            crop_h = 0
            if y_offset + strip.fill > visible_end:
                crop_h = visible_end - y_offset

            # Re-transmit if pixel data changed since last update_image
            needs_transmit = strip.fill != strip.transmitted_fill
            layout.append(
                _StripPlacement(
                    key=strip.key,
                    y=y_offset,
                    crop_h=crop_h,
                    transmit_data=strip.buffer[: strip.fill] if needs_transmit else None,
                    x=x_offset,
                )
            )
            y_offset += strip.fill

        return layout

    @traced("image_emit_commands")
    def _emit_strip_commands(self, layout: list[_StripPlacement]) -> None:
        """Emit kitty commands for computed layout. Remove off-screen strips."""
        for placement in layout:
            if placement.remove:
                self._kitty.remove_image(placement.key)
                self._image_strips[:] = [s for s in self._image_strips if s.key != placement.key]
            elif placement.hide:
                self._kitty.hide_image(placement.key)
                for s in self._image_strips:
                    if s.key == placement.key:
                        s.transmitted_fill = 0
                        break
            elif placement.transmit_data is not None:
                self._kitty.update_image(
                    placement.key,
                    placement.transmit_data,
                    x=placement.x,
                    y=placement.y,
                )
                for s in self._image_strips:
                    if s.key == placement.key:
                        s.transmitted_fill = len(placement.transmit_data)
                        break
            else:
                self._kitty.place_image(
                    placement.key,
                    x=placement.x,
                    y=placement.y,
                    crop_h=placement.crop_h,
                )

    def _on_image_mode_enabled(self) -> None:
        self._strips = []
        self.refresh()
        if self.current_event:
            self._render_waterfall_image(self.current_event)

    def _on_image_mode_disabled(self) -> None:
        self._clear_image_strips()

    def _clear_image_strips(self) -> None:
        """Remove all image strips and reset state."""
        for strip in self._image_strips:
            self._kitty.remove_image(strip.key)
        self._image_strips.clear()
        self._image_scroll = 0
        self._strip_counter = 0

    @traced("text_add_line")
    def _add_line(self, spectrum: NDArray[np.float32]) -> None:
        """Add a new spectrum line to the rolling buffer."""
        width = self.size.width
        height = (self.size.height - 2) * 2  # 2 rows per terminal line (half-blocks)

        if width < 10 or height < 4:
            return

        # Ensure buffer exists with correct dimensions
        if self._buffer is None or self._buffer.shape != (height, width):
            self._buffer = np.zeros((height, width), dtype=np.uint8)
            self._write_pos = 0
            self._strip_cache = {}

        zoomed = zoom_spectrum(spectrum, self.zoom)
        line = decimate_spectrum(zoomed, width)
        normalized = normalize_spectrum(line, self.db_min, self.db_max)
        color_indices = (normalized * 255).astype(np.uint8)

        # Write to circular buffer
        self._buffer[self._write_pos] = color_indices
        self._write_pos = (self._write_pos + 1) % self._buffer.shape[0]

    @traced("text_rebuild_strips")
    def _rebuild_strips(self) -> None:
        """Rebuild Strip objects from the rolling buffer.

        Uses a cache keyed by (top_buffer_idx, bottom_buffer_idx) so only
        strips referencing the just-written buffer position are rebuilt.
        """
        event = self.current_event
        base = self.rich_style
        if event is None:
            self._strips = [Strip([Segment("Waterfall: No data", base)])]
            return

        width = self.size.width
        height = self.size.height

        if width < 10 or height < 5:
            self._strips = [Strip([Segment("Too small", base)])]
            return

        if self._buffer is None:
            self._strips = [Strip([Segment("Waterfall: No data", base)])]
            return

        strips: list[Strip] = [
            self._status_strip(),
            Strip([Segment("─" * width, base)], width),
        ]

        buffer_height = self._buffer.shape[0]
        display_rows = height - 2
        written_pos = (self._write_pos - 1) % buffer_height
        old_cache = self._strip_cache
        new_cache: dict[tuple[int, int], Strip] = {}

        for row in range(display_rows):
            top_idx = (self._write_pos - 1 - row * 2) % buffer_height
            bottom_idx = (self._write_pos - 2 - row * 2) % buffer_height
            key = (top_idx, bottom_idx)

            if top_idx != written_pos and bottom_idx != written_pos:
                cached = old_cache.get(key)
                if cached is not None:
                    new_cache[key] = cached
                    strips.append(cached)
                    continue

            strip = self._render_half_block_row(
                self._buffer[top_idx], self._buffer[bottom_idx], width
            )
            new_cache[key] = strip
            strips.append(strip)

        self._strip_cache = new_cache
        self._strips = strips

    def _render_half_block_row(
        self,
        top_colors: NDArray[np.uint8],
        bottom_colors: NDArray[np.uint8],
        width: int,
    ) -> Strip:
        """Render a single row using half-block characters with fg/bg colors.

        Vectorized: numpy finds run boundaries, then a short Python loop
        over runs (~20-40) builds segments with cached Style objects.
        """
        # Combine top+bottom into single uint16 for vectorized comparison
        combined = top_colors.astype(np.uint16) << 8 | bottom_colors
        changes = np.nonzero(np.diff(combined))[0] + 1

        # Build boundary array: [0, change1, change2, ..., width]
        n_runs = len(changes) + 1
        boundaries = np.empty(n_runs + 1, dtype=np.intp)
        boundaries[0] = 0
        boundaries[1:-1] = changes
        boundaries[-1] = len(top_colors)

        base = self.rich_style
        segments: list[Segment] = []
        for i in range(n_runs):
            start = int(boundaries[i])
            length = int(boundaries[i + 1]) - start
            top = int(top_colors[start])
            bottom = int(bottom_colors[start])
            char, style = _get_cell(top, bottom)
            # When a half is index 0 the cell style has no bgcolor, so it
            # won't inherit the widget bg. Merge with base so unpainted
            # halves paint the widget's own bg rather than the terminal's.
            if top == 0 or bottom == 0:
                style = base + style
            segments.append(Segment(char * length, style))

        if len(top_colors) < width:
            segments.append(Segment(" " * (width - len(top_colors)), base))

        return Strip(segments, width)
