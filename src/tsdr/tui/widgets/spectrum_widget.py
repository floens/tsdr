import logging
from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.events import Click, MouseScrollDown, MouseScrollUp
from textual.strip import Strip
from textual.timer import Timer
from textual.widget import Widget

from tsdr.core.bandplans import Bandplan, band_type_color, contrast_fg
from tsdr.core.events.events import (
    BandplanChangedEvent,
    FFTUpdateEvent,
    MemoriesChangedEvent,
)
from tsdr.core.memories import Memory, get_memory_store, memory_color, recall_memory
from tsdr.core.preferences import save_device, save_ui_state
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.tracing import traced
from tsdr.core.tuning import save_previous_tune_state
from tsdr.core.units import format_hz
from tsdr.tui.inline_edit import InlineEditBuffer
from tsdr.tui.state import UIState
from tsdr.tui.widgets.dsp_utils import (
    decimate_spectrum,
    normalize_spectrum,
    render_spectrum_to_buf,
    status_strip,
    zoom_spectrum,
)
from tsdr.tui.widgets.image_mode_mixin import ImageModeMixin
from tsdr.tui.widgets.waterfall_widget import WaterfallWidget

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _MemoryEdit:
    """Active inline edit state for a memory label."""

    memory_id: str
    buffer: InlineEditBuffer


# Quadrant block lookup: index by 4-bit pattern
# (upper_left << 3) | (upper_right << 2) | (lower_left << 1) | lower_right
_QUADRANT_BLOCKS = (
    " ",
    "▗",
    "▖",
    "▄",
    "▝",
    "▐",
    "▞",
    "▟",
    "▘",
    "▚",
    "▌",
    "▙",
    "▀",
    "▜",
    "▛",
    "█",
)

_STYLE_NONE = Style()
_STYLE_DIM = Style(dim=True)
_STYLE_GREEN = Style(color="green")
_STYLE_BLUE = Style(color="blue")

# Image mode colors (RGBA)
_TRACE_COLOR = np.array([0x40, 0xA0, 0xFF, 0xFF], dtype=np.uint8)  # bright blue
_FILL_COLOR = np.array([0x20, 0x50, 0x80, 0x60], dtype=np.uint8)  # semi-transparent blue
_BW_BOX_COLOR = np.array([40, 40, 40, 255], dtype=np.uint8)  # dark grey bandwidth box
_CENTER_LINE_COLOR = np.array([255, 0, 0, 200], dtype=np.uint8)  # red center line


@dataclass(frozen=True, slots=True)
class SpectrumFrame:
    """Pre-computed frame ready for rendering."""

    header: str
    cells: tuple[tuple[int, ...], ...]  # rows of cell indices (0-15)
    width: int
    height: int
    freq_axis_labels: tuple[tuple[int, str], ...]  # (col, label) pairs
    bandwidth_range: tuple[int, int] | None  # (col_low, col_high) or None
    memory_labels: tuple[
        tuple[tuple[int, str, str, str], ...], ...
    ] = ()  # rows of (col, name, color_hex, memory_id)
    bandplan_segments: tuple[
        tuple[int, int, str, str], ...
    ] = ()  # (col_start, col_end_exclusive, label, color_hex)


class SpectrumWidget(ImageModeMixin, Widget):
    """Display power spectrum with blue bars, bandwidth indicator, and frequency axis.

    Uses quadrant block characters for 2x2 resolution per terminal cell.
    Supports Kitty image mode for line plot rendering.
    """

    def __init__(self, ui_state: UIState) -> None:
        super().__init__()
        self._ui_state = ui_state
        self.current_event: FFTUpdateEvent | None = None
        self._channel_bandwidth: float | None = None
        self._memories: tuple[Memory, ...] = ()
        self._bandplan: Bandplan | None = None
        self._strips: list[Strip] = []
        self._image_key = "spectrum"
        self._edit: _MemoryEdit | None = None
        self._edit_blink_timer: Timer | None = None

    def on_mount(self) -> None:
        self._mount_kitty()
        self._read_config()
        # Render overlays on a blank spectrum so bandplan / memories / freq axis
        # are visible before the first FFT arrives.
        if not self.image_mode:
            self._rebuild_strips()
        self.refresh()

    def update_spectrum(self, event: FFTUpdateEvent) -> None:
        """Update spectrum display from event."""
        self.current_event = event
        if self.image_mode:
            self._render_spectrum_image(event)
        else:
            self._rebuild_strips()
            self.refresh()

    def _read_config(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return
        # Prefer explicit config value; fall back to demodulator default
        if device.config.channel_bandwidth is not None:
            self._channel_bandwidth = device.config.channel_bandwidth
        else:
            demod_info = device.active_demod_info
            self._channel_bandwidth = demod_info.channel_bandwidth if demod_info else None

    def update_config(self) -> None:
        """Config changed - bars shift within visible range, overlays follow new config."""
        self._read_config()
        self._refresh_display()

    def update_memories(self, event: MemoriesChangedEvent) -> None:
        """Update memory labels from event snapshot."""
        self._memories = tuple(event.memories)  # type: ignore[arg-type]
        if self._edit is not None and not any(m.id == self._edit.memory_id for m in self._memories):
            self.cancel_edit()
            return
        self._refresh_display()

    def update_bandplan(self, event: BandplanChangedEvent) -> None:
        """Update active bandplan from event."""
        self._bandplan = event.bandplan  # type: ignore[assignment]
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Re-render after a non-FFT state change (config, memories, bandplan)."""
        if self.image_mode:
            if self.current_event is not None:
                self._render_spectrum_image(self.current_event)
            else:
                self.refresh()
        else:
            self._rebuild_strips()
            self.refresh()

    def on_resize(self) -> None:
        if self.image_mode:
            self._kitty.remove_image(self._image_key)
            if self.current_event is not None:
                self._render_spectrum_image(self.current_event)
        elif self.current_event is not None:
            self._rebuild_strips()

    def invalidate_frame_buffer(self) -> None:
        """Invalidate after zoom/dB change."""
        self._strips = []
        if self.image_mode and self.current_event is not None:
            self._render_spectrum_image(self.current_event)
        self.refresh()

    # Inline memory editing

    @property
    def is_editing(self) -> bool:
        return self._edit is not None

    def start_edit(self, memory: Memory) -> None:
        self._edit = _MemoryEdit(memory.id, InlineEditBuffer(memory.name))
        self._edit_blink_timer = self.set_interval(0.53, self._toggle_edit_cursor)
        self._rebuild_strips()
        self.refresh()

    def confirm_edit(self) -> None:
        if self._edit is None:
            return
        store = get_memory_store()
        store.rename(self._edit.memory_id, self._edit.buffer.value)
        engine = get_engine()
        engine.event_bus.publish(MemoriesChangedEvent(memories=tuple(store.all())))
        self._end_edit()

    def cancel_edit(self) -> None:
        self._end_edit()

    def _end_edit(self) -> None:
        if self._edit_blink_timer is not None:
            self._edit_blink_timer.stop()
            self._edit_blink_timer = None
        self._edit = None
        self._rebuild_strips()
        self.refresh()

    def _toggle_edit_cursor(self) -> None:
        if self._edit is not None:
            self._edit.buffer.toggle_cursor()
            self._rebuild_strips()
            self.refresh()

    def _reset_edit_cursor(self) -> None:
        if self._edit is not None:
            self._edit.buffer.reset_cursor()
            if self._edit_blink_timer is not None:
                self._edit_blink_timer.reset()

    def handle_edit_key(self, event: events.Key) -> None:
        """Handle keyboard input during inline edit mode."""
        if self._edit is None:
            return
        buf = self._edit.buffer
        if event.key == "enter":
            self.confirm_edit()
        elif event.key == "escape":
            self.cancel_edit()
        elif event.key == "backspace":
            buf.backspace()
            self._reset_edit_cursor()
            self._rebuild_strips()
            self.refresh()
        elif event.key == "delete":
            buf.delete()
            self._reset_edit_cursor()
            self._rebuild_strips()
            self.refresh()
        elif event.key == "left":
            buf.move_left()
            self._reset_edit_cursor()
            self._rebuild_strips()
            self.refresh()
        elif event.key == "right":
            buf.move_right()
            self._reset_edit_cursor()
            self._rebuild_strips()
            self.refresh()
        elif event.key == "home":
            buf.home()
            self._reset_edit_cursor()
            self._rebuild_strips()
            self.refresh()
        elif event.key == "end":
            buf.end()
            self._reset_edit_cursor()
            self._rebuild_strips()
            self.refresh()
        elif event.character and event.is_printable:
            buf.insert(event.character)
            self._reset_edit_cursor()
            self._rebuild_strips()
            self.refresh()
        event.prevent_default()
        event.stop()

    def render_line(self, y: int) -> Strip:
        if self.image_mode:
            if y == 0:
                return self._status_strip()
            live = self._display_range()
            if live is not None:
                freq_min, freq_max = live
                width = self.size.width
                if y in (1, 2) and self._memories:
                    label_rows = self._compute_memory_labels(width, freq_min, freq_max)
                    row_idx = y - 1
                    if row_idx < len(label_rows) and label_rows[row_idx]:
                        return self._render_memory_label_strip(width, label_rows[row_idx])
                if y == self.size.height - 2:
                    segments = self._compute_bandplan_segments(width, freq_min, freq_max)
                    return self._render_bandplan_strip(width, segments)
                if y == self.size.height - 1:
                    labels = self._compute_freq_labels(width, freq_min, freq_max)
                    return self._render_freq_axis_strip(width, labels)
            # Image-area rows: transparent cells so the Kitty image (drawn at
            # z=-1073741825, under text) shows through. The image buffer's own
            # RGB already fills unpainted pixels with $app-bg.
            return Strip.blank(self.size.width)
        if y < len(self._strips):
            return self._strips[y]
        return Strip.blank(self.size.width, self.rich_style)

    def _status_strip(self) -> Strip:
        return status_strip(self.size.width, self._ui_state, self.rich_style)

    # Image mode rendering

    @traced("spectrum_image")
    def _render_spectrum_image(self, event: FFTUpdateEvent) -> None:
        full_w, full_h = self._kitty.full_pixel_size
        occ = self._kitty.occlusion_insets
        w = full_w - occ.left - occ.right
        h = full_h - occ.top - occ.bottom
        if w <= 0 or h <= 0:
            return

        cell_h = self._kitty.cell_height_px
        top_rows = 1
        bottom_rows = 2  # bandplan strip + frequency axis
        plot_h = h - (top_rows + bottom_rows) * cell_h
        if plot_h <= 0:
            return

        live = self._display_range()
        if live is None:
            return
        freq_min, freq_max = live

        zoomed = zoom_spectrum(event.spectrum, self._ui_state.zoom)
        e_fmin, e_fmax = self._actual_freq_range(event)
        line = self._shift_spectrum_to_live(zoomed, e_fmin, e_fmax, freq_min, freq_max, w)
        normalized = normalize_spectrum(line, self._ui_state.db_min, self._ui_state.db_max)

        buf = np.zeros((plot_h, w, 4), dtype=np.uint8)

        y_vals = (plot_h - 1) - (normalized * (plot_h - 1)).astype(np.intp)
        bw_range = self._compute_bandwidth_range(w, freq_min, freq_max)
        bw_low, bw_high = bw_range if bw_range else (-1, -1)
        # Config center always sits at the middle of the visible range.
        center_x = w // 2

        render_spectrum_to_buf(
            buf,
            y_vals,
            _FILL_COLOR,
            _TRACE_COLOR,
            bw_low,
            bw_high,
            _BW_BOX_COLOR,
            center_x,
            _CENTER_LINE_COLOR,
        )

        span = freq_max - freq_min
        if self._memories and span > 0:
            # Pre-compute dashed row indices (3px on, 3px off)
            dash_rows = np.concatenate(
                [np.arange(r, min(r + 3, plot_h)) for r in range(0, plot_h, 6)]
            )
            for m in self._memories:
                if m.frequency < freq_min or m.frequency > freq_max:
                    continue
                mx = int((m.frequency - freq_min) / span * w)
                if 0 <= mx < w:
                    color_hex = memory_color(m)
                    r = int(color_hex[1:3], 16)
                    g = int(color_hex[3:5], 16)
                    b = int(color_hex[5:7], 16)
                    buf[dash_rows, mx] = [r, g, b, 200]

        self._kitty.update_image(self._image_key, buf, x=occ.left, y=occ.top + top_rows * cell_h)
        self.refresh()

    def _on_image_mode_enabled(self) -> None:
        self._strips = []
        self.refresh()
        if self.current_event:
            self._render_spectrum_image(self.current_event)

    def _on_image_mode_disabled(self) -> None:
        self._kitty.remove_image(self._image_key)

    # Text mode rendering

    @traced("spectrum_strips")
    def _rebuild_strips(self) -> None:
        """Pre-compute all Strip objects from current data."""
        base = self.rich_style
        width = self.size.width
        height = self.size.height

        if width < 10 or height < 6:
            self._strips = [Strip([Segment("Too small", base)])]
            return

        live = self._display_range()
        if live is None:
            self._strips = [Strip([Segment("Spectrum: No device", base)])]
            return

        frame = self._prepare_frame(self.current_event, width, height, live)
        self._strips = self._render_frame(frame)

    @traced("prepare_frame")
    def _prepare_frame(
        self,
        event: FFTUpdateEvent | None,
        width: int,
        height: int,
        live_range: tuple[float, float],
    ) -> SpectrumFrame:
        """Transform spectrum data into frame buffer.

        Overlays always use `live_range` (config-led). Bars shift from the event's
        freq range into `live_range`; gaps are blank.
        """
        bars_height = max(1, height - 4)
        freq_min, freq_max = live_range
        target = width * 2

        if event is None:
            normalized = np.zeros(target, dtype=np.float32)
        else:
            zoomed = zoom_spectrum(event.spectrum, self._ui_state.zoom)
            e_fmin, e_fmax = self._actual_freq_range(event)
            spectrum = self._shift_spectrum_to_live(
                zoomed, e_fmin, e_fmax, freq_min, freq_max, target
            )
            normalized = normalize_spectrum(spectrum, self._ui_state.db_min, self._ui_state.db_max)

        # Build cell grid using vectorized operations
        rows_arr = np.arange(bars_height)
        threshold_upper = 1.0 - (rows_arr * 2 + 1) / (bars_height * 2)
        threshold_lower = 1.0 - (rows_arr * 2 + 2) / (bars_height * 2)

        left_vals = normalized[0::2]
        right_vals = normalized[1::2]

        # Broadcast comparisons: (width, 1) > (bars_height,) -> (width, bars_height)
        upper_left = left_vals[:, np.newaxis] > threshold_upper
        upper_right = right_vals[:, np.newaxis] > threshold_upper
        lower_left = left_vals[:, np.newaxis] > threshold_lower
        lower_right = right_vals[:, np.newaxis] > threshold_lower

        cells = (
            upper_left.astype(np.uint8) << 3
            | upper_right.astype(np.uint8) << 2
            | lower_left.astype(np.uint8) << 1
            | lower_right.astype(np.uint8)
        ).T

        rows = tuple(tuple(row) for row in cells)

        freq_axis_labels = self._compute_freq_labels(width, freq_min, freq_max)
        bandwidth_range = self._compute_bandwidth_range(width, freq_min, freq_max)
        header = f"Zoom: {self._ui_state.zoom:.1f}x | Min: {self._ui_state.db_min:.0f} dB | Max: {self._ui_state.db_max:.0f} dB"
        memory_labels = self._compute_memory_labels(width, freq_min, freq_max)
        bandplan_segments = self._compute_bandplan_segments(width, freq_min, freq_max)

        return SpectrumFrame(
            header=header,
            cells=tuple(rows),
            width=width,
            height=bars_height,
            freq_axis_labels=freq_axis_labels,
            bandwidth_range=bandwidth_range,
            memory_labels=memory_labels,
            bandplan_segments=bandplan_segments,
        )

    @traced("render_frame")
    def _render_frame(self, frame: SpectrumFrame) -> list[Strip]:
        """Render frame buffer to Strip objects."""
        strips: list[Strip] = []
        width = frame.width
        base = self.rich_style

        # Header strip
        strips.append(Strip([Segment(frame.header.ljust(width)[:width], base)], width))

        for label_row in frame.memory_labels:
            if label_row:
                strips.append(self._render_memory_label_strip(width, label_row))

        # Bandwidth indicator strip
        strips.append(self._render_bandwidth_strip(width, frame.bandwidth_range))

        for row_cells in frame.cells:
            strips.append(self._render_bar_row(row_cells, width))

        # Bandplan row (always present; blank when inactive)
        strips.append(self._render_bandplan_strip(width, frame.bandplan_segments))

        # Frequency axis strip
        strips.append(self._render_freq_axis_strip(width, frame.freq_axis_labels))

        return strips

    def _render_bar_row(self, cells: tuple[int, ...], width: int) -> Strip:
        """Render a single bar row with run-length encoding."""
        segments: list[Segment] = []
        base = self.rich_style
        run_char: str | None = None
        run_style: Style | None = None
        run_len = 0

        for cell_idx in cells:
            char = _QUADRANT_BLOCKS[cell_idx]
            style = _STYLE_BLUE if cell_idx != 0 else base

            if char == run_char and style is run_style:
                run_len += 1
            else:
                if run_len > 0 and run_char is not None:
                    segments.append(Segment(run_char * run_len, run_style))
                run_char = char
                run_style = style
                run_len = 1

        # Flush last run
        if run_len > 0 and run_char is not None:
            segments.append(Segment(run_char * run_len, run_style))

        # Pad if needed
        num_cols = len(cells)
        if num_cols < width:
            segments.append(Segment(" " * (width - num_cols), base))

        return Strip(segments, width)

    def _render_bandwidth_strip(self, width: int, bandwidth_range: tuple[int, int] | None) -> Strip:
        """Render bandwidth indicator strip."""
        base = self.rich_style
        dim = base + _STYLE_DIM
        if bandwidth_range is None:
            return Strip([Segment("─" * width, dim)], width)

        col_low, col_high = bandwidth_range
        segments: list[Segment] = []
        if col_low > 0:
            segments.append(Segment("─" * col_low, dim))
        if col_high > col_low:
            segments.append(Segment("█" * (col_high - col_low), base + _STYLE_GREEN))
        if col_high < width:
            segments.append(Segment("─" * (width - col_high), dim))

        return Strip(segments, width)

    def _render_bandplan_strip(
        self, width: int, segments: tuple[tuple[int, int, str, str], ...]
    ) -> Strip:
        """Render bandplan row: each band is a solid color block with contrasting
        label text on top (same bg across the whole band, including letters).
        """
        base = self.rich_style
        if not segments:
            return Strip.blank(width, base)

        chars: list[str] = [" "] * width
        styles: list[Style] = [base] * width

        for col_start, col_end, name, color in segments:
            seg_style = base + Style(color=contrast_fg(color), bgcolor=color)
            for c in range(col_start, col_end):
                chars[c] = " "
                styles[c] = seg_style
            seg_w = col_end - col_start
            if name and seg_w >= len(name) + 2:
                label_start = col_start + (seg_w - len(name)) // 2
                for i, ch in enumerate(name):
                    chars[label_start + i] = ch
            elif name and seg_w >= 5:
                truncated = name[: seg_w - 3] + "…"
                label_start = col_start + (seg_w - len(truncated)) // 2
                for i, ch in enumerate(truncated):
                    chars[label_start + i] = ch

        segs: list[Segment] = []
        i = 0
        while i < width:
            j = i + 1
            while j < width and chars[j] == chars[i] and styles[j] is styles[i]:
                j += 1
            segs.append(Segment(chars[i] * (j - i), styles[i]))
            i = j
        return Strip(segs, width)

    def _render_freq_axis_strip(self, width: int, labels: tuple[tuple[int, str], ...]) -> Strip:
        """Render frequency axis strip."""
        base = self.rich_style
        dim = base + _STYLE_DIM
        segments: list[Segment] = []
        pos = 0
        for start, label in labels:
            if start > pos:
                segments.append(Segment("─" * (start - pos), dim))
            segments.append(Segment(label, base))
            pos = start + len(label)
        if pos < width:
            segments.append(Segment("─" * (width - pos), dim))

        return Strip(segments, width)

    def _compute_memory_labels(
        self, width: int, freq_min: float, freq_max: float
    ) -> tuple[tuple[tuple[int, str, str, str], ...], ...]:
        """Compute visible memory label positions across up to 2 rows."""
        if not self._memories:
            return ()
        span = freq_max - freq_min
        if span <= 0:
            return ()

        edit = self._edit
        rows: list[list[tuple[int, str, str, str]]] = [[], []]
        occupied_ends = [-1, -1]
        for m in sorted(self._memories, key=lambda m: m.frequency):
            if m.frequency < freq_min or m.frequency > freq_max:
                continue
            col = int((m.frequency - freq_min) / span * width)
            col = max(0, min(width - 1, col))
            is_editing = edit is not None and m.id == edit.memory_id
            name = edit.buffer.value if edit is not None and is_editing else m.name
            label = f"▼{name}"
            if col + len(label) > width:
                label = label[: width - col]
            if not label:
                continue
            color = memory_color(m)
            # +1 for cursor space when editing
            end = col + len(label) + (1 if is_editing else 0)
            # Try row 0, then row 1, skip if both occupied
            for r in range(2):
                if col > occupied_ends[r]:
                    rows[r].append((col, label, color, m.id))
                    occupied_ends[r] = end
                    break

        result = tuple(tuple(row) for row in rows if row)
        return result

    def _render_memory_label_strip(
        self, width: int, labels: tuple[tuple[int, str, str, str], ...]
    ) -> Strip:
        """Render memory labels as a colored strip."""
        base = self.rich_style
        segments: list[Segment] = []
        pos = 0
        for col, label, color, memory_id in labels:
            if col > pos:
                segments.append(Segment(" " * (col - pos), base))
            if self._edit is not None and memory_id == self._edit.memory_id:
                # Render ▼ prefix + editable name with cursor
                mem_style = base + Style(color=color)
                segments.append(Segment("▼", mem_style))
                edit_segments = self._edit.buffer.render_segments(mem_style)
                segments.extend(edit_segments)
                pos = col + 1 + sum(len(s.text) for s in edit_segments)
            else:
                segments.append(Segment(label, base + Style(color=color)))
                pos = col + len(label)
        if pos < width:
            segments.append(Segment(" " * (width - pos), base))
        return Strip(segments, width)

    # Shared helpers

    def on_click(self, event: Click) -> None:
        if self.current_event is None:
            return
        freq_min, freq_max = self._actual_freq_range(self.current_event)
        if self.image_mode:
            occ = self._kitty.occlusion_insets
            w = self._kitty.full_pixel_size[0] - occ.left - occ.right
            x = event.x * self._kitty.cell_width_px - occ.left
        else:
            w = self.size.width
            x = event.x
        if w <= 0 or x < 0 or x >= w:
            return
        freq = freq_min + (x / w) * (freq_max - freq_min)

        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        # Check if click is near a memory label
        if self._memories:
            span = freq_max - freq_min
            # TODO: hardcoded pixels
            tolerance = span / w * 15 if w > 0 else 0
            for m in self._memories:
                if abs(m.frequency - freq) <= tolerance:
                    try:
                        recall_memory(m, device.device_id)
                    except SDRException as e:
                        self.app._show_error(str(e))
                    event.stop()
                    return

        # Snap to the pixel grid so click and scroll agree on a column.
        pixel_hz = (freq_max - freq_min) / w if w > 0 else 1.0
        snap = max(pixel_hz, 1.0)
        freq = round(freq / snap) * snap
        save_previous_tune_state(device)
        engine.update_device_config(device.device_id, center_frequency=freq)
        save_device(engine)
        event.stop()

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        if event.shift:
            self._scroll_zoom(1)
        else:
            self._scroll_tune(-1)
        event.stop()

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        if event.shift:
            self._scroll_zoom(-1)
        else:
            self._scroll_tune(1)
        event.stop()

    def _scroll_zoom(self, direction: int) -> None:
        self._ui_state.adjust_zoom(direction)
        self.invalidate_frame_buffer()
        self.screen.query_one(WaterfallWidget).invalidate_text_buffer()
        save_ui_state(self._ui_state)

    def _scroll_tune(self, direction: int) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return
        # Pixel-relative step: visible span / display width × 2 (2 pixels per notch).
        # Falls back to 1 Hz minimum so we never lose visible motion.
        sample_rate = device.config.sample_rate
        visible_span = sample_rate / self._ui_state.zoom
        w = self.size.width
        pixel_hz = visible_span / w if w > 0 else 1.0
        step = max(pixel_hz * 2, 1.0)
        freq = device.config.center_frequency
        new_freq = freq + direction * step
        new_freq = round(new_freq / step) * step
        engine.update_device_config(device.device_id, center_frequency=new_freq)
        save_device(engine)

    def _actual_freq_range(self, event: FFTUpdateEvent) -> tuple[float, float]:
        """Return (freq_min, freq_max) for the captured FFT event, accounting for zoom."""
        visible_bw = event.sample_rate / self._ui_state.zoom
        freq_min = event.center_frequency - visible_bw / 2
        freq_max = event.center_frequency + visible_bw / 2
        return freq_min, freq_max

    def _display_range(self) -> tuple[float, float] | None:
        """Return the freq range used for rendering both bars and overlays.

        When RUNNING and we have an FFT event: use the event's range. Overlays
        only update when a new FFT lands, matching what the spectrum shows; no
        transient shift during SDR retune.

        Otherwise (stopped, pre-first-FFT, or no focused device with a
        `current_event`): use the live config range so retunes update overlays
        immediately. Stale bars get shifted/clipped by `_shift_spectrum_to_live`
        into this window, disappearing where there's no overlap.
        """
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return None
        if device.state == DeviceState.RUNNING and self.current_event is not None:
            return self._actual_freq_range(self.current_event)
        cf = device.config.center_frequency
        sr = device.config.sample_rate
        visible_bw = sr / self._ui_state.zoom
        return cf - visible_bw / 2, cf + visible_bw / 2

    def _shift_spectrum_to_live(
        self,
        zoomed: np.ndarray,
        e_fmin: float,
        e_fmax: float,
        l_fmin: float,
        l_fmax: float,
        target: int,
    ) -> np.ndarray:
        """Map `zoomed` (covering [e_fmin, e_fmax]) into `target` bins over [l_fmin, l_fmax].

        Bins outside the event's freq range are filled with -inf so they clip to 0
        after normalization: bars disappear where we have no captured data.
        """
        if e_fmin == l_fmin and e_fmax == l_fmax:
            return decimate_spectrum(zoomed, target)

        out = np.full(target, -np.inf, dtype=np.float32)
        e_span = e_fmax - e_fmin
        l_span = l_fmax - l_fmin
        n = len(zoomed)
        if e_span <= 0 or l_span <= 0 or n == 0:
            return out

        l_freqs = l_fmin + (np.arange(target) + 0.5) / target * l_span
        src = ((l_freqs - e_fmin) / e_span * n).astype(np.intp)
        mask = (src >= 0) & (src < n)
        out[mask] = zoomed[src[mask]]
        return out

    def _compute_freq_labels(
        self, width: int, freq_min: float, freq_max: float
    ) -> tuple[tuple[int, str], ...]:
        """Compute frequency axis label positions."""
        span = freq_max - freq_min

        nice_intervals = [1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3, 200e3, 500e3, 1e6, 2e6, 5e6, 10e6]

        # Pick the smallest nice interval where labels don't overlap.
        # Estimate label width from a representative tick, add min gap.
        min_gap = 3
        interval = nice_intervals[-1]
        for ni in nice_intervals:
            sample_label = self._format_frequency(freq_min + ni, ni)
            cols_per_label = len(sample_label) + min_gap
            tick_spacing = ni / span * width
            if tick_spacing >= cols_per_label:
                interval = ni
                break

        labels: list[tuple[int, str]] = []
        first_tick = ((freq_min // interval) + 1) * interval
        freq = first_tick
        while freq < freq_max:
            col = int((freq - freq_min) / span * width)
            if 0 <= col < width:
                label = self._format_frequency(freq, interval)
                start = col - len(label) // 2
                end = start + len(label)
                if start >= 0 and end <= width:
                    labels.append((start, label))
            freq += interval

        return tuple(labels)

    def _compute_bandplan_segments(
        self, width: int, freq_min: float, freq_max: float
    ) -> tuple[tuple[int, int, str, str], ...]:
        """Return (col_start, col_end_exclusive, label, color_hex) for each visible band.

        Narrower bands sort later so they overdraw wider overlapping bands
        when painted in order.
        """
        bp = self._bandplan
        if bp is None:
            return ()
        span = freq_max - freq_min
        if span <= 0:
            return ()

        raw: list[tuple[int, int, str, str, int]] = []
        for band in bp.bands:
            if band.end <= freq_min or band.start >= freq_max:
                continue
            f_lo = max(band.start, freq_min)
            f_hi = min(band.end, freq_max)
            col_start = max(0, floor((f_lo - freq_min) / span * width))
            col_end = min(width, ceil((f_hi - freq_min) / span * width))
            if col_end <= col_start:
                continue
            color = band_type_color(band.type)
            # Un-clipped Hz width is the stacking key: a wide band clipped to
            # a narrow visible slice must still sort as wide.
            raw.append((col_start, col_end, band.name, color, band.end - band.start))

        raw.sort(key=lambda t: -t[4])
        return tuple((cs, ce, name, color) for cs, ce, name, color, _ in raw)

    def _compute_bandwidth_range(
        self, width: int, freq_min: float, freq_max: float
    ) -> tuple[int, int] | None:
        """Compute bandwidth indicator column range."""
        channel_bw = self._channel_bandwidth

        if channel_bw is None:
            return None

        center = (freq_min + freq_max) / 2
        span = freq_max - freq_min
        col_low = int((center - channel_bw / 2 - freq_min) / span * width)
        col_high = int((center + channel_bw / 2 - freq_min) / span * width)
        col_low = max(0, col_low)
        col_high = min(width, col_high)

        return (col_low, col_high)

    def _format_frequency(self, hz: float, interval: float | None = None) -> str:
        """Format frequency in Hz to human-readable string.

        When interval is provided, precision adapts so adjacent ticks differ.
        """
        return format_hz(hz, interval=interval)
