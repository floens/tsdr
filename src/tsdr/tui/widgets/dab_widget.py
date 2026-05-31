import io
import logging

import numpy as np
from PIL import Image
from textual import on
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from tsdr.core.events.events import DecoderOutputEvent
from tsdr.radio.decoders.dab import DABData, DABServiceInfo, DABSlide
from tsdr.tui.commands import registry
from tsdr.tui.markup import escape_forced
from tsdr.tui.widgets.kitty_image import KittyImageWidget
from tsdr.tui.widgets.panel import PanelWidget
from tsdr.tui.widgets.utils import NonFocusableOptionList

logger = logging.getLogger(__name__)

_SLIDE_IMAGE_KEY = "dab-slide"


def _service_prompt(svc: DABServiceInfo, selected: bool) -> str:
    """Build Rich markup for a service option."""
    marker = "[bold green]>[/bold green] " if selected else "  "
    kind = "♪" if svc.is_audio else "d"
    label = escape_forced(svc.label)
    sid = f"[dim]{svc.service_id:#06x}[/dim]"
    parts = [marker, kind, " ", label, " ", sid]
    if svc.protection_level is not None:
        parts.append(f" [dim]EEP-{svc.protection_level + 1}[/dim]")
    return "".join(parts)


def _decode_slide_to_rgba(slide: DABSlide) -> np.ndarray | None:
    """Decode JPEG/PNG slide to RGBA uint8 numpy array."""
    try:
        img = Image.open(io.BytesIO(slide.data))
        img = img.convert("RGBA")
        return np.array(img, dtype=np.uint8)
    except OSError, ValueError:
        logger.debug("dab_slide_decode_failed", exc_info=True)
        return None


class DABWidget(Horizontal, PanelWidget):
    """Display DAB ensemble info, signal stats, and service list.

    Col 1: selected station + DLS text.
    Col 2: slide image (kitty) or alt text.
    Col 3: ensemble metadata (label, ID, FIB quality).
    Col 4: signal info (frames, freq offset, audio codec).
    Col 5: interactive service list - click to select.

    Reactive props:
      image_mode: bool — when True, slide column shows the kitty image; when False, shows alt-text.
    """

    image_mode = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._current: DABData | None = None
        self._col_station = Static("", id="dab-station")
        self._col_slide_text = Static("", id="dab-slide-text")
        self._col_info = Static("Waiting...", id="dab-info")
        self._col_signal = Static("", id="dab-signal")
        self._service_list = NonFocusableOptionList(id="dab-services")
        self._kitty = KittyImageWidget(id="dab-slide-img")
        self._last_slide: DABSlide | None = None
        self._last_services: tuple[DABServiceInfo, ...] = ()
        self._last_selected_id: int | None = None

    def compose(self):
        yield self._col_signal
        yield self._col_info
        yield self._service_list
        yield self._col_station
        yield self._col_slide_text
        yield self._kitty

    def on_mount(self) -> None:
        self.border_title = "DAB"
        # Apply the initial image_mode (reactive watcher fires once on mount if set)
        self._apply_image_mode(self.image_mode)

    def watch_image_mode(self, image_mode: bool) -> None:
        self._apply_image_mode(image_mode)

    def _apply_image_mode(self, enabled: bool) -> None:
        if enabled:
            self._col_slide_text.display = False
            self._kitty.display = True
            if self._last_slide is not None:
                # Defer render until after layout so the widget has a screen region
                self.call_after_refresh(self._render_slide, self._last_slide)
        else:
            self._kitty.display = False
            self._kitty.remove_image(_SLIDE_IMAGE_KEY)
            self._col_slide_text.display = True
            self._refresh_slide_text()

    def update_messages(self, event: DecoderOutputEvent) -> None:
        dab_data = None
        for msg in event.messages:
            if isinstance(msg.data, DABData):
                dab_data = msg.data

        if dab_data is None:
            return

        self._current = dab_data
        self._refresh_display()

    def _refresh_display(self) -> None:
        if self._current is None:
            self._col_station.update("")
            self._col_slide_text.update("")
            self._col_info.update("Waiting...")
            self._col_signal.update("")
            self._service_list.clear_options()
            return

        data = self._current

        # Col 1: station + DLS
        station_lines: list[str] = []
        selected_svc = None
        if data.selected_service_id is not None:
            for svc in data.services:
                if svc.service_id == data.selected_service_id:
                    selected_svc = svc
                    break
        if selected_svc is not None:
            station_lines.append(f"[bold]{escape_forced(selected_svc.label)}[/bold]")
        if data.dynamic_label:
            station_lines.append(escape_forced(data.dynamic_label))
        if data.audio_sample_rate is not None:
            codec_parts = [f"{data.audio_channels}ch"]
            if data.core_sample_rate is not None:
                codec_parts.append(f"{data.core_sample_rate / 1000:.0f}k")
            if data.sbr:
                codec_parts.append("SBR")
            if data.ps:
                codec_parts.append("PS")
            station_lines.append(
                f"Audio: {data.audio_sample_rate / 1000:.1f} kHz {' '.join(codec_parts)}"
            )
        self._col_station.update("\n".join(station_lines))

        # Clear slide when service changes
        if data.selected_service_id != self._last_selected_id:
            self._last_slide = None
            self._kitty.remove_image(_SLIDE_IMAGE_KEY)
            self._col_slide_text.update("")

        # Col 2: slide
        if data.slide is not None and data.slide is not self._last_slide:
            self._last_slide = data.slide
            if self.image_mode:
                self._render_slide(data.slide)
            else:
                self._refresh_slide_text()

        # Col 3: ensemble info
        info: list[str] = []
        info.append(f"[bold]{escape_forced(data.ensemble_label)}[/bold]")
        info.append(f"ID: {data.ensemble_id:#06x}")
        n_audio = sum(1 for s in data.services if s.is_audio)
        n_data = len(data.services) - n_audio
        info.append(f"Services: {n_audio}A {n_data}D")
        self._col_info.update("\n".join(info))

        # Col 4: signal info
        signal: list[str] = []
        signal.append(f"Frames: {data.frames_processed}")
        offset = data.freq_offset_hz
        off_color = "green" if abs(offset) < 200 else "yellow" if abs(offset) < 500 else "red"
        signal.append(f"Freq: [{off_color}]{offset:+.1f} Hz[/{off_color}]")
        crc_pct = data.fib_crc_rate * 100
        crc_color = "green" if crc_pct > 90 else "yellow" if crc_pct > 50 else "red"
        signal.append(f"FIB CRC: [{crc_color}]{crc_pct:.0f}%[/{crc_color}]")
        self._col_signal.update("\n".join(signal))

        # Col 5: service list (only rebuild when services or selection changed)
        if (
            data.services != self._last_services
            or data.selected_service_id != self._last_selected_id
        ):
            self._last_services = data.services
            self._last_selected_id = data.selected_service_id
            prev_idx = self._service_list.highlighted
            self._service_list.clear_options()
            for svc in data.services:
                selected = svc.service_id == data.selected_service_id
                prompt = _service_prompt(svc, selected)
                self._service_list.add_option(
                    Option(prompt, id=f"{svc.service_id:x}", disabled=not svc.is_audio)
                )
            if prev_idx is not None and 0 <= prev_idx < self._service_list.option_count:
                self._service_list.highlighted = prev_idx

    def _render_slide(self, slide: DABSlide) -> None:
        rgba = _decode_slide_to_rgba(slide)
        if rgba is None:
            return
        w_px, h_px = self._kitty.available_pixel_size
        if w_px > 0 and h_px > 0:
            img = Image.fromarray(rgba)
            img.thumbnail((w_px, h_px), Image.LANCZOS)
            rgba = np.array(img, dtype=np.uint8)
        self._kitty.update_image(_SLIDE_IMAGE_KEY, rgba)

    def _refresh_slide_text(self) -> None:
        if self._last_slide is None:
            self._col_slide_text.update("")
            return
        alt = self._last_slide.content_name or self._last_slide.category_title or "Slide"
        self._col_slide_text.update(f"[dim]📷 {escape_forced(alt)}[/dim]")

    @on(OptionList.OptionSelected, "#dab-services")
    def _on_service_selected(self, event: OptionList.OptionSelected) -> None:
        """Select a DAB service when the user clicks."""
        if event.option_id is not None:
            result = registry.execute(f"dab select {event.option_id}")
            self.app.show_status(result)
