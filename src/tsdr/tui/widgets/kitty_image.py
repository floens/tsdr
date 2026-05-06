import itertools
import logging
from base64 import standard_b64encode
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from textual import errors as textual_errors
from textual.dom import NoScreen
from textual.events import Resize
from textual.geometry import Region
from textual.strip import Strip
from textual.widget import Widget

from tsdr.core.platform import tty_window_spec

_id_counter = itertools.count(1)
logger = logging.getLogger(__name__)


@dataclass
class OcclusionInsets:
    """Pixel insets occluded from each edge by higher-layer widgets."""

    top: int = 0
    right: int = 0
    bottom: int = 0
    left: int = 0


@dataclass
class _ImageEntry:
    key: str
    image_id: int
    shm_name: str
    width: int = 0
    height: int = 0
    x: int = 0
    y: int = 0
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0
    crop_h: int = 0
    needs_transmit: bool = False
    has_image: bool = False
    pending_shm: SharedMemory | None = field(default=None, repr=False)


class KittyImageWidget(Widget):
    """Displays images using Kitty terminal graphics protocol with shared memory."""

    # TODO: IO-bug
    # Number of extra frames to re-send delete commands after hide/remove.
    #
    # Kitty graphics delete commands (a=d,d=I) are sometimes silently lost,
    # leaving "ghost" images on screen after layout changes. The root cause
    # is unknown; the following were investigated and ruled out:
    #
    #  - Terminal bug: verified delete handling in both Kitty (graphics.c)
    #    and Ghostty (graphics_exec.zig) source - both implement d=I correctly
    #    and the bug reproduces on both terminals.
    #  - Bytes not reaching terminal: TAPPED_WRITE confirms bytes pass through
    #    WriterThread. Replacing Python's TextIOWrapper -> BufferedWriter ->
    #    FileIO stack with direct os.write() (retry loop) did not help.
    #  - Python I/O buffering: PYTHONUNBUFFERED has no effect. CPython #85393
    #    (TextIOWrapper silently ignores partial writes) was investigated but
    #    the direct fd writer bypass ruled this out.
    #  - Non-blocking stderr: setting O_NONBLOCK on the fd did not help.
    #  - OOB ordering: deletes are queued before transmits in _end_update and
    #    written inside the synchronized output block (BSU…ESU).
    #  - Image lifecycle: per-event logging (IMAGE_CREATE/UPDATE/PLACE/
    #    HIDE/REMOVE) shows no re-creation of deleted image IDs.
    #  - SHM reuse: each image gets a unique shm name (tsdr_<id>), monotonic
    #    IDs, no reuse after delete.
    #  - Separate placement + data delete: splitting into d=i,p=1 then d=I
    #    did not help.
    #  - q=0 responses: delete commands produce no terminal response per the
    #    kitty protocol spec (confirmed in both terminal sources), so receipt
    #    cannot be verified.
    #
    # TODO: use the PTY capture tool (scripts/pty_capture.py), together with a
    # debug implementation of ghostty/kitty, to inspect the exact byte stream
    # the terminal receives - verify whether the delete APC sequences actually
    # appear in the captured output and are correctly formed when interleaved
    # with Textual's render data.
    #
    # Workaround: re-send the delete for N additional frames.
    _DELETE_RETRIES = 16

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._images: dict[str, _ImageEntry] = {}
        self._cell_width_px = 8
        self._cell_height_px = 16
        self._pending_deletes: list[tuple[int, int]] = []  # (image_id, remaining_frames)

    @property
    def cell_width_px(self) -> int:
        return self._cell_width_px

    @property
    def cell_height_px(self) -> int:
        return self._cell_height_px

    @property
    def full_pixel_size(self) -> tuple[int, int]:
        """Full widget pixel dimensions (ignoring occlusion)."""
        region = self.content_region
        return region.width * self._cell_width_px, region.height * self._cell_height_px

    @property
    def available_pixel_size(self) -> tuple[int, int]:
        """Return (width_px, height_px) of this widget in pixels."""
        region = self.content_region
        if region.width == 0 or region.height == 0:
            logger.warning(
                "available_pixel_size content_region is zero, widget is not laid out yet."
            )
        return region.width * self._cell_width_px, region.height * self._cell_height_px

    @property
    def occlusion_insets(self) -> OcclusionInsets:
        """Pixel insets occluded from each edge."""
        return self._get_occlusion()

    def _get_occlusion(self) -> OcclusionInsets:
        """Compute pixel insets occluded by higher-layer widgets.

        Two-pass approach:
        1. Collect all higher-layer widgets that overlap our region.
        2. Filter to root occluders only (skip children whose ancestor is
           also an occluder - e.g. ConsoleInput inside ConsoleWidget).
        3. Classify edges: an overlap only counts for an edge if it doesn't
           span to the opposite edge (a full-width bottom bar produces
           bottom insets only, not left+right).
        """
        try:
            my_geom = self.screen.find_widget(self)
        except NoScreen, LookupError:
            return OcclusionInsets()

        compositor = self.screen._compositor
        if compositor is None or not hasattr(compositor, "layers"):
            return OcclusionInsets()

        my = my_geom.region
        my_order = my_geom.order

        # Pass 1: collect all overlapping higher-layer widgets
        occluders: list[tuple[Widget, Region]] = []
        for widget, geom in compositor.layers:
            if geom.order <= my_order or not my.overlaps(geom.region):
                continue
            occluders.append((widget, my.intersection(geom.region)))

        # Pass 2: keep only root occluders, classify edges
        occluder_ids = {id(w) for w, _ in occluders}
        top = right = bottom = left = 0

        for widget, overlap in occluders:
            if widget.styles.overlay != "screen" and any(
                id(a) in occluder_ids for a in widget.ancestors
            ):
                continue

            touches_top = overlap.y <= my.y
            touches_bottom = overlap.y + overlap.height >= my.y + my.height
            touches_left = overlap.x <= my.x
            touches_right = overlap.x + overlap.width >= my.x + my.width

            if touches_top and not touches_bottom:
                top = max(top, overlap.y + overlap.height - my.y)
            if touches_right and not touches_left:
                right = max(right, (my.x + my.width) - overlap.x)
            if touches_bottom and not touches_top:
                bottom = max(bottom, (my.y + my.height) - overlap.y)
            if touches_left and not touches_right:
                left = max(left, overlap.x + overlap.width - my.x)

        cw, ch = self._cell_width_px, self._cell_height_px
        return OcclusionInsets(
            top=top * ch,
            right=right * cw,
            bottom=bottom * ch,
            left=left * cw,
        )

    def on_resize(self, event: Resize) -> None:
        self._detect_cell_pixel_size(event)

    def _detect_cell_pixel_size(self, event: Resize) -> None:
        if event.pixel_size is not None and event.size.width > 0 and event.size.height > 0:
            self._cell_width_px = event.pixel_size.width // event.size.width
            self._cell_height_px = event.pixel_size.height // event.size.height
            return
        spec = tty_window_spec()
        if spec is not None:
            self._cell_width_px = spec.cell_width_px
            self._cell_height_px = spec.cell_height_px

    def _queue_cmd(self, cmd: str) -> None:
        self.app.queue_oob_escape(cmd)
        self.refresh()

    def _schedule_delete_retries(self, image_id: int) -> None:
        """Schedule delete commands to be re-sent for the next N frames."""
        self._pending_deletes.append((image_id, self._DELETE_RETRIES))
        self.refresh()

    def flush_pending_deletes(self) -> None:
        """Re-queue delete commands for images still pending. Call each frame."""
        if not self._pending_deletes:
            return
        surviving: list[tuple[int, int]] = []
        for image_id, remaining in self._pending_deletes:
            self.app.queue_oob_escape(f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\")
            if remaining > 1:
                surviving.append((image_id, remaining - 1))
        self._pending_deletes = surviving
        if surviving:
            self.refresh()

    def _screen_offset(self) -> tuple[int, int]:
        """Return (x_px, y_px) of widget's top-left corner in terminal pixels."""
        try:
            region = self.screen.find_widget(self).region
        except NoScreen, textual_errors.NoWidget:
            return 0, 0
        return region.x * self._cell_width_px, region.y * self._cell_height_px

    def _build_extra(self, entry: _ImageEntry, sub_x: int = 0, sub_y: int = 0) -> str:
        """Build crop/sub-pixel offset params string."""
        extra = ""
        if sub_x:
            extra += f",X={sub_x}"
        if sub_y:
            extra += f",Y={sub_y}"
        if entry.crop_x:
            extra += f",x={entry.crop_x}"
        if entry.crop_y:
            extra += f",y={entry.crop_y}"
        if entry.crop_w:
            extra += f",w={entry.crop_w}"
        if entry.crop_h:
            extra += f",h={entry.crop_h}"
        return extra

    def _build_position_prefix(self, entry: _ImageEntry) -> tuple[str, int, int]:
        """Build cursor move + compute sub-pixel offsets for an entry."""
        ox, oy = self._screen_offset()
        abs_x = ox + entry.x
        abs_y = oy + entry.y
        cell_col = abs_x // self._cell_width_px
        cell_row = abs_y // self._cell_height_px
        sub_x = abs_x % self._cell_width_px
        sub_y = abs_y % self._cell_height_px
        move = f"\x1b[{cell_row + 1};{cell_col + 1}H"
        return move, sub_x, sub_y

    def update_image(
        self,
        key: str,
        data: np.ndarray,
        *,
        x: int = 0,
        y: int = 0,
        crop_x: int = 0,
        crop_y: int = 0,
        crop_w: int = 0,
        crop_h: int = 0,
    ) -> None:
        """Create or update an image. data must be RGBA uint8 array with shape (H, W, 4)."""
        entry = self._images.get(key)
        if entry is None:
            image_id = next(_id_counter)
            entry = _ImageEntry(key=key, image_id=image_id, shm_name=f"tsdr_{image_id}")
            self._images[key] = entry

        entry.width = data.shape[1]
        entry.height = data.shape[0]
        entry.x = x
        entry.y = y
        entry.crop_x = crop_x
        entry.crop_y = crop_y
        entry.crop_w = crop_w
        entry.crop_h = crop_h

        self._write_shm(entry, data.tobytes())

        # Build transmit command
        move, sub_x, sub_y = self._build_position_prefix(entry)
        extra = self._build_extra(entry, sub_x, sub_y)
        shm_path = f"/{entry.shm_name}"
        b64_name = standard_b64encode(shm_path.encode()).decode()
        cmd = (
            f"{move}\x1b_Gf=32,t=s,s={entry.width},v={entry.height},"
            f"a=T,i={entry.image_id},p=1,"
            f"z=-1073741825,C=1,q=0{extra};{b64_name}\x1b\\"
        )
        self._queue_cmd(cmd)

        if entry.pending_shm is not None:
            entry.pending_shm.close()
            entry.pending_shm = None
        entry.needs_transmit = False
        entry.has_image = True

    def place_image(
        self,
        key: str,
        *,
        x: int | None = None,
        y: int | None = None,
        crop_x: int | None = None,
        crop_y: int | None = None,
        crop_w: int | None = None,
        crop_h: int | None = None,
    ) -> None:
        """Update position/crop of existing image without re-transmitting data."""
        entry = self._images.get(key)
        if entry is None or not entry.has_image:
            return
        if x is not None:
            entry.x = x
        if y is not None:
            entry.y = y
        if crop_x is not None:
            entry.crop_x = crop_x
        if crop_y is not None:
            entry.crop_y = crop_y
        if crop_w is not None:
            entry.crop_w = crop_w
        if crop_h is not None:
            entry.crop_h = crop_h

        move, sub_x, sub_y = self._build_position_prefix(entry)
        extra = self._build_extra(entry, sub_x, sub_y)
        cmd = f"{move}\x1b_Ga=p,i={entry.image_id},p=1,z=-1073741825,C=1,q=0{extra}\x1b\\"
        self._queue_cmd(cmd)

    def hide_image(self, key: str) -> None:
        """Remove image from terminal display but preserve entry for re-display.

        Occlusion cannot be handled transparently inside update_image/place_image
        because callers (e.g. waterfall) track transmit state independently - silent
        suppression would desync their bookkeeping. Callers use this explicitly.
        """
        entry = self._images.get(key)
        if entry is None or not entry.has_image:
            return
        self._queue_cmd(f"\x1b_Ga=d,d=I,i={entry.image_id},q=0\x1b\\")
        self._schedule_delete_retries(entry.image_id)
        if entry.pending_shm is not None:
            entry.pending_shm.close()
            entry.pending_shm = None
        entry.has_image = False

    def remove_image(self, key: str) -> None:
        entry = self._images.pop(key, None)
        if entry is None:
            logger.warning("REMOVE_MISS key=%s (not in _images)", key)
            return
        self._queue_cmd(f"\x1b_Ga=d,d=I,i={entry.image_id},q=0\x1b\\")
        self._schedule_delete_retries(entry.image_id)
        if entry.pending_shm is not None:
            entry.pending_shm.close()

    def render_lines(self, crop: Region) -> list[Strip]:
        self.flush_pending_deletes()
        return [Strip([])] * crop.height

    def on_unmount(self) -> None:
        for entry in self._images.values():
            self._queue_cmd(f"\x1b_Ga=d,d=I,i={entry.image_id},q=0\x1b\\")
            if entry.pending_shm is not None:
                entry.pending_shm.close()
        self._images.clear()

    def _write_shm(self, entry: _ImageEntry, raw: bytes) -> None:
        if entry.pending_shm is not None:
            entry.pending_shm.close()
            entry.pending_shm = None
        try:
            shm = SharedMemory(create=True, name=entry.shm_name, size=len(raw), track=False)
        except FileExistsError:
            try:
                stale = SharedMemory(name=entry.shm_name, track=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError, OSError:
                pass
            shm = SharedMemory(create=True, name=entry.shm_name, size=len(raw), track=False)
        shm.buf[: len(raw)] = raw  # type: ignore[index]
        entry.pending_shm = shm
