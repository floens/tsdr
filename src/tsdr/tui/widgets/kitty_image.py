import itertools
import logging
import sys
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

from tsdr.tui.tty import cell_pixel_size, shm_payload_name

_id_counter = itertools.count(1)
_shm_seq = itertools.count(1)
logger = logging.getLogger(__name__)

# How this widget uploads pixels (see update_image: f=32, t=s). Human-readable
# description of the wire format TSDR *transmits with* — distinct from the set of
# transports a terminal reports it *supports*. The doctor surfaces it in its
# report/export/UI; kept here, beside the actual command, so it can't go stale.
KITTY_TRANSPORT_DESC = "shared memory (kitty t=s, f=32 RGBA)"


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
    shm_capacity: int = 0  # Windows: bytes the persistent mapping can hold


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
    #  - SHM reuse: image IDs are monotonic with no reuse after delete (and on
    #    Windows the shm name is unique per frame; see _write_shm).
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

    # Windows: frames to keep a *reallocated* (grown) mapping open before closing,
    # so the terminal's async read of the previous frame can't lose the race.
    # Only grows trigger this (rare with geometric capacity), so the list is tiny.
    _SHM_RETIRE_FRAMES = 4

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._images: dict[str, _ImageEntry] = {}
        self._cell_width_px = 8
        self._cell_height_px = 16
        self._pending_deletes: list[tuple[int, int]] = []  # (image_id, remaining_frames)
        self._retired_shms: list[tuple[SharedMemory, int]] = []  # (shm, frames_left)
        self._visible = True

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
            logger.warning("kitty_image_zero_region reason=not_laid_out")
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

    def on_hide(self) -> None:
        """Widget left the visible tree (tab switch / scrolled out of view).

        Kitty images are painted at absolute terminal pixels, independent of
        Textual's cell grid, so hiding the widget does not remove the image —
        without this it would ghost over whatever is shown next. We delete the
        *placement* but keep the stored image data (lowercase ``d=i``) so
        ``on_show`` can re-place it without re-uploading pixels.
        """
        if not self._visible:
            return
        self._visible = False
        for entry in self._images.values():
            if entry.has_image:
                self._queue_cmd(f"\x1b_Ga=d,d=i,i={entry.image_id},p=1,q=2\x1b\\")

    def on_show(self) -> None:
        """Widget became visible again: re-place every image at its current
        on-screen position (pixel data is still resident in the terminal)."""
        if self._visible:
            return
        self._visible = True
        for key in list(self._images):
            self.place_image(key)

    def _detect_cell_pixel_size(self, event: Resize) -> None:
        pixels = (
            (event.pixel_size.width, event.pixel_size.height)
            if event.pixel_size is not None
            else None
        )
        self._cell_width_px, self._cell_height_px = cell_pixel_size(
            event.size.width, event.size.height, pixels
        )

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

        # One contiguous uint8 view, copied straight into shared memory (no
        # intermediate bytes object on the hot path).
        flat = np.ascontiguousarray(data).reshape(-1)
        n = flat.nbytes
        self._write_shm(entry, flat)

        # Build transmit command (shared memory, kitty t=s — no base64 on the hot
        # path). a=T transmits *and* displays; re-transmitting the same id makes
        # the terminal re-read the (overwritten) shared buffer and replace it.
        # S= is the exact byte count to read: a Windows mapping is rounded up to a
        # whole page (and our buffer may be over-allocated), so without it the
        # terminal reads the padded size and rejects the frame ("data len doesn't
        # match width*height*4").
        move, sub_x, sub_y = self._build_position_prefix(entry)
        extra = self._build_extra(entry, sub_x, sub_y)
        b64_name = standard_b64encode(shm_payload_name(entry.shm_name).encode()).decode()
        cmd = (
            f"{move}\x1b_Gf=32,t=s,S={n},s={entry.width},v={entry.height},"
            f"a=T,i={entry.image_id},p=1,"
            f"z=0,C=1,q=0{extra};{b64_name}\x1b\\"
        )
        self._queue_cmd(cmd)
        self._release_shm(entry)
        entry.needs_transmit = False
        entry.has_image = True

    def _release_shm(self, entry: _ImageEntry) -> None:
        """Drop our handle after sending, where the platform allows it.

        POSIX hands ownership to the terminal, which unlinks the object once it has
        read it, so the object outlives our immediate close. On Windows the terminal
        only *closes* its own handle (it cannot unlink), so ours is what keeps the
        mapping alive — hold it open and reuse it (see ``_write_shm``).
        """
        if sys.platform != "win32" and entry.pending_shm is not None:
            entry.pending_shm.close()
            entry.pending_shm = None

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
        cmd = f"{move}\x1b_Ga=p,i={entry.image_id},p=1,z=0,C=1,q=0{extra}\x1b\\"
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
            entry.shm_capacity = 0
        entry.has_image = False

    def remove_image(self, key: str) -> None:
        entry = self._images.pop(key, None)
        if entry is None:
            logger.warning("kitty_image_remove_miss key=%s", key)
            return
        self._queue_cmd(f"\x1b_Ga=d,d=I,i={entry.image_id},q=0\x1b\\")
        self._schedule_delete_retries(entry.image_id)
        if entry.pending_shm is not None:
            entry.pending_shm.close()

    def render_lines(self, crop: Region) -> list[Strip]:
        self.flush_pending_deletes()
        self._age_retired_shms()
        return [Strip([])] * crop.height

    def _age_retired_shms(self) -> None:
        """Close grown-out Windows mappings once they've survived a few frames."""
        if not self._retired_shms:
            return
        survivors: list[tuple[SharedMemory, int]] = []
        for shm, frames_left in self._retired_shms:
            if frames_left > 1:
                survivors.append((shm, frames_left - 1))
            else:
                shm.close()
        self._retired_shms = survivors

    def on_unmount(self) -> None:
        for entry in self._images.values():
            self._queue_cmd(f"\x1b_Ga=d,d=I,i={entry.image_id},q=0\x1b\\")
            if entry.pending_shm is not None:
                entry.pending_shm.close()
        for shm, _ in self._retired_shms:
            shm.close()
        self._retired_shms.clear()
        self._images.clear()

    def _write_shm(self, entry: _ImageEntry, flat: np.ndarray) -> None:
        if sys.platform == "win32":
            self._write_shm_persistent(entry, flat)
        else:
            self._write_shm_recreate(entry, flat)

    def _write_shm_persistent(self, entry: _ImageEntry, flat: np.ndarray) -> None:
        """Windows: keep one mapping alive and overwrite it in place each frame.

        The terminal never unlinks on Windows (kitty spec — it "just closes" its
        own handle) and reads asynchronously, so recreating per frame races the
        read and ``unlink()`` is a no-op that can't clear a stale name. Instead we
        hold the handle for the image's lifetime; the mapping always exists for the
        terminal to open, and a re-sent ``a=T`` re-reads it (with S= bounding the
        read to the live bytes). Capacity grows geometrically so a steadily growing
        image (e.g. a filling waterfall strip) reallocates O(log) times, not every
        frame; the superseded mapping is retired (closed a few frames later) so the
        terminal's read of the previous frame can't lose the race.
        """
        n = flat.nbytes
        if entry.pending_shm is None or entry.shm_capacity < n:
            if entry.pending_shm is not None:
                self._retired_shms.append((entry.pending_shm, self._SHM_RETIRE_FRAMES))
            cap = max(n, entry.shm_capacity * 2)
            entry.pending_shm = self._create_unique_shm(entry, cap)
            entry.shm_capacity = cap
        entry.pending_shm.buf[:n] = flat  # type: ignore[index]

    def _create_unique_shm(self, entry: _ImageEntry, size: int) -> SharedMemory:
        """Create a fresh-named Windows mapping, skipping any name still in use.

        A mapping leaked by a crashed prior tsdr can stay alive if the terminal
        still holds its handle; reusing that exact name would raise
        ERROR_ALREADY_EXISTS. Advance the counter until an unused name is found.
        """
        while True:
            entry.shm_name = f"tsdr_{entry.image_id}_{next(_shm_seq)}"
            try:
                return SharedMemory(create=True, name=entry.shm_name, size=size, track=False)
            except FileExistsError:
                continue

    def _write_shm_recreate(self, entry: _ImageEntry, flat: np.ndarray) -> None:
        """POSIX: recreate the named object each frame.

        The terminal unlinks it after reading, so the name is gone by the next
        frame; the object survives our close (done in ``_release_shm``) until the
        terminal unlinks it. The except branch clears a stale object the terminal
        never read.
        """
        n = flat.nbytes
        if entry.pending_shm is not None:
            entry.pending_shm.close()
            entry.pending_shm = None
        try:
            shm = SharedMemory(create=True, name=entry.shm_name, size=n, track=False)
        except FileExistsError:
            try:
                stale = SharedMemory(name=entry.shm_name, track=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError, OSError:
                pass
            shm = SharedMemory(create=True, name=entry.shm_name, size=n, track=False)
        shm.buf[:n] = flat  # type: ignore[index]
        entry.pending_shm = shm
