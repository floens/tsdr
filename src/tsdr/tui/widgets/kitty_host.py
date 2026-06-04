"""Shared host plumbing for apps that display `KittyImageWidget`(s).

`KittyImageWidget` paints via the kitty graphics protocol — escape sequences that
must reach the terminal *outside* Textual's normal cell rendering. Widgets queue
them with ``queue_oob_escape``; this mixin writes the batch straight to the driver
once per frame in ``_end_update`` (Textual's per-update hook) before deferring to
the base App.

Any App that hosts a `KittyImageWidget` must inherit `KittyHostMixin`, listed
before ``App`` in the bases so its ``_end_update`` takes precedence.

At runtime the base is ``object`` (a plain mixin); type checkers see ``App`` so
``_driver`` / ``_end_update`` resolve (mirrors ``tui/_mixin_base.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App as _Base
else:
    _Base = object


class KittyHostMixin(_Base):  # type: ignore[misc]
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pending_oob_escapes: list[str] = []

    def queue_oob_escape(self, cmd: str) -> None:
        self._pending_oob_escapes.append(cmd)

    def _end_update(self) -> None:
        if self._pending_oob_escapes and self._driver is not None:
            self._driver.write("".join(self._pending_oob_escapes))
            self._pending_oob_escapes.clear()
        super()._end_update()
