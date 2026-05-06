from collections.abc import Iterable

import textual.drivers.linux_driver as _ld
from textual._xterm_parser import XTermParser
from textual.message import Message


class _APCAwareXTermParser(XTermParser):
    """XTermParser subclass that strips kitty graphics APC responses.

    Kitty sends responses as APC sequences: \\x1b_Gi=<id>;OK\\x1b\\
    Textual's parser doesn't handle APC, so it would mangle them into
    key events. We strip them before they reach the parser.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._apc_buffer = ""

    def feed(self, data: str) -> Iterable[Message]:
        data = self._extract_apc_responses(data)
        if not data:
            return ()
        result: Iterable[Message] = super().feed(data)
        return result

    def _extract_apc_responses(self, data: str) -> str:
        if self._apc_buffer:
            data = self._apc_buffer + data
            self._apc_buffer = ""

        result: list[str] = []
        i = 0
        while i < len(data):
            if data[i : i + 3] == "\x1b_G":
                end = data.find("\x1b\\", i + 3)
                if end == -1:
                    self._apc_buffer = data[i:]
                    break
                i = end + 2
            else:
                result.append(data[i])
                i += 1
        return "".join(result)


# Monkey-patch before LinuxDriver.run_input_thread creates the parser
_ld.XTermParser = _APCAwareXTermParser  # type: ignore[attr-defined]
