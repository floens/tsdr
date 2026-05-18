from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.tui.commands.base import Completion


def get_focused_device_id() -> str:
    engine = get_engine()
    if engine.focused_device is None:
        raise SDRException("No device focused. Use 'add' to add a device.")
    return engine.focused_device


def device_id_completions(prefix: str) -> list[Completion]:
    try:
        engine = get_engine()
    except RuntimeError:
        return []
    return [Completion(did) for did in engine.devices if did.startswith(prefix)]


def parse_endpoint(spec: str) -> tuple[str, int | None]:
    """Parse `host`, `host:port`, or `scheme://host[:port]` into (host, port).

    Port is None if not present. Raises ValueError when the port part is
    non-numeric.
    """
    rest = spec
    if "://" in rest:
        rest = rest.split("://", 1)[1]
    rest = rest.split("/", 1)[0]
    if ":" not in rest:
        return rest, None
    host, _, port_str = rest.rpartition(":")
    if not port_str.isdigit():
        raise ValueError(f"invalid endpoint {spec!r}")
    return host, int(port_str)
