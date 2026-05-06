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
