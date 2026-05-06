from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tsdr.core.events.events import Event


@dataclass(frozen=True)
class Subscription:
    """Active event subscription. Returned by EventBus.subscribe and used to unsubscribe."""

    event_type: type[Event]
    handler: Callable[[Event], None]
