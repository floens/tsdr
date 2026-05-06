"""Event system for worker communication.

Workers publish immutable Event dataclasses on EventBus; UI/state subscribes
by event type. See `events.py` for the full set of event types: import them
from `tsdr.core.events.events` directly.
"""

from tsdr.core.events.bus import EventBus
from tsdr.core.events.subscription import Subscription

__all__ = ["EventBus", "Subscription"]
