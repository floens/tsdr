import logging
import threading
from collections.abc import Callable

from tsdr.core.events.events import Event
from tsdr.core.events.subscription import Subscription
from tsdr.core.tracing import span

logger = logging.getLogger(__name__)


class EventBus:
    """Thread-safe publish-subscribe event bus.

    Subscribers are indexed by event type so publish is O(K) in the number of
    subscribers for that specific type, not the total.

    Delivery is synchronous on the publisher's thread.
    """

    def __init__(self) -> None:
        self._subscriptions: dict[type[Event], list[Subscription]] = {}
        self._lock = threading.RLock()

    def publish(self, event: Event) -> None:
        with span(f"event_bus.publish.{type(event).__name__}"):
            with self._lock:
                subs = self._subscriptions.get(type(event))
                if not subs:
                    return
                # Snapshot so handlers can subscribe/unsubscribe without mutating
                # the list mid-iteration.
                handlers = [sub.handler for sub in subs]

            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    # Broad: one bad handler must not break delivery to others.
                    logger.error(
                        f"Error in event handler for {type(event).__name__}: {e}", exc_info=True
                    )

    def subscribe(
        self,
        event_type: type[Event],
        handler: Callable[[Event], None],
    ) -> Subscription:
        subscription = Subscription(event_type=event_type, handler=handler)
        with self._lock:
            self._subscriptions.setdefault(event_type, []).append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Remove a subscription. Returns True if it was present."""
        with self._lock:
            subs = self._subscriptions.get(subscription.event_type)
            if not subs:
                return False
            try:
                subs.remove(subscription)
            except ValueError:
                return False
            if not subs:
                del self._subscriptions[subscription.event_type]
            return True
