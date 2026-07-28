"""
Smart Cabin Platform - Event Bus

Thread-safe pub/sub system for inter-plugin communication.

Features:
- Subscribe by EventType or wildcard (all events)
- Thread-safe publish (non-blocking, dispatches to ThreadPoolExecutor)
- Pydantic validation on publish (rejects invalid events)
- Event history (bounded buffer for late subscribers / debugging)
- Subscriber error isolation (one handler crash doesn't affect others)

Usage:
    from edge.core.event_bus import EventBus
    from shared.event_schemas import FaceRecognizedEvent, EventType

    bus = EventBus()

    # Subscribe to specific event type
    def on_face(event):
        print(f"Recognized: {event.person_name}")

    bus.subscribe(EventType.FACE_RECOGNIZED, on_face)

    # Subscribe to ALL events
    bus.subscribe("*", lambda e: print(e.event_type))

    # Publish
    bus.publish(FaceRecognizedEvent(
        source="face_recognition",
        person_id="001",
        person_name="John",
        confidence=0.92,
        bbox=[100, 50, 200, 200],
    ))
"""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from shared.event_schemas import BaseEvent, EventType
from edge.core.logging_setup import get_logger

logger = get_logger("system")

# Type alias for event handler
EventHandler = Callable[[BaseEvent], None]

# Wildcard subscription key
WILDCARD = "*"


class EventBus:
    """
    Thread-safe event bus with pub/sub pattern.

    Subscribers register for specific EventType or wildcard "*".
    Published events are validated (must be BaseEvent subclass) and
    dispatched to matching handlers via thread pool.
    """

    def __init__(self, max_history: int = 100, max_workers: int = 4):
        """
        Args:
            max_history: Number of recent events to keep for debugging
            max_workers: Thread pool size for handler dispatch
        """
        self._subscribers: dict[str, list[EventHandler]] = {}
        self._lock = threading.Lock()
        self._history: deque[BaseEvent] = deque(maxlen=max_history)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="eventbus"
        )
        self._stats = EventBusStats()

    @property
    def stats(self) -> "EventBusStats":
        """Current event bus statistics."""
        return self._stats

    @property
    def history(self) -> list[BaseEvent]:
        """Recent event history (newest last)."""
        return list(self._history)

    def subscribe(self, event_type: EventType | str, handler: EventHandler) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: EventType enum value, or "*" for all events
            handler: Callable that accepts a BaseEvent
        """
        key = event_type.value if isinstance(event_type, EventType) else str(event_type)

        with self._lock:
            if key not in self._subscribers:
                self._subscribers[key] = []
            if handler not in self._subscribers[key]:
                self._subscribers[key].append(handler)

        logger.info(
            "event=bus_subscribe | event_type={etype} | handler={name}",
            etype=key, name=handler.__name__,
        )

    def unsubscribe(self, event_type: EventType | str, handler: EventHandler) -> None:
        """
        Unsubscribe a handler from an event type.

        Args:
            event_type: EventType enum value, or "*" for all events
            handler: Previously registered handler
        """
        key = event_type.value if isinstance(event_type, EventType) else str(event_type)

        with self._lock:
            if key in self._subscribers:
                self._subscribers[key] = [
                    h for h in self._subscribers[key] if h is not handler
                ]

        logger.info(
            "event=bus_unsubscribe | event_type={etype} | handler={name}",
            etype=key, name=handler.__name__,
        )

    def publish(self, event: BaseEvent) -> None:
        """
        Publish an event to all matching subscribers.

        Validates the event (must be BaseEvent instance), stores in history,
        and dispatches to handlers via thread pool (non-blocking).

        Args:
            event: A BaseEvent instance (or subclass)

        Raises:
            TypeError: If event is not a BaseEvent instance
        """
        if not isinstance(event, BaseEvent):
            raise TypeError(
                f"Event must be a BaseEvent instance, got {type(event).__name__}"
            )

        # Store in history
        self._history.append(event)
        self._stats.events_published += 1

        # Find matching handlers
        event_key = event.event_type.value
        handlers = []

        with self._lock:
            # Exact match subscribers
            if event_key in self._subscribers:
                handlers.extend(self._subscribers[event_key])
            # Wildcard subscribers
            if WILDCARD in self._subscribers:
                handlers.extend(self._subscribers[WILDCARD])

        if not handlers:
            return

        # Dispatch to handlers (non-blocking)
        for handler in handlers:
            self._executor.submit(self._invoke_handler, handler, event)

    def _invoke_handler(self, handler: EventHandler, event: BaseEvent) -> None:
        """Invoke a single handler with error isolation."""
        try:
            handler(event)
            self._stats.events_delivered += 1
        except Exception as e:
            self._stats.handler_errors += 1
            logger.error(
                "event=bus_handler_error | handler={name} | event_type={etype} | error={err}",
                name=handler.__name__,
                etype=event.event_type.value,
                err=str(e),
            )

    def get_subscriber_count(self, event_type: EventType | str | None = None) -> int:
        """
        Get number of subscribers.

        Args:
            event_type: Specific type to count, or None for total
        """
        with self._lock:
            if event_type is None:
                return sum(len(handlers) for handlers in self._subscribers.values())
            key = event_type.value if isinstance(event_type, EventType) else str(event_type)
            return len(self._subscribers.get(key, []))

    def clear(self) -> None:
        """Remove all subscribers and clear history."""
        with self._lock:
            self._subscribers.clear()
        self._history.clear()
        self._stats = EventBusStats()

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=False)


class EventBusStats:
    """Simple stats counter for the event bus."""

    def __init__(self):
        self.events_published: int = 0
        self.events_delivered: int = 0
        self.handler_errors: int = 0
