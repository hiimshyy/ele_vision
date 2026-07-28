"""
Tests for edge/core/event_bus.py - Event Bus System.

Covers:
- Publish/subscribe pattern (specific type + wildcard)
- Multiple subscribers for same event type
- Event validation (reject non-BaseEvent objects)
- Handler error isolation (one crash doesn't affect others)
- Thread safety (publish from multiple threads)
- Event history
- Unsubscribe
- Stats tracking
"""

import threading
import time

import pytest

from edge.core.event_bus import EventBus, WILDCARD
from shared.event_schemas import (
    BaseEvent,
    EventType,
    FaceDetectedEvent,
    FaceRecognizedEvent,
    FaceUnknownEvent,
    SystemErrorEvent,
)


# --- Fixtures ---


@pytest.fixture
def bus():
    """Create a fresh EventBus instance."""
    b = EventBus(max_history=50)
    yield b
    b.shutdown()


def make_face_recognized(person_id="001", name="John", confidence=0.9):
    """Helper to create a FaceRecognizedEvent."""
    return FaceRecognizedEvent(
        source="face_recognition",
        person_id=person_id,
        person_name=name,
        confidence=confidence,
        bbox=[100, 50, 200, 200],
    )


def make_face_detected(confidence=0.95):
    """Helper to create a FaceDetectedEvent."""
    return FaceDetectedEvent(
        source="face_recognition",
        bbox=[100, 50, 200, 200],
        confidence=confidence,
        frame_id=42,
    )


def make_system_error(message="test error"):
    """Helper to create a SystemErrorEvent."""
    return SystemErrorEvent(
        source="system",
        error_message=message,
        error_type="RuntimeError",
        recoverable=True,
    )


# --- Test: Publish & Subscribe ---


class TestPubSub:
    """Tests for basic publish/subscribe pattern."""

    def test_subscribe_and_receive(self, bus):
        """Subscriber should receive published event of matching type."""
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.FACE_RECOGNIZED, handler)
        bus.publish(make_face_recognized())

        time.sleep(0.1)  # Wait for async dispatch
        assert len(received) == 1
        assert received[0].person_id == "001"

    def test_no_receive_for_different_type(self, bus):
        """Subscriber should NOT receive events of different type."""
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.FACE_RECOGNIZED, handler)
        bus.publish(make_face_detected())  # Different type

        time.sleep(0.1)
        assert len(received) == 0

    def test_wildcard_receives_all(self, bus):
        """Wildcard subscriber should receive ALL event types."""
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe("*", handler)

        bus.publish(make_face_recognized())
        bus.publish(make_face_detected())
        bus.publish(make_system_error())

        time.sleep(0.1)
        assert len(received) == 3

    def test_multiple_subscribers_same_type(self, bus):
        """Multiple subscribers to same event type should all receive."""
        counts = {"a": 0, "b": 0, "c": 0}

        def handler_a(event):
            counts["a"] += 1

        def handler_b(event):
            counts["b"] += 1

        def handler_c(event):
            counts["c"] += 1

        bus.subscribe(EventType.FACE_RECOGNIZED, handler_a)
        bus.subscribe(EventType.FACE_RECOGNIZED, handler_b)
        bus.subscribe(EventType.FACE_RECOGNIZED, handler_c)

        bus.publish(make_face_recognized())

        time.sleep(0.1)
        assert counts["a"] == 1
        assert counts["b"] == 1
        assert counts["c"] == 1

    def test_wildcard_and_specific_both_receive(self, bus):
        """Both wildcard and specific subscribers should receive."""
        specific = []
        wildcard = []

        def specific_handler(event):
            specific.append(event)

        def wildcard_handler(event):
            wildcard.append(event)

        bus.subscribe(EventType.FACE_RECOGNIZED, specific_handler)
        bus.subscribe("*", wildcard_handler)

        bus.publish(make_face_recognized())

        time.sleep(0.1)
        assert len(specific) == 1
        assert len(wildcard) == 1

    def test_multiple_publishes(self, bus):
        """Subscriber should receive all published events."""
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.FACE_RECOGNIZED, handler)

        for i in range(10):
            bus.publish(make_face_recognized(person_id=str(i)))

        time.sleep(0.2)
        assert len(received) == 10


# --- Test: Validation ---


class TestValidation:
    """Tests for event validation on publish."""

    def test_reject_non_base_event(self, bus):
        """Publishing non-BaseEvent should raise TypeError."""
        with pytest.raises(TypeError):
            bus.publish({"event_type": "fake"})

    def test_reject_string(self, bus):
        """Publishing a string should raise TypeError."""
        with pytest.raises(TypeError):
            bus.publish("not an event")

    def test_reject_none(self, bus):
        """Publishing None should raise TypeError."""
        with pytest.raises(TypeError):
            bus.publish(None)

    def test_accept_base_event_subclass(self, bus):
        """Any BaseEvent subclass should be accepted."""
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe("*", handler)

        bus.publish(make_face_recognized())
        bus.publish(make_face_detected())
        bus.publish(make_system_error())

        time.sleep(0.1)
        assert len(received) == 3


# --- Test: Unsubscribe ---


class TestUnsubscribe:
    """Tests for unsubscribe behavior."""

    def test_unsubscribe_stops_receiving(self, bus):
        """Unsubscribed handler should stop receiving events."""
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.FACE_RECOGNIZED, handler)
        bus.publish(make_face_recognized())
        time.sleep(0.1)
        assert len(received) == 1

        bus.unsubscribe(EventType.FACE_RECOGNIZED, handler)
        bus.publish(make_face_recognized())
        time.sleep(0.1)
        assert len(received) == 1  # No new events

    def test_unsubscribe_does_not_affect_others(self, bus):
        """Unsubscribing one handler should not affect other handlers."""
        counts = {"a": 0, "b": 0}

        def handler_a(event):
            counts["a"] += 1

        def handler_b(event):
            counts["b"] += 1

        bus.subscribe(EventType.FACE_RECOGNIZED, handler_a)
        bus.subscribe(EventType.FACE_RECOGNIZED, handler_b)

        bus.unsubscribe(EventType.FACE_RECOGNIZED, handler_a)
        bus.publish(make_face_recognized())

        time.sleep(0.1)
        assert counts["a"] == 0
        assert counts["b"] == 1

    def test_duplicate_subscribe_ignored(self, bus):
        """Subscribing same handler twice should not duplicate."""
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.FACE_RECOGNIZED, handler)
        bus.subscribe(EventType.FACE_RECOGNIZED, handler)

        bus.publish(make_face_recognized())
        time.sleep(0.1)
        assert len(received) == 1


# --- Test: Error Isolation ---


class TestErrorIsolation:
    """Tests for handler error isolation."""

    def test_crashing_handler_does_not_affect_others(self, bus):
        """A crashing handler should not prevent other handlers from receiving."""
        counts = {"good": 0}

        def crashing_handler(event):
            raise RuntimeError("handler crash!")

        def good_handler(event):
            counts["good"] += 1

        bus.subscribe(EventType.FACE_RECOGNIZED, crashing_handler)
        bus.subscribe(EventType.FACE_RECOGNIZED, good_handler)

        bus.publish(make_face_recognized())

        time.sleep(0.1)
        assert counts["good"] == 1

    def test_error_counted_in_stats(self, bus):
        """Handler errors should be tracked in stats."""

        def crashing_handler(event):
            raise RuntimeError("crash!")

        bus.subscribe(EventType.FACE_RECOGNIZED, crashing_handler)

        for _ in range(3):
            bus.publish(make_face_recognized())

        time.sleep(0.2)
        assert bus.stats.handler_errors == 3


# --- Test: Event History ---


class TestEventHistory:
    """Tests for event history buffer."""

    def test_history_stores_events(self, bus):
        """Published events should appear in history."""
        bus.publish(make_face_recognized(person_id="001"))
        bus.publish(make_face_recognized(person_id="002"))

        assert len(bus.history) == 2
        assert bus.history[0].person_id == "001"
        assert bus.history[1].person_id == "002"

    def test_history_bounded(self):
        """History should not exceed max_history."""
        bus = EventBus(max_history=5)

        for i in range(10):
            bus.publish(make_face_recognized(person_id=str(i)))

        assert len(bus.history) == 5
        # Should contain last 5 events
        assert bus.history[0].person_id == "5"
        assert bus.history[4].person_id == "9"
        bus.shutdown()

    def test_history_not_affected_by_no_subscribers(self, bus):
        """Events should be stored in history even without subscribers."""
        bus.publish(make_face_recognized())
        assert len(bus.history) == 1


# --- Test: Stats ---


class TestStats:
    """Tests for EventBusStats tracking."""

    def test_publish_count(self, bus):
        """Stats should track published event count."""
        bus.publish(make_face_recognized())
        bus.publish(make_face_detected())
        bus.publish(make_system_error())

        assert bus.stats.events_published == 3

    def test_delivered_count(self, bus):
        """Stats should track delivered event count."""
        def handler(event):
            pass

        bus.subscribe("*", handler)

        bus.publish(make_face_recognized())
        bus.publish(make_face_detected())

        time.sleep(0.1)
        assert bus.stats.events_delivered == 2

    def test_subscriber_count(self, bus):
        """get_subscriber_count should return correct counts."""
        def h1(event): pass
        def h2(event): pass
        def h3(event): pass

        bus.subscribe(EventType.FACE_RECOGNIZED, h1)
        bus.subscribe(EventType.FACE_RECOGNIZED, h2)
        bus.subscribe(EventType.FACE_DETECTED, h3)

        assert bus.get_subscriber_count(EventType.FACE_RECOGNIZED) == 2
        assert bus.get_subscriber_count(EventType.FACE_DETECTED) == 1
        assert bus.get_subscriber_count() == 3


# --- Test: Thread Safety ---


class TestThreadSafety:
    """Tests for concurrent publish/subscribe."""

    def test_concurrent_publish(self, bus):
        """Multiple threads publishing simultaneously should be safe."""
        received = []
        lock = threading.Lock()

        def handler(event):
            with lock:
                received.append(event)

        bus.subscribe(EventType.FACE_RECOGNIZED, handler)

        def publisher(n):
            for i in range(10):
                bus.publish(make_face_recognized(person_id=f"{n}-{i}"))

        threads = [threading.Thread(target=publisher, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        time.sleep(0.5)
        assert len(received) == 50  # 5 threads × 10 events

    def test_subscribe_during_publish(self, bus):
        """Subscribing while events are being published should be safe."""
        received = []

        def handler(event):
            received.append(event)

        # Start publishing
        def publisher():
            for _ in range(20):
                bus.publish(make_face_recognized())
                time.sleep(0.01)

        t = threading.Thread(target=publisher)
        t.start()

        # Subscribe mid-stream
        time.sleep(0.05)
        bus.subscribe(EventType.FACE_RECOGNIZED, handler)

        t.join()
        time.sleep(0.2)

        # Should have received some (not all, since subscribed late)
        assert len(received) > 0
        assert len(received) <= 20
