"""
Tests for MQTT Cloud Sync.

Tests cover:
- OfflineBuffer: push, pop, count, clear
- CloudSync: initialization, event publishing, heartbeat, command handling
- Event bridge: EventBus → MQTT forwarding

Note: Most tests mock the MQTT client since we don't require a running broker.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from edge.core.cloud_sync import CloudSync, OfflineBuffer, SyncStats
from edge.core.event_bus import EventBus
from shared.event_schemas import (
    FaceRecognizedEvent, FaceUnknownEvent, EventType,
)


# --- Test: OfflineBuffer ---


class TestOfflineBuffer:
    """Tests for SQLite offline message buffer."""

    def test_empty_buffer(self, tmp_path):
        """New buffer should be empty."""
        buf = OfflineBuffer(tmp_path / "buf.db")
        assert buf.count == 0
        assert buf.pop_all() == []
        buf.close()

    def test_push_and_count(self, tmp_path):
        """Push should increment count."""
        buf = OfflineBuffer(tmp_path / "buf.db")
        buf.push("topic/a", '{"msg": 1}', qos=1)
        buf.push("topic/b", '{"msg": 2}', qos=0)
        assert buf.count == 2
        buf.close()

    def test_pop_all_returns_fifo(self, tmp_path):
        """pop_all should return messages in FIFO order."""
        buf = OfflineBuffer(tmp_path / "buf.db")
        buf.push("t1", "p1", 1)
        buf.push("t2", "p2", 0)
        buf.push("t3", "p3", 1)

        messages = buf.pop_all()
        assert len(messages) == 3
        assert messages[0] == ("t1", "p1", 1)
        assert messages[1] == ("t2", "p2", 0)
        assert messages[2] == ("t3", "p3", 1)

        # Should be empty after pop
        assert buf.count == 0
        buf.close()

    def test_pop_all_clears_buffer(self, tmp_path):
        """After pop_all, buffer should be empty."""
        buf = OfflineBuffer(tmp_path / "buf.db")
        buf.push("t", "p", 1)
        buf.pop_all()
        assert buf.count == 0
        buf.close()

    def test_clear(self, tmp_path):
        """Clear should remove all messages."""
        buf = OfflineBuffer(tmp_path / "buf.db")
        buf.push("t1", "p1", 1)
        buf.push("t2", "p2", 1)
        buf.clear()
        assert buf.count == 0
        buf.close()

    def test_persistence(self, tmp_path):
        """Buffer should persist across close/reopen."""
        db_path = tmp_path / "persist.db"
        buf = OfflineBuffer(db_path)
        buf.push("topic", "payload", 1)
        buf.close()

        # Reopen
        buf2 = OfflineBuffer(db_path)
        assert buf2.count == 1
        messages = buf2.pop_all()
        assert messages[0] == ("topic", "payload", 1)
        buf2.close()


# --- Test: CloudSync Init ---


class TestCloudSyncInit:
    """Tests for CloudSync initialization."""

    def test_default_properties(self, tmp_path):
        """Should have correct defaults."""
        sync = CloudSync(
            broker_host="test.local",
            broker_port=1883,
            device_id="test-001",
            buffer_path=str(tmp_path / "buf.db"),
        )
        assert sync.device_id == "test-001"
        assert sync.is_connected is False
        assert sync.stats.messages_published == 0

    def test_stats_initial(self, tmp_path):
        """Stats should start at zero."""
        sync = CloudSync(buffer_path=str(tmp_path / "buf.db"))
        assert sync.stats.messages_published == 0
        assert sync.stats.messages_buffered == 0
        assert sync.stats.heartbeats_sent == 0
        assert sync.stats.commands_received == 0


# --- Test: CloudSync Publishing ---


class TestCloudSyncPublishing:
    """Tests for event publishing (mocked MQTT client)."""

    def test_publish_event_when_disconnected_buffers(self, tmp_path):
        """Should buffer event when not connected."""
        sync = CloudSync(buffer_path=str(tmp_path / "buf.db"))
        # Not connected → should buffer
        event = FaceRecognizedEvent(
            source="test", person_id="p001", person_name="Alice",
            confidence=0.9, bbox=[10, 20, 100, 150],
        )
        result = sync.publish_event(event)
        assert result is True
        assert sync.stats.messages_buffered == 1
        assert sync._buffer.count == 1

    def test_publish_raw_when_disconnected_buffers(self, tmp_path):
        """Should buffer raw message when not connected."""
        sync = CloudSync(buffer_path=str(tmp_path / "buf.db"))
        result = sync.publish_raw("test/topic", {"key": "value"})
        assert result is True
        assert sync.stats.messages_buffered == 1

    def test_publish_event_unknown(self, tmp_path):
        """Should handle FaceUnknownEvent correctly."""
        sync = CloudSync(buffer_path=str(tmp_path / "buf.db"))
        event = FaceUnknownEvent(
            source="test", confidence=0.8, bbox=[50, 60, 150, 200],
        )
        result = sync.publish_event(event)
        assert result is True


# --- Test: Command Handling ---


class TestCloudSyncCommands:
    """Tests for command registration and handling."""

    def test_register_command(self, tmp_path):
        """Should register command handler."""
        sync = CloudSync(buffer_path=str(tmp_path / "buf.db"))
        handler = MagicMock()
        sync.register_command("test_cmd", handler)
        assert "test_cmd" in sync._command_handlers

    def test_command_dispatch(self, tmp_path):
        """Should dispatch received command to handler."""
        sync = CloudSync(device_id="test-001", buffer_path=str(tmp_path / "buf.db"))
        handler = MagicMock()
        sync.register_command("restart_plugin", handler)

        # Simulate incoming message
        msg = MagicMock()
        msg.topic = "cabin/test-001/command/restart_plugin"
        msg.payload = json.dumps({"plugin": "face_recognition"}).encode()

        sync._on_message(None, None, msg)
        handler.assert_called_once_with({"plugin": "face_recognition"})
        assert sync.stats.commands_received == 1

    def test_command_unknown_handler(self, tmp_path):
        """Should log warning for unknown command."""
        sync = CloudSync(device_id="test-001", buffer_path=str(tmp_path / "buf.db"))

        msg = MagicMock()
        msg.topic = "cabin/test-001/command/unknown_cmd"
        msg.payload = b'{}'

        # Should not crash
        sync._on_message(None, None, msg)
        assert sync.stats.commands_received == 0


# --- Test: Event Bus Bridge ---


class TestEventBusBridge:
    """Tests for EventBus → MQTT bridge."""

    def test_bridge_subscribes_to_events(self, tmp_path):
        """Bridge should subscribe to publishable event types."""
        sync = CloudSync(buffer_path=str(tmp_path / "buf.db"))
        bus = EventBus()

        sync.bridge_event_bus(bus)

        # Should have subscribers for face events
        assert bus.get_subscriber_count(EventType.FACE_RECOGNIZED) > 0
        assert bus.get_subscriber_count(EventType.FACE_UNKNOWN) > 0
        bus.shutdown()

    def test_bridge_forwards_events(self, tmp_path):
        """Published EventBus event should be buffered in MQTT."""
        sync = CloudSync(buffer_path=str(tmp_path / "buf.db"))
        bus = EventBus()

        sync.bridge_event_bus(bus)

        # Publish event via bus
        event = FaceRecognizedEvent(
            source="face_recognition", person_id="p001",
            person_name="Alice", confidence=0.85, bbox=[10, 20, 100, 150],
        )
        bus.publish(event)

        # Wait for async dispatch
        time.sleep(0.3)

        # Should be buffered (not connected)
        assert sync.stats.messages_buffered >= 1
        assert sync._buffer.count >= 1

        bus.shutdown()

    def test_bridge_event_payload_format(self, tmp_path):
        """Buffered event should have correct JSON payload."""
        sync = CloudSync(device_id="cabin-001", buffer_path=str(tmp_path / "buf.db"))

        event = FaceRecognizedEvent(
            source="face_recognition", person_id="p001",
            person_name="Alice", confidence=0.85, bbox=[10, 20, 100, 150],
        )
        sync.publish_event(event)

        messages = sync._buffer.pop_all()
        assert len(messages) == 1

        topic, payload_str, qos = messages[0]
        assert "cabin-001/face/recognized" in topic
        payload = json.loads(payload_str)
        assert payload["person_id"] == "p001"
        assert payload["person_name"] == "Alice"
        assert payload["confidence"] == 0.85
