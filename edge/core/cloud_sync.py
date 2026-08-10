"""
Smart Cabin Platform - Cloud Sync (MQTT Client)

Bridges Edge EventBus to cloud via MQTT. Handles:
- Event publishing (face recognized/unknown → MQTT)
- Heartbeat (periodic system stats)
- Offline buffering (SQLite queue when disconnected)
- Command handling (cloud → edge)

Usage:
    from edge.core.cloud_sync import CloudSync

    sync = CloudSync(config.mqtt, event_bus)
    sync.start()
    ...
    sync.stop()
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil
import paho.mqtt.client as mqtt

from edge.core.logging_setup import get_logger
from shared.event_schemas import (
    BaseEvent, EventType, FaceRecognizedEvent, FaceUnknownEvent,
    get_mqtt_topic, MQTT_TOPIC_MAP,
)
from shared.mqtt_topics import (
    TOPIC_HEARTBEAT, TOPIC_SYSTEM_START, TOPIC_SYSTEM_STOP,
)

logger = get_logger("system")


# --- Offline Buffer ---


class OfflineBuffer:
    """
    SQLite-backed message queue for offline buffering.

    Stores MQTT messages when disconnected, flushes when reconnected.
    """

    def __init__(self, db_path: str | Path = "data/db/mqtt_buffer.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._initialize()

    def _initialize(self) -> None:
        """Create buffer table."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS mqtt_buffer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                payload TEXT NOT NULL,
                qos INTEGER DEFAULT 1,
                created_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def push(self, topic: str, payload: str, qos: int = 1) -> None:
        """Add message to buffer."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO mqtt_buffer (topic, payload, qos, created_at) VALUES (?, ?, ?, ?)",
                (topic, payload, qos, time.time()),
            )
            self._conn.commit()

    def pop_all(self) -> list[tuple[str, str, int]]:
        """Get and remove all buffered messages (FIFO order)."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, topic, payload, qos FROM mqtt_buffer ORDER BY id"
            )
            rows = cursor.fetchall()
            if rows:
                ids = [row[0] for row in rows]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"DELETE FROM mqtt_buffer WHERE id IN ({placeholders})", ids
                )
                self._conn.commit()
            return [(row[1], row[2], row[3]) for row in rows]

    @property
    def count(self) -> int:
        """Number of buffered messages."""
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM mqtt_buffer")
            return cursor.fetchone()[0]

    def clear(self) -> None:
        """Clear all buffered messages."""
        with self._lock:
            self._conn.execute("DELETE FROM mqtt_buffer")
            self._conn.commit()

    def close(self) -> None:
        """Close database."""
        if self._conn:
            self._conn.close()
            self._conn = None


# --- Cloud Sync ---


class CloudSync:
    """
    MQTT-based cloud synchronization.

    Subscribes to EventBus events and publishes them to MQTT broker.
    Handles reconnection, offline buffering, heartbeat, and commands.

    Args:
        broker_host: MQTT broker hostname
        broker_port: MQTT broker port
        device_id: Device identifier for topic namespacing
        username: MQTT username (optional)
        password: MQTT password (optional)
        keepalive: MQTT keepalive interval in seconds
        heartbeat_interval: Seconds between heartbeat publishes
        buffer_path: Path for offline buffer SQLite DB
    """

    def __init__(self,
                 broker_host: str = "localhost",
                 broker_port: int = 1883,
                 device_id: str = "cabin-001",
                 username: str = "",
                 password: str = "",
                 keepalive: int = 60,
                 heartbeat_interval: int = 30,
                 buffer_path: str = "data/db/mqtt_buffer.db"):
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._device_id = device_id
        self._username = username
        self._password = password
        self._keepalive = keepalive
        self._heartbeat_interval = heartbeat_interval

        # MQTT client (paho-mqtt v2)
        self._client: mqtt.Client | None = None
        self._connected = False
        self._started = False
        self._start_time = 0.0

        # Offline buffer
        self._buffer = OfflineBuffer(buffer_path)

        # Heartbeat thread
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()

        # Command handlers
        self._command_handlers: dict[str, Callable[[dict], None]] = {}

        # Stats
        self._stats = SyncStats()

    @property
    def is_connected(self) -> bool:
        """Whether currently connected to broker."""
        return self._connected

    @property
    def stats(self) -> "SyncStats":
        """Sync statistics."""
        return self._stats

    @property
    def device_id(self) -> str:
        return self._device_id

    def start(self) -> bool:
        """
        Connect to MQTT broker and start sync.

        Returns:
            True if connection initiated (may still be connecting async)
        """
        if self._started:
            return True

        try:
            # Create MQTT client (paho-mqtt v2 API)
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"smart-cabin-{self._device_id}",
                protocol=mqtt.MQTTv5,
            )

            # Auth
            if self._username:
                self._client.username_pw_set(self._username, self._password)

            # Callbacks
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message

            # Last will (published by broker if we disconnect unexpectedly)
            will_topic = TOPIC_SYSTEM_STOP.format(device_id=self._device_id)
            will_payload = json.dumps({
                "device_id": self._device_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "unexpected_disconnect",
            })
            self._client.will_set(will_topic, will_payload, qos=1, retain=False)

            # Connect (non-blocking)
            self._client.connect_async(self._broker_host, self._broker_port, self._keepalive)
            self._client.loop_start()
            self._started = True
            self._start_time = time.time()

            # Start heartbeat thread
            self._heartbeat_stop.clear()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True, name="mqtt-heartbeat"
            )
            self._heartbeat_thread.start()

            logger.info(
                "event=cloud_sync_starting | broker={h}:{p} | device_id={d}",
                h=self._broker_host, p=self._broker_port, d=self._device_id,
            )
            return True

        except Exception as e:
            logger.error("event=cloud_sync_start_failed | error={err}", err=str(e))
            return False

    def stop(self) -> None:
        """Disconnect from broker and stop sync."""
        if not self._started:
            return

        # Stop heartbeat
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=3)

        # Publish stop event
        if self._connected:
            self._publish_system_event("stop", {"reason": "shutdown"})

        # Disconnect
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()

        self._buffer.close()
        self._started = False
        self._connected = False

        logger.info(
            "event=cloud_sync_stopped | messages_sent={s} | messages_buffered={b}",
            s=self._stats.messages_published, b=self._stats.messages_buffered,
        )

    # --- Event Publishing ---

    def publish_event(self, event: BaseEvent) -> bool:
        """
        Publish an event to MQTT.

        If disconnected, buffers the message for later delivery.

        Args:
            event: BaseEvent instance to publish

        Returns:
            True if published or buffered successfully
        """
        topic = get_mqtt_topic(event)
        if topic is None:
            return False

        # Serialize event to JSON
        payload = event.model_dump_json()

        if self._connected and self._client:
            try:
                result = self._client.publish(topic, payload, qos=1)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    self._stats.messages_published += 1
                    return True
            except Exception as e:
                logger.warning("event=mqtt_publish_failed | error={err}", err=str(e))

        # Buffer if not connected or publish failed
        self._buffer.push(topic, payload, qos=1)
        self._stats.messages_buffered += 1
        return True

    def publish_raw(self, topic: str, payload: dict, qos: int = 1) -> bool:
        """
        Publish a raw message (dict → JSON) to a topic.

        Args:
            topic: MQTT topic (with device_id already formatted)
            payload: Dict to serialize as JSON
            qos: MQTT QoS level

        Returns:
            True if published or buffered
        """
        payload_str = json.dumps(payload)

        if self._connected and self._client:
            try:
                result = self._client.publish(topic, payload_str, qos=qos)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    self._stats.messages_published += 1
                    return True
            except Exception:
                pass

        self._buffer.push(topic, payload_str, qos=qos)
        self._stats.messages_buffered += 1
        return True

    # --- Command Handling ---

    def register_command(self, command_name: str, handler: Callable[[dict], None]) -> None:
        """
        Register a handler for a cloud command.

        Args:
            command_name: Command name (e.g., "sync_faces", "restart_plugin")
            handler: Callable that receives the command payload dict
        """
        self._command_handlers[command_name] = handler
        logger.info("event=command_registered | command={cmd}", cmd=command_name)

    # --- MQTT Callbacks ---

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        """Called when connected to broker."""
        if reason_code == 0:
            self._connected = True
            self._stats.reconnect_count += 1

            logger.info(
                "event=mqtt_connected | broker={h}:{p} | device_id={d}",
                h=self._broker_host, p=self._broker_port, d=self._device_id,
            )

            # Subscribe to command topics
            cmd_topic = f"cabin/{self._device_id}/command/+"
            client.subscribe(cmd_topic, qos=1)
            logger.info("event=mqtt_subscribed | topic={t}", t=cmd_topic)

            # Publish start event
            self._publish_system_event("start", {"version": "1.0.0"})

            # Flush offline buffer
            self._flush_buffer()
        else:
            logger.error(
                "event=mqtt_connect_failed | reason_code={rc}",
                rc=reason_code,
            )

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        """Called when disconnected from broker."""
        self._connected = False
        logger.warning(
            "event=mqtt_disconnected | reason_code={rc} | buffer_count={bc}",
            rc=reason_code, bc=self._buffer.count,
        )

    def _on_message(self, client, userdata, msg):
        """Called when a message is received (commands from cloud)."""
        try:
            # Parse topic: cabin/{device_id}/command/{command_name}
            parts = msg.topic.split("/")
            if len(parts) >= 4 and parts[2] == "command":
                command_name = parts[3]
                payload = json.loads(msg.payload.decode()) if msg.payload else {}

                logger.info(
                    "event=command_received | command={cmd} | payload_size={s}",
                    cmd=command_name, s=len(msg.payload),
                )

                # Dispatch to handler
                handler = self._command_handlers.get(command_name)
                if handler:
                    try:
                        handler(payload)
                        self._stats.commands_received += 1
                    except Exception as e:
                        logger.error(
                            "event=command_handler_error | command={cmd} | error={err}",
                            cmd=command_name, err=str(e),
                        )
                else:
                    logger.warning(
                        "event=command_no_handler | command={cmd}",
                        cmd=command_name,
                    )
        except Exception as e:
            logger.error("event=mqtt_message_error | error={err}", err=str(e))

    # --- Heartbeat ---

    def _heartbeat_loop(self) -> None:
        """Background thread: publish heartbeat every N seconds."""
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            self._publish_heartbeat()

    def _publish_heartbeat(self) -> None:
        """Publish system stats as heartbeat."""
        uptime = time.time() - self._start_time if self._start_time > 0 else 0

        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            payload = {
                "device_id": self._device_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_s": round(uptime, 1),
                "cpu_percent": cpu_percent,
                "ram_used_mb": round(mem.used / (1024 * 1024), 1),
                "ram_percent": mem.percent,
                "mqtt_connected": self._connected,
                "buffer_count": self._buffer.count,
                "messages_published": self._stats.messages_published,
            }
        except Exception:
            payload = {
                "device_id": self._device_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_s": round(uptime, 1),
                "mqtt_connected": self._connected,
            }

        topic = TOPIC_HEARTBEAT.format(device_id=self._device_id)
        self.publish_raw(topic, payload, qos=0)
        self._stats.heartbeats_sent += 1

    # --- System Events ---

    def _publish_system_event(self, event_type: str, extra: dict = None) -> None:
        """Publish a system event (start/stop)."""
        if event_type == "start":
            topic = TOPIC_SYSTEM_START.format(device_id=self._device_id)
        else:
            topic = TOPIC_SYSTEM_STOP.format(device_id=self._device_id)

        payload = {
            "device_id": self._device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(extra or {}),
        }
        self.publish_raw(topic, payload, qos=1)

    # --- Buffer Flush ---

    def _flush_buffer(self) -> None:
        """Flush all buffered messages to broker."""
        messages = self._buffer.pop_all()
        if not messages:
            return

        flushed = 0
        for topic, payload, qos in messages:
            try:
                result = self._client.publish(topic, payload, qos=qos)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    flushed += 1
            except Exception:
                # Re-buffer if flush fails
                self._buffer.push(topic, payload, qos)
                break

        if flushed > 0:
            self._stats.messages_flushed += flushed
            logger.info(
                "event=buffer_flushed | count={n} | remaining={r}",
                n=flushed, r=self._buffer.count,
            )

    # --- Event Bus Bridge ---

    def bridge_event_bus(self, event_bus) -> None:
        """
        Subscribe to EventBus and auto-publish events to MQTT.

        Subscribes to all publishable event types and forwards to MQTT.

        Args:
            event_bus: EventBus instance to bridge
        """
        from shared.event_schemas import EventType

        # Subscribe to events that have MQTT topic mappings
        publishable_types = [
            EventType.FACE_RECOGNIZED,
            EventType.FACE_UNKNOWN,
            EventType.SYSTEM_ERROR,
        ]

        for event_type in publishable_types:
            event_bus.subscribe(event_type, self.publish_event)

        logger.info(
            "event=event_bus_bridged | event_types={n}",
            n=len(publishable_types),
        )


# --- Stats ---


class SyncStats:
    """MQTT sync statistics."""

    def __init__(self):
        self.messages_published: int = 0
        self.messages_buffered: int = 0
        self.messages_flushed: int = 0
        self.heartbeats_sent: int = 0
        self.commands_received: int = 0
        self.reconnect_count: int = 0
