"""
Smart Cabin Platform - Event Schemas

Defines the event models used for inter-plugin communication via the Event Bus
and for Edge-Cloud synchronization via MQTT.

All events inherit from BaseEvent and include metadata (timestamp, source, device_id).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --- Event Types ---


class EventType(str, Enum):
    """All supported event types in the platform."""

    # Face Recognition events
    FACE_DETECTED = "face.detected"
    FACE_RECOGNIZED = "face.recognized"
    FACE_UNKNOWN = "face.unknown"

    # People Counter events (future)
    PERSON_ENTERED = "person.entered"
    PERSON_EXITED = "person.exited"
    PERSON_COUNT_UPDATED = "person.count_updated"

    # Elevator Control events (future)
    FLOOR_REQUESTED = "elevator.floor_requested"
    FLOOR_ARRIVED = "elevator.floor_arrived"

    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    PLUGIN_LOADED = "system.plugin_loaded"
    PLUGIN_ERROR = "system.plugin_error"


# --- Base Event ---


class BaseEvent(BaseModel):
    """Base event model. All events inherit from this."""

    event_type: EventType = Field(description="Type of the event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when event was created",
    )
    source: str = Field(description="Module/plugin that generated this event")
    device_id: str = Field(default="cabin-001", description="Device identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional unstructured metadata"
    )


# --- Face Recognition Events ---


class FaceDetectedEvent(BaseEvent):
    """Emitted when a face is detected in a frame (before recognition)."""

    event_type: EventType = EventType.FACE_DETECTED
    bbox: list[int] = Field(description="Bounding box [x, y, w, h]")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    frame_id: int = Field(default=0, description="Frame sequence number")


class FaceRecognizedEvent(BaseEvent):
    """Emitted when a detected face is matched to a known person."""

    event_type: EventType = EventType.FACE_RECOGNIZED
    person_id: str = Field(description="Matched person identifier")
    person_name: str = Field(default="", description="Person display name")
    confidence: float = Field(ge=0.0, le=1.0, description="Recognition confidence")
    bbox: list[int] = Field(description="Bounding box [x, y, w, h]")


class FaceUnknownEvent(BaseEvent):
    """Emitted when a detected face does not match any known person."""

    event_type: EventType = EventType.FACE_UNKNOWN
    confidence: float = Field(
        ge=0.0, le=1.0, description="Best match confidence (below threshold)"
    )
    bbox: list[int] = Field(description="Bounding box [x, y, w, h]")


# --- System Events ---


class SystemErrorEvent(BaseEvent):
    """Emitted when a system-level error occurs."""

    event_type: EventType = EventType.SYSTEM_ERROR
    error_message: str = Field(description="Error description")
    error_type: str = Field(default="", description="Exception class name")
    recoverable: bool = Field(default=True, description="Whether system can recover")


# --- MQTT Topic Mapping ---

MQTT_TOPIC_MAP: dict[EventType, str] = {
    EventType.FACE_RECOGNIZED: "cabin/{device_id}/face/recognized",
    EventType.FACE_UNKNOWN: "cabin/{device_id}/face/unknown",
    EventType.FACE_DETECTED: "cabin/{device_id}/face/detected",
    EventType.PERSON_COUNT_UPDATED: "cabin/{device_id}/people/count",
    EventType.SYSTEM_ERROR: "cabin/{device_id}/system/error",
    EventType.SYSTEM_START: "cabin/{device_id}/system/start",
    EventType.SYSTEM_STOP: "cabin/{device_id}/system/stop",
}


def get_mqtt_topic(event: BaseEvent) -> str | None:
    """Get the MQTT topic for a given event, with device_id interpolated."""
    template = MQTT_TOPIC_MAP.get(event.event_type)
    if template is None:
        return None
    return template.format(device_id=event.device_id)
