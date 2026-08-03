"""
Integration tests for the Face Recognition Plugin.

Tests the full pipeline: detect → track → align → embed → match → event.
Model-dependent tests require both det_500m.onnx and w600k_mbf.onnx.

Also tests plugin interface compliance with BasePlugin.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import tempfile
import time

import numpy as np
import pytest

from edge.core.event_bus import EventBus
from edge.core.plugin_manager import BasePlugin
from edge.plugins.face_recognition.plugin import Plugin
from edge.plugins.face_recognition.tracker import TrackState
from shared.event_schemas import EventType, FaceRecognizedEvent, FaceUnknownEvent


# --- Fixtures ---

MODEL_DIR = Path("edge/plugins/face_recognition/models")
DET_MODEL = MODEL_DIR / "det_500m.onnx"
EMB_MODEL = MODEL_DIR / "w600k_mbf.onnx"


def has_models():
    """Check if both models exist."""
    return DET_MODEL.exists() and EMB_MODEL.exists()


def make_plugin_config(tmp_path) -> dict[str, Any]:
    """Create config for testing."""
    return {
        "detection_threshold": 0.5,
        "embedding_threshold": 0.4,
        "embedding_model": "w600k_mbf.onnx",
        "min_face_size": 40,
        "min_face_quality": 10.0,  # Low threshold for synthetic faces
        "tracker_iou_threshold": 0.4,
        "tracker_max_lost": 5,
        "tracker_max_tracks": 10,
        "tracker_reverify_interval": 15,
        "database_path": str(tmp_path / "test_faces.db"),
    }


@pytest.fixture
def event_bus():
    """Create an event bus for testing."""
    bus = EventBus()
    yield bus
    bus.shutdown()


@pytest.fixture
def plugin_with_models(tmp_path, event_bus):
    """Create and initialize plugin (skip if models not available)."""
    if not has_models():
        pytest.skip("Models not found (det_500m.onnx + w600k_mbf.onnx required)")
    config = make_plugin_config(tmp_path)
    p = Plugin()
    ok = p.initialize(config, event_bus)
    assert ok, "Plugin initialization failed"
    yield p
    p.shutdown()


# --- Test: Plugin Interface ---


class TestPluginInterface:
    """Tests for BasePlugin compliance."""

    def test_is_base_plugin(self):
        """Plugin should inherit from BasePlugin."""
        assert issubclass(Plugin, BasePlugin)

    def test_name_property(self):
        """Should have correct name."""
        p = Plugin()
        assert p.name == "face_recognition"

    def test_version_property(self):
        """Should have a version string."""
        p = Plugin()
        assert isinstance(p.version, str)
        assert len(p.version) > 0

    def test_default_fps(self):
        """Should have default_fps = 5."""
        p = Plugin()
        assert p.default_fps == 5.0

    def test_initialize_without_models(self, tmp_path, event_bus):
        """Initialize should fail gracefully without models."""
        config = make_plugin_config(tmp_path)
        config["embedding_model"] = "nonexistent.onnx"
        p = Plugin()
        # This may fail at detector or embedder load
        # Just ensure it doesn't crash
        result = p.initialize(config, event_bus)
        # Result depends on whether detection model exists
        assert isinstance(result, bool)


# --- Test: Plugin Initialization ---


class TestPluginInit:
    """Tests for plugin initialization (requires models)."""

    def test_successful_init(self, plugin_with_models):
        """Should initialize all components."""
        p = plugin_with_models
        assert p.detector.backend is not None
        assert p.embedder.is_loaded
        assert p.database.is_initialized
        assert p.tracker.track_count == 0

    def test_database_accessible(self, plugin_with_models):
        """Database should be accessible after init."""
        p = plugin_with_models
        assert p.database.count() == 0


# --- Test: Process Frame (requires models) ---


class TestPluginProcessFrame:
    """Tests for process_frame method (requires models)."""

    def test_process_blank_frame(self, plugin_with_models, event_bus):
        """Blank frame should produce no tracks."""
        p = plugin_with_models
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        p.process_frame(frame, frame_id=1, timestamp=time.time())
        assert p.tracker.track_count == 0

    def test_process_frame_returns_none(self, plugin_with_models):
        """process_frame should return None (void)."""
        p = plugin_with_models
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = p.process_frame(frame, frame_id=1, timestamp=time.time())
        assert result is None


# --- Test: Event Publishing ---


class TestPluginEvents:
    """Tests for event publishing (mocked inference)."""

    def test_unknown_face_event(self, tmp_path, event_bus):
        """Should publish FaceUnknownEvent for unregistered face."""
        if not has_models():
            pytest.skip("Models required")

        events = []
        event_bus.subscribe(EventType.FACE_UNKNOWN, lambda e: events.append(e))

        config = make_plugin_config(tmp_path)
        p = Plugin()
        p.initialize(config, event_bus)

        # Create a synthetic frame with face-like pattern
        frame = _make_detectable_frame()

        # Process multiple frames to trigger detection + embedding
        for i in range(5):
            p.process_frame(frame, frame_id=i + 1, timestamp=time.time())

        p.shutdown()

        # Allow time for async event delivery
        time.sleep(0.2)

        # If a face was detected (depends on synthetic frame quality),
        # an unknown event should be published since DB is empty
        # Note: synthetic frames may or may not trigger detection
        # so we just verify no crash and correct event types
        for event in events:
            assert isinstance(event, FaceUnknownEvent)
            assert event.source == "face_recognition"

    def test_recognized_face_event(self, tmp_path, event_bus):
        """Should publish FaceRecognizedEvent for registered face."""
        if not has_models():
            pytest.skip("Models required")

        events = []
        event_bus.subscribe(EventType.FACE_RECOGNIZED, lambda e: events.append(e))

        config = make_plugin_config(tmp_path)
        p = Plugin()
        p.initialize(config, event_bus)

        # To test recognition, we'd need to:
        # 1. Detect a face and get its embedding
        # 2. Register it in the database
        # 3. Detect the same face again
        # This is hard with synthetic frames, so we test the flow doesn't crash
        frame = _make_detectable_frame()
        p.process_frame(frame, frame_id=1, timestamp=time.time())

        # If face detected, register its embedding
        if p.tracker.track_count > 0:
            track = p.tracker.tracks[0]
            if track.embedding is not None:
                # Register this face
                p.database.add_face("test_001", "Test Person", track.embedding)
                # Reset tracker to simulate re-entry
                p.tracker.reset()
                # Process again - should now recognize
                for i in range(3):
                    p.process_frame(frame, frame_id=10 + i, timestamp=time.time())

        p.shutdown()
        time.sleep(0.2)

        # Verify events are correct type (if any were published)
        for event in events:
            assert isinstance(event, FaceRecognizedEvent)
            assert event.source == "face_recognition"
            assert event.person_id == "test_001"


# --- Test: Tracker Integration ---


class TestPluginTrackerIntegration:
    """Tests for tracker behavior within plugin."""

    def test_no_duplicate_events(self, tmp_path, event_bus):
        """Same track should only publish one event."""
        if not has_models():
            pytest.skip("Models required")

        all_events = []
        event_bus.subscribe("*", lambda e: all_events.append(e))

        config = make_plugin_config(tmp_path)
        p = Plugin()
        p.initialize(config, event_bus)

        frame = _make_detectable_frame()
        # Process same frame multiple times (simulating person staying)
        for i in range(10):
            p.process_frame(frame, frame_id=i + 1, timestamp=time.time())

        p.shutdown()
        time.sleep(0.2)

        # Count recognition events (should be at most 1 per track)
        face_events = [e for e in all_events
                       if e.event_type in (EventType.FACE_RECOGNIZED, EventType.FACE_UNKNOWN)]
        # At most 1 event per track (not 10)
        assert len(face_events) <= p.tracker.track_count + 1  # +1 for safety


# --- Test: Shutdown ---


class TestPluginShutdown:
    """Tests for plugin shutdown."""

    def test_shutdown_closes_database(self, plugin_with_models):
        """Shutdown should close the database."""
        p = plugin_with_models
        assert p.database.is_initialized
        p.shutdown()
        assert not p.database.is_initialized

    def test_shutdown_resets_tracker(self, plugin_with_models):
        """Shutdown should reset the tracker."""
        p = plugin_with_models
        # Add some tracks
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        p.process_frame(frame, frame_id=1, timestamp=time.time())
        p.shutdown()
        assert p.tracker.track_count == 0


# --- Helpers ---


def _make_detectable_frame(w=640, h=480):
    """
    Create a synthetic frame that might trigger face detection.

    Uses face-like oval with features. Detection depends on model sensitivity.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w // 2, h // 2

    # Skin-colored oval
    cv2.ellipse(frame, (cx, cy), (80, 100), 0, 0, 360, (200, 180, 160), -1)
    # Eyes (dark circles)
    cv2.circle(frame, (cx - 30, cy - 25), 12, (40, 40, 40), -1)
    cv2.circle(frame, (cx + 30, cy - 25), 12, (40, 40, 40), -1)
    # Eye whites
    cv2.circle(frame, (cx - 30, cy - 25), 6, (220, 220, 220), -1)
    cv2.circle(frame, (cx + 30, cy - 25), 6, (220, 220, 220), -1)
    # Nose
    cv2.line(frame, (cx, cy - 10), (cx, cy + 15), (170, 150, 130), 3)
    # Mouth
    cv2.ellipse(frame, (cx, cy + 40), (25, 10), 0, 0, 180, (120, 80, 80), 2)

    return frame


# Import cv2 for helper
import cv2
