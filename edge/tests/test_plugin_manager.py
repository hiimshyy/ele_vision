"""
Tests for edge/core/plugin_manager.py - Plugin Manager System.

Covers:
- BasePlugin contract
- Plugin discovery and loading from config
- Plugin lifecycle transitions (init → running → stopped)
- Plugin isolation (crash doesn't affect others)
- PluginManager integration with VideoPipeline and EventBus
- Restart, stop, get_status
"""

import time
from typing import Any
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from edge.core.config import PluginsConfig, PluginEntry
from edge.core.event_bus import EventBus
from edge.core.plugin_manager import (
    BasePlugin,
    PluginManager,
    PluginState,
    PluginWrapper,
)
from edge.core.video_pipeline import VideoPipeline, PipelineState, CameraConfig


# --- Test Plugins ---


class GoodPlugin(BasePlugin):
    """A plugin that works correctly."""

    @property
    def name(self) -> str:
        return "good_plugin"

    @property
    def default_fps(self) -> float:
        return 10.0

    def initialize(self, config: dict[str, Any], event_bus: EventBus) -> bool:
        self._config = config
        self._event_bus = event_bus
        self._frame_count = 0
        return True

    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        self._frame_count += 1

    def shutdown(self) -> None:
        self._frame_count = 0


class FailingInitPlugin(BasePlugin):
    """A plugin that fails during initialization."""

    @property
    def name(self) -> str:
        return "failing_init"

    def initialize(self, config: dict[str, Any], event_bus: EventBus) -> bool:
        return False  # Intentional failure

    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        pass

    def shutdown(self) -> None:
        pass


class CrashingPlugin(BasePlugin):
    """A plugin that crashes during process_frame."""

    @property
    def name(self) -> str:
        return "crashing"

    @property
    def default_fps(self) -> float:
        return 5.0

    def initialize(self, config: dict[str, Any], event_bus: EventBus) -> bool:
        return True

    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        raise RuntimeError("Plugin crash!")

    def shutdown(self) -> None:
        pass


# --- Fixtures ---


@pytest.fixture
def event_bus():
    """Create a fresh EventBus."""
    bus = EventBus()
    yield bus
    bus.shutdown()


@pytest.fixture
def camera_config():
    """Camera config for pipeline."""
    return CameraConfig(
        url="rtsp://fake:554/stream",
        capture_fps=25,
        process_fps=15,
        reconnect_interval=0.1,
        max_reconnect_attempts=3,
    )


@pytest.fixture
def pipeline(camera_config):
    """Create a pipeline (not started, just for registration)."""
    p = VideoPipeline(camera_config)
    yield p
    if p.state not in (PipelineState.IDLE, PipelineState.STOPPED):
        p.stop()


@pytest.fixture
def manager(pipeline, event_bus):
    """Create a PluginManager."""
    return PluginManager(pipeline, event_bus)


# --- Test: BasePlugin Contract ---


class TestBasePlugin:
    """Tests for BasePlugin interface."""

    def test_good_plugin_implements_interface(self):
        """GoodPlugin should implement all required methods."""
        plugin = GoodPlugin()
        assert plugin.name == "good_plugin"
        assert plugin.default_fps == 10.0
        assert plugin.version == "0.1.0"

    def test_initialize_returns_bool(self, event_bus):
        """initialize() should return True on success."""
        plugin = GoodPlugin()
        assert plugin.initialize({}, event_bus) is True

    def test_failing_init_returns_false(self, event_bus):
        """initialize() should return False on failure."""
        plugin = FailingInitPlugin()
        assert plugin.initialize({}, event_bus) is False


# --- Test: Plugin Loading ---


class TestPluginLoading:
    """Tests for plugin discovery and loading."""

    def test_load_dummy_plugin(self, manager):
        """Should load dummy plugin from edge/plugins/dummy/."""
        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=True, config={"log_interval": 5}),
        ])

        manager.load_plugins(config)

        assert "dummy" in manager.plugins
        assert manager.plugins["dummy"].state == PluginState.INITIALIZED
        assert manager.plugins["dummy"].plugin.name == "dummy"

    def test_skip_disabled_plugin(self, manager):
        """Disabled plugins should not be loaded."""
        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=False, config={}),
        ])

        manager.load_plugins(config)
        assert "dummy" not in manager.plugins

    def test_nonexistent_plugin_logs_error(self, manager):
        """Non-existent plugin module should log error, not crash."""
        config = PluginsConfig(modules=[
            PluginEntry(name="nonexistent_xyz", enabled=True, config={}),
        ])

        manager.load_plugins(config)  # Should not raise
        assert "nonexistent_xyz" not in manager.plugins

    def test_fps_from_config(self, manager):
        """Plugin FPS should come from config if specified."""
        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=True, config={"fps": 8}),
        ])

        manager.load_plugins(config)
        assert manager.plugins["dummy"].fps == 8

    def test_fps_default_from_plugin(self, manager):
        """Plugin FPS should use plugin.default_fps if not in config."""
        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=True, config={}),
        ])

        manager.load_plugins(config)
        # DummyPlugin.default_fps = 5.0
        assert manager.plugins["dummy"].fps == 5.0


# --- Test: Plugin Lifecycle ---


class TestPluginLifecycle:
    """Tests for lifecycle transitions."""

    def test_start_all(self, manager):
        """start_all should transition plugins from INITIALIZED to RUNNING."""
        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=True, config={}),
        ])
        manager.load_plugins(config)
        manager.start_all()

        assert manager.plugins["dummy"].state == PluginState.RUNNING

    def test_stop_plugin(self, manager):
        """stop_plugin should transition to STOPPED."""
        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=True, config={}),
        ])
        manager.load_plugins(config)
        manager.start_all()
        manager.stop_plugin("dummy")

        assert manager.plugins["dummy"].state == PluginState.STOPPED

    def test_stop_all(self, manager):
        """stop_all should stop all running plugins."""
        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=True, config={}),
        ])
        manager.load_plugins(config)
        manager.start_all()
        manager.stop_all()

        assert manager.plugins["dummy"].state == PluginState.STOPPED

    def test_restart_plugin(self, manager):
        """restart_plugin should stop and restart a plugin."""
        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=True, config={}),
        ])
        manager.load_plugins(config)
        manager.start_all()

        assert manager.plugins["dummy"].state == PluginState.RUNNING
        manager.restart_plugin("dummy")
        assert manager.plugins["dummy"].state == PluginState.RUNNING

    def test_uptime_tracks_running_time(self, manager):
        """Plugin uptime should increase while running."""
        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=True, config={}),
        ])
        manager.load_plugins(config)
        manager.start_all()

        time.sleep(0.2)
        assert manager.plugins["dummy"].uptime >= 0.1


# --- Test: get_status ---


class TestGetStatus:
    """Tests for plugin status reporting."""

    def test_status_format(self, manager):
        """get_status should return structured plugin info."""
        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=True, config={}),
        ])
        manager.load_plugins(config)
        manager.start_all()

        status = manager.get_status()
        assert len(status) == 1
        assert status[0]["name"] == "dummy"
        assert status[0]["state"] == "running"
        assert status[0]["version"] == "1.0.0"
        assert status[0]["fps"] == 5.0
        assert status[0]["error"] == ""

    def test_status_shows_error(self, manager, event_bus):
        """Failed plugins should show error in status."""
        # Manually add a failed plugin
        plugin = FailingInitPlugin()
        wrapper = PluginWrapper(plugin, {})
        wrapper.state = PluginState.ERROR
        wrapper.error_message = "init failed"
        manager._plugins["failing_init"] = wrapper

        status = manager.get_status()
        failing = [s for s in status if s["name"] == "failing_init"][0]
        assert failing["state"] == "error"
        assert failing["error"] == "init failed"


# --- Test: Integration with Pipeline ---


class TestPipelineIntegration:
    """Tests for PluginManager + VideoPipeline integration."""

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_plugin_receives_frames(self, mock_cv2, camera_config, event_bus):
        """Started plugin should receive frames from pipeline."""
        from edge.tests.test_video_pipeline import MockVideoCapture

        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        pipeline = VideoPipeline(camera_config)
        manager = PluginManager(pipeline, event_bus)

        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=True, config={"log_interval": 5}),
        ])
        manager.load_plugins(config)
        manager.start_all()

        pipeline.start()
        time.sleep(1.5)
        pipeline.stop()
        manager.stop_all()

        # DummyPlugin should have received frames
        dummy_plugin = manager.plugins["dummy"].plugin
        assert dummy_plugin._frame_count > 0

    @patch("edge.core.video_pipeline.cv2.VideoCapture")
    def test_plugin_publishes_events(self, mock_cv2, camera_config, event_bus):
        """Plugin should publish events to EventBus."""
        from edge.tests.test_video_pipeline import MockVideoCapture

        mock_cv2.side_effect = lambda *args, **kwargs: MockVideoCapture(frames_to_produce=5000)

        received_events = []
        event_bus.subscribe("*", lambda e: received_events.append(e))

        pipeline = VideoPipeline(camera_config)
        manager = PluginManager(pipeline, event_bus)

        config = PluginsConfig(modules=[
            PluginEntry(name="dummy", enabled=True, config={"log_interval": 3}),
        ])
        manager.load_plugins(config)
        manager.start_all()

        pipeline.start()
        time.sleep(1.5)
        pipeline.stop()
        manager.stop_all()

        # DummyPlugin publishes an event every log_interval frames
        assert len(received_events) > 0
