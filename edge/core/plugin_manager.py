"""
Smart Cabin Platform - Plugin Manager

Manages plugin lifecycle: discovery, loading, initialization, frame routing, and shutdown.

Architecture:
    PluginManager
        ├── loads plugins from config (edge/plugins/{name}/)
        ├── manages lifecycle: INIT → RUNNING → PAUSED → STOPPED / ERROR
        ├── registers each plugin's process_frame with VideoPipeline scheduler
        └── provides health monitoring + auto-restart on failure

Plugin Contract (BasePlugin):
    - initialize(config, event_bus) → bool
    - process_frame(frame, frame_id, timestamp) → None
    - shutdown() → None
    - name, version, default_fps properties

Usage:
    manager = PluginManager(pipeline, event_bus)
    manager.load_plugins(config.plugins)
    manager.start_all()
    ...
    manager.stop_all()
"""

import importlib
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np

from edge.core.config import PluginsConfig, PluginEntry
from edge.core.event_bus import EventBus
from edge.core.video_pipeline import VideoPipeline
from edge.core.logging_setup import get_logger

logger = get_logger("plugin")


# --- Plugin States ---


class PluginState(str, Enum):
    """Plugin lifecycle states."""

    UNLOADED = "unloaded"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


# --- Base Plugin ---


class BasePlugin(ABC):
    """
    Abstract base class for all Smart Cabin plugins.

    Subclass this and implement:
    - initialize(config, event_bus): setup resources, return True on success
    - process_frame(frame, frame_id, timestamp): handle each frame
    - shutdown(): cleanup resources
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin identifier."""
        ...

    @property
    def version(self) -> str:
        """Plugin version string."""
        return "0.1.0"

    @property
    def default_fps(self) -> float:
        """Default target FPS for this plugin."""
        return 5.0

    @abstractmethod
    def initialize(self, config: dict[str, Any], event_bus: EventBus) -> bool:
        """
        Initialize plugin with config and event bus.

        Args:
            config: Plugin-specific configuration dict from YAML
            event_bus: Shared event bus for publishing events

        Returns:
            True if initialization succeeded, False otherwise
        """
        ...

    @abstractmethod
    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        """
        Process a single frame.

        Called by the scheduler at the plugin's target FPS.
        Must be thread-safe (called from thread pool).

        Args:
            frame: BGR image as numpy array
            frame_id: Monotonically increasing frame counter
            timestamp: Time when frame was captured
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanup resources (models, connections, files)."""
        ...


# --- Plugin Wrapper ---


class PluginWrapper:
    """Wraps a BasePlugin instance with lifecycle state and metadata."""

    def __init__(self, plugin: BasePlugin, config: dict[str, Any]):
        self.plugin = plugin
        self.config = config
        self.state = PluginState.UNLOADED
        self.error_message: str = ""
        self.started_at: float = 0.0

    @property
    def name(self) -> str:
        return self.plugin.name

    @property
    def fps(self) -> float:
        """Target FPS from config or plugin default."""
        return self.config.get("fps", self.plugin.default_fps)

    @property
    def uptime(self) -> float:
        """Seconds since plugin started running."""
        if self.started_at > 0 and self.state == PluginState.RUNNING:
            return time.time() - self.started_at
        return 0.0


# --- Plugin Manager ---


class PluginManager:
    """
    Manages plugin lifecycle and integrates with VideoPipeline + EventBus.

    Responsibilities:
    - Load plugins from config (import module, instantiate class)
    - Initialize plugins with their config + event bus
    - Register process_frame with pipeline scheduler at correct FPS
    - Track plugin state and provide health info
    - Clean shutdown of all plugins
    """

    def __init__(self, pipeline: VideoPipeline, event_bus: EventBus):
        self._pipeline = pipeline
        self._event_bus = event_bus
        self._plugins: dict[str, PluginWrapper] = {}

    @property
    def plugins(self) -> dict[str, PluginWrapper]:
        """All loaded plugins (name → PluginWrapper)."""
        return dict(self._plugins)

    def load_plugins(self, config: PluginsConfig) -> None:
        """
        Load and initialize plugins from config.

        Args:
            config: PluginsConfig with modules list
        """
        for entry in config.modules:
            if not entry.enabled:
                logger.info(
                    "event=plugin_skipped | name={name} | reason=disabled",
                    name=entry.name,
                )
                continue

            self._load_plugin(entry)

    def _load_plugin(self, entry: PluginEntry) -> None:
        """Load a single plugin from its module path."""
        try:
            # Import the plugin module: edge.plugins.{name}.plugin
            module_path = f"edge.plugins.{entry.name}.plugin"
            module = importlib.import_module(module_path)

            # Expect a `Plugin` class in the module
            if not hasattr(module, "Plugin"):
                logger.error(
                    "event=plugin_load_error | name={name} | error=no Plugin class found in {path}",
                    name=entry.name, path=module_path,
                )
                return

            plugin_class = module.Plugin
            if not issubclass(plugin_class, BasePlugin):
                logger.error(
                    "event=plugin_load_error | name={name} | error=Plugin class does not inherit BasePlugin",
                    name=entry.name,
                )
                return

            # Instantiate
            plugin_instance = plugin_class()
            wrapper = PluginWrapper(plugin_instance, entry.config)

            # Initialize
            success = plugin_instance.initialize(entry.config, self._event_bus)
            if success:
                wrapper.state = PluginState.INITIALIZED
                self._plugins[entry.name] = wrapper
                logger.info(
                    "event=plugin_loaded | name={name} | version={ver} | fps={fps}",
                    name=entry.name, ver=plugin_instance.version, fps=wrapper.fps,
                )
            else:
                wrapper.state = PluginState.ERROR
                wrapper.error_message = "initialize() returned False"
                self._plugins[entry.name] = wrapper
                logger.error(
                    "event=plugin_init_failed | name={name} | error=initialize returned False",
                    name=entry.name,
                )

        except ImportError as e:
            logger.error(
                "event=plugin_load_error | name={name} | error=import failed: {err}",
                name=entry.name, err=str(e),
            )
        except Exception as e:
            logger.error(
                "event=plugin_load_error | name={name} | error={err}",
                name=entry.name, err=str(e),
            )

    def start_all(self) -> None:
        """Start all initialized plugins (register with pipeline scheduler)."""
        for name, wrapper in self._plugins.items():
            if wrapper.state == PluginState.INITIALIZED:
                self._start_plugin(wrapper)

    def _start_plugin(self, wrapper: PluginWrapper) -> None:
        """Register plugin's process_frame with pipeline scheduler."""
        self._pipeline.register_callback(
            wrapper.plugin.process_frame,
            fps=wrapper.fps,
        )
        wrapper.state = PluginState.RUNNING
        wrapper.started_at = time.time()
        logger.info(
            "event=plugin_started | name={name} | fps={fps}",
            name=wrapper.name, fps=wrapper.fps,
        )

    def stop_plugin(self, name: str) -> None:
        """Stop a specific plugin."""
        wrapper = self._plugins.get(name)
        if wrapper is None:
            return

        if wrapper.state == PluginState.RUNNING:
            self._pipeline.unregister_callback(wrapper.plugin.process_frame)
            try:
                wrapper.plugin.shutdown()
            except Exception as e:
                logger.error(
                    "event=plugin_shutdown_error | name={name} | error={err}",
                    name=name, err=str(e),
                )
            wrapper.state = PluginState.STOPPED
            logger.info("event=plugin_stopped | name={name}", name=name)

    def restart_plugin(self, name: str) -> None:
        """Restart a plugin (stop → re-initialize → start)."""
        wrapper = self._plugins.get(name)
        if wrapper is None:
            logger.warning("event=plugin_restart_failed | name={name} | reason=not_found", name=name)
            return

        # Stop if running
        if wrapper.state == PluginState.RUNNING:
            self._pipeline.unregister_callback(wrapper.plugin.process_frame)
            try:
                wrapper.plugin.shutdown()
            except Exception:
                pass

        # Re-initialize
        try:
            success = wrapper.plugin.initialize(wrapper.config, self._event_bus)
            if success:
                wrapper.state = PluginState.INITIALIZED
                self._start_plugin(wrapper)
                logger.info("event=plugin_restarted | name={name}", name=name)
            else:
                wrapper.state = PluginState.ERROR
                wrapper.error_message = "reinitialize failed"
                logger.error("event=plugin_restart_failed | name={name} | reason=init_failed", name=name)
        except Exception as e:
            wrapper.state = PluginState.ERROR
            wrapper.error_message = str(e)
            logger.error(
                "event=plugin_restart_failed | name={name} | error={err}",
                name=name, err=str(e),
            )

    def stop_all(self) -> None:
        """Stop and shutdown all plugins."""
        for name in list(self._plugins.keys()):
            self.stop_plugin(name)

    def get_status(self) -> list[dict[str, Any]]:
        """Get status of all plugins for API/monitoring."""
        result = []
        for name, wrapper in self._plugins.items():
            result.append({
                "name": name,
                "version": wrapper.plugin.version,
                "state": wrapper.state.value,
                "fps": wrapper.fps,
                "uptime": wrapper.uptime,
                "error": wrapper.error_message,
            })
        return result
