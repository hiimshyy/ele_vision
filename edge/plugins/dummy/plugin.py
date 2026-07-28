"""
Smart Cabin - Dummy Plugin

A simple plugin for testing the plugin system.
Counts frames and publishes periodic events via EventBus.
"""

from typing import Any

import numpy as np

from edge.core.plugin_manager import BasePlugin
from edge.core.event_bus import EventBus
from edge.core.logging_setup import get_logger
from shared.event_schemas import BaseEvent, EventType

logger = get_logger("plugin")


class Plugin(BasePlugin):
    """Dummy plugin that counts frames and publishes events."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def default_fps(self) -> float:
        return 5.0

    def initialize(self, config: dict[str, Any], event_bus: EventBus) -> bool:
        """Initialize dummy plugin."""
        self._event_bus = event_bus
        self._frame_count = 0
        self._log_interval = config.get("log_interval", 10)
        logger.info(
            "event=dummy_init | log_interval={interval}",
            interval=self._log_interval,
        )
        return True

    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        """Process frame: count and log periodically."""
        self._frame_count += 1

        if self._frame_count % self._log_interval == 0:
            h, w = frame.shape[:2]
            logger.info(
                "event=dummy_tick | frame_count={count} | frame_id={fid} | resolution={w}x{h}",
                count=self._frame_count, fid=frame_id, w=w, h=h,
            )

            # Publish a system event
            self._event_bus.publish(BaseEvent(
                event_type=EventType.PLUGIN_LOADED,
                source="dummy",
                metadata={"frame_count": self._frame_count},
            ))

    def shutdown(self) -> None:
        """Cleanup."""
        logger.info(
            "event=dummy_shutdown | total_frames={count}",
            count=self._frame_count,
        )
