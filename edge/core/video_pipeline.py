"""
Smart Cabin Platform - Video Pipeline

Captures frames from RTSP stream and distributes them to registered consumers
(plugins) via callback pattern. Features:
- Thread-safe ring buffer to decouple capture from processing
- Configurable FPS throttling (capture at native FPS, distribute at lower FPS)
- Automatic reconnection with backoff on stream failure
- Observer pattern for frame distribution to multiple consumers
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import cv2
import numpy as np

from edge.core.config import CameraConfig

logger = logging.getLogger(__name__)


# --- Types ---

# Frame callback signature: (frame: np.ndarray, frame_id: int, timestamp: float) -> None
FrameCallback = Callable[[np.ndarray, int, float], None]


class PipelineState(str, Enum):
    """Video pipeline lifecycle states."""

    IDLE = "idle"
    CONNECTING = "connecting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class PipelineStats:
    """Runtime statistics for the video pipeline."""

    frames_captured: int = 0
    frames_distributed: int = 0
    frames_dropped: int = 0
    reconnect_count: int = 0
    current_capture_fps: float = 0.0
    current_distribute_fps: float = 0.0
    last_frame_time: float = 0.0
    start_time: float = 0.0


@dataclass
class FrameData:
    """Container for a captured frame with metadata."""

    frame: np.ndarray
    frame_id: int
    timestamp: float


# --- Video Pipeline ---


class VideoPipeline:
    """
    RTSP video capture pipeline with frame distribution.

    Runs a capture thread that reads frames from camera and places them
    into a ring buffer. A distributor thread consumes from the buffer
    at the configured process_fps and invokes registered callbacks.
    """

    def __init__(self, config: CameraConfig):
        self._config = config
        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()

        # Ring buffer (bounded deque)
        self._buffer: deque[FrameData] = deque(maxlen=30)
        self._buffer_lock = threading.Lock()
        self._buffer_event = threading.Event()

        # Frame consumers (observers)
        self._callbacks: list[FrameCallback] = []
        self._callbacks_lock = threading.Lock()

        # Threading
        self._capture_thread: threading.Thread | None = None
        self._distribute_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Stats
        self._stats = PipelineStats()
        self._frame_counter = 0

        # Capture handle
        self._capture: cv2.VideoCapture | None = None

    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""
        with self._state_lock:
            return self._state

    @property
    def stats(self) -> PipelineStats:
        """Current pipeline statistics (snapshot)."""
        return PipelineStats(
            frames_captured=self._stats.frames_captured,
            frames_distributed=self._stats.frames_distributed,
            frames_dropped=self._stats.frames_dropped,
            reconnect_count=self._stats.reconnect_count,
            current_capture_fps=self._stats.current_capture_fps,
            current_distribute_fps=self._stats.current_distribute_fps,
            last_frame_time=self._stats.last_frame_time,
            start_time=self._stats.start_time,
        )

    def register_callback(self, callback: FrameCallback) -> None:
        """Register a frame consumer callback."""
        with self._callbacks_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
                logger.info(f"Registered frame callback: {callback.__name__}")

    def unregister_callback(self, callback: FrameCallback) -> None:
        """Unregister a frame consumer callback."""
        with self._callbacks_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                logger.info(f"Unregistered frame callback: {callback.__name__}")

    def start(self) -> None:
        """Start the video pipeline (capture + distribution threads)."""
        if self.state in (PipelineState.RUNNING, PipelineState.CONNECTING):
            logger.warning("Pipeline already running")
            return

        self._stop_event.clear()
        self._stats = PipelineStats(start_time=time.time())
        self._frame_counter = 0

        self._set_state(PipelineState.CONNECTING)

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="video-capture",
            daemon=True,
        )
        self._distribute_thread = threading.Thread(
            target=self._distribute_loop,
            name="video-distribute",
            daemon=True,
        )

        self._capture_thread.start()
        self._distribute_thread.start()
        logger.info(f"Video pipeline started: {self._config.url}")

    def stop(self) -> None:
        """Stop the video pipeline gracefully."""
        if self.state == PipelineState.STOPPED:
            return

        logger.info("Stopping video pipeline...")
        self._stop_event.set()
        self._buffer_event.set()  # Wake up distributor if waiting

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)

        if self._distribute_thread and self._distribute_thread.is_alive():
            self._distribute_thread.join(timeout=5.0)

        self._release_capture()
        self._set_state(PipelineState.STOPPED)
        uptime = time.time() - self._stats.start_time if self._stats.start_time > 0 else 0
        logger.info(
            f"Video pipeline stopped (uptime: {uptime:.1f}s, "
            f"captured: {self._stats.frames_captured}, "
            f"distributed: {self._stats.frames_distributed}, "
            f"reconnects: {self._stats.reconnect_count})"
        )

    def _set_state(self, new_state: PipelineState) -> None:
        """Thread-safe state transition."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            if old_state != new_state:
                logger.debug(f"Pipeline state: {old_state} -> {new_state}")

    # --- Capture Thread ---

    def _capture_loop(self) -> None:
        """Main capture loop - reads frames from RTSP and places into buffer."""
        while not self._stop_event.is_set():
            # Connect to stream
            if not self._connect():
                # If max attempts reached (ERROR state) or stop requested, exit
                if self._stop_event.is_set() or self.state == PipelineState.ERROR:
                    break
                continue

            self._set_state(PipelineState.RUNNING)
            fps_timer = time.time()
            fps_count = 0

            # Read frames
            while not self._stop_event.is_set():
                ret, frame = self._capture.read()

                if not ret:
                    uptime = time.time() - self._stats.start_time
                    logger.warning(
                        f"Frame read failed, attempting reconnection "
                        f"(uptime: {uptime:.1f}s, frames_captured: {self._stats.frames_captured})"
                    )
                    self._release_capture()
                    self._set_state(PipelineState.RECONNECTING)
                    self._stats.reconnect_count += 1
                    break

                # Update stats
                self._frame_counter += 1
                self._stats.frames_captured += 1
                self._stats.last_frame_time = time.time()
                fps_count += 1

                # Calculate capture FPS every second
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    self._stats.current_capture_fps = fps_count / elapsed
                    fps_count = 0
                    fps_timer = time.time()

                # Put frame into ring buffer
                frame_data = FrameData(
                    frame=frame,
                    frame_id=self._frame_counter,
                    timestamp=time.time(),
                )

                with self._buffer_lock:
                    if len(self._buffer) == self._buffer.maxlen:
                        self._stats.frames_dropped += 1
                    self._buffer.append(frame_data)

                self._buffer_event.set()

    def _connect(self) -> bool:
        """Attempt to connect to the RTSP stream with retry logic."""
        attempts = 0
        max_attempts = self._config.max_reconnect_attempts

        # Support integer camera index (e.g., "0" for webcam)
        source = self._config.url
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass

        while not self._stop_event.is_set():
            attempts += 1
            self._set_state(PipelineState.CONNECTING)
            logger.info(
                f"Connecting to camera: {self._config.url} (attempt {attempts})"
            )

            # Set connection timeout before opening
            timeout_ms = int(self._config.connection_timeout * 1000)
            self._capture = cv2.VideoCapture(
                source,
                cv2.CAP_ANY,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
                ],
            )

            if self._capture.isOpened():
                logger.info("Camera connected successfully")
                return True

            self._release_capture()
            logger.warning(
                f"Connection failed, retrying in {self._config.reconnect_interval}s"
            )

            if max_attempts > 0 and attempts >= max_attempts:
                logger.error(
                    f"Max reconnection attempts ({max_attempts}) reached"
                )
                self._set_state(PipelineState.ERROR)
                return False

            # Wait before retry (interruptible)
            self._stop_event.wait(timeout=self._config.reconnect_interval)

        return False

    def _release_capture(self) -> None:
        """Release OpenCV capture safely."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    # --- Distribution Thread ---

    def _distribute_loop(self) -> None:
        """Distribute frames to registered callbacks at configured process_fps."""
        frame_interval = 1.0 / self._config.process_fps
        last_distribute_time = 0.0
        fps_timer = time.time()
        fps_count = 0

        while not self._stop_event.is_set():
            # Wait for frames to be available
            self._buffer_event.wait(timeout=1.0)

            if self._stop_event.is_set():
                break

            # FPS throttling
            now = time.time()
            elapsed_since_last = now - last_distribute_time
            if elapsed_since_last < frame_interval:
                sleep_time = frame_interval - elapsed_since_last
                time.sleep(sleep_time)

            # Get latest frame from buffer
            frame_data = None
            with self._buffer_lock:
                if self._buffer:
                    frame_data = self._buffer[-1]  # Latest frame
                    self._buffer.clear()  # Drop older frames
                else:
                    self._buffer_event.clear()
                    continue

            if frame_data is None:
                self._buffer_event.clear()
                continue

            last_distribute_time = time.time()

            # Distribute to callbacks
            self._invoke_callbacks(frame_data)

            # Update distribute FPS
            fps_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self._stats.current_distribute_fps = fps_count / elapsed
                fps_count = 0
                fps_timer = time.time()

            self._stats.frames_distributed += 1

    def _invoke_callbacks(self, frame_data: FrameData) -> None:
        """Invoke all registered callbacks with the frame."""
        with self._callbacks_lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback(frame_data.frame, frame_data.frame_id, frame_data.timestamp)
            except Exception as e:
                logger.error(
                    f"Error in frame callback {callback.__name__}: {e}",
                    exc_info=True,
                )


# --- CLI Demo ---

if __name__ == "__main__":
    import sys

    from edge.core.config import load_config

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config(config_path=config_path)

    # Create pipeline
    pipeline = VideoPipeline(config.camera)

    # Register a demo callback
    def demo_callback(frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        if frame_id % 10 == 0:  # Log every 10 frames
            h, w = frame.shape[:2]
            stats = pipeline.stats
            print(
                f"  Frame #{frame_id}: {w}x{h} | "
                f"Capture FPS: {stats.current_capture_fps:.1f} | "
                f"Distribute FPS: {stats.current_distribute_fps:.1f} | "
                f"Dropped: {stats.frames_dropped}"
            )

    pipeline.register_callback(demo_callback)

    # Run
    print("Smart Cabin - Video Pipeline Demo")
    print("=" * 50)
    print(f"RTSP URL: {config.camera.url}")
    print(f"Process FPS: {config.camera.process_fps}")
    print("Press Ctrl+C to stop\n")

    pipeline.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        pipeline.stop()

    stats = pipeline.stats
    print(f"\nFinal stats:")
    print(f"  Frames captured: {stats.frames_captured}")
    print(f"  Frames distributed: {stats.frames_distributed}")
    print(f"  Frames dropped: {stats.frames_dropped}")
    print(f"  Reconnections: {stats.reconnect_count}")
