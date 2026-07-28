"""
Smart Cabin Platform - Video Pipeline

Architecture:
    Camera (RTSP) → Capture Thread → Latest Frame → Frame Scheduler → Per-plugin callbacks

Features:
- Single latest frame buffer (no ring buffer, minimal RAM)
- Frame Scheduler with per-callback FPS control
- Automatic reconnection with configurable timeout
- Periodic stats logging (FPS, CPU, RAM, decode time, latency)
"""

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import cv2
import numpy as np
import psutil

from edge.core.config import CameraConfig
from edge.core.logging_setup import get_logger

logger = get_logger("camera")
sched_logger = get_logger("scheduler")


# --- Helpers ---


def _mask_url(url: str) -> str:
    """Mask credentials in RTSP URL for safe logging."""
    return re.sub(r"(://[^:]+:)[^@]+(@)", r"\1***\2", url)


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
    # Video metrics
    resolution: tuple[int, int] = (0, 0)  # (width, height)
    decode_time_ms: float = 0.0  # Time to decode one frame
    buffer_latency_ms: float = 0.0  # Latest frame age when scheduled
    queue_length: int = 0  # Always 0 or 1 (latest frame buffer)


@dataclass
class FrameData:
    """Container for a captured frame with metadata."""

    frame: np.ndarray
    frame_id: int
    timestamp: float


@dataclass
class ScheduledCallback:
    """A registered callback with its own FPS target."""

    callback: FrameCallback
    target_fps: float
    interval: float  # 1.0 / target_fps
    last_invoked: float = 0.0  # Timestamp of last invocation
    # Error tracking
    consecutive_errors: int = 0
    total_errors: int = 0
    max_errors: int = 5  # Disable after N consecutive errors
    disabled: bool = False
    # Performance tracking
    invocation_count: int = 0
    total_process_time: float = 0.0  # Total processing time (seconds)
    missed_deadlines: int = 0  # Times processing took longer than interval


# --- Video Pipeline ---


class VideoPipeline:
    """
    RTSP video capture pipeline with per-callback frame scheduling.

    Architecture:
    - Capture thread: reads frames, stores latest frame (atomic swap)
    - Scheduler thread: checks timing per callback, invokes at target FPS
    - Stats thread: logs periodic metrics every 30s
    """

    def __init__(self, config: CameraConfig):
        self._config = config
        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()

        # Latest frame (single buffer, atomic swap)
        self._latest_frame: FrameData | None = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()

        # Scheduled callbacks (observers with per-callback FPS)
        self._callbacks: list[ScheduledCallback] = []
        self._callbacks_lock = threading.Lock()

        # Threading
        self._capture_thread: threading.Thread | None = None
        self._scheduler_thread: threading.Thread | None = None
        self._stats_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="plugin")

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
            resolution=self._stats.resolution,
            decode_time_ms=self._stats.decode_time_ms,
            buffer_latency_ms=self._stats.buffer_latency_ms,
            queue_length=self._stats.queue_length,
        )

    def register_callback(self, callback: FrameCallback, fps: float = 15.0) -> None:
        """
        Register a frame consumer callback with target FPS.

        Args:
            callback: Function to receive frames
            fps: Target frames per second for this callback (default 15)
        """
        with self._callbacks_lock:
            # Check if already registered
            for sc in self._callbacks:
                if sc.callback is callback:
                    return

            scheduled = ScheduledCallback(
                callback=callback,
                target_fps=fps,
                interval=1.0 / fps,
                last_invoked=0.0,
            )
            self._callbacks.append(scheduled)
            sched_logger.info(
                "event=callback_registered | name={name} | target_fps={fps}",
                name=callback.__name__, fps=fps,
            )

    def unregister_callback(self, callback: FrameCallback) -> None:
        """Unregister a frame consumer callback."""
        with self._callbacks_lock:
            self._callbacks = [
                sc for sc in self._callbacks if sc.callback is not callback
            ]
            sched_logger.info(
                "event=callback_unregistered | name={name}",
                name=callback.__name__,
            )

    def start(self) -> None:
        """Start the video pipeline (capture + scheduler + stats threads)."""
        if self.state in (PipelineState.RUNNING, PipelineState.CONNECTING):
            logger.warning("event=start_ignored | reason=already_running")
            return

        self._stop_event.clear()
        self._stats = PipelineStats(start_time=time.time())
        self._frame_counter = 0
        self._latest_frame = None

        self._set_state(PipelineState.CONNECTING)

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="video-capture",
            daemon=True,
        )
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="video-scheduler",
            daemon=True,
        )
        self._stats_thread = threading.Thread(
            target=self._stats_loop,
            name="video-stats",
            daemon=True,
        )

        self._capture_thread.start()
        self._scheduler_thread.start()
        self._stats_thread.start()
        logger.info(
            "event=pipeline_started | url={url} | capture_fps={cfps}",
            url=_mask_url(self._config.url),
            cfps=self._config.capture_fps,
        )

    def stop(self) -> None:
        """Stop the video pipeline gracefully."""
        if self.state == PipelineState.STOPPED:
            return

        logger.info("event=pipeline_stopping")
        self._stop_event.set()
        self._frame_event.set()  # Wake up scheduler if waiting

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)

        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)

        self._release_capture()
        self._executor.shutdown(wait=False)
        self._set_state(PipelineState.STOPPED)
        uptime = time.time() - self._stats.start_time if self._stats.start_time > 0 else 0
        logger.info(
            "event=pipeline_stopped | uptime_s={uptime:.1f} | captured={captured} "
            "| distributed={distributed} | dropped={dropped} | reconnects={reconnects}",
            uptime=uptime,
            captured=self._stats.frames_captured,
            distributed=self._stats.frames_distributed,
            dropped=self._stats.frames_dropped,
            reconnects=self._stats.reconnect_count,
        )

    def _set_state(self, new_state: PipelineState) -> None:
        """Thread-safe state transition."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            if old_state != new_state:
                logger.debug(
                    "event=state_change | from={old} | to={new}",
                    old=old_state.value, new=new_state.value,
                )

    # --- Capture Thread ---

    def _capture_loop(self) -> None:
        """Capture loop: decode frames, store latest frame only."""
        while not self._stop_event.is_set():
            # Connect to stream
            if not self._connect():
                if self._stop_event.is_set() or self.state == PipelineState.ERROR:
                    break
                continue

            self._set_state(PipelineState.RUNNING)
            fps_timer = time.time()
            fps_count = 0

            # Read frames
            while not self._stop_event.is_set():
                t_start = time.time()
                ret, frame = self._capture.read()
                decode_time = (time.time() - t_start) * 1000  # ms

                if not ret:
                    uptime = time.time() - self._stats.start_time
                    logger.warning(
                        "event=stream_lost | uptime_s={uptime:.1f} | frames_captured={fc}",
                        uptime=uptime, fc=self._stats.frames_captured,
                    )
                    self._release_capture()
                    self._set_state(PipelineState.RECONNECTING)
                    self._stats.reconnect_count += 1
                    break

                # Update stats
                self._frame_counter += 1
                self._stats.frames_captured += 1
                self._stats.last_frame_time = time.time()
                self._stats.decode_time_ms = decode_time
                fps_count += 1

                # Track resolution (log on change)
                h, w = frame.shape[:2]
                if self._stats.resolution != (w, h):
                    self._stats.resolution = (w, h)
                    logger.info(
                        "event=resolution_detected | width={w} | height={h}",
                        w=w, h=h,
                    )

                # Calculate capture FPS every second
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    self._stats.current_capture_fps = fps_count / elapsed
                    fps_count = 0
                    fps_timer = time.time()

                # Store latest frame (atomic swap)
                frame_data = FrameData(
                    frame=frame,
                    frame_id=self._frame_counter,
                    timestamp=time.time(),
                )

                with self._frame_lock:
                    self._latest_frame = frame_data

                self._frame_event.set()

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
                "event=connecting | url={url} | attempt={attempt}",
                url=_mask_url(self._config.url), attempt=attempts,
            )

            # Set connection timeout
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
                logger.info("event=connected")
                return True

            self._release_capture()
            logger.warning(
                "event=connection_failed | retry_in_s={interval}",
                interval=self._config.reconnect_interval,
            )

            if max_attempts > 0 and attempts >= max_attempts:
                logger.error(
                    "event=max_reconnects_reached | max_attempts={max}",
                    max=max_attempts,
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

    # --- Scheduler Thread ---

    def _scheduler_loop(self) -> None:
        """
        Frame scheduler: invoke each callback at its own target FPS.

        Ticks at the fastest registered FPS rate, checks each callback's
        timing independently.
        """
        fps_timer = time.time()
        fps_count = 0

        while not self._stop_event.is_set():
            # Wait until first frame arrives
            if self._latest_frame is None:
                self._frame_event.wait(timeout=1.0)
                self._frame_event.clear()
                if self._stop_event.is_set():
                    break
                continue

            # Get latest frame
            with self._frame_lock:
                frame_data = self._latest_frame

            if frame_data is None:
                time.sleep(0.01)
                continue

            # Track buffer latency
            now = time.time()
            self._stats.buffer_latency_ms = (now - frame_data.timestamp) * 1000
            self._stats.queue_length = 1 if self._latest_frame else 0

            # Check each callback's timing
            invoked_any = False
            with self._callbacks_lock:
                callbacks = list(self._callbacks)

            for sc in callbacks:
                if sc.disabled:
                    continue
                if now - sc.last_invoked >= sc.interval:
                    sc.last_invoked = now
                    self._executor.submit(self._invoke_callback, sc, frame_data)
                    invoked_any = True

            if invoked_any:
                self._stats.frames_distributed += 1
                fps_count += 1

            # Calculate distribute FPS every second
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self._stats.current_distribute_fps = fps_count / elapsed
                fps_count = 0
                fps_timer = time.time()

            # Sleep until next tick (shortest interval among callbacks)
            # Recalculate with fresh time after processing
            now_after = time.time()
            next_due = self._get_min_interval()
            for sc in callbacks:
                time_since_last = now_after - sc.last_invoked
                remaining = sc.interval - time_since_last
                if remaining > 0:
                    next_due = min(next_due, remaining)

            if next_due > 0.001:
                time.sleep(next_due)

    def _get_min_interval(self) -> float:
        """Get the smallest interval among registered callbacks."""
        with self._callbacks_lock:
            if not self._callbacks:
                return 0.1  # Default 10fps tick when no callbacks
            return min(sc.interval for sc in self._callbacks)

    def _invoke_callback(self, sc: ScheduledCallback, frame_data: FrameData) -> None:
        """Invoke a single callback in a thread pool worker (non-blocking)."""
        t_start = time.time()
        try:
            sc.callback(frame_data.frame, frame_data.frame_id, frame_data.timestamp)
            # Reset error count on success
            if sc.consecutive_errors > 0:
                sc.consecutive_errors = 0
            # Track performance
            process_time = time.time() - t_start
            sc.invocation_count += 1
            sc.total_process_time += process_time
            if process_time > sc.interval:
                sc.missed_deadlines += 1
        except Exception as e:
            sc.consecutive_errors += 1
            sc.total_errors += 1
            if sc.consecutive_errors >= sc.max_errors:
                sc.disabled = True
                sched_logger.warning(
                    "event=callback_disabled | name={name} | reason=consecutive_errors "
                    "| error_count={count} | last_error={err}",
                    name=sc.callback.__name__,
                    count=sc.consecutive_errors,
                    err=str(e),
                )
            else:
                sched_logger.error(
                    "event=callback_error | name={name} | error={err} "
                    "| consecutive={count}/{max}",
                    name=sc.callback.__name__,
                    err=str(e),
                    count=sc.consecutive_errors,
                    max=sc.max_errors,
                )

    # --- Stats Thread ---

    def _stats_loop(self) -> None:
        """Log periodic stats (FPS, CPU, RAM, video metrics) every 30 seconds."""
        stats_interval = 30

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=stats_interval)

            if self._stop_event.is_set():
                break

            if self.state != PipelineState.RUNNING:
                continue

            # Gather system metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            uptime = time.time() - self._stats.start_time

            logger.info(
                "event=periodic_stats | uptime_s={uptime:.0f} "
                "| capture_fps={cfps:.1f} | distribute_fps={dfps:.1f} "
                "| resolution={res} | decode_ms={decode:.1f} "
                "| buffer_latency_ms={lat:.1f} "
                "| reconnects={reconnects} "
                "| cpu_percent={cpu:.1f} | ram_used_mb={ram_used:.0f} | ram_percent={ram_pct:.1f}",
                uptime=uptime,
                cfps=self._stats.current_capture_fps,
                dfps=self._stats.current_distribute_fps,
                res=f"{self._stats.resolution[0]}x{self._stats.resolution[1]}",
                decode=self._stats.decode_time_ms,
                lat=self._stats.buffer_latency_ms,
                reconnects=self._stats.reconnect_count,
                cpu=cpu_percent,
                ram_used=memory.used / (1024 * 1024),
                ram_pct=memory.percent,
            )

            # Per-plugin stats
            with self._callbacks_lock:
                callbacks = list(self._callbacks)

            for sc in callbacks:
                avg_ms = (
                    (sc.total_process_time / sc.invocation_count * 1000)
                    if sc.invocation_count > 0
                    else 0.0
                )
                actual_fps = sc.invocation_count / uptime if uptime > 0 else 0.0

                sched_logger.info(
                    "event=plugin_stats | plugin={name} | target_fps={tfps} "
                    "| actual_fps={afps:.1f} | avg_process_ms={avg:.1f} "
                    "| missed_deadlines={missed} | errors={errors} | disabled={disabled}",
                    name=sc.callback.__name__,
                    tfps=sc.target_fps,
                    afps=actual_fps,
                    avg=avg_ms,
                    missed=sc.missed_deadlines,
                    errors=sc.total_errors,
                    disabled=sc.disabled,
                )


# --- CLI Demo ---

if __name__ == "__main__":
    import sys

    from edge.core.config import load_config
    from edge.core.logging_setup import setup_logging

    setup_logging("DEBUG")

    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config(config_path=config_path)

    # Create pipeline
    pipeline = VideoPipeline(config.camera)

    # Register a demo callback
    def demo_callback(frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        if frame_id % 50 == 0:
            h, w = frame.shape[:2]
            stats = pipeline.stats
            print(
                f"  Frame #{frame_id}: {w}x{h} | "
                f"Capture FPS: {stats.current_capture_fps:.1f} | "
                f"Distribute FPS: {stats.current_distribute_fps:.1f} | "
                f"Decode: {stats.decode_time_ms:.1f}ms | "
                f"Latency: {stats.buffer_latency_ms:.1f}ms"
            )

    pipeline.register_callback(demo_callback, fps=5)

    # Run
    print("Smart Cabin - Video Pipeline Demo")
    print("=" * 50)
    print(f"RTSP URL: {_mask_url(config.camera.url)}")
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
    print(f"  Reconnections: {stats.reconnect_count}")
