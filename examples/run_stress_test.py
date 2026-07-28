"""
Smart Cabin - Stress Test

Tests pipeline resilience with simulated plugins:
- SlowPlugin (300ms processing at 2fps)
- CrashPlugin (always throws exception at 5fps)
- Display (normal rendering at 15fps)

Verifies:
- Slow plugin doesn't block others
- Crashing plugin gets auto-disabled after 5 errors
- Display remains smooth
- Logs capture all events for analysis

Usage:
    python examples/run_stress_test.py --url "rtsp://user:pass@IP:554/stream"
    python examples/run_stress_test.py --url 0 --duration 60
"""

import argparse
import sys
import time

import cv2
import numpy as np

from edge.core.config import CameraConfig, load_config
from edge.core.video_pipeline import VideoPipeline, PipelineState
from edge.core.logging_setup import setup_logging, get_logger

logger = get_logger("system")
plugin_logger = get_logger("plugin")


# --- Simulated Plugins ---


def slow_plugin(frame: np.ndarray, frame_id: int, timestamp: float) -> None:
    """Simulates heavy AI processing (300ms per frame)."""
    time.sleep(0.3)
    plugin_logger.info(
        "event=slow_plugin_done | frame_id={fid} | process_ms=300",
        fid=frame_id,
    )


def crashing_plugin(frame: np.ndarray, frame_id: int, timestamp: float) -> None:
    """Plugin that always crashes - tests error isolation."""
    raise RuntimeError(f"Intentional crash at frame {frame_id}")


class DisplayPlugin:
    """Simple display with minimal overlay for stress test monitoring."""

    def __init__(self, pipeline: VideoPipeline, scale: float = 0.5):
        self._pipeline = pipeline
        self._scale = scale
        self._latest_frame: np.ndarray | None = None
        self._frame_id = 0
        self._latency: float = 0.0

    def on_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        self._latest_frame = frame.copy()
        self._frame_id = frame_id
        self._latency = time.time() - timestamp

    def run(self, duration: float = 0) -> None:
        """Run display. If duration > 0, auto-quit after N seconds."""
        cv2.namedWindow("Stress Test", cv2.WINDOW_AUTOSIZE)
        start_time = time.time()

        while True:
            if self._latest_frame is not None:
                frame = self._latest_frame.copy()
                stats = self._pipeline.stats
                h, w = frame.shape[:2]

                # Minimal overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (500, 85), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                elapsed = time.time() - start_time
                cv2.putText(frame, f"STRESS TEST | Elapsed: {elapsed:.0f}s | FPS: {stats.current_distribute_fps:.1f}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.putText(frame, f"Latency: {self._latency*1000:.1f}ms | Reconnects: {stats.reconnect_count}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

                new_w = int(w * self._scale)
                new_h = int(h * self._scale)
                frame = cv2.resize(frame, (new_w, new_h))
                cv2.imshow("Stress Test", frame)
            else:
                waiting = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for stream...",
                            (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("Stress Test", waiting)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break

            if cv2.getWindowProperty("Stress Test", cv2.WND_PROP_VISIBLE) < 1:
                break

            # Auto-quit after duration
            if duration > 0 and (time.time() - start_time) >= duration:
                print(f"\n  Auto-quit after {duration}s")
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Smart Cabin - Stress Test")
    parser.add_argument("--url", type=str, help="RTSP URL or camera index")
    parser.add_argument("--duration", type=float, default=0, help="Auto-quit after N seconds (0=manual)")
    parser.add_argument("--scale", type=float, default=0.5, help="Display scale")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    # Load config
    try:
        config = load_config(config_path=args.config)
    except FileNotFoundError:
        config = None

    # Setup logging
    if config:
        setup_logging(config.logging)
    else:
        setup_logging("INFO")

    # Build camera config
    if args.url is not None:
        camera_config = CameraConfig(
            url=str(args.url),
            capture_fps=config.camera.capture_fps if config else 25,
            process_fps=15,
            reconnect_interval=config.camera.reconnect_interval if config else 5.0,
            max_reconnect_attempts=0,
            connection_timeout=config.camera.connection_timeout if config else 10.0,
        )
    elif config:
        camera_config = config.camera
    else:
        print("ERROR: No RTSP URL provided and no config.yaml found.")
        sys.exit(1)

    # Info
    print("=" * 60)
    print("  Smart Cabin - Stress Test")
    print("=" * 60)
    print(f"  URL: {camera_config.url}")
    print(f"  Duration: {'manual' if args.duration == 0 else f'{args.duration}s'}")
    print(f"  Plugins:")
    print(f"    - Display       @ 15fps")
    print(f"    - SlowPlugin    @ 2fps  (300ms processing)")
    print(f"    - CrashPlugin   @ 5fps  (always throws)")
    print("=" * 60)
    print("  Controls: [q/ESC] Quit")
    print("=" * 60)
    print()

    # Create pipeline
    pipeline = VideoPipeline(camera_config)

    # Register plugins
    display = DisplayPlugin(pipeline, scale=args.scale)
    pipeline.register_callback(display.on_frame, fps=15)
    pipeline.register_callback(slow_plugin, fps=2)
    pipeline.register_callback(crashing_plugin, fps=5)

    # Run
    pipeline.start()
    try:
        display.run(duration=args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        pipeline.stop()

    # Summary
    stats = pipeline.stats
    print(f"\n{'='*60}")
    print("  STRESS TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Uptime: {time.time() - stats.start_time:.1f}s")
    print(f"  Captured: {stats.frames_captured}")
    print(f"  Distributed: {stats.frames_distributed}")
    print(f"  Capture FPS: {stats.current_capture_fps:.1f}")
    print(f"  Distribute FPS: {stats.current_distribute_fps:.1f}")
    print(f"  Reconnects: {stats.reconnect_count}")
    print(f"{'='*60}")
    print(f"  Check logs: cat logs/scheduler.log | grep plugin_stats")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
