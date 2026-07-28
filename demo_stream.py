"""
Smart Cabin - RTSP Stream Demo

Demonstrates the Video Pipeline by connecting to a real RTSP camera
and displaying the live video feed with stats overlay.

Usage:
    uv run python edge/demo_stream.py                    # Use config.yaml
    uv run python edge/demo_stream.py --url rtsp://IP:554/stream  # Override URL
    uv run python edge/demo_stream.py --url 0            # Use local webcam (index 0)

Controls:
    q / ESC  - Quit
    s        - Print stats
"""

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from edge.core.config import CameraConfig, load_config
from edge.core.video_pipeline import VideoPipeline, PipelineState
from edge.core.logging_setup import setup_logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo_stream")


class VideoDisplay:
    """Displays frames in an OpenCV window with stats overlay."""

    def __init__(self, pipeline: VideoPipeline, scale: float = 0.5, window_name: str = "Smart Cabin - RTSP Stream"):
        self._pipeline = pipeline
        self._window_name = window_name
        self._scale = scale
        self._latest_frame: np.ndarray | None = None
        self._frame_id = 0
        self._frame_timestamp: float = 0.0
        self._latency: float = 0.0

    def on_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        """Callback: receives frames from pipeline."""
        self._latest_frame = frame.copy()
        self._frame_id = frame_id
        self._frame_timestamp = timestamp
        self._latency = time.time() - timestamp

    def run(self) -> None:
        """Main display loop."""
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)

        logger.info(f"Display window opened: {self._window_name} (scale={self._scale})")
        logger.info("Press 'q' or ESC to quit, 's' for stats")

        while True:
            # Display frame if available
            if self._latest_frame is not None:
                display_frame = self._draw_overlay(self._latest_frame.copy())
                # Scale frame maintaining aspect ratio
                h, w = display_frame.shape[:2]
                new_w = int(w * self._scale)
                new_h = int(h * self._scale)
                display_frame = cv2.resize(display_frame, (new_w, new_h))
                cv2.imshow(self._window_name, display_frame)
            else:
                # Show waiting screen
                waiting = np.zeros((360, 640, 3), dtype=np.uint8)
                state = self._pipeline.state.value
                cv2.putText(
                    waiting,
                    f"Waiting for stream... ({state})",
                    (120, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(self._window_name, waiting)

            # Handle key input (wait 30ms)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord("s"):
                self._print_stats()

            # Check if window was closed
            if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw stats overlay on frame."""
        stats = self._pipeline.stats
        state = self._pipeline.state.value
        h, w = frame.shape[:2]

        # Format uptime
        uptime_sec = time.time() - self._pipeline.stats.start_time if self._pipeline.stats.start_time > 0 else 0
        hours, remainder = divmod(int(uptime_sec), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 160), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Draw stats
        y = 35
        color = (0, 255, 0) if state == "running" else (0, 255, 255)
        texts = [
            f"FPS: {stats.current_distribute_fps:.1f} (capture: {stats.current_capture_fps:.1f})",
            f"Resolution: {w}x{h}",
            f"Uptime: {uptime_str}",
            f"Latency: {self._latency*1000:.1f} ms",
            f"Frame: #{self._frame_id} | Dropped: {stats.frames_dropped} | Reconnects: {stats.reconnect_count}",
        ]

        for text in texts:
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y += 25

        return frame

    def _print_stats(self) -> None:
        """Print detailed stats to console."""
        stats = self._pipeline.stats
        elapsed = time.time() - stats.start_time if stats.start_time > 0 else 0
        print(f"\n{'='*50}")
        print(f"  Pipeline State: {self._pipeline.state.value}")
        print(f"  Uptime: {elapsed:.1f}s")
        print(f"  Frames Captured: {stats.frames_captured}")
        print(f"  Frames Distributed: {stats.frames_distributed}")
        print(f"  Frames Dropped: {stats.frames_dropped}")
        print(f"  Capture FPS: {stats.current_capture_fps:.1f}")
        print(f"  Distribute FPS: {stats.current_distribute_fps:.1f}")
        print(f"  Reconnections: {stats.reconnect_count}")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Smart Cabin - RTSP Stream Demo")
    parser.add_argument("--url", type=str, help="RTSP URL or camera index (e.g., 0 for webcam)")
    parser.add_argument("--fps", type=int, default=None, help="Process FPS (default: from config)")
    parser.add_argument("--scale", type=float, default=0.5, help="Display scale factor (default: 0.5)")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    # Load config
    try:
        config = load_config(config_path=args.config)
    except FileNotFoundError:
        logger.warning("No config.yaml found, using defaults")
        config = None

    # Setup file logging
    if config:
        setup_logging(config.logging)

    # Build camera config
    if args.url is not None:
        # Check if it's a webcam index
        try:
            cam_index = int(args.url)
            url = cam_index  # OpenCV accepts int for local cameras
        except ValueError:
            url = args.url

        camera_config = CameraConfig(
            url=str(url),
            capture_fps=config.camera.capture_fps if config else 25,
            process_fps=args.fps or (config.camera.process_fps if config else 10),
            reconnect_interval=config.camera.reconnect_interval if config else 5.0,
            max_reconnect_attempts=config.camera.max_reconnect_attempts if config else 0,
        )
    elif config:
        camera_config = config.camera
        if args.fps:
            camera_config = CameraConfig(
                url=camera_config.url,
                capture_fps=camera_config.capture_fps,
                process_fps=args.fps,
                reconnect_interval=camera_config.reconnect_interval,
                max_reconnect_attempts=camera_config.max_reconnect_attempts,
            )
    else:
        print("ERROR: No RTSP URL provided and no config.yaml found.")
        print("Usage: uv run python edge/demo_stream.py --url rtsp://IP:554/stream")
        sys.exit(1)

    # Print info
    print("=" * 60)
    print("  Smart Cabin Platform - RTSP Stream Demo")
    print("=" * 60)
    print(f"  URL: {camera_config.url}")
    print(f"  Capture FPS: {camera_config.capture_fps}")
    print(f"  Process FPS: {camera_config.process_fps}")
    print(f"  Reconnect interval: {camera_config.reconnect_interval}s")
    print(f"  Display scale: {args.scale}")
    print("=" * 60)
    print("  Controls: [q/ESC] Quit | [s] Print stats")
    print("=" * 60)
    print()

    # Create pipeline
    pipeline = VideoPipeline(camera_config)

    # Create display
    display = VideoDisplay(pipeline, scale=args.scale)
    pipeline.register_callback(display.on_frame)

    # Start pipeline
    pipeline.start()

    try:
        display.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.stop()
        display._print_stats()
        print("Demo ended.")


if __name__ == "__main__":
    main()
