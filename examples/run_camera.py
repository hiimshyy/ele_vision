"""
Smart Cabin - Camera Only

Demonstrates video pipeline with display overlay.
No plugins, no AI — just camera streaming with stats.

Usage:
    python examples/run_camera.py --url "rtsp://user:pass@IP:554/stream"
    python examples/run_camera.py --url 0 --scale 0.75
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


class VideoDisplay:
    """Displays frames in an OpenCV window with stats overlay."""

    def __init__(self, pipeline: VideoPipeline, scale: float = 0.5, window_name: str = "Smart Cabin - Camera"):
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

        logger.info("event=display_opened | window={name} | scale={scale}",
                    name=self._window_name, scale=self._scale)

        while True:
            if self._latest_frame is not None:
                display_frame = self._draw_overlay(self._latest_frame.copy())
                h, w = display_frame.shape[:2]
                new_w = int(w * self._scale)
                new_h = int(h * self._scale)
                display_frame = cv2.resize(display_frame, (new_w, new_h))
                cv2.imshow(self._window_name, display_frame)
            else:
                waiting = np.zeros((360, 640, 3), dtype=np.uint8)
                state = self._pipeline.state.value
                cv2.putText(waiting, f"Waiting for stream... ({state})",
                            (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(self._window_name, waiting)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("s"):
                self._print_stats()

            if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw stats overlay on frame."""
        stats = self._pipeline.stats
        state = self._pipeline.state.value
        h, w = frame.shape[:2]

        uptime_sec = time.time() - stats.start_time if stats.start_time > 0 else 0
        hours, remainder = divmod(int(uptime_sec), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 160), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

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
        print(f"  Capture FPS: {stats.current_capture_fps:.1f}")
        print(f"  Distribute FPS: {stats.current_distribute_fps:.1f}")
        print(f"  Decode Time: {stats.decode_time_ms:.1f}ms")
        print(f"  Buffer Latency: {stats.buffer_latency_ms:.1f}ms")
        print(f"  Reconnections: {stats.reconnect_count}")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Smart Cabin - Camera Only")
    parser.add_argument("--url", type=str, help="RTSP URL or camera index (0 for webcam)")
    parser.add_argument("--fps", type=int, default=None, help="Display FPS (default: from config)")
    parser.add_argument("--scale", type=float, default=0.5, help="Display scale (default: 0.5)")
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
            process_fps=args.fps or (config.camera.process_fps if config else 15),
            reconnect_interval=config.camera.reconnect_interval if config else 5.0,
            max_reconnect_attempts=config.camera.max_reconnect_attempts if config else 0,
            connection_timeout=config.camera.connection_timeout if config else 10.0,
        )
    elif config:
        camera_config = config.camera
    else:
        print("ERROR: No RTSP URL provided and no config.yaml found.")
        print("Usage: python examples/run_camera.py --url rtsp://IP:554/stream")
        sys.exit(1)

    # Info
    print("=" * 60)
    print("  Smart Cabin - Camera Only")
    print("=" * 60)
    print(f"  URL: {camera_config.url}")
    print(f"  Display FPS: {args.fps or camera_config.process_fps}")
    print(f"  Scale: {args.scale}")
    print("=" * 60)
    print("  Controls: [q/ESC] Quit | [s] Stats")
    print("=" * 60)
    print()

    # Create pipeline
    pipeline = VideoPipeline(camera_config)

    # Display callback
    display = VideoDisplay(pipeline, scale=args.scale)
    pipeline.register_callback(display.on_frame, fps=args.fps or camera_config.process_fps)

    # Run
    pipeline.start()
    try:
        display.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        pipeline.stop()
        display._print_stats()


if __name__ == "__main__":
    main()
