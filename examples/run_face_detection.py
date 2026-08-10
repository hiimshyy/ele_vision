"""
Smart Cabin - Real-time Face Detection

Demonstrates Camera + Face Detection with bounding boxes and landmarks overlay.
Uses FaceDetector (OpenCV fallback or C++ NCNN if built).

Usage:
    python examples/run_face_detection.py --url "rtsp://user:pass@IP:554/stream"
    python examples/run_face_detection.py --url 0 --scale 0.75
    python examples/run_face_detection.py --url 0 --det-fps 5

Controls:
    q / ESC  - Quit
    s        - Print stats
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

from edge.core.config import CameraConfig, load_config
from edge.core.video_pipeline import VideoPipeline, PipelineState
from edge.core.logging_setup import setup_logging, get_logger
from edge.plugins.face_recognition.detector import FaceDetector, FaceInfo

logger = get_logger("system")
plugin_logger = get_logger("plugin")

# Default model path
MODEL_PATH = Path("edge/plugins/face_recognition/models/det_500m.onnx")


class FaceDetectionDisplay:
    """Displays frames with face detection bounding boxes and landmarks."""

    def __init__(self, pipeline: VideoPipeline, detector: FaceDetector,
                 scale: float = 0.5, conf_threshold: float = 0.7):
        self._pipeline = pipeline
        self._detector = detector
        self._scale = scale
        self._conf_threshold = conf_threshold

        # Shared state (written by detection callback, read by display)
        self._latest_frame: np.ndarray | None = None
        self._latest_faces: list[FaceInfo] = []
        self._frame_id = 0
        self._det_time_ms: float = 0.0
        self._lock = threading.Lock()

    def on_display_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        """Display callback: just stores latest frame."""
        with self._lock:
            self._latest_frame = frame.copy()
            self._frame_id = frame_id

    def on_detect_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        """Detection callback: runs face detection and stores results."""
        faces = self._detector.detect(frame, conf_threshold=self._conf_threshold)
        with self._lock:
            self._latest_faces = faces
            self._det_time_ms = self._detector.inference_time_ms

        if faces:
            plugin_logger.info(
                "event=faces_detected | frame_id={fid} | count={n} | inference_ms={ms:.1f}",
                fid=frame_id, n=len(faces), ms=self._detector.inference_time_ms,
            )

    def run(self) -> None:
        """Main display loop."""
        cv2.namedWindow("Face Detection", cv2.WINDOW_AUTOSIZE)
        logger.info("event=display_opened | window=Face Detection | scale={s}", s=self._scale)

        while True:
            with self._lock:
                frame = self._latest_frame.copy() if self._latest_frame is not None else None
                faces = list(self._latest_faces)
                frame_id = self._frame_id
                det_time = self._det_time_ms

            if frame is not None:
                # Draw detections
                display_frame = self._draw_detections(frame, faces)
                # Draw stats overlay
                display_frame = self._draw_overlay(display_frame, len(faces), det_time)
                # Scale
                h, w = display_frame.shape[:2]
                new_w = int(w * self._scale)
                new_h = int(h * self._scale)
                display_frame = cv2.resize(display_frame, (new_w, new_h))
                cv2.imshow("Face Detection", display_frame)
            else:
                waiting = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for stream...",
                            (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("Face Detection", waiting)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("s"):
                self._print_stats()

            if cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    def _draw_detections(self, frame: np.ndarray, faces: list[FaceInfo]) -> np.ndarray:
        """Draw bounding boxes and landmarks on frame."""
        for face in faces:
            x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Confidence + size label
            label = f"{face.score:.2f} {int(face.width)}x{int(face.height)}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 5-point landmarks
            colors = [
                (255, 0, 0),    # right eye - blue
                (0, 0, 255),    # left eye - red
                (0, 255, 0),    # nose - green
                (255, 0, 255),  # right mouth - magenta
                (0, 255, 255),  # left mouth - yellow
            ]
            for j in range(5):
                px = int(face.landmarks[j * 2])
                py = int(face.landmarks[j * 2 + 1])
                cv2.circle(frame, (px, py), 3, colors[j], -1)

        return frame

    def _draw_overlay(self, frame: np.ndarray, face_count: int, det_time: float) -> np.ndarray:
        """Draw stats overlay."""
        stats = self._pipeline.stats
        h, w = frame.shape[:2]

        uptime_sec = time.time() - stats.start_time if stats.start_time > 0 else 0
        hours, remainder = divmod(int(uptime_sec), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 135), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        y = 32
        color = (0, 255, 0)
        texts = [
            f"FPS: {stats.current_distribute_fps:.1f} | Uptime: {uptime_str}",
            f"Resolution: {w}x{h} | Backend: {self._detector.backend}",
            f"Faces: {face_count} | Detection: {det_time:.1f}ms",
            f"Reconnects: {stats.reconnect_count}",
        ]

        for text in texts:
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 25

        return frame

    def _print_stats(self) -> None:
        """Print stats to console."""
        stats = self._pipeline.stats
        elapsed = time.time() - stats.start_time if stats.start_time > 0 else 0
        print(f"\n{'='*50}")
        print(f"  Pipeline: {self._pipeline.state.value}")
        print(f"  Uptime: {elapsed:.1f}s")
        print(f"  Capture FPS: {stats.current_capture_fps:.1f}")
        print(f"  Distribute FPS: {stats.current_distribute_fps:.1f}")
        print(f"  Detection backend: {self._detector.backend}")
        print(f"  Detection time: {self._det_time_ms:.1f}ms")
        print(f"  Reconnects: {stats.reconnect_count}")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Smart Cabin - Real-time Face Detection")
    parser.add_argument("--url", type=str, help="RTSP URL or camera index")
    parser.add_argument("--scale", type=float, default=0.5, help="Display scale (default: 0.5)")
    parser.add_argument("--det-fps", type=int, default=5, help="Detection FPS (default: 5)")
    parser.add_argument("--display-fps", type=int, default=15, help="Display FPS (default: 15)")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold (default: 0.7)")
    parser.add_argument("--model", type=str, default=None, help="Model path (default: auto-detect)")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    # Load config
    try:
        config = load_config(config_path=args.config)
    except FileNotFoundError:
        config = None

    if config:
        setup_logging(config.logging)
    else:
        setup_logging("INFO")

    # Camera config
    if args.url is not None:
        camera_config = CameraConfig(
            url=str(args.url),
            capture_fps=config.camera.capture_fps if config else 25,
            process_fps=args.display_fps,
            reconnect_interval=config.camera.reconnect_interval if config else 5.0,
            max_reconnect_attempts=0,
            connection_timeout=config.camera.connection_timeout if config else 10.0,
        )
    elif config:
        camera_config = config.camera
    else:
        print("ERROR: No RTSP URL provided and no config.yaml found.")
        sys.exit(1)

    # Load face detector (auto-detect model: SCRFD > YuNet)
    model_path = Path(args.model) if args.model else None
    detector = FaceDetector()
    if not detector.load(model_path=model_path, input_size=640):
        print(f"ERROR: Failed to load face detection model")
        print("  Download models: bash edge/inference/download_models.sh")
        sys.exit(1)

    # Info
    print("=" * 60)
    print("  Smart Cabin - Real-time Face Detection")
    print("=" * 60)
    print(f"  URL: {camera_config.url}")
    print(f"  Display FPS: {args.display_fps}")
    print(f"  Detection FPS: {args.det_fps}")
    print(f"  Confidence: {args.conf}")
    print(f"  Model: {detector.model_name}")
    print(f"  Backend: {detector.backend}")
    print(f"  Scale: {args.scale}")
    print("=" * 60)
    print("  Controls: [q/ESC] Quit | [s] Stats")
    print("=" * 60)
    print()

    # Create pipeline
    pipeline = VideoPipeline(camera_config)

    # Create display + detector integration
    display = FaceDetectionDisplay(pipeline, detector, scale=args.scale, conf_threshold=args.conf)

    # Register callbacks at different FPS
    pipeline.register_callback(display.on_display_frame, fps=args.display_fps)
    pipeline.register_callback(display.on_detect_frame, fps=args.det_fps)

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
