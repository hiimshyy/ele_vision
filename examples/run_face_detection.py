"""
Smart Cabin - Face Detection (Placeholder)

Will demonstrate Camera + Face Detection plugin once Task 5-7 are complete.
Currently just shows camera with a placeholder "detection" callback.

Usage:
    python examples/run_face_detection.py --url "rtsp://user:pass@IP:554/stream"
    python examples/run_face_detection.py --url 0
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


def face_detection_placeholder(frame: np.ndarray, frame_id: int, timestamp: float) -> None:
    """
    Placeholder for face detection plugin.
    Will be replaced with real NCNN inference in Task 5-7.
    """
    # Simulate lightweight processing (~5ms)
    h, w = frame.shape[:2]
    # Just log periodically to verify it's receiving frames
    if frame_id % 50 == 0:
        plugin_logger.info(
            "event=face_detection_tick | frame_id={fid} | resolution={w}x{h}",
            fid=frame_id, w=w, h=h,
        )


def main():
    parser = argparse.ArgumentParser(description="Smart Cabin - Face Detection")
    parser.add_argument("--url", type=str, help="RTSP URL or camera index")
    parser.add_argument("--scale", type=float, default=0.5, help="Display scale")
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

    print("=" * 60)
    print("  Smart Cabin - Face Detection (Placeholder)")
    print("=" * 60)
    print(f"  URL: {camera_config.url}")
    print(f"  Face Detection: 5fps (placeholder)")
    print(f"  Display: 15fps")
    print("  NOTE: Real detection will be added in Task 5-7")
    print("=" * 60)
    print()

    # Pipeline
    pipeline = VideoPipeline(camera_config)

    # Display
    latest_frame = {"frame": None, "id": 0, "ts": 0.0}

    def display_callback(frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        latest_frame["frame"] = frame.copy()
        latest_frame["id"] = frame_id
        latest_frame["ts"] = timestamp

    pipeline.register_callback(display_callback, fps=15)
    pipeline.register_callback(face_detection_placeholder, fps=5)

    pipeline.start()

    cv2.namedWindow("Face Detection", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            if latest_frame["frame"] is not None:
                frame = latest_frame["frame"].copy()
                h, w = frame.shape[:2]

                # Placeholder text
                cv2.putText(frame, "[Face Detection - Placeholder]",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Frame #{latest_frame['id']} | Waiting for NCNN engine (Task 5-7)",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                new_w = int(w * args.scale)
                new_h = int(h * args.scale)
                frame = cv2.resize(frame, (new_w, new_h))
                cv2.imshow("Face Detection", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
            if cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()


if __name__ == "__main__":
    main()
