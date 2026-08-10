"""
Smart Cabin - Data Recorder

CLI tool for collecting training data from camera:
- Record video segments (configurable duration, auto file rotation)
- Periodic snapshot (full-frame capture every N seconds)
- Continuous mode (video + snapshot combined)

Runs standalone (no plugin system required).

Usage:
    # Record video in 60-second segments
    python -m edge.tools.data_recorder --mode video --url "rtsp://..." --duration 60

    # Capture snapshots every 5 seconds
    python -m edge.tools.data_recorder --mode snapshot --url "rtsp://..." --interval 5

    # Both video + snapshots
    python -m edge.tools.data_recorder --mode continuous --url "rtsp://..." --duration 60 --interval 5

    # From webcam
    python -m edge.tools.data_recorder --mode snapshot --url 0 --interval 3

Output structure:
    data/
    ├── videos/YYYY-MM-DD/cabin-001_HH-MM-SS.mp4
    └── frames/YYYY-MM-DD/cabin-001_HH-MM-SS_frame.jpg
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from edge.core.logging_setup import setup_logging, get_logger
from edge.tools.storage_manager import StorageManager

logger = get_logger("system")

# Defaults
DEFAULT_DATA_DIR = Path("data")
DEFAULT_DEVICE_ID = "cabin-001"


class DataRecorder:
    """
    Records video and/or snapshots from camera stream.

    Args:
        url: Camera URL or device index
        data_dir: Base output directory
        device_id: Device identifier for filenames
        max_disk_mb: Max disk usage before cleanup (0 = no limit)
    """

    def __init__(self,
                 url: str | int,
                 data_dir: Path = DEFAULT_DATA_DIR,
                 device_id: str = DEFAULT_DEVICE_ID,
                 max_disk_mb: int = 1000):
        self._url = url
        self._data_dir = data_dir
        self._device_id = device_id
        self._cap: cv2.VideoCapture | None = None
        self._storage = StorageManager(data_dir, max_disk_mb=max_disk_mb)

        # Directories
        self._video_dir = data_dir / "videos"
        self._frame_dir = data_dir / "frames"

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def connect(self) -> bool:
        """Open camera connection."""
        url = int(self._url) if str(self._url).isdigit() else self._url
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            logger.error("event=recorder_connect_failed | url={u}", u=self._url)
            return False

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        logger.info(
            "event=recorder_connected | url={u} | resolution={w}x{h} | fps={fps}",
            u=self._url, w=w, h=h, fps=fps,
        )
        return True

    def disconnect(self) -> None:
        """Release camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def record_video(self, duration: int = 60, max_segments: int = 0) -> None:
        """
        Record video in segments of `duration` seconds.

        Args:
            duration: Seconds per video file
            max_segments: Max segments to record (0 = infinite, Ctrl+C to stop)
        """
        if not self.is_connected:
            print("  ERROR: Not connected to camera")
            return

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0

        segment = 0
        print(f"  Recording video: {duration}s segments (Ctrl+C to stop)")

        try:
            while max_segments == 0 or segment < max_segments:
                # Storage check
                self._storage.cleanup_if_needed()

                # Create output path
                now = datetime.now()
                date_dir = self._video_dir / now.strftime("%Y-%m-%d")
                date_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{self._device_id}_{now.strftime('%H-%M-%S')}.mp4"
                filepath = date_dir / filename

                # Open writer
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h))

                if not writer.isOpened():
                    logger.error("event=video_writer_failed | path={p}", p=str(filepath))
                    break

                segment += 1
                start_time = time.time()
                frame_count = 0

                logger.info(
                    "event=video_segment_start | segment={seg} | path={p} | duration={d}s",
                    seg=segment, p=str(filepath), d=duration,
                )
                print(f"  [{_timestamp()}] Segment {segment}: {filepath.name}")

                # Record for duration
                while time.time() - start_time < duration:
                    ret, frame = self._cap.read()
                    if not ret:
                        logger.warning("event=recorder_frame_lost")
                        time.sleep(0.1)
                        continue
                    writer.write(frame)
                    frame_count += 1

                writer.release()
                elapsed = time.time() - start_time

                # Write metadata sidecar
                meta = {
                    "device_id": self._device_id,
                    "timestamp": now.isoformat(),
                    "duration_s": round(elapsed, 1),
                    "frames": frame_count,
                    "resolution": f"{w}x{h}",
                    "fps": round(fps, 1),
                    "file_size_mb": round(filepath.stat().st_size / 1024 / 1024, 2),
                }
                meta_path = filepath.with_suffix(".json")
                meta_path.write_text(json.dumps(meta, indent=2))

                logger.info(
                    "event=video_segment_done | segment={seg} | frames={fc} | "
                    "duration={dur:.1f}s | size_mb={sz:.2f}",
                    seg=segment, fc=frame_count, dur=elapsed,
                    sz=meta["file_size_mb"],
                )

        except KeyboardInterrupt:
            print(f"\n  Stopped after {segment} segment(s).")

    def capture_snapshots(self, interval: float = 5.0, max_count: int = 0) -> None:
        """
        Capture periodic snapshots.

        Args:
            interval: Seconds between captures
            max_count: Max snapshots (0 = infinite)
        """
        if not self.is_connected:
            print("  ERROR: Not connected to camera")
            return

        count = 0
        print(f"  Capturing snapshots every {interval}s (Ctrl+C to stop)")

        try:
            while max_count == 0 or count < max_count:
                # Storage check
                self._storage.cleanup_if_needed()

                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue

                # Save frame
                now = datetime.now()
                date_dir = self._frame_dir / now.strftime("%Y-%m-%d")
                date_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{self._device_id}_{now.strftime('%H-%M-%S')}_frame.jpg"
                filepath = date_dir / filename

                cv2.imwrite(str(filepath), frame)
                count += 1

                h, w = frame.shape[:2]
                logger.info(
                    "event=snapshot_captured | count={c} | path={p} | resolution={w}x{h}",
                    c=count, p=str(filepath), w=w, h=h,
                )
                print(f"  [{_timestamp()}] Snapshot #{count}: {filepath.name}")

                # Wait for interval (drain frames to stay current)
                wait_until = time.time() + interval
                while time.time() < wait_until:
                    self._cap.read()  # Drain buffer
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print(f"\n  Stopped after {count} snapshot(s).")

    def record_continuous(self, duration: int = 60, interval: float = 5.0) -> None:
        """
        Record video + capture snapshots simultaneously.

        Video records in segments, snapshots captured every `interval` seconds.
        """
        if not self.is_connected:
            print("  ERROR: Not connected to camera")
            return

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0

        print(f"  Continuous: video ({duration}s segments) + snapshots (every {interval}s)")
        print(f"  Ctrl+C to stop\n")

        segment = 0
        snapshot_count = 0
        last_snapshot_time = 0

        try:
            while True:
                self._storage.cleanup_if_needed()

                # Start new video segment
                now = datetime.now()
                date_dir = self._video_dir / now.strftime("%Y-%m-%d")
                date_dir.mkdir(parents=True, exist_ok=True)
                video_path = date_dir / f"{self._device_id}_{now.strftime('%H-%M-%S')}.mp4"

                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                if not writer.isOpened():
                    break

                segment += 1
                start_time = time.time()
                frame_count = 0

                print(f"  [{_timestamp()}] Video segment {segment}: {video_path.name}")

                while time.time() - start_time < duration:
                    ret, frame = self._cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue

                    writer.write(frame)
                    frame_count += 1

                    # Snapshot check
                    current_time = time.time()
                    if current_time - last_snapshot_time >= interval:
                        snap_now = datetime.now()
                        snap_dir = self._frame_dir / snap_now.strftime("%Y-%m-%d")
                        snap_dir.mkdir(parents=True, exist_ok=True)
                        snap_path = snap_dir / f"{self._device_id}_{snap_now.strftime('%H-%M-%S')}_frame.jpg"
                        cv2.imwrite(str(snap_path), frame)
                        snapshot_count += 1
                        last_snapshot_time = current_time
                        print(f"  [{_timestamp()}]   Snapshot #{snapshot_count}: {snap_path.name}")

                writer.release()

                logger.info(
                    "event=continuous_segment | segment={seg} | frames={fc} | snapshots={sc}",
                    seg=segment, fc=frame_count, sc=snapshot_count,
                )

        except KeyboardInterrupt:
            print(f"\n  Stopped: {segment} video segments, {snapshot_count} snapshots.")


# --- Helpers ---


def _timestamp():
    return time.strftime("%H:%M:%S")


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Smart Cabin - Data Recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m edge.tools.data_recorder --mode video --url 0 --duration 30
  python -m edge.tools.data_recorder --mode snapshot --url "rtsp://..." --interval 3
  python -m edge.tools.data_recorder --mode continuous --url 0 --duration 60 --interval 5
""",
    )
    parser.add_argument("--mode", required=True, choices=["video", "snapshot", "continuous"],
                        help="Recording mode")
    parser.add_argument("--url", required=True, help="Camera URL or device index (0)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Video segment duration in seconds (default: 60)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Snapshot interval in seconds (default: 5)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Output directory (default: data/)")
    parser.add_argument("--device-id", type=str, default=DEFAULT_DEVICE_ID,
                        help="Device ID for filenames")
    parser.add_argument("--max-disk-mb", type=int, default=1000,
                        help="Max disk usage in MB before cleanup (default: 1000)")
    parser.add_argument("--max-segments", type=int, default=0,
                        help="Max video segments (0 = infinite)")
    parser.add_argument("--max-snapshots", type=int, default=0,
                        help="Max snapshots (0 = infinite)")

    args = parser.parse_args()
    setup_logging("INFO")

    print(f"\n  Smart Cabin - Data Recorder")
    print(f"  Mode: {args.mode}")
    print(f"  URL:  {args.url}")
    print(f"  Output: {args.data_dir}/")

    recorder = DataRecorder(
        url=args.url,
        data_dir=Path(args.data_dir),
        device_id=args.device_id,
        max_disk_mb=args.max_disk_mb,
    )

    if not recorder.connect():
        print(f"  ERROR: Cannot connect to {args.url}")
        sys.exit(1)

    try:
        if args.mode == "video":
            recorder.record_video(duration=args.duration, max_segments=args.max_segments)
        elif args.mode == "snapshot":
            recorder.capture_snapshots(interval=args.interval, max_count=args.max_snapshots)
        elif args.mode == "continuous":
            recorder.record_continuous(duration=args.duration, interval=args.interval)
    finally:
        recorder.disconnect()
        print("  Done.")


if __name__ == "__main__":
    main()
