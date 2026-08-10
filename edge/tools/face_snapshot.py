"""
Smart Cabin - Auto Face Snapshot

Automatically saves face crops and full frames when faces are detected.
Integrates with the Face Recognition Plugin to capture data for:
- Fine-tuning models
- Reviewing false positives/negatives
- Building training datasets

Output structure:
    data/snapshots/
    ├── faces/
    │   ├── recognized_person001_1722700000.jpg   # Aligned 112x112 face
    │   └── unknown_1722700100.jpg
    └── full/
        ├── recognized_person001_1722700000.jpg   # Full frame with bbox
        └── unknown_1722700100.jpg

Config (in plugin config):
    snapshot_enabled: true
    snapshot_dir: "data/snapshots"
    snapshot_max_per_person_per_day: 10
    snapshot_save_full_frame: true
"""

import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from edge.core.logging_setup import get_logger

logger = get_logger("plugin")


class FaceSnapshot:
    """
    Auto face snapshot manager.

    Saves aligned face crops and optionally full frames when
    the recognition plugin processes faces.

    Args:
        snapshot_dir: Output directory for snapshots
        max_per_person_per_day: Max snapshots per person per day (prevents spam)
        save_full_frame: Also save full frame with bbox annotation
    """

    def __init__(self,
                 snapshot_dir: str | Path = "data/snapshots",
                 max_per_person_per_day: int = 10,
                 save_full_frame: bool = True):
        self._snapshot_dir = Path(snapshot_dir)
        self._faces_dir = self._snapshot_dir / "faces"
        self._full_dir = self._snapshot_dir / "full"
        self._max_per_person = max_per_person_per_day
        self._save_full = save_full_frame

        # Daily counters: {person_id: count} — reset each day
        self._daily_counts: dict[str, int] = defaultdict(int)
        self._current_date: str = ""

        # Create directories
        self._faces_dir.mkdir(parents=True, exist_ok=True)
        if self._save_full:
            self._full_dir.mkdir(parents=True, exist_ok=True)

    @property
    def snapshot_dir(self) -> Path:
        return self._snapshot_dir

    @property
    def total_snapshots(self) -> int:
        """Count total snapshot files."""
        count = 0
        if self._faces_dir.exists():
            count += sum(1 for _ in self._faces_dir.rglob("*.jpg"))
        return count

    def save_snapshot(self,
                      aligned_face: np.ndarray | None,
                      full_frame: np.ndarray | None,
                      person_id: str | None,
                      person_name: str = "",
                      confidence: float = 0.0,
                      bbox: tuple[float, float, float, float] | None = None) -> bool:
        """
        Save face snapshot (aligned crop + optional full frame).

        Args:
            aligned_face: 112x112 aligned face image (BGR)
            full_frame: Full camera frame (BGR)
            person_id: Person ID if recognized, None if unknown
            person_name: Person name (for logging)
            confidence: Recognition confidence
            bbox: Face bounding box (x1, y1, x2, y2) for annotation

        Returns:
            True if snapshot was saved
        """
        # Reset daily counters if date changed
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._daily_counts.clear()
            self._current_date = today

        # Check daily limit
        key = person_id or "unknown"
        if self._daily_counts[key] >= self._max_per_person:
            return False

        # Generate filename
        timestamp = int(time.time())
        if person_id:
            prefix = f"recognized_{person_id}_{timestamp}"
        else:
            prefix = f"unknown_{timestamp}"

        saved = False

        # Save aligned face crop
        if aligned_face is not None:
            face_path = self._faces_dir / f"{prefix}.jpg"
            cv2.imwrite(str(face_path), aligned_face)
            saved = True

        # Save full frame with bbox annotation
        if self._save_full and full_frame is not None and bbox is not None:
            annotated = full_frame.copy()
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Draw bbox
            color = (0, 200, 0) if person_id else (0, 0, 220)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{person_name} ({confidence:.2f})" if person_id else f"Unknown ({confidence:.2f})"
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            full_path = self._full_dir / f"{prefix}.jpg"
            cv2.imwrite(str(full_path), annotated)
            saved = True

        if saved:
            self._daily_counts[key] += 1
            logger.debug(
                "event=snapshot_saved | person={key} | count_today={cnt} | prefix={pfx}",
                key=key, cnt=self._daily_counts[key], pfx=prefix,
            )

        return saved

    def get_daily_count(self, person_id: str | None = None) -> int:
        """Get snapshot count for today."""
        key = person_id or "unknown"
        return self._daily_counts.get(key, 0)

    def reset_daily_counts(self) -> None:
        """Manually reset daily counters."""
        self._daily_counts.clear()
