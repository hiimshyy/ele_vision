"""
Smart Cabin - Face Recognition Plugin

Full pipeline: detect → track → align → embed → match → publish events.

Integrates:
- FaceDetector (SCRFD-500M detection + 5-point landmarks)
- FaceTracker (IoU-based tracking, reduces redundant embeddings)
- FaceEmbedder (MobileFaceNet w600k_mbf, 512-dim embeddings)
- FaceDatabase (SQLite, person matching via cosine similarity)

Events published:
- face.recognized: known person identified (once per track entry)
- face.unknown: unknown face detected (once per track entry)
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edge.core.plugin_manager import BasePlugin
from edge.core.event_bus import EventBus
from edge.core.logging_setup import get_logger
from edge.plugins.face_recognition.detector import FaceDetector
from edge.plugins.face_recognition.alignment import align_face
from edge.plugins.face_recognition.embedder import FaceEmbedder
from edge.plugins.face_recognition.tracker import FaceTracker, TrackState
from edge.plugins.face_recognition.database import FaceDatabase
from edge.tools.face_snapshot import FaceSnapshot
from shared.event_schemas import FaceRecognizedEvent, FaceUnknownEvent

logger = get_logger("plugin")


class Plugin(BasePlugin):
    """
    Face Recognition plugin.

    Pipeline per frame:
    1. Detect faces (SCRFD)
    2. Update tracker (IoU matching)
    3. For NEW/unidentified tracks: align → embed → match database
    4. Publish events for newly-identified tracks (once per entry)
    """

    @property
    def name(self) -> str:
        return "face_recognition"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def default_fps(self) -> float:
        return 5.0

    def initialize(self, config: dict[str, Any], event_bus: EventBus) -> bool:
        """
        Initialize all sub-components.

        Config keys:
            detection_threshold: float (default 0.7)
            embedding_threshold: float (default 0.4)
            embedding_model: str (default "w600k_mbf.onnx")
            min_face_size: int (default 80)
            min_face_quality: float (default 50.0, Laplacian variance)
            tracker_iou_threshold: float (default 0.4)
            tracker_max_lost: int (default 15)
            tracker_max_tracks: int (default 10)
            tracker_reverify_interval: int (default 15)
            database_path: str (default "faces.db")
            snapshot_enabled: bool (default False)
            snapshot_dir: str (default "data/snapshots")
            snapshot_max_per_person_per_day: int (default 10)
            snapshot_save_full_frame: bool (default True)
        """
        self._config = config
        self._event_bus = event_bus
        self._frame_count = 0

        # Config values
        self._det_threshold = config.get("detection_threshold", 0.7)
        self._emb_threshold = config.get("embedding_threshold", 0.4)
        self._min_face_size = config.get("min_face_size", 80)
        self._min_face_quality = config.get("min_face_quality", 50.0)

        # Initialize detector
        self._detector = FaceDetector()
        if not self._detector.load():
            logger.error("event=plugin_init_failed | reason=detector load failed")
            return False

        # Initialize embedder
        model_name = config.get("embedding_model", "w600k_mbf.onnx")
        model_path = Path("edge/plugins/face_recognition/models") / model_name
        self._embedder = FaceEmbedder()
        if not self._embedder.load(model_path):
            logger.error("event=plugin_init_failed | reason=embedder load failed")
            return False

        # Initialize tracker
        self._tracker = FaceTracker(
            iou_threshold=config.get("tracker_iou_threshold", 0.4),
            max_lost=config.get("tracker_max_lost", 15),
            max_tracks=config.get("tracker_max_tracks", 10),
            reverify_interval=config.get("tracker_reverify_interval", 15),
        )

        # Initialize database
        db_path = config.get("database_path", "faces.db")
        self._database = FaceDatabase(db_path)
        if not self._database.initialize():
            logger.error("event=plugin_init_failed | reason=database init failed")
            return False

        # Initialize auto-snapshot (optional)
        self._snapshot_enabled = config.get("snapshot_enabled", False)
        self._snapshot: FaceSnapshot | None = None
        if self._snapshot_enabled:
            self._snapshot = FaceSnapshot(
                snapshot_dir=config.get("snapshot_dir", "data/snapshots"),
                max_per_person_per_day=config.get("snapshot_max_per_person_per_day", 10),
                save_full_frame=config.get("snapshot_save_full_frame", True),
            )
            logger.info(
                "event=snapshot_enabled | dir={d} | max_per_day={m}",
                d=config.get("snapshot_dir", "data/snapshots"),
                m=config.get("snapshot_max_per_person_per_day", 10),
            )

        logger.info(
            "event=face_recognition_init | det_model={det} | emb_model={emb} | "
            "det_threshold={dt} | emb_threshold={et} | db_faces={n}",
            det=self._detector.model_name, emb=self._embedder.model_name,
            dt=self._det_threshold, et=self._emb_threshold,
            n=self._database.count(),
        )
        return True

    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        """
        Process a single frame through the recognition pipeline.

        Steps:
        1. Detect faces
        2. Filter small/low-quality faces
        3. Update tracker
        4. Extract embeddings for tracks that need it
        5. Match against database
        6. Publish events for new identifications
        """
        self._frame_count += 1

        # Step 1: Detect faces
        faces = self._detector.detect(frame, conf_threshold=self._det_threshold)

        # Step 2: Filter and convert to tracker format
        detections = []
        for face in faces:
            # Skip small faces
            if face.width < self._min_face_size or face.height < self._min_face_size:
                continue
            detections.append({
                "bbox": (face.x1, face.y1, face.x2, face.y2),
                "landmarks": face.landmarks,
                "confidence": face.score,
            })

        # Step 3: Update tracker
        tracks = self._tracker.update(detections, frame_id)

        # Step 4: Process tracks needing embedding
        tracks_to_embed = self._tracker.get_tracks_needing_embedding(frame_id)
        embeddings_extracted = 0

        for track in tracks_to_embed:
            # Quality check: Laplacian variance (blur detection)
            if not self._check_face_quality(frame, track.bbox):
                logger.debug(
                    "event=face_quality_rejected | track_id={tid} | reason=blur",
                    tid=track.track_id,
                )
                continue

            # Align face
            aligned = align_face(frame, track.landmarks)
            if aligned is None:
                logger.debug(
                    "event=face_align_failed | track_id={tid}",
                    tid=track.track_id,
                )
                continue

            # Extract embedding
            embedding = self._embedder.extract(aligned)
            if embedding is None:
                continue

            embeddings_extracted += 1

            # Match against database
            match = self._database.find_match(embedding, threshold=self._emb_threshold)

            if match is not None:
                track.set_embedding(
                    embedding=embedding,
                    identity=match.person_id,
                    identity_name=match.name,
                    confidence=match.similarity,
                    frame_id=frame_id,
                )
            else:
                track.set_embedding(
                    embedding=embedding,
                    identity=None,
                    identity_name="",
                    confidence=0.0,
                    frame_id=frame_id,
                )

        # Step 5: Publish events for newly-identified tracks
        for track in tracks:
            if track.event_published:
                continue
            if track.state != TrackState.ACTIVE:
                continue
            if track.embedding is None:
                continue

            # Publish recognition event (once per track)
            if track.identity is not None:
                self._event_bus.publish(FaceRecognizedEvent(
                    source=self.name,
                    person_id=track.identity,
                    person_name=track.identity_name,
                    confidence=track.identity_confidence,
                    bbox=track.bbox_xywh,
                ))
                logger.info(
                    "event=face_recognized | track_id={tid} | person={pid} | "
                    "name={name} | confidence={conf:.3f}",
                    tid=track.track_id, pid=track.identity,
                    name=track.identity_name, conf=track.identity_confidence,
                )
            else:
                self._event_bus.publish(FaceUnknownEvent(
                    source=self.name,
                    confidence=track.confidence,
                    bbox=track.bbox_xywh,
                ))
                logger.info(
                    "event=face_unknown | track_id={tid} | det_confidence={conf:.3f}",
                    tid=track.track_id, conf=track.confidence,
                )

            track.event_published = True

            # Auto-snapshot (save face crop + full frame)
            if self._snapshot is not None:
                aligned_for_snap = align_face(frame, track.landmarks)
                self._snapshot.save_snapshot(
                    aligned_face=aligned_for_snap,
                    full_frame=frame,
                    person_id=track.identity,
                    person_name=track.identity_name,
                    confidence=track.identity_confidence if track.identity else track.confidence,
                    bbox=track.bbox,
                )

        # Periodic stats (every 25 frames = 5s at 5fps)
        if self._frame_count % 25 == 0:
            active = len(self._tracker.active_tracks)
            total_tracks = self._tracker.track_count
            logger.info(
                "event=recognition_stats | frames_processed={fp} | "
                "faces_detected={fd} | embeddings_extracted={ee} | "
                "active_tracks={at} | total_tracks={tt} | "
                "det_ms={dms:.1f} | emb_ms={ems:.1f} | db_persons={db}",
                fp=self._frame_count, fd=len(detections), ee=embeddings_extracted,
                at=active, tt=total_tracks,
                dms=self._detector.inference_time_ms,
                ems=self._embedder.inference_time_ms,
                db=self._database.count_persons(),
            )

    def _check_face_quality(self, frame: np.ndarray,
                            bbox: tuple[float, float, float, float]) -> bool:
        """
        Check face quality using Laplacian variance (blur detection).

        Args:
            frame: Full BGR frame
            bbox: Face bounding box (x1, y1, x2, y2)

        Returns:
            True if face quality is acceptable
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return False

        face_crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        return laplacian_var >= self._min_face_quality

    @property
    def tracker(self) -> FaceTracker:
        """Access tracker (for testing/monitoring)."""
        return self._tracker

    @property
    def database(self) -> FaceDatabase:
        """Access database (for enrollment/management)."""
        return self._database

    @property
    def detector(self) -> FaceDetector:
        """Access detector (for testing)."""
        return self._detector

    @property
    def embedder(self) -> FaceEmbedder:
        """Access embedder (for testing)."""
        return self._embedder

    def shutdown(self) -> None:
        """Cleanup resources."""
        self._database.close()
        self._tracker.reset()
        logger.info(
            "event=face_recognition_shutdown | frames_processed={n}",
            n=self._frame_count,
        )
