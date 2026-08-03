"""
Smart Cabin - Lightweight Face Tracker

IoU-based tracker for face detection results. Maintains track identities
across frames to avoid redundant embedding extraction.

Design:
    - Simple IoU matching (no Kalman filter — overkill for 5fps cabin camera)
    - Track states: NEW → ACTIVE → LOST → REMOVED
    - Embedding only extracted for NEW/unidentified tracks
    - Re-verify existing tracks periodically

Usage:
    tracker = FaceTracker(iou_threshold=0.5, max_lost=15)
    tracks = tracker.update(detections, frame_id)
    for track in tracks:
        if track.needs_embedding:
            embedding = embedder.extract(aligned_face)
            track.set_embedding(embedding, identity, confidence)
"""

from dataclasses import dataclass, field

import numpy as np

from edge.core.logging_setup import get_logger

logger = get_logger("plugin")


# --- Track State ---


class TrackState:
    """Track lifecycle states."""
    NEW = "new"          # Just created, needs embedding
    ACTIVE = "active"    # Identified or being tracked
    LOST = "lost"        # Not matched this frame, counting down
    REMOVED = "removed"  # Exceeded max_lost, ready for cleanup


# --- Track ---


@dataclass
class Track:
    """A single tracked face across frames."""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    landmarks: np.ndarray  # (10,) flat landmarks
    confidence: float  # detection confidence
    state: str = TrackState.NEW

    # Identity (set after embedding + matching)
    identity: str | None = None        # person_id or None
    identity_name: str = ""            # person display name
    identity_confidence: float = 0.0   # matching similarity score
    embedding: np.ndarray | None = None

    # Tracking counters
    frame_count: int = 0         # frames since track created
    lost_count: int = 0          # consecutive frames not matched
    first_seen_frame: int = 0    # frame_id when track was created
    last_seen_frame: int = 0     # frame_id when last matched
    last_embed_frame: int = 0    # frame_id when embedding was last extracted
    event_published: bool = False  # whether recognition event was published

    @property
    def needs_embedding(self) -> bool:
        """Whether this track needs embedding extraction."""
        # New track always needs embedding
        if self.state == TrackState.NEW:
            return True
        # Unidentified track needs retry
        if self.identity is None and self.embedding is None:
            return True
        return False

    def needs_reverify(self, current_frame: int, interval: int) -> bool:
        """Whether this track should re-extract embedding for verification."""
        if self.last_embed_frame == 0:
            return True
        return (current_frame - self.last_embed_frame) >= interval

    def set_embedding(self, embedding: np.ndarray | None,
                      identity: str | None = None,
                      identity_name: str = "",
                      confidence: float = 0.0,
                      frame_id: int = 0) -> None:
        """Update track with new embedding and identity."""
        self.embedding = embedding
        self.identity = identity
        self.identity_name = identity_name
        self.identity_confidence = confidence
        self.last_embed_frame = frame_id
        if self.state == TrackState.NEW:
            self.state = TrackState.ACTIVE

    @property
    def bbox_xywh(self) -> list[int]:
        """Bounding box as [x, y, w, h] integers (for event publishing)."""
        x1, y1, x2, y2 = self.bbox
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


# --- IoU Computation ---


def compute_iou(box_a: tuple[float, float, float, float],
                box_b: tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union between two bboxes.

    Args:
        box_a: (x1, y1, x2, y2)
        box_b: (x1, y1, x2, y2)

    Returns:
        IoU value in [0.0, 1.0]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area < 1e-6:
        return 0.0

    return inter_area / union_area


def compute_iou_matrix(detections: list[tuple], tracks: list[Track]) -> np.ndarray:
    """
    Compute IoU matrix between detections and existing tracks.

    Args:
        detections: List of (x1, y1, x2, y2) bboxes
        tracks: List of Track objects

    Returns:
        (N_det, N_track) IoU matrix
    """
    n_det = len(detections)
    n_trk = len(tracks)
    iou_matrix = np.zeros((n_det, n_trk), dtype=np.float32)

    for i, det_bbox in enumerate(detections):
        for j, track in enumerate(tracks):
            iou_matrix[i, j] = compute_iou(det_bbox, track.bbox)

    return iou_matrix


# --- Face Tracker ---


class FaceTracker:
    """
    Lightweight IoU-based face tracker.

    Matches new detections to existing tracks using IoU overlap.
    Designed for low-FPS cabin camera (5fps, max 5-6 people).

    Args:
        iou_threshold: Minimum IoU to match detection to track (default 0.4)
        max_lost: Frames before removing lost track (default 15 = 3s at 5fps)
        max_tracks: Maximum simultaneous tracks (default 10)
        reverify_interval: Frames between re-verification (default 15 = 3s at 5fps)
    """

    def __init__(self,
                 iou_threshold: float = 0.4,
                 max_lost: int = 15,
                 max_tracks: int = 10,
                 reverify_interval: int = 15):
        self._iou_threshold = iou_threshold
        self._max_lost = max_lost
        self._max_tracks = max_tracks
        self._reverify_interval = reverify_interval
        self._tracks: list[Track] = []
        self._next_id: int = 1

    @property
    def tracks(self) -> list[Track]:
        """All active tracks (not REMOVED)."""
        return [t for t in self._tracks if t.state != TrackState.REMOVED]

    @property
    def active_tracks(self) -> list[Track]:
        """Tracks in ACTIVE state."""
        return [t for t in self._tracks if t.state == TrackState.ACTIVE]

    @property
    def track_count(self) -> int:
        """Number of non-removed tracks."""
        return len(self.tracks)

    def update(self, detections: list[dict], frame_id: int) -> list[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of dicts with keys: bbox (x1,y1,x2,y2), landmarks (10,), confidence
            frame_id: Current frame number

        Returns:
            List of all active Track objects (matched + new)
        """
        # Step 1: Mark all existing tracks as potentially lost
        for track in self._tracks:
            if track.state in (TrackState.NEW, TrackState.ACTIVE):
                track.state = TrackState.LOST
                track.lost_count += 1
            elif track.state == TrackState.LOST:
                track.lost_count += 1

        if not detections:
            # No detections → just age tracks
            self._cleanup_lost_tracks()
            return self.tracks

        # Step 2: Compute IoU between detections and existing tracks
        active_tracks = [t for t in self._tracks if t.state != TrackState.REMOVED]

        if active_tracks:
            det_bboxes = [d["bbox"] for d in detections]
            iou_matrix = compute_iou_matrix(det_bboxes, active_tracks)

            # Greedy matching (Hungarian would be overkill for <10 faces)
            matched_dets = set()
            matched_trks = set()

            # Sort by IoU descending for greedy assignment
            pairs = []
            for i in range(len(detections)):
                for j in range(len(active_tracks)):
                    if iou_matrix[i, j] >= self._iou_threshold:
                        pairs.append((iou_matrix[i, j], i, j))
            pairs.sort(reverse=True)

            for iou_val, det_idx, trk_idx in pairs:
                if det_idx in matched_dets or trk_idx in matched_trks:
                    continue
                # Match found
                track = active_tracks[trk_idx]
                det = detections[det_idx]
                track.bbox = det["bbox"]
                track.landmarks = det["landmarks"]
                track.confidence = det["confidence"]
                track.state = TrackState.ACTIVE
                track.lost_count = 0
                track.frame_count += 1
                track.last_seen_frame = frame_id
                matched_dets.add(det_idx)
                matched_trks.add(trk_idx)

            # Step 3: Create new tracks for unmatched detections
            for i, det in enumerate(detections):
                if i not in matched_dets:
                    self._create_track(det, frame_id)
        else:
            # No existing tracks → all detections are new
            for det in detections:
                self._create_track(det, frame_id)

        # Step 4: Cleanup lost tracks
        self._cleanup_lost_tracks()

        return self.tracks

    def _create_track(self, detection: dict, frame_id: int) -> Track | None:
        """Create a new track from a detection."""
        if self.track_count >= self._max_tracks:
            return None

        track = Track(
            track_id=self._next_id,
            bbox=detection["bbox"],
            landmarks=detection["landmarks"],
            confidence=detection["confidence"],
            state=TrackState.NEW,
            first_seen_frame=frame_id,
            last_seen_frame=frame_id,
        )
        self._next_id += 1
        self._tracks.append(track)

        logger.info(
            "event=track_created | track_id={tid} | frame={fid} | bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})",
            tid=track.track_id, fid=frame_id,
            x1=track.bbox[0], y1=track.bbox[1], x2=track.bbox[2], y2=track.bbox[3],
        )
        return track

    def _cleanup_lost_tracks(self) -> None:
        """Remove tracks that have been lost too long."""
        for track in self._tracks:
            if track.state == TrackState.LOST and track.lost_count > self._max_lost:
                track.state = TrackState.REMOVED
                logger.info(
                    "event=track_removed | track_id={tid} | lost_frames={lf} | "
                    "identity={ident} | total_frames={tf}",
                    tid=track.track_id, lf=track.lost_count,
                    ident=track.identity or "unknown", tf=track.frame_count,
                )

        # Purge removed tracks from list to prevent unbounded growth
        self._tracks = [t for t in self._tracks if t.state != TrackState.REMOVED]

    def get_tracks_needing_embedding(self, frame_id: int) -> list[Track]:
        """
        Get tracks that need embedding extraction.

        Returns tracks that are:
        - NEW (never embedded)
        - ACTIVE but unidentified
        - ACTIVE and due for re-verification

        Args:
            frame_id: Current frame number

        Returns:
            List of tracks needing embedding
        """
        result = []
        for track in self._tracks:
            if track.state == TrackState.REMOVED:
                continue
            if track.needs_embedding:
                result.append(track)
            elif (track.state == TrackState.ACTIVE
                  and track.needs_reverify(frame_id, self._reverify_interval)):
                result.append(track)
        return result

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 1
