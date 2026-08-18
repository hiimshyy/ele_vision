"""
Smart Cabin - ByteTrack Face Tracker

Production-grade multi-object tracker based on ByteTrack algorithm:
- Kalman Filter: predict next position, smooth trajectory
- 2-Stage Matching: high-confidence first, then low-confidence (recover occluded)
- Hungarian Algorithm: optimal assignment (scipy linear_sum_assignment)
- Track lifecycle: NEW → ACTIVE → LOST → REMOVED

Reference: ByteTrack (ECCV 2022) - Zhang et al.
Adapted for face tracking in cabin camera (5fps, max 10 faces).

Usage:
    tracker = FaceTracker()
    tracks = tracker.update(detections, frame_id)
    for track in tracks:
        if track.needs_embedding:
            ...
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from edge.core.logging_setup import get_logger

logger = get_logger("plugin")


# --- Track State ---


class TrackState:
    """Track lifecycle states."""
    NEW = "new"          # Just created, needs embedding
    ACTIVE = "active"    # Identified or being tracked
    LOST = "lost"        # Not matched this frame, counting down
    REMOVED = "removed"  # Exceeded max_lost, ready for cleanup


# --- Kalman Filter ---


class KalmanFilter:
    """
    Simple 2D Kalman filter for bounding box tracking.

    State: [cx, cy, area, aspect_ratio, vx, vy, v_area, v_aspect]
    Measurement: [cx, cy, area, aspect_ratio]
    """

    def __init__(self):
        # State transition matrix (constant velocity model)
        self._dt = 1.0  # Normalized time step
        self._F = np.eye(8, dtype=np.float64)
        self._F[:4, 4:] = self._dt * np.eye(4)

        # Measurement matrix
        self._H = np.eye(4, 8, dtype=np.float64)

        # Process noise
        self._Q = np.eye(8, dtype=np.float64)
        self._Q[4:, 4:] *= 0.01  # Velocity components have less noise

        # Measurement noise
        self._R = np.eye(4, dtype=np.float64) * 1.0

        # State and covariance
        self._x = np.zeros(8, dtype=np.float64)
        self._P = np.eye(8, dtype=np.float64) * 10.0

    def initialize(self, bbox: tuple[float, float, float, float]) -> None:
        """Initialize state from bounding box (x1, y1, x2, y2)."""
        cx, cy, area, ar = self._bbox_to_measurement(bbox)
        self._x[:4] = [cx, cy, area, ar]
        self._x[4:] = 0  # Zero initial velocity
        self._P = np.eye(8, dtype=np.float64) * 10.0
        self._P[4:, 4:] *= 100.0  # High uncertainty for velocity

    def predict(self) -> tuple[float, float, float, float]:
        """Predict next state and return predicted bbox."""
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        # Ensure area and aspect ratio stay positive
        self._x[2] = max(self._x[2], 1.0)
        self._x[3] = max(self._x[3], 0.1)

        return self._state_to_bbox()

    def update(self, bbox: tuple[float, float, float, float]) -> None:
        """Update state with measurement."""
        z = np.array(self._bbox_to_measurement(bbox), dtype=np.float64)

        # Innovation
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R

        # Kalman gain
        K = self._P @ self._H.T @ np.linalg.inv(S)

        # Update
        self._x = self._x + K @ y
        self._P = (np.eye(8) - K @ self._H) @ self._P

        # Ensure area and aspect ratio stay positive
        self._x[2] = max(self._x[2], 1.0)
        self._x[3] = max(self._x[3], 0.1)

    def get_state_bbox(self) -> tuple[float, float, float, float]:
        """Get current state as bbox (x1, y1, x2, y2)."""
        return self._state_to_bbox()

    def _bbox_to_measurement(self, bbox):
        """Convert (x1, y1, x2, y2) to (cx, cy, area, aspect_ratio)."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        area = w * h
        ar = w / max(h, 1e-6)
        return cx, cy, area, ar

    def _state_to_bbox(self):
        """Convert state (cx, cy, area, aspect_ratio) to (x1, y1, x2, y2)."""
        cx, cy, area, ar = self._x[:4]
        area = max(area, 1.0)
        ar = max(ar, 0.1)
        h = np.sqrt(area / ar)
        w = ar * h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return (x1, y1, x2, y2)


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
    identity: str | None = None
    identity_name: str = ""
    identity_confidence: float = 0.0
    embedding: np.ndarray | None = None

    # Tracking counters
    frame_count: int = 0
    lost_count: int = 0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    last_embed_frame: int = 0
    event_published: bool = False

    # Kalman filter (initialized post-creation)
    _kalman: KalmanFilter = field(default_factory=KalmanFilter, repr=False)

    def __post_init__(self):
        """Initialize Kalman filter with initial bbox."""
        self._kalman.initialize(self.bbox)

    @property
    def needs_embedding(self) -> bool:
        """Whether this track needs embedding extraction."""
        if self.state == TrackState.NEW:
            return True
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
        """Bounding box as [x, y, w, h] integers."""
        x1, y1, x2, y2 = self.bbox
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    def predict(self) -> tuple[float, float, float, float]:
        """Kalman predict next position. Updates internal predicted bbox for matching."""
        predicted = self._kalman.predict()
        # Store predicted bbox for IoU matching (not for display)
        self._predicted_bbox = predicted
        return predicted

    @property
    def predicted_bbox(self) -> tuple[float, float, float, float]:
        """Kalman-predicted bbox (used for IoU matching)."""
        return getattr(self, "_predicted_bbox", self.bbox)

    def update_bbox(self, bbox: tuple[float, float, float, float],
                    landmarks: np.ndarray, confidence: float, frame_id: int) -> None:
        """Update track with new detection (Kalman update + state)."""
        self._kalman.update(bbox)
        # Use detector bbox directly for display (smooth but responsive)
        self.bbox = bbox
        self.landmarks = landmarks
        self.confidence = confidence
        self.state = TrackState.ACTIVE
        self.lost_count = 0
        self.frame_count += 1
        self.last_seen_frame = frame_id


# --- IoU Computation ---


def compute_iou(box_a: tuple[float, float, float, float],
                box_b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two bboxes (x1, y1, x2, y2)."""
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
    """Compute IoU matrix between detection bboxes and track predicted bboxes."""
    n_det = len(detections)
    n_trk = len(tracks)
    iou_matrix = np.zeros((n_det, n_trk), dtype=np.float32)

    for i, det_bbox in enumerate(detections):
        for j, track in enumerate(tracks):
            # Use Kalman-predicted bbox for matching (better association)
            iou_matrix[i, j] = compute_iou(det_bbox, track.predicted_bbox)

    return iou_matrix


# --- Hungarian Matching ---


def linear_assignment(cost_matrix: np.ndarray,
                      threshold: float) -> tuple[list[tuple], list[int], list[int]]:
    """
    Solve assignment problem using Hungarian algorithm.

    Args:
        cost_matrix: (N_det, N_trk) cost matrix (1 - IoU)
        threshold: Maximum cost to accept a match

    Returns:
        matches: list of (det_idx, trk_idx) pairs
        unmatched_dets: list of unmatched detection indices
        unmatched_trks: list of unmatched track indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_dets = set(range(cost_matrix.shape[0]))
    unmatched_trks = set(range(cost_matrix.shape[1]))

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] <= threshold:
            matches.append((row, col))
            unmatched_dets.discard(row)
            unmatched_trks.discard(col)

    return matches, list(unmatched_dets), list(unmatched_trks)


# --- ByteTrack Face Tracker ---


class FaceTracker:
    """
    ByteTrack-based face tracker.

    Features:
    - Kalman filter for motion prediction
    - 2-stage matching (high-conf → low-conf)
    - Hungarian algorithm for optimal assignment
    - Track lifecycle management (NEW → ACTIVE → LOST → REMOVED)

    Args:
        high_threshold: Confidence threshold for high-conf detections (default 0.6)
        low_threshold: Confidence threshold for low-conf detections (default 0.3)
        iou_threshold: IoU threshold for matching (default 0.4)
        max_lost: Frames before removing lost track (default 15)
        max_tracks: Maximum simultaneous tracks (default 10)
        reverify_interval: Frames between re-verification (default 15)
    """

    def __init__(self,
                 high_threshold: float = 0.5,
                 low_threshold: float = 0.3,
                 iou_threshold: float = 0.4,
                 max_lost: int = 15,
                 max_tracks: int = 10,
                 reverify_interval: int = 15,
                 # Legacy params (ignored, kept for backward compat)
                 centroid_dist_threshold: float = 150.0):
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold
        self._iou_threshold = iou_threshold
        self._max_lost = max_lost
        self._max_tracks = max_tracks
        self._reverify_interval = reverify_interval
        self._tracks: list[Track] = []
        self._next_id: int = 1

    @property
    def tracks(self) -> list[Track]:
        """All non-removed tracks."""
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
        Update tracker with new detections (ByteTrack algorithm).

        Args:
            detections: List of dicts with keys: bbox, landmarks, confidence
            frame_id: Current frame number

        Returns:
            List of all active Track objects
        """
        # Step 1: Kalman predict for all existing tracks + mark as unmatched
        for track in self._tracks:
            if track.state != TrackState.REMOVED:
                track.predict()
                track.lost_count += 1  # Assume unmatched; reset if matched

        if not detections:
            # No detections → mark all as LOST, cleanup
            for track in self._tracks:
                if track.state in (TrackState.NEW, TrackState.ACTIVE):
                    track.state = TrackState.LOST
            self._cleanup_lost_tracks()
            return self.tracks

        # Step 2: Split detections into high and low confidence
        high_dets = []
        low_dets = []
        high_indices = []
        low_indices = []

        for i, det in enumerate(detections):
            if det["confidence"] >= self._high_threshold:
                high_dets.append(det)
                high_indices.append(i)
            elif det["confidence"] >= self._low_threshold:
                low_dets.append(det)
                low_indices.append(i)

        # Get matchable tracks
        active_trks = [t for t in self._tracks if t.state != TrackState.REMOVED]

        # Step 3: First association — high-conf detections vs all tracks
        matched_global_det_indices = set()
        matched_trk_set = set()

        if high_dets and active_trks:
            det_bboxes = [d["bbox"] for d in high_dets]
            iou_matrix = compute_iou_matrix(det_bboxes, active_trks)
            cost_matrix = 1.0 - iou_matrix

            matches, unmatched_d, unmatched_t = linear_assignment(
                cost_matrix, threshold=1.0 - self._iou_threshold
            )

            for det_idx, trk_idx in matches:
                det = high_dets[det_idx]
                track = active_trks[trk_idx]
                track.update_bbox(det["bbox"], det["landmarks"],
                                  det["confidence"], frame_id)
                matched_trk_set.add(id(track))
                matched_global_det_indices.add(high_indices[det_idx])

            unmatched_trk_list = [active_trks[i] for i in unmatched_t]
        else:
            unmatched_trk_list = list(active_trks)

        # Step 4: Second association — low-conf detections vs unmatched tracks
        if low_dets and unmatched_trk_list:
            det_bboxes = [d["bbox"] for d in low_dets]
            iou_matrix = compute_iou_matrix(det_bboxes, unmatched_trk_list)
            cost_matrix = 1.0 - iou_matrix

            matches2, _, _ = linear_assignment(
                cost_matrix, threshold=1.0 - (self._iou_threshold * 0.7)
            )

            for det_idx, trk_idx in matches2:
                det = low_dets[det_idx]
                track = unmatched_trk_list[trk_idx]
                track.update_bbox(det["bbox"], det["landmarks"],
                                  det["confidence"], frame_id)
                matched_trk_set.add(id(track))
                matched_global_det_indices.add(low_indices[det_idx])

        # Step 5: Create new tracks for unmatched high-conf detections
        new_track_ids = set()
        for i, det in enumerate(high_dets):
            if high_indices[i] not in matched_global_det_indices:
                new_track = self._create_track(det, frame_id)
                if new_track:
                    new_track_ids.add(id(new_track))

        # Step 6: Update states for unmatched tracks (skip newly created)
        for track in self._tracks:
            if track.state == TrackState.REMOVED:
                continue
            if id(track) in matched_trk_set:
                continue
            if id(track) in new_track_ids:
                continue
            # This track was not matched → mark as LOST
            if track.state in (TrackState.NEW, TrackState.ACTIVE):
                track.state = TrackState.LOST

        # Step 7: Cleanup
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
        """Remove tracks exceeding max_lost frames."""
        for track in self._tracks:
            if track.state != TrackState.REMOVED and track.lost_count > self._max_lost:
                track.state = TrackState.REMOVED
                logger.info(
                    "event=track_removed | track_id={tid} | lost_frames={lf} | "
                    "identity={ident} | total_frames={tf}",
                    tid=track.track_id, lf=track.lost_count,
                    ident=track.identity or "unknown", tf=track.frame_count,
                )

        # Purge removed tracks
        self._tracks = [t for t in self._tracks if t.state != TrackState.REMOVED]

    def get_tracks_needing_embedding(self, frame_id: int) -> list[Track]:
        """Get tracks that need embedding extraction."""
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
