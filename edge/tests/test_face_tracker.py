"""
Tests for the lightweight IoU face tracker.

Tests cover:
- IoU computation
- Track creation and lifecycle
- IoU matching across frames
- Stale track cleanup
- Embedding trigger logic
- Re-verification interval
"""

import numpy as np
import pytest

from edge.plugins.face_recognition.tracker import (
    FaceTracker,
    Track,
    TrackState,
    compute_iou,
    compute_iou_matrix,
)


# --- Helpers ---


def make_detection(x1, y1, x2, y2, confidence=0.9):
    """Create a detection dict."""
    return {
        "bbox": (x1, y1, x2, y2),
        "landmarks": np.zeros(10, dtype=np.float32),
        "confidence": confidence,
    }


# --- Test: IoU Computation ---


class TestComputeIoU:
    """Tests for compute_iou function."""

    def test_identical_boxes(self):
        """Identical boxes should have IoU = 1.0."""
        box = (10, 20, 100, 150)
        assert abs(compute_iou(box, box) - 1.0) < 1e-6

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        a = (0, 0, 50, 50)
        b = (100, 100, 200, 200)
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        """Partially overlapping boxes should have 0 < IoU < 1."""
        a = (0, 0, 100, 100)
        b = (50, 50, 150, 150)
        iou = compute_iou(a, b)
        # Intersection: 50*50=2500, Union: 10000+10000-2500=17500
        assert abs(iou - 2500 / 17500) < 1e-5

    def test_contained_box(self):
        """Smaller box inside larger should have IoU < 1."""
        outer = (0, 0, 200, 200)
        inner = (50, 50, 100, 100)
        iou = compute_iou(outer, inner)
        # Intersection = inner area = 2500, Union = 40000
        assert abs(iou - 2500 / 40000) < 1e-5

    def test_zero_area_box(self):
        """Zero area box should return IoU = 0."""
        a = (10, 10, 10, 10)  # zero width/height
        b = (0, 0, 100, 100)
        assert compute_iou(a, b) == 0.0

    def test_touching_edges(self):
        """Boxes sharing only an edge should have IoU = 0."""
        a = (0, 0, 50, 50)
        b = (50, 0, 100, 50)
        assert compute_iou(a, b) == 0.0


class TestComputeIoUMatrix:
    """Tests for compute_iou_matrix function."""

    def test_matrix_shape(self):
        """Matrix should have shape (n_detections, n_tracks)."""
        dets = [(0, 0, 50, 50), (100, 100, 200, 200)]
        tracks = [
            Track(track_id=1, bbox=(0, 0, 50, 50), landmarks=np.zeros(10), confidence=0.9),
            Track(track_id=2, bbox=(100, 100, 200, 200), landmarks=np.zeros(10), confidence=0.9),
            Track(track_id=3, bbox=(300, 300, 400, 400), landmarks=np.zeros(10), confidence=0.9),
        ]
        matrix = compute_iou_matrix(dets, tracks)
        assert matrix.shape == (2, 3)

    def test_matrix_values(self):
        """Matching detections and tracks should have high IoU."""
        dets = [(0, 0, 50, 50), (100, 100, 200, 200)]
        tracks = [
            Track(track_id=1, bbox=(0, 0, 50, 50), landmarks=np.zeros(10), confidence=0.9),
            Track(track_id=2, bbox=(100, 100, 200, 200), landmarks=np.zeros(10), confidence=0.9),
        ]
        matrix = compute_iou_matrix(dets, tracks)
        assert matrix[0, 0] == 1.0  # det 0 matches track 0
        assert matrix[1, 1] == 1.0  # det 1 matches track 1
        assert matrix[0, 1] == 0.0  # det 0 doesn't match track 1
        assert matrix[1, 0] == 0.0  # det 1 doesn't match track 0


# --- Test: Track Dataclass ---


class TestTrack:
    """Tests for Track dataclass."""

    def test_needs_embedding_new(self):
        """NEW track should need embedding."""
        track = Track(track_id=1, bbox=(0, 0, 50, 50),
                      landmarks=np.zeros(10), confidence=0.9, state=TrackState.NEW)
        assert track.needs_embedding is True

    def test_needs_embedding_active_identified(self):
        """ACTIVE track with identity should NOT need embedding."""
        track = Track(track_id=1, bbox=(0, 0, 50, 50),
                      landmarks=np.zeros(10), confidence=0.9, state=TrackState.ACTIVE)
        track.identity = "person_001"
        track.embedding = np.ones(512)
        assert track.needs_embedding is False

    def test_needs_embedding_active_unidentified(self):
        """ACTIVE track without identity/embedding should need embedding."""
        track = Track(track_id=1, bbox=(0, 0, 50, 50),
                      landmarks=np.zeros(10), confidence=0.9, state=TrackState.ACTIVE)
        assert track.needs_embedding is True

    def test_needs_reverify(self):
        """Track should need re-verify after interval."""
        track = Track(track_id=1, bbox=(0, 0, 50, 50),
                      landmarks=np.zeros(10), confidence=0.9, state=TrackState.ACTIVE)
        track.last_embed_frame = 10
        assert track.needs_reverify(current_frame=25, interval=15) is True
        assert track.needs_reverify(current_frame=20, interval=15) is False

    def test_set_embedding(self):
        """set_embedding should update state from NEW to ACTIVE."""
        track = Track(track_id=1, bbox=(0, 0, 50, 50),
                      landmarks=np.zeros(10), confidence=0.9, state=TrackState.NEW)
        emb = np.random.randn(512).astype(np.float32)
        track.set_embedding(emb, identity="p001", identity_name="John",
                            confidence=0.85, frame_id=5)
        assert track.state == TrackState.ACTIVE
        assert track.identity == "p001"
        assert track.identity_name == "John"
        assert track.identity_confidence == 0.85
        assert track.last_embed_frame == 5
        assert np.array_equal(track.embedding, emb)

    def test_bbox_xywh(self):
        """bbox_xywh should convert (x1,y1,x2,y2) to [x,y,w,h]."""
        track = Track(track_id=1, bbox=(10, 20, 110, 170),
                      landmarks=np.zeros(10), confidence=0.9)
        assert track.bbox_xywh == [10, 20, 100, 150]


# --- Test: FaceTracker ---


class TestFaceTrackerCreation:
    """Tests for FaceTracker initialization and track creation."""

    def test_empty_tracker(self):
        """New tracker should have no tracks."""
        tracker = FaceTracker()
        assert tracker.track_count == 0
        assert tracker.tracks == []

    def test_single_detection_creates_track(self):
        """A single detection should create one track."""
        tracker = FaceTracker()
        dets = [make_detection(10, 20, 100, 150)]
        tracks = tracker.update(dets, frame_id=1)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1
        assert tracks[0].state == TrackState.NEW
        assert tracks[0].bbox == (10, 20, 100, 150)

    def test_multiple_detections_create_tracks(self):
        """Multiple detections should create multiple tracks."""
        tracker = FaceTracker()
        dets = [
            make_detection(0, 0, 50, 50),
            make_detection(200, 200, 300, 300),
        ]
        tracks = tracker.update(dets, frame_id=1)
        assert len(tracks) == 2

    def test_max_tracks_limit(self):
        """Should not exceed max_tracks."""
        tracker = FaceTracker(max_tracks=2)
        dets = [
            make_detection(0, 0, 50, 50),
            make_detection(100, 100, 150, 150),
            make_detection(200, 200, 250, 250),
        ]
        tracks = tracker.update(dets, frame_id=1)
        assert len(tracks) == 2


class TestFaceTrackerMatching:
    """Tests for IoU matching across frames."""

    def test_same_position_matches(self):
        """Detection at same position should match existing track."""
        tracker = FaceTracker(iou_threshold=0.4)
        # Frame 1
        dets1 = [make_detection(100, 100, 200, 200)]
        tracks1 = tracker.update(dets1, frame_id=1)
        tid = tracks1[0].track_id

        # Frame 2 (same position)
        dets2 = [make_detection(100, 100, 200, 200)]
        tracks2 = tracker.update(dets2, frame_id=2)
        assert len(tracks2) == 1
        assert tracks2[0].track_id == tid  # Same track

    def test_slight_movement_matches(self):
        """Small movement should still match (IoU > threshold)."""
        tracker = FaceTracker(iou_threshold=0.4)
        # Frame 1
        dets1 = [make_detection(100, 100, 200, 200)]
        tracker.update(dets1, frame_id=1)

        # Frame 2 (shifted 10px)
        dets2 = [make_detection(110, 110, 210, 210)]
        tracks2 = tracker.update(dets2, frame_id=2)
        assert len(tracks2) == 1
        assert tracks2[0].track_id == 1

    def test_large_movement_creates_new_track(self):
        """Large movement (low IoU) should create a new track."""
        tracker = FaceTracker(iou_threshold=0.4)
        # Frame 1
        dets1 = [make_detection(0, 0, 50, 50)]
        tracker.update(dets1, frame_id=1)

        # Frame 2 (completely different position)
        dets2 = [make_detection(300, 300, 400, 400)]
        tracks2 = tracker.update(dets2, frame_id=2)
        # Old track goes lost, new track created
        assert any(t.track_id == 2 for t in tracks2)

    def test_multiple_tracks_correct_matching(self):
        """Multiple tracks should match to correct detections."""
        tracker = FaceTracker(iou_threshold=0.4)
        # Frame 1: two faces
        dets1 = [
            make_detection(0, 0, 100, 100),
            make_detection(200, 200, 300, 300),
        ]
        tracks1 = tracker.update(dets1, frame_id=1)
        id_a = tracks1[0].track_id
        id_b = tracks1[1].track_id

        # Frame 2: same positions (slightly moved)
        dets2 = [
            make_detection(5, 5, 105, 105),
            make_detection(205, 205, 305, 305),
        ]
        tracks2 = tracker.update(dets2, frame_id=2)
        assert len(tracks2) == 2
        ids = {t.track_id for t in tracks2}
        assert id_a in ids
        assert id_b in ids

    def test_frame_count_increments(self):
        """frame_count should increment on each match."""
        tracker = FaceTracker()
        dets = [make_detection(10, 10, 100, 100)]
        tracker.update(dets, frame_id=1)
        tracker.update(dets, frame_id=2)
        tracks = tracker.update(dets, frame_id=3)
        assert tracks[0].frame_count == 2  # Matched 2 times after creation


class TestFaceTrackerLostTracks:
    """Tests for lost track handling."""

    def test_no_detection_increments_lost(self):
        """Empty detections should increment lost_count."""
        tracker = FaceTracker(max_lost=5)
        dets = [make_detection(10, 10, 100, 100)]
        tracker.update(dets, frame_id=1)

        # No detections
        tracks = tracker.update([], frame_id=2)
        assert tracks[0].state == TrackState.LOST
        assert tracks[0].lost_count == 1

    def test_track_removed_after_max_lost(self):
        """Track should be removed after max_lost frames without match."""
        tracker = FaceTracker(max_lost=3)
        dets = [make_detection(10, 10, 100, 100)]
        tracker.update(dets, frame_id=1)

        # Miss 4 frames (> max_lost=3)
        for i in range(4):
            tracker.update([], frame_id=2 + i)

        assert tracker.track_count == 0

    def test_recovered_track_resets_lost(self):
        """Re-matched track should reset lost_count."""
        tracker = FaceTracker(max_lost=5)
        dets = [make_detection(10, 10, 100, 100)]
        tracker.update(dets, frame_id=1)

        # Miss 2 frames
        tracker.update([], frame_id=2)
        tracker.update([], frame_id=3)

        # Re-appear
        tracks = tracker.update(dets, frame_id=4)
        assert tracks[0].lost_count == 0
        assert tracks[0].state == TrackState.ACTIVE


class TestFaceTrackerEmbeddingLogic:
    """Tests for embedding trigger logic."""

    def test_new_track_needs_embedding(self):
        """New tracks should need embedding."""
        tracker = FaceTracker()
        dets = [make_detection(10, 10, 100, 100)]
        tracker.update(dets, frame_id=1)
        to_embed = tracker.get_tracks_needing_embedding(frame_id=1)
        assert len(to_embed) == 1

    def test_identified_track_no_embedding(self):
        """Identified track should NOT need embedding (until reverify)."""
        tracker = FaceTracker(reverify_interval=15)
        dets = [make_detection(10, 10, 100, 100)]
        tracker.update(dets, frame_id=1)

        # Simulate identification
        track = tracker.tracks[0]
        track.set_embedding(
            np.ones(512, dtype=np.float32),
            identity="p001", identity_name="Test",
            confidence=0.8, frame_id=1,
        )

        # Frame 5: should NOT need embedding (only 4 frames since embed)
        tracker.update(dets, frame_id=5)
        to_embed = tracker.get_tracks_needing_embedding(frame_id=5)
        assert len(to_embed) == 0

    def test_reverify_interval_triggers(self):
        """Track should need re-embedding after reverify_interval."""
        tracker = FaceTracker(reverify_interval=10)
        dets = [make_detection(10, 10, 100, 100)]
        tracker.update(dets, frame_id=1)

        # Identify
        track = tracker.tracks[0]
        track.set_embedding(
            np.ones(512, dtype=np.float32),
            identity="p001", identity_name="Test",
            confidence=0.8, frame_id=1,
        )

        # Frame 12: should need re-verify (11 frames since embed, > interval=10)
        tracker.update(dets, frame_id=12)
        to_embed = tracker.get_tracks_needing_embedding(frame_id=12)
        assert len(to_embed) == 1
        assert to_embed[0].track_id == track.track_id


class TestFaceTrackerReset:
    """Tests for tracker reset."""

    def test_reset_clears_all(self):
        """Reset should clear all tracks."""
        tracker = FaceTracker()
        dets = [make_detection(10, 10, 100, 100)]
        tracker.update(dets, frame_id=1)
        assert tracker.track_count == 1

        tracker.reset()
        assert tracker.track_count == 0
        assert tracker.tracks == []
