"""
Tests for Face Enrollment Tool.

Covers:
- EnrollmentValidator: single face, size, blur, duplicate detection
- FaceEnroller: image enrollment, batch, list, remove
- Edge cases: no face, multiple faces, invalid paths
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from edge.tools.enroll_face import EnrollmentValidator, FaceEnroller
from edge.plugins.face_recognition.database import FaceDatabase
from edge.plugins.face_recognition.detector import FaceInfo


# --- Helpers ---


def make_face_info(x1=100, y1=100, x2=200, y2=250, score=0.95):
    """Create a FaceInfo object."""
    landmarks = np.array([
        130, 150, 170, 150, 150, 180, 135, 210, 165, 210
    ], dtype=np.float32)
    return FaceInfo(x1=x1, y1=y1, x2=x2, y2=y2, score=score, landmarks=landmarks)


def make_clear_frame(w=640, h=480):
    """Create a frame with texture (won't be 'blurry')."""
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return frame


def make_blurry_frame(w=640, h=480):
    """Create a smooth/blurry frame (low Laplacian variance)."""
    frame = np.ones((h, w, 3), dtype=np.uint8) * 128
    # Very smooth gradient - low Laplacian variance
    for y in range(h):
        frame[y, :, :] = int(128 + 20 * np.sin(y / 50))
    return frame


def make_embedding(seed=42, dim=512):
    """Create a normalized embedding."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


# --- Test: EnrollmentValidator ---


class TestEnrollmentValidator:
    """Tests for face validation logic."""

    def test_no_faces_rejected(self):
        """Should reject when no face detected."""
        validator = EnrollmentValidator()
        frame = make_clear_frame()
        valid, msg = validator.validate(frame, [])
        assert valid is False
        assert "No face" in msg

    def test_multiple_faces_rejected(self):
        """Should reject when multiple faces detected."""
        validator = EnrollmentValidator()
        frame = make_clear_frame()
        faces = [make_face_info(0, 0, 100, 100), make_face_info(200, 200, 300, 300)]
        valid, msg = validator.validate(frame, faces)
        assert valid is False
        assert "Multiple" in msg

    def test_small_face_rejected(self):
        """Should reject face smaller than min_face_size."""
        validator = EnrollmentValidator(min_face_size=100)
        frame = make_clear_frame()
        face = make_face_info(100, 100, 150, 150)  # 50x50 px
        valid, msg = validator.validate(frame, [face])
        assert valid is False
        assert "too small" in msg

    def test_large_face_accepted(self):
        """Should accept face larger than min_face_size."""
        validator = EnrollmentValidator(min_face_size=60)
        frame = make_clear_frame()
        face = make_face_info(100, 100, 200, 250)  # 100x150 px
        valid, msg = validator.validate(frame, [face])
        assert valid is True
        assert msg == ""

    def test_blurry_face_rejected(self):
        """Should reject blurry face (low Laplacian variance)."""
        validator = EnrollmentValidator(min_quality=50.0)
        frame = make_blurry_frame()
        face = make_face_info(100, 100, 400, 400)  # large enough
        valid, msg = validator.validate(frame, [face])
        assert valid is False
        assert "blurry" in msg.lower()

    def test_clear_face_accepted(self):
        """Should accept face with good quality."""
        validator = EnrollmentValidator(min_quality=10.0)
        frame = make_clear_frame()
        face = make_face_info(100, 100, 300, 350)
        valid, msg = validator.validate(frame, [face])
        assert valid is True

    def test_duplicate_detection(self, tmp_path):
        """Should detect duplicate embeddings."""
        validator = EnrollmentValidator()
        db = FaceDatabase(tmp_path / "test.db")
        db.initialize()

        emb = make_embedding(1)
        db.add_face("p001", "Alice", emb)

        # Same embedding → duplicate
        is_dup, msg = validator.check_duplicate(emb, db, threshold=0.7)
        assert is_dup is True
        assert "Alice" in msg

        # Different embedding → not duplicate
        diff_emb = make_embedding(999)
        is_dup, msg = validator.check_duplicate(diff_emb, db, threshold=0.7)
        assert is_dup is False

        db.close()

    def test_no_duplicate_empty_db(self, tmp_path):
        """Should not detect duplicate in empty database."""
        validator = EnrollmentValidator()
        db = FaceDatabase(tmp_path / "test.db")
        db.initialize()

        emb = make_embedding(1)
        is_dup, msg = validator.check_duplicate(emb, db, threshold=0.7)
        assert is_dup is False

        db.close()


# --- Test: FaceEnroller ---


class TestFaceEnrollerInit:
    """Tests for FaceEnroller initialization."""

    def test_initialize_without_models(self, tmp_path):
        """Should fail gracefully if models not found."""
        enroller = FaceEnroller(db_path=tmp_path / "test.db")
        # This depends on whether models exist on disk
        # Just ensure it doesn't crash
        result = enroller.initialize()
        assert isinstance(result, bool)
        enroller.close()


class TestFaceEnrollerImageMode:
    """Tests for image enrollment (requires models)."""

    @pytest.fixture
    def enroller(self, tmp_path):
        """Create enroller with real models (skip if not available)."""
        from edge.plugins.face_recognition.detector import SCRFD_MODEL
        from edge.plugins.face_recognition.embedder import EMBEDDING_MODEL
        if not SCRFD_MODEL.exists() or not EMBEDDING_MODEL.exists():
            pytest.skip("Models not found")
        e = FaceEnroller(db_path=tmp_path / "test.db")
        assert e.initialize()
        yield e
        e.close()

    def test_enroll_invalid_path(self, enroller):
        """Should fail for non-existent image."""
        result = enroller.enroll_image("/nonexistent.jpg", "p001", "Test")
        assert result is False

    def test_enroll_blank_image(self, enroller, tmp_path):
        """Should fail for image with no face."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        path = str(tmp_path / "blank.jpg")
        cv2.imwrite(path, blank)
        result = enroller.enroll_image(path, "p001", "Test")
        assert result is False


class TestFaceEnrollerBatch:
    """Tests for batch enrollment."""

    @pytest.fixture
    def enroller(self, tmp_path):
        """Create enroller (skip if models not available)."""
        from edge.plugins.face_recognition.detector import SCRFD_MODEL
        from edge.plugins.face_recognition.embedder import EMBEDDING_MODEL
        if not SCRFD_MODEL.exists() or not EMBEDDING_MODEL.exists():
            pytest.skip("Models not found")
        e = FaceEnroller(db_path=tmp_path / "test.db")
        assert e.initialize()
        yield e
        e.close()

    def test_batch_empty_folder(self, enroller, tmp_path):
        """Should return 0 for empty folder."""
        empty = tmp_path / "empty"
        empty.mkdir()
        count = enroller.enroll_batch(str(empty))
        assert count == 0

    def test_batch_nonexistent_folder(self, enroller):
        """Should return 0 for non-existent folder."""
        count = enroller.enroll_batch("/nonexistent/folder")
        assert count == 0


class TestFaceEnrollerListRemove:
    """Tests for list and remove operations."""

    @pytest.fixture
    def enroller(self, tmp_path):
        """Create enroller (skip if models not available)."""
        from edge.plugins.face_recognition.detector import SCRFD_MODEL
        from edge.plugins.face_recognition.embedder import EMBEDDING_MODEL
        if not SCRFD_MODEL.exists() or not EMBEDDING_MODEL.exists():
            pytest.skip("Models not found")
        e = FaceEnroller(db_path=tmp_path / "test.db")
        assert e.initialize()
        yield e
        e.close()

    def test_list_empty(self, enroller, capsys):
        """List should show empty message."""
        enroller.list_faces()
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_remove_nonexistent(self, enroller, capsys):
        """Remove non-existent person should show not found."""
        enroller.remove_face("nobody")
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()
