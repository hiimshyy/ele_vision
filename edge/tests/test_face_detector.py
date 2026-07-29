"""
Tests for face detection (Python wrapper).

Tests the FaceDetector class which supports NCNN (C++) and OpenCV (fallback).
These tests use OpenCV backend since C++ module requires native build.

Prerequisites:
    Download model: bash edge/inference/download_models.sh
    Or manually download yunet.onnx to edge/plugins/face_recognition/models/
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from edge.plugins.face_recognition.detector import FaceDetector, FaceInfo


# --- Fixtures ---

MODEL_DIR = Path("edge/plugins/face_recognition/models")
YUNET_ONNX = MODEL_DIR / "yunet.onnx"


def has_model():
    """Check if model file exists."""
    return YUNET_ONNX.exists()


@pytest.fixture
def detector():
    """Create and load a face detector (skip if no model)."""
    if not has_model():
        pytest.skip("Model not downloaded. Run: bash edge/inference/download_models.sh")
    det = FaceDetector()
    ok = det.load(YUNET_ONNX, input_width=320, input_height=320)
    assert ok, "Failed to load model"
    return det


def make_blank_frame(w=640, h=480):
    """Create a blank frame (no faces)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_synthetic_face_frame(w=640, h=480):
    """Create a frame with a synthetic face-like pattern (oval + features)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Draw oval face
    center = (w // 2, h // 2)
    cv2.ellipse(frame, center, (80, 100), 0, 0, 360, (200, 180, 160), -1)
    # Eyes
    cv2.circle(frame, (w // 2 - 30, h // 2 - 20), 10, (50, 50, 50), -1)
    cv2.circle(frame, (w // 2 + 30, h // 2 - 20), 10, (50, 50, 50), -1)
    # Mouth
    cv2.ellipse(frame, (w // 2, h // 2 + 40), (25, 10), 0, 0, 360, (100, 50, 50), -1)
    return frame


# --- Test: Detector Loading ---


class TestDetectorLoading:
    """Tests for model loading."""

    def test_load_opencv_backend(self):
        """Should load with OpenCV backend when ONNX file exists."""
        if not has_model():
            pytest.skip("Model not downloaded")

        det = FaceDetector()
        ok = det.load(YUNET_ONNX)
        assert ok
        assert det.backend == "opencv"

    def test_load_nonexistent_model(self):
        """Should return False for non-existent model."""
        det = FaceDetector()
        ok = det.load("/nonexistent/model.onnx")
        assert not ok
        assert det.backend is None

    def test_ncnn_fallback_to_opencv(self):
        """When NCNN not available, should fallback to OpenCV."""
        if not has_model():
            pytest.skip("Model not downloaded")

        det = FaceDetector()
        ok = det.load(YUNET_ONNX)
        assert ok
        # On dev machine without C++ build, should use opencv
        assert det.backend in ("ncnn", "opencv")


# --- Test: Face Detection ---


class TestFaceDetection:
    """Tests for face detection inference."""

    def test_detect_blank_frame(self, detector):
        """Blank frame should produce no faces."""
        frame = make_blank_frame()
        faces = detector.detect(frame, conf_threshold=0.7)
        assert len(faces) == 0

    def test_detect_returns_face_info(self, detector):
        """Detected faces should be FaceInfo objects with correct fields."""
        # Use synthetic face (may or may not detect depending on model)
        frame = make_synthetic_face_frame()
        faces = detector.detect(frame, conf_threshold=0.3)
        # Even if no face detected, verify return type
        assert isinstance(faces, list)
        for face in faces:
            assert isinstance(face, FaceInfo)
            assert 0.0 <= face.score <= 1.0
            assert face.x2 > face.x1
            assert face.y2 > face.y1
            assert face.landmarks.shape == (10,)

    def test_inference_time_tracked(self, detector):
        """Inference time should be tracked after detection."""
        frame = make_blank_frame()
        detector.detect(frame)
        assert detector.inference_time_ms > 0

    def test_different_frame_sizes(self, detector):
        """Detector should handle different input sizes."""
        for size in [(320, 240), (640, 480), (1280, 720), (1920, 1080)]:
            w, h = size
            frame = make_blank_frame(w, h)
            faces = detector.detect(frame, conf_threshold=0.9)
            assert isinstance(faces, list)

    def test_confidence_threshold_filtering(self, detector):
        """Higher threshold should produce fewer or equal detections."""
        frame = make_synthetic_face_frame()
        faces_low = detector.detect(frame, conf_threshold=0.1)
        faces_high = detector.detect(frame, conf_threshold=0.9)
        assert len(faces_high) <= len(faces_low)


# --- Test: FaceInfo dataclass ---


class TestFaceInfo:
    """Tests for FaceInfo dataclass properties."""

    def test_bbox_property(self):
        info = FaceInfo(x1=10, y1=20, x2=100, y2=150, score=0.9)
        assert info.bbox == (10, 20, 100, 150)

    def test_width_height(self):
        info = FaceInfo(x1=10, y1=20, x2=100, y2=150, score=0.9)
        assert info.width == 90
        assert info.height == 130

    def test_area(self):
        info = FaceInfo(x1=0, y1=0, x2=100, y2=100, score=0.9)
        assert info.area == 10000

    def test_landmarks_default(self):
        info = FaceInfo(x1=0, y1=0, x2=100, y2=100, score=0.9)
        assert info.landmarks.shape == (10,)
        assert np.all(info.landmarks == 0)
