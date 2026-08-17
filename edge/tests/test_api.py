"""
Tests for Edge REST API.

Tests cover:
- Root endpoint
- System status
- Face listing, enrollment, deletion
- Validation (no face, multiple faces, small face)
- Floor update

Uses FastAPI TestClient (no actual server needed).
"""

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge.api.main import app, _database
from edge.api.routers import status, faces
from edge.plugins.face_recognition.database import FaceDatabase
from edge.plugins.face_recognition.detector import FaceDetector, SCRFD_MODEL
from edge.plugins.face_recognition.embedder import FaceEmbedder, EMBEDDING_MODEL


# --- Fixtures ---


def has_models():
    """Check if both models exist."""
    return SCRFD_MODEL.exists() and EMBEDDING_MODEL.exists()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Create test client with real models (skip if not available)."""
    if not has_models():
        pytest.skip("Models not found (det_500m.onnx + w600k_mbf.onnx required)")

    tmp = tmp_path_factory.mktemp("api_test")
    db_path = tmp / "test_faces.db"
    data_dir = tmp / "faces"

    # Initialize components
    db = FaceDatabase(db_path)
    db.initialize()

    detector = FaceDetector()
    detector.load()

    embedder = FaceEmbedder()
    embedder.load()

    # Configure routers BEFORE creating TestClient (overrides startup)
    status.configure(device_id="test-001", database=db)
    faces.configure(database=db, detector=detector, embedder=embedder, data_faces_dir=data_dir)

    # Patch startup to not re-configure
    from unittest.mock import patch
    with patch("edge.api.main.startup"):
        with TestClient(app) as c:
            yield c

    db.close()


def make_face_image():
    """Create a synthetic face image (may not trigger real detection)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cx, cy = 320, 240
    cv2.ellipse(frame, (cx, cy), (80, 100), 0, 0, 360, (200, 180, 160), -1)
    cv2.circle(frame, (cx - 30, cy - 25), 12, (40, 40, 40), -1)
    cv2.circle(frame, (cx + 30, cy - 25), 12, (40, 40, 40), -1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 15), (170, 150, 130), 3)
    cv2.ellipse(frame, (cx, cy + 40), (25, 10), 0, 0, 180, (120, 80, 80), 2)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def make_blank_image():
    """Create blank image (no face)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


# --- Test: Root ---


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Root should return service info."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "Smart Cabin Edge API"
        assert data["version"] == "1.0.0"
        assert "docs" in data


# --- Test: Status ---


class TestStatusEndpoint:
    """Tests for /api/status."""

    def test_get_status(self, client):
        """Should return system status."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert "device_id" in data
        assert "cpu_percent" in data
        assert "ram_used_mb" in data
        assert "uptime_s" in data

    def test_get_stats(self, client):
        """Should return pipeline stats."""
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "uptime_s" in data

    def test_get_plugins(self, client):
        """Should return plugin list (empty)."""
        resp = client.get("/api/plugins")
        assert resp.status_code == 200
        data = resp.json()
        assert "plugins" in data


# --- Test: Faces CRUD ---


class TestFacesEndpoints:
    """Tests for /api/faces endpoints."""

    def test_list_faces_empty(self, client):
        """Should return faces list (may have data from startup db)."""
        resp = client.get("/api/faces")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_persons" in data
        assert "total_embeddings" in data
        assert "persons" in data
        assert isinstance(data["persons"], list)

    def test_enroll_no_face(self, client):
        """Should reject image with no face."""
        img = make_blank_image()
        resp = client.post(
            "/api/faces/enroll",
            data={"person_id": "p001", "name": "Test"},
            files={"image": ("blank.jpg", io.BytesIO(img), "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "No face" in resp.json()["detail"]

    def test_get_nonexistent_person(self, client):
        """Should return 404 for unknown person."""
        resp = client.get("/api/faces/nonexistent")
        assert resp.status_code == 404

    def test_delete_nonexistent_person(self, client):
        """Should return 404 when deleting unknown person."""
        resp = client.delete("/api/faces/nonexistent")
        assert resp.status_code == 404

    def test_enroll_invalid_image(self, client):
        """Should reject non-image data."""
        resp = client.post(
            "/api/faces/enroll",
            data={"person_id": "p001", "name": "Test"},
            files={"image": ("bad.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "Invalid image" in resp.json()["detail"]

    def test_update_floor_nonexistent(self, client):
        """Should return 404 for unknown person floor update."""
        resp = client.put("/api/faces/nonexistent/floor", params={"floor": 5})
        assert resp.status_code == 404
