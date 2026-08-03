"""
Tests for face embedding (alignment + embedder + cosine similarity).

Tests the alignment utility, FaceEmbedder class, and similarity functions.
Model-dependent tests require w600k_mbf.onnx (skip if not present).

Prerequisites:
    Download model from InsightFace buffalo_s pack:
    Place w600k_mbf.onnx in edge/plugins/face_recognition/models/
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from edge.plugins.face_recognition.alignment import (
    ARCFACE_REF_LANDMARKS,
    align_face,
    estimate_similarity_transform,
)
from edge.plugins.face_recognition.embedder import (
    EMBEDDING_MODEL,
    FaceEmbedder,
    cosine_similarity,
    cosine_similarity_batch,
)


# --- Fixtures ---

MODEL_PATH = Path("edge/plugins/face_recognition/models/w600k_mbf.onnx")


def has_embedding_model():
    """Check if embedding model exists."""
    return MODEL_PATH.exists()


@pytest.fixture
def embedder():
    """Create and load an embedder (skip if no model)."""
    if not has_embedding_model():
        pytest.skip("Embedding model not found. Place w600k_mbf.onnx in models/")
    emb = FaceEmbedder()
    ok = emb.load()
    assert ok, "Failed to load embedding model"
    return emb


def make_synthetic_face(w=200, h=250):
    """Create a synthetic face-like image with known landmark positions."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w // 2, h // 2

    # Draw oval face
    cv2.ellipse(frame, (cx, cy), (60, 80), 0, 0, 360, (200, 180, 160), -1)
    # Eyes
    cv2.circle(frame, (cx - 25, cy - 20), 8, (50, 50, 80), -1)
    cv2.circle(frame, (cx + 25, cy - 20), 8, (50, 50, 80), -1)
    # Nose
    cv2.circle(frame, (cx, cy + 5), 5, (180, 160, 140), -1)
    # Mouth
    cv2.ellipse(frame, (cx, cy + 30), (20, 8), 0, 0, 360, (100, 80, 80), -1)

    # Landmarks matching the drawn features
    landmarks = np.array([
        cx - 25, cy - 20,   # left eye
        cx + 25, cy - 20,   # right eye
        cx, cy + 5,         # nose
        cx - 15, cy + 30,   # mouth left
        cx + 15, cy + 30,   # mouth right
    ], dtype=np.float32)

    return frame, landmarks


def make_random_aligned_face(seed=42):
    """Create a random 112x112 'face' image for embedding tests."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)


# --- Test: Similarity Transform ---


class TestSimilarityTransform:
    """Tests for estimate_similarity_transform."""

    def test_identity_transform(self):
        """Same src and dst should produce identity-like transform."""
        pts = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float64)
        M = estimate_similarity_transform(pts, pts)
        # Should be close to identity
        expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        np.testing.assert_allclose(M, expected, atol=1e-6)

    def test_translation_only(self):
        """Pure translation should be captured correctly."""
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        dst = src + np.array([10, 20])
        M = estimate_similarity_transform(src, dst)
        # Translation should be [10, 20]
        assert abs(M[0, 2] - 10) < 1e-6
        assert abs(M[1, 2] - 20) < 1e-6

    def test_scale_transform(self):
        """Scaling should be captured correctly."""
        src = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        dst = src * 2.0
        M = estimate_similarity_transform(src, dst)
        # Scale factor should be ~2
        scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
        assert abs(scale - 2.0) < 1e-6

    def test_output_shape(self):
        """Output should be 2x3 matrix."""
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        dst = np.array([[0, 0], [2, 0], [0, 2]], dtype=np.float64)
        M = estimate_similarity_transform(src, dst)
        assert M.shape == (2, 3)


# --- Test: Face Alignment ---


class TestFaceAlignment:
    """Tests for align_face function."""

    def test_align_basic(self):
        """Should produce 112x112 aligned output."""
        frame, landmarks = make_synthetic_face()
        aligned = align_face(frame, landmarks)
        assert aligned is not None
        assert aligned.shape == (112, 112, 3)
        assert aligned.dtype == np.uint8

    def test_align_custom_size(self):
        """Should support custom output sizes."""
        frame, landmarks = make_synthetic_face()
        aligned = align_face(frame, landmarks, output_size=224)
        assert aligned is not None
        assert aligned.shape == (224, 224, 3)

    def test_align_5x2_landmarks(self):
        """Should accept (5, 2) shaped landmarks."""
        frame, landmarks = make_synthetic_face()
        lmk_2d = landmarks.reshape(5, 2)
        aligned = align_face(frame, lmk_2d)
        assert aligned is not None
        assert aligned.shape == (112, 112, 3)

    def test_align_none_landmarks(self):
        """Should return None for None landmarks."""
        frame, _ = make_synthetic_face()
        aligned = align_face(frame, None)
        assert aligned is None

    def test_align_zero_landmarks(self):
        """Should return None for all-zero landmarks."""
        frame, _ = make_synthetic_face()
        zeros = np.zeros(10, dtype=np.float32)
        aligned = align_face(frame, zeros)
        assert aligned is None

    def test_align_invalid_landmark_length(self):
        """Should return None for wrong landmark count."""
        frame, _ = make_synthetic_face()
        bad = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        aligned = align_face(frame, bad)
        assert aligned is None

    def test_align_invalid_landmark_shape(self):
        """Should return None for wrong 2D shape."""
        frame, _ = make_synthetic_face()
        bad = np.zeros((3, 2), dtype=np.float32)
        aligned = align_face(frame, bad)
        assert aligned is None

    def test_align_not_all_black(self):
        """Aligned face should contain non-zero pixels (actual content)."""
        frame, landmarks = make_synthetic_face()
        aligned = align_face(frame, landmarks)
        assert aligned is not None
        # Should have some non-zero content
        assert np.sum(aligned) > 0

    def test_arcface_ref_landmarks_shape(self):
        """Reference landmarks should be (5, 2)."""
        assert ARCFACE_REF_LANDMARKS.shape == (5, 2)
        assert ARCFACE_REF_LANDMARKS.dtype == np.float32


# --- Test: Cosine Similarity ---


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        v = np.array([1.0, 2.0, 3.0, 4.0])
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        v = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(v, -v)
        assert abs(sim - (-1.0)) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sim = cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_zero_vector(self):
        """Zero vector should return 0.0 (not NaN)."""
        a = np.array([1.0, 2.0, 3.0])
        z = np.zeros(3)
        sim = cosine_similarity(a, z)
        assert sim == 0.0

    def test_normalized_vectors(self):
        """Pre-normalized vectors should work correctly."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.707, 0.707, 0.0])
        sim = cosine_similarity(a, b)
        assert abs(sim - 0.707) < 0.01

    def test_high_dim_similar(self):
        """Similar 512-dim vectors should have high similarity."""
        rng = np.random.default_rng(42)
        base = rng.random(512)
        noise = rng.random(512) * 0.01
        sim = cosine_similarity(base, base + noise)
        assert sim > 0.99

    def test_high_dim_different(self):
        """Random 512-dim vectors should have low similarity."""
        rng = np.random.default_rng(42)
        # Use centered distribution [-1, 1] to avoid positive bias
        a = rng.standard_normal(512)
        b = rng.standard_normal(512)
        sim = cosine_similarity(a, b)
        # Random normal vectors in high dim tend toward 0
        assert abs(sim) < 0.2


# --- Test: Cosine Similarity Batch ---


class TestCosineSimilarityBatch:
    """Tests for cosine_similarity_batch function."""

    def test_batch_single(self):
        """Single gallery entry should match scalar result."""
        query = np.array([1.0, 2.0, 3.0])
        gallery = np.array([[1.0, 2.0, 3.0]])
        result = cosine_similarity_batch(query, gallery)
        expected = cosine_similarity(query, gallery[0])
        assert abs(result[0] - expected) < 1e-6

    def test_batch_multiple(self):
        """Multiple gallery entries should return correct shape."""
        rng = np.random.default_rng(42)
        query = rng.random(512)
        gallery = rng.random((10, 512))
        result = cosine_similarity_batch(query, gallery)
        assert result.shape == (10,)

    def test_batch_consistency(self):
        """Batch results should match individual computations."""
        rng = np.random.default_rng(42)
        query = rng.random(128)
        gallery = rng.random((5, 128))
        batch_result = cosine_similarity_batch(query, gallery)
        for i in range(5):
            individual = cosine_similarity(query, gallery[i])
            assert abs(batch_result[i] - individual) < 1e-6

    def test_batch_zero_query(self):
        """Zero query should return all zeros."""
        query = np.zeros(128)
        gallery = np.random.default_rng(42).random((5, 128))
        result = cosine_similarity_batch(query, gallery)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_batch_finds_best_match(self):
        """Best match should be the identical vector."""
        rng = np.random.default_rng(42)
        query = rng.random(256)
        gallery = rng.random((10, 256))
        gallery[7] = query  # Insert exact match at index 7
        result = cosine_similarity_batch(query, gallery)
        assert np.argmax(result) == 7
        assert abs(result[7] - 1.0) < 1e-6


# --- Test: FaceEmbedder Loading ---


class TestEmbedderLoading:
    """Tests for FaceEmbedder model loading."""

    def test_load_nonexistent_model(self):
        """Should return False for non-existent model."""
        emb = FaceEmbedder()
        ok = emb.load("/nonexistent/model.onnx")
        assert not ok
        assert not emb.is_loaded

    def test_default_properties(self):
        """Default properties before loading."""
        emb = FaceEmbedder()
        assert emb.model_name == ""
        assert emb.inference_time_ms == 0.0
        assert emb.embedding_dim == 512
        assert emb.input_size == 112
        assert not emb.is_loaded

    def test_load_model(self):
        """Should load model successfully (skip if not present)."""
        if not has_embedding_model():
            pytest.skip("Embedding model not found")
        emb = FaceEmbedder()
        ok = emb.load()
        assert ok
        assert emb.is_loaded
        assert emb.model_name == "w600k_mbf.onnx"
        assert emb.embedding_dim == 512

    def test_extract_without_load(self):
        """Should return None if model not loaded."""
        emb = FaceEmbedder()
        face = make_random_aligned_face()
        result = emb.extract(face)
        assert result is None


# --- Test: FaceEmbedder Inference ---


class TestEmbedderInference:
    """Tests for embedding extraction (requires model)."""

    def test_extract_basic(self, embedder):
        """Should produce normalized 512-dim vector."""
        face = make_random_aligned_face(seed=1)
        embedding = embedder.extract(face)
        assert embedding is not None
        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32
        # Should be L2-normalized (norm ≈ 1.0)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5

    def test_extract_deterministic(self, embedder):
        """Same input should produce same output."""
        face = make_random_aligned_face(seed=2)
        emb1 = embedder.extract(face)
        emb2 = embedder.extract(face)
        assert emb1 is not None and emb2 is not None
        np.testing.assert_allclose(emb1, emb2, atol=1e-6)

    def test_extract_different_faces(self, embedder):
        """Different inputs should produce different embeddings."""
        face1 = make_random_aligned_face(seed=10)
        face2 = make_random_aligned_face(seed=20)
        emb1 = embedder.extract(face1)
        emb2 = embedder.extract(face2)
        assert emb1 is not None and emb2 is not None
        # Should not be identical
        assert not np.allclose(emb1, emb2, atol=1e-3)

    def test_extract_none_input(self, embedder):
        """Should return None for None input."""
        result = embedder.extract(None)
        assert result is None

    def test_extract_invalid_shape(self, embedder):
        """Should return None for invalid input shape."""
        bad = np.zeros((50, 50), dtype=np.uint8)  # 2D, not 3D
        result = embedder.extract(bad)
        assert result is None

    def test_extract_auto_resize(self, embedder):
        """Should handle non-112x112 input by resizing."""
        face = make_random_aligned_face(seed=3)
        # Resize to different size
        big_face = cv2.resize(face, (224, 224))
        emb_big = embedder.extract(big_face)
        assert emb_big is not None
        assert emb_big.shape == (512,)

    def test_inference_time_tracked(self, embedder):
        """Inference time should be tracked."""
        face = make_random_aligned_face(seed=4)
        embedder.extract(face)
        assert embedder.inference_time_ms > 0

    def test_extract_batch(self, embedder):
        """Batch extraction should work."""
        faces = [make_random_aligned_face(seed=i) for i in range(3)]
        embeddings = embedder.extract_batch(faces)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb is not None
            assert emb.shape == (512,)

    def test_extract_batch_with_none(self, embedder):
        """Batch with None entries should handle gracefully."""
        faces = [make_random_aligned_face(seed=1), None, make_random_aligned_face(seed=2)]
        embeddings = embedder.extract_batch(faces)
        assert len(embeddings) == 3
        assert embeddings[0] is not None
        assert embeddings[1] is None
        assert embeddings[2] is not None


# --- Test: End-to-End Alignment + Embedding ---


class TestAlignmentEmbeddingPipeline:
    """Integration tests: alignment → embedding (requires model)."""

    def test_full_pipeline(self, embedder):
        """Align + extract should work end-to-end."""
        frame, landmarks = make_synthetic_face()
        aligned = align_face(frame, landmarks)
        assert aligned is not None

        embedding = embedder.extract(aligned)
        assert embedding is not None
        assert embedding.shape == (512,)
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5

    def test_same_face_consistent_embedding(self, embedder):
        """Same face aligned from same frame should give consistent embedding."""
        frame, landmarks = make_synthetic_face()
        aligned1 = align_face(frame, landmarks)
        aligned2 = align_face(frame, landmarks)

        emb1 = embedder.extract(aligned1)
        emb2 = embedder.extract(aligned2)
        assert emb1 is not None and emb2 is not None

        sim = cosine_similarity(emb1, emb2)
        assert sim > 0.999  # Same input → same output
