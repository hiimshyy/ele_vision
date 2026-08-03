"""
Tests for the face embedding database (SQLite).

Tests cover:
- Database initialization
- CRUD operations (add, remove, get, count)
- find_match with cosine similarity
- Multiple embeddings per person
- Edge cases (empty db, zero vectors)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from edge.plugins.face_recognition.database import FaceDatabase, FaceRecord, MatchResult


# --- Fixtures ---


@pytest.fixture
def db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_faces.db"
    database = FaceDatabase(db_path)
    assert database.initialize()
    yield database
    database.close()


def make_embedding(seed=42, dim=512):
    """Create a random L2-normalized embedding."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


# --- Test: Initialization ---


class TestDatabaseInit:
    """Tests for database initialization."""

    def test_initialize_creates_file(self, tmp_path):
        """Should create database file."""
        db_path = tmp_path / "new.db"
        db = FaceDatabase(db_path)
        assert db.initialize()
        assert db_path.exists()
        db.close()

    def test_initialize_creates_parent_dirs(self, tmp_path):
        """Should create parent directories if needed."""
        db_path = tmp_path / "subdir" / "nested" / "faces.db"
        db = FaceDatabase(db_path)
        assert db.initialize()
        assert db_path.exists()
        db.close()

    def test_is_initialized(self, tmp_path):
        """is_initialized should reflect state."""
        db_path = tmp_path / "test.db"
        db = FaceDatabase(db_path)
        assert not db.is_initialized
        db.initialize()
        assert db.is_initialized
        db.close()
        assert not db.is_initialized

    def test_reinitialize_preserves_data(self, tmp_path):
        """Closing and reopening should preserve data."""
        db_path = tmp_path / "persist.db"
        db = FaceDatabase(db_path)
        db.initialize()
        db.add_face("p001", "Alice", make_embedding(1))
        db.close()

        # Reopen
        db2 = FaceDatabase(db_path)
        db2.initialize()
        assert db2.count() == 1
        records = db2.get_all()
        assert records[0].person_id == "p001"
        db2.close()


# --- Test: CRUD Operations ---


class TestDatabaseCRUD:
    """Tests for add, remove, get operations."""

    def test_add_face(self, db):
        """Should add a face record."""
        emb = make_embedding(1)
        row_id = db.add_face("p001", "Alice", emb)
        assert row_id > 0
        assert db.count() == 1

    def test_add_multiple_faces(self, db):
        """Should add multiple face records."""
        db.add_face("p001", "Alice", make_embedding(1))
        db.add_face("p002", "Bob", make_embedding(2))
        db.add_face("p003", "Charlie", make_embedding(3))
        assert db.count() == 3
        assert db.count_persons() == 3

    def test_add_multiple_embeddings_per_person(self, db):
        """Should support multiple embeddings for same person."""
        db.add_face("p001", "Alice", make_embedding(1))
        db.add_face("p001", "Alice", make_embedding(2))
        db.add_face("p001", "Alice", make_embedding(3))
        assert db.count() == 3
        assert db.count_persons() == 1

    def test_remove_face(self, db):
        """Should remove all embeddings for a person."""
        db.add_face("p001", "Alice", make_embedding(1))
        db.add_face("p001", "Alice", make_embedding(2))
        db.add_face("p002", "Bob", make_embedding(3))
        removed = db.remove_face("p001")
        assert removed == 2
        assert db.count() == 1
        assert db.count_persons() == 1

    def test_remove_nonexistent(self, db):
        """Removing non-existent person should return 0."""
        removed = db.remove_face("nobody")
        assert removed == 0

    def test_get_all(self, db):
        """get_all should return all records."""
        db.add_face("p001", "Alice", make_embedding(1))
        db.add_face("p002", "Bob", make_embedding(2))
        records = db.get_all()
        assert len(records) == 2
        assert all(isinstance(r, FaceRecord) for r in records)

    def test_get_person(self, db):
        """get_person should return only that person's records."""
        db.add_face("p001", "Alice", make_embedding(1))
        db.add_face("p001", "Alice", make_embedding(2))
        db.add_face("p002", "Bob", make_embedding(3))
        records = db.get_person("p001")
        assert len(records) == 2
        assert all(r.person_id == "p001" for r in records)

    def test_get_person_nonexistent(self, db):
        """get_person for non-existent person should return empty list."""
        records = db.get_person("nobody")
        assert records == []

    def test_face_record_properties(self, db):
        """FaceRecord should have correct properties."""
        emb = make_embedding(1)
        db.add_face("p001", "Alice", emb)
        records = db.get_all()
        r = records[0]
        assert r.person_id == "p001"
        assert r.name == "Alice"
        assert r.embedding_dim == 512
        assert r.created_at > 0
        assert r.updated_at > 0
        np.testing.assert_allclose(r.embedding, emb, atol=1e-6)


# --- Test: find_match ---


class TestDatabaseMatching:
    """Tests for find_match (cosine similarity matching)."""

    def test_exact_match(self, db):
        """Same embedding should match with similarity ~1.0."""
        emb = make_embedding(1)
        db.add_face("p001", "Alice", emb)
        match = db.find_match(emb, threshold=0.4)
        assert match is not None
        assert match.person_id == "p001"
        assert match.name == "Alice"
        assert match.similarity > 0.99

    def test_similar_match(self, db):
        """Similar embedding should match above threshold."""
        emb1 = make_embedding(1)
        noise = np.random.default_rng(99).standard_normal(512).astype(np.float32) * 0.1
        emb2 = emb1 + noise
        emb2 /= np.linalg.norm(emb2)

        db.add_face("p001", "Alice", emb1)
        match = db.find_match(emb2, threshold=0.4)
        assert match is not None
        assert match.person_id == "p001"
        assert match.similarity > 0.4

    def test_no_match_below_threshold(self, db):
        """Very different embedding should not match."""
        db.add_face("p001", "Alice", make_embedding(1))
        # Completely different vector
        query = make_embedding(999)
        match = db.find_match(query, threshold=0.4)
        # Random normal vectors in high dim have ~0 cosine similarity
        # So match should be None or below threshold
        if match is not None:
            assert match.similarity < 0.4

    def test_find_match_empty_db(self, db):
        """Empty database should return None."""
        match = db.find_match(make_embedding(1), threshold=0.4)
        assert match is None

    def test_find_best_match_multiple_persons(self, db):
        """Should return the best matching person."""
        emb_alice = make_embedding(1)
        emb_bob = make_embedding(2)
        db.add_face("p001", "Alice", emb_alice)
        db.add_face("p002", "Bob", emb_bob)

        # Query very close to Alice
        noise = np.random.default_rng(77).standard_normal(512).astype(np.float32) * 0.05
        query = emb_alice + noise
        query /= np.linalg.norm(query)

        match = db.find_match(query, threshold=0.3)
        assert match is not None
        assert match.person_id == "p001"

    def test_find_match_zero_query(self, db):
        """Zero vector query should return None."""
        db.add_face("p001", "Alice", make_embedding(1))
        match = db.find_match(np.zeros(512, dtype=np.float32), threshold=0.4)
        assert match is None

    def test_find_match_returns_match_result(self, db):
        """Match result should have correct type and fields."""
        emb = make_embedding(1)
        db.add_face("p001", "Alice", emb)
        match = db.find_match(emb, threshold=0.3)
        assert isinstance(match, MatchResult)
        assert hasattr(match, "person_id")
        assert hasattr(match, "name")
        assert hasattr(match, "similarity")
        assert hasattr(match, "embedding")
