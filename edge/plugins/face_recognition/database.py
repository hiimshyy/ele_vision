"""
Smart Cabin - Face Database (SQLite)

Local face embedding database for recognition matching.
Stores person info + embedding vectors in SQLite.

Usage:
    db = FaceDatabase("faces.db")
    db.initialize()
    db.add_face("person_001", "Nguyen Van A", embedding_vector)
    match = db.find_match(query_embedding, threshold=0.4)
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edge.core.logging_setup import get_logger

logger = get_logger("plugin")


@dataclass
class FaceRecord:
    """A registered face in the database."""

    person_id: str
    name: str
    embedding: np.ndarray  # (512,) float32
    created_at: float      # unix timestamp
    updated_at: float      # unix timestamp
    default_floor: int | None = None  # Default floor for elevator control

    @property
    def embedding_dim(self) -> int:
        return self.embedding.shape[0]


@dataclass
class MatchResult:
    """Result from a face matching query."""

    person_id: str
    name: str
    similarity: float
    embedding: np.ndarray


class FaceDatabase:
    """
    SQLite-backed face embedding database.

    Stores face embeddings as BLOBs. Supports multiple embeddings per person
    (uses highest similarity match).

    Args:
        db_path: Path to SQLite database file (created if not exists)
    """

    def __init__(self, db_path: str | Path = "faces.db"):
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def is_initialized(self) -> bool:
        return self._conn is not None

    def initialize(self) -> bool:
        """
        Initialize database (create tables if not exist).

        Returns:
            True if successful
        """
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._create_tables()
            logger.info("event=database_initialized | path={p}", p=str(self._db_path))
            return True
        except Exception as e:
            logger.error("event=database_init_failed | error={err}", err=str(e))
            return False

    def _create_tables(self) -> None:
        """Create database schema."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL DEFAULT 512,
                default_floor INTEGER DEFAULT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id)
        """)
        self._conn.commit()

        # Migration: add default_floor column if missing (existing databases)
        self._migrate_add_floor_column()

    def add_face(self, person_id: str, name: str, embedding: np.ndarray,
                 default_floor: int | None = None) -> int:
        """
        Add a face embedding to the database.

        Args:
            person_id: Unique person identifier
            name: Display name
            embedding: Normalized embedding vector (512,)
            default_floor: Default floor for elevator control (optional)

        Returns:
            Row ID of inserted record
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized")

        now = time.time()
        embedding_blob = embedding.astype(np.float32).tobytes()

        cursor = self._conn.execute(
            "INSERT INTO faces (person_id, name, embedding, embedding_dim, default_floor, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (person_id, name, embedding_blob, embedding.shape[0], default_floor, now, now),
        )
        self._conn.commit()

        logger.info(
            "event=face_added | person_id={pid} | name={name} | dim={dim} | floor={f}",
            pid=person_id, name=name, dim=embedding.shape[0], f=default_floor,
        )
        return cursor.lastrowid

    def remove_face(self, person_id: str) -> int:
        """
        Remove all embeddings for a person.

        Args:
            person_id: Person to remove

        Returns:
            Number of rows deleted
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized")

        cursor = self._conn.execute(
            "DELETE FROM faces WHERE person_id = ?", (person_id,)
        )
        self._conn.commit()
        count = cursor.rowcount

        logger.info(
            "event=face_removed | person_id={pid} | rows_deleted={n}",
            pid=person_id, n=count,
        )
        return count

    def get_all(self) -> list[FaceRecord]:
        """Get all face records."""
        if self._conn is None:
            return []

        cursor = self._conn.execute(
            "SELECT person_id, name, embedding, created_at, updated_at, default_floor FROM faces"
        )
        records = []
        for row in cursor.fetchall():
            embedding = np.frombuffer(row[2], dtype=np.float32).copy()
            records.append(FaceRecord(
                person_id=row[0],
                name=row[1],
                embedding=embedding,
                created_at=row[3],
                updated_at=row[4],
                default_floor=row[5],
            ))
        return records

    def get_person(self, person_id: str) -> list[FaceRecord]:
        """Get all embeddings for a specific person."""
        if self._conn is None:
            return []

        cursor = self._conn.execute(
            "SELECT person_id, name, embedding, created_at, updated_at, default_floor FROM faces "
            "WHERE person_id = ?", (person_id,)
        )
        records = []
        for row in cursor.fetchall():
            embedding = np.frombuffer(row[2], dtype=np.float32).copy()
            records.append(FaceRecord(
                person_id=row[0],
                name=row[1],
                embedding=embedding,
                created_at=row[3],
                updated_at=row[4],
                default_floor=row[5],
            ))
        return records

    def count(self) -> int:
        """Get total number of face records."""
        if self._conn is None:
            return 0
        cursor = self._conn.execute("SELECT COUNT(*) FROM faces")
        return cursor.fetchone()[0]

    def count_persons(self) -> int:
        """Get number of unique persons."""
        if self._conn is None:
            return 0
        cursor = self._conn.execute("SELECT COUNT(DISTINCT person_id) FROM faces")
        return cursor.fetchone()[0]

    def find_match(self, query_embedding: np.ndarray,
                   threshold: float = 0.4) -> MatchResult | None:
        """
        Find the best matching person for a query embedding.

        Computes cosine similarity against all stored embeddings.
        Returns the best match above threshold, or None.

        Args:
            query_embedding: Normalized query embedding (512,)
            threshold: Minimum similarity for a match

        Returns:
            MatchResult or None if no match above threshold
        """
        if self._conn is None:
            return None

        records = self.get_all()
        if not records:
            return None

        query = query_embedding.flatten().astype(np.float64)
        norm_q = np.linalg.norm(query)
        if norm_q < 1e-10:
            return None

        best_match: MatchResult | None = None
        best_sim = -1.0

        for record in records:
            emb = record.embedding.astype(np.float64)
            norm_e = np.linalg.norm(emb)
            if norm_e < 1e-10:
                continue
            sim = float(np.dot(query, emb) / (norm_q * norm_e))

            if sim > best_sim:
                best_sim = sim
                best_match = MatchResult(
                    person_id=record.person_id,
                    name=record.name,
                    similarity=sim,
                    embedding=record.embedding,
                )

        if best_match is not None and best_match.similarity >= threshold:
            return best_match
        return None

    def _migrate_add_floor_column(self) -> None:
        """Migration: add default_floor column if not exists (for existing DBs)."""
        try:
            cursor = self._conn.execute("PRAGMA table_info(faces)")
            columns = [row[1] for row in cursor.fetchall()]
            if "default_floor" not in columns:
                self._conn.execute("ALTER TABLE faces ADD COLUMN default_floor INTEGER DEFAULT NULL")
                self._conn.commit()
                logger.info("event=db_migration | added_column=default_floor")
        except Exception as e:
            logger.warning("event=db_migration_skipped | error={err}", err=str(e))

    def update_person_floor(self, person_id: str, floor: int | None) -> int:
        """
        Update default floor for a person (all their embeddings).

        Args:
            person_id: Person identifier
            floor: Floor number (None to clear)

        Returns:
            Number of rows updated
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized")

        cursor = self._conn.execute(
            "UPDATE faces SET default_floor = ?, updated_at = ? WHERE person_id = ?",
            (floor, time.time(), person_id),
        )
        self._conn.commit()
        logger.info(
            "event=floor_updated | person_id={pid} | floor={f} | rows={n}",
            pid=person_id, f=floor, n=cursor.rowcount,
        )
        return cursor.rowcount

    def get_person_floor(self, person_id: str) -> int | None:
        """
        Get default floor for a person.

        Returns:
            Floor number or None if not set
        """
        if self._conn is None:
            return None

        cursor = self._conn.execute(
            "SELECT default_floor FROM faces WHERE person_id = ? LIMIT 1",
            (person_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return row[0]

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("event=database_closed | path={p}", p=str(self._db_path))
