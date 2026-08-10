"""
Tests for data collection tools: StorageManager, FaceSnapshot, DataRecorder.

Covers:
- Storage manager: disk usage tracking, cleanup logic
- Face snapshot: save crops, daily limits, filename format
- Data recorder: instantiation, directory creation (no real camera needed)
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from edge.tools.storage_manager import StorageManager
from edge.tools.face_snapshot import FaceSnapshot
from edge.tools.data_recorder import DataRecorder


# --- Test: Storage Manager ---


class TestStorageManager:
    """Tests for StorageManager disk cleanup."""

    def test_empty_dir_usage(self, tmp_path):
        """Empty directory should have 0 usage."""
        sm = StorageManager(tmp_path, max_disk_mb=100)
        assert sm.get_usage_bytes() == 0
        assert sm.get_usage_mb() == 0.0

    def test_nonexistent_dir(self, tmp_path):
        """Non-existent directory should have 0 usage."""
        sm = StorageManager(tmp_path / "nonexistent", max_disk_mb=100)
        assert sm.get_usage_bytes() == 0

    def test_usage_calculation(self, tmp_path):
        """Should correctly calculate file sizes."""
        # Create 10 files of 1KB each
        for i in range(10):
            f = tmp_path / f"file_{i}.bin"
            f.write_bytes(b"x" * 1024)

        sm = StorageManager(tmp_path, max_disk_mb=100)
        usage = sm.get_usage_bytes()
        assert usage == 10 * 1024

    def test_not_over_limit(self, tmp_path):
        """Should not be over limit when under threshold."""
        (tmp_path / "small.bin").write_bytes(b"x" * 100)
        sm = StorageManager(tmp_path, max_disk_mb=1)
        assert not sm.is_over_limit()

    def test_over_limit(self, tmp_path):
        """Should detect when over limit."""
        # Create 2MB of data with 1MB limit
        for i in range(20):
            (tmp_path / f"file_{i}.bin").write_bytes(b"x" * (100 * 1024))

        sm = StorageManager(tmp_path, max_disk_mb=1)
        assert sm.is_over_limit()

    def test_cleanup_removes_oldest(self, tmp_path):
        """Cleanup should remove oldest files first."""
        # Create files with increasing mtime
        for i in range(5):
            f = tmp_path / f"file_{i}.bin"
            f.write_bytes(b"x" * (300 * 1024))  # 300KB each = 1.5MB total
            time.sleep(0.01)  # Ensure different mtime

        sm = StorageManager(tmp_path, max_disk_mb=1, cleanup_ratio=0.5)
        deleted = sm.force_cleanup()
        assert deleted > 0

        # Oldest files should be gone, newest should remain
        remaining = sorted(tmp_path.glob("*.bin"))
        assert len(remaining) < 5
        # file_4 (newest) should still exist
        assert (tmp_path / "file_4.bin").exists()

    def test_cleanup_if_needed_throttled(self, tmp_path):
        """cleanup_if_needed should respect check interval."""
        for i in range(20):
            (tmp_path / f"file_{i}.bin").write_bytes(b"x" * (100 * 1024))

        sm = StorageManager(tmp_path, max_disk_mb=1)
        sm._check_interval = 0  # Disable throttle for test
        deleted = sm.cleanup_if_needed()
        assert deleted > 0

    def test_no_limit_skips_cleanup(self, tmp_path):
        """max_disk_mb=0 should never cleanup."""
        (tmp_path / "big.bin").write_bytes(b"x" * (2 * 1024 * 1024))
        sm = StorageManager(tmp_path, max_disk_mb=0)
        assert not sm.is_over_limit()
        assert sm.force_cleanup() == 0


# --- Test: Face Snapshot ---


class TestFaceSnapshot:
    """Tests for FaceSnapshot auto-capture."""

    def test_save_aligned_face(self, tmp_path):
        """Should save aligned face image."""
        snap = FaceSnapshot(snapshot_dir=tmp_path, max_per_person_per_day=10)
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

        result = snap.save_snapshot(
            aligned_face=face,
            full_frame=None,
            person_id="p001",
            person_name="Alice",
            confidence=0.8,
            bbox=None,
        )
        assert result is True

        # Check file exists
        files = list((tmp_path / "faces").glob("*.jpg"))
        assert len(files) == 1
        assert "recognized_p001" in files[0].name

    def test_save_unknown_face(self, tmp_path):
        """Should save unknown face with 'unknown' prefix."""
        snap = FaceSnapshot(snapshot_dir=tmp_path, max_per_person_per_day=10)
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

        result = snap.save_snapshot(
            aligned_face=face,
            full_frame=None,
            person_id=None,
            confidence=0.9,
            bbox=None,
        )
        assert result is True

        files = list((tmp_path / "faces").glob("*.jpg"))
        assert len(files) == 1
        assert "unknown_" in files[0].name

    def test_save_full_frame(self, tmp_path):
        """Should save annotated full frame when enabled."""
        snap = FaceSnapshot(snapshot_dir=tmp_path, save_full_frame=True)
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        snap.save_snapshot(
            aligned_face=face,
            full_frame=frame,
            person_id="p001",
            person_name="Alice",
            confidence=0.75,
            bbox=(100, 50, 200, 200),
        )

        full_files = list((tmp_path / "full").glob("*.jpg"))
        assert len(full_files) == 1

    def test_daily_limit(self, tmp_path):
        """Should stop saving after max_per_person_per_day."""
        snap = FaceSnapshot(snapshot_dir=tmp_path, max_per_person_per_day=3)
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

        for i in range(5):
            result = snap.save_snapshot(
                aligned_face=face,
                full_frame=None,
                person_id="p001",
                confidence=0.8,
                bbox=None,
            )
            if i < 3:
                assert result is True
            else:
                assert result is False

        assert snap.get_daily_count("p001") == 3

    def test_different_persons_separate_limits(self, tmp_path):
        """Each person should have independent daily limit."""
        snap = FaceSnapshot(snapshot_dir=tmp_path, max_per_person_per_day=2)
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

        # Person A: 2 snapshots
        for _ in range(2):
            snap.save_snapshot(aligned_face=face, full_frame=None,
                               person_id="a", confidence=0.8, bbox=None)
        # Person B: still has quota
        result = snap.save_snapshot(aligned_face=face, full_frame=None,
                                     person_id="b", confidence=0.8, bbox=None)
        assert result is True
        assert snap.get_daily_count("a") == 2
        assert snap.get_daily_count("b") == 1

    def test_no_aligned_face_no_save(self, tmp_path):
        """Should not crash with None aligned_face."""
        snap = FaceSnapshot(snapshot_dir=tmp_path)
        result = snap.save_snapshot(
            aligned_face=None,
            full_frame=None,
            person_id="p001",
            confidence=0.8,
            bbox=None,
        )
        assert result is False

    def test_total_snapshots_count(self, tmp_path):
        """total_snapshots should count files."""
        snap = FaceSnapshot(snapshot_dir=tmp_path)
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

        snap.save_snapshot(aligned_face=face, full_frame=None,
                           person_id="p001", confidence=0.8, bbox=None)
        snap.save_snapshot(aligned_face=face, full_frame=None,
                           person_id="p002", confidence=0.7, bbox=None)

        assert snap.total_snapshots == 2

    def test_reset_daily_counts(self, tmp_path):
        """reset_daily_counts should clear all counters."""
        snap = FaceSnapshot(snapshot_dir=tmp_path, max_per_person_per_day=5)
        face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

        snap.save_snapshot(aligned_face=face, full_frame=None,
                           person_id="p001", confidence=0.8, bbox=None)
        assert snap.get_daily_count("p001") == 1

        snap.reset_daily_counts()
        assert snap.get_daily_count("p001") == 0


# --- Test: Data Recorder ---


class TestDataRecorder:
    """Tests for DataRecorder (no real camera, just instantiation and config)."""

    def test_instantiation(self, tmp_path):
        """Should instantiate without error."""
        recorder = DataRecorder(
            url="0",
            data_dir=tmp_path,
            device_id="test-001",
            max_disk_mb=500,
        )
        assert not recorder.is_connected

    def test_connect_invalid_url(self, tmp_path):
        """Should fail gracefully for invalid camera URL."""
        recorder = DataRecorder(
            url="/nonexistent/stream.mp4",
            data_dir=tmp_path,
        )
        result = recorder.connect()
        assert result is False
        assert not recorder.is_connected

    def test_disconnect_safe(self, tmp_path):
        """Disconnect should be safe even when not connected."""
        recorder = DataRecorder(url="0", data_dir=tmp_path)
        recorder.disconnect()  # Should not raise
