"""
Smart Cabin - Storage Manager

Manages disk usage for data collection outputs.
Auto-deletes oldest files when disk usage exceeds threshold.

Usage:
    manager = StorageManager(Path("data"), max_disk_mb=1000)
    manager.cleanup_if_needed()  # Call periodically
"""

import os
from pathlib import Path

from edge.core.logging_setup import get_logger

logger = get_logger("system")


class StorageManager:
    """
    Manages disk space for collected data.

    Monitors total size of a directory and removes oldest files
    when the threshold is exceeded.

    Args:
        data_dir: Root directory to monitor
        max_disk_mb: Maximum allowed disk usage in MB (0 = no limit)
        cleanup_ratio: Remove files until usage drops to this ratio of max (default 0.8)
    """

    def __init__(self,
                 data_dir: Path,
                 max_disk_mb: int = 1000,
                 cleanup_ratio: float = 0.8):
        self._data_dir = Path(data_dir)
        self._max_bytes = max_disk_mb * 1024 * 1024
        self._target_bytes = int(self._max_bytes * cleanup_ratio)
        self._last_check_time = 0.0
        self._check_interval = 30.0  # Only check every 30 seconds

    @property
    def max_disk_mb(self) -> int:
        """Maximum allowed disk usage in MB."""
        return self._max_bytes // (1024 * 1024)

    def get_usage_bytes(self) -> int:
        """Calculate total disk usage of data directory."""
        if not self._data_dir.exists():
            return 0

        total = 0
        for f in self._data_dir.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    def get_usage_mb(self) -> float:
        """Get disk usage in MB."""
        return self.get_usage_bytes() / (1024 * 1024)

    def is_over_limit(self) -> bool:
        """Check if disk usage exceeds the max threshold."""
        if self._max_bytes == 0:
            return False
        return self.get_usage_bytes() > self._max_bytes

    def cleanup_if_needed(self) -> int:
        """
        Check disk usage and cleanup oldest files if over limit.

        Only checks every 30 seconds to avoid excessive I/O.

        Returns:
            Number of files deleted
        """
        import time

        now = time.time()
        if now - self._last_check_time < self._check_interval:
            return 0
        self._last_check_time = now

        if self._max_bytes == 0:
            return 0

        current_usage = self.get_usage_bytes()
        if current_usage <= self._max_bytes:
            return 0

        return self._do_cleanup(current_usage)

    def _do_cleanup(self, current_usage: int) -> int:
        """Delete oldest files until usage drops below target."""
        # Get all files sorted by modification time (oldest first)
        files = []
        for f in self._data_dir.rglob("*"):
            if f.is_file():
                try:
                    files.append((f.stat().st_mtime, f.stat().st_size, f))
                except OSError:
                    pass

        files.sort()  # Oldest first

        deleted = 0
        freed = 0
        target = current_usage - self._target_bytes

        for mtime, size, filepath in files:
            if freed >= target:
                break
            try:
                filepath.unlink()
                freed += size
                deleted += 1

                # Also remove JSON sidecar if exists
                sidecar = filepath.with_suffix(".json")
                if sidecar.exists():
                    sidecar.unlink()
                    deleted += 1
            except OSError:
                pass

        # Remove empty directories
        self._remove_empty_dirs()

        logger.info(
            "event=storage_cleanup | files_deleted={n} | freed_mb={mb:.1f} | "
            "usage_before_mb={before:.1f} | usage_after_mb={after:.1f}",
            n=deleted, mb=freed / (1024 * 1024),
            before=current_usage / (1024 * 1024),
            after=(current_usage - freed) / (1024 * 1024),
        )

        return deleted

    def _remove_empty_dirs(self) -> None:
        """Remove empty subdirectories."""
        if not self._data_dir.exists():
            return

        for dirpath in sorted(self._data_dir.rglob("*"), reverse=True):
            if dirpath.is_dir():
                try:
                    dirpath.rmdir()  # Only removes if empty
                except OSError:
                    pass

    def force_cleanup(self) -> int:
        """Force cleanup regardless of timing interval."""
        if self._max_bytes == 0:
            return 0
        current_usage = self.get_usage_bytes()
        if current_usage <= self._max_bytes:
            return 0
        return self._do_cleanup(current_usage)
