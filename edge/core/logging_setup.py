"""
Smart Cabin Platform - Logging Setup (Loguru)

Structured logging with:
- Module-based log files (camera.log, system.log, etc.)
- Key-value format for easy parsing
- Console output for development
- Rotating files (10MB, 5 backups)
"""

import sys
from pathlib import Path

from loguru import logger

# Remove default loguru handler
logger.remove()

# Log directory
LOG_DIR = Path("logs")


def setup_logging(level: str = "INFO", log_dir: str | Path = "logs") -> None:
    """
    Configure loguru logging with module-based files.

    Creates separate log files:
    - camera.log: Video pipeline events
    - system.log: General system events (start/stop, config, plugins)
    - all.log: Everything combined

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
    """
    global LOG_DIR
    LOG_DIR = Path(log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Clear existing handlers
    logger.remove()

    # Key-value format
    kv_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {extra[module]:<10} | {message}"
    )

    # Console handler (colored, for development)
    logger.add(
        sys.stderr,
        level=level,
        format=kv_format,
        filter=lambda record: record["extra"].get("module", "system"),
    )

    # All logs combined
    logger.add(
        LOG_DIR / "all.log",
        level=level,
        format=kv_format,
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
    )

    # Camera module log
    logger.add(
        LOG_DIR / "camera.log",
        level=level,
        format=kv_format,
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        filter=lambda record: record["extra"].get("module") == "camera",
    )

    # System module log
    logger.add(
        LOG_DIR / "system.log",
        level=level,
        format=kv_format,
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        filter=lambda record: record["extra"].get("module") == "system",
    )


def get_logger(module: str):
    """
    Get a module-scoped logger.

    Args:
        module: Module name (e.g., "camera", "system", "face_recognition")

    Returns:
        Loguru logger bound with module context
    """
    return logger.bind(module=module)
