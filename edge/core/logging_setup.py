"""
Smart Cabin Platform - Logging Setup

Configures structured logging to both console and file.
Log file includes timestamps for all events (reconnects, errors, start/stop)
so they can be reviewed later without real-time monitoring.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from edge.core.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """
    Configure root logger based on config.

    - Always logs to console (stdout)
    - If config.file is set, also logs to rotating file
    - Creates log directory if needed
    """
    level = getattr(logging, config.level.upper(), logging.INFO)
    fmt = config.format

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers (avoid duplicates on reload)
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(console_handler)

    # File handler (if configured)
    log_file = config.file
    if not log_file:
        # Default log file location
        log_file = "logs/smart_cabin.log"

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(file_handler)

    logging.info(f"Logging initialized: level={config.level}, file={log_path}")
