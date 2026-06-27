"""Package-wide rotating file logger setup."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_INITIALIZED = False


def setup_logging() -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return

    log_dir = Path(os.getenv("LOG_DIR", "./logs"))
    log_file = os.getenv("LOG_FILE", "doc_schema_extractor.log")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    log_dir.mkdir(parents=True, exist_ok=True)
    file_path = log_dir / log_file

    root = logging.getLogger("doc_schema_extractor")
    root.setLevel(getattr(logging, log_level, logging.INFO))
    root.propagate = False

    if not root.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _LOGGER_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"doc_schema_extractor.{name}")
