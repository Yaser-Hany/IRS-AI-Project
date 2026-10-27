"""Utility helpers for logging, directories, and serialization."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any


def setup_logger(name: str = "rf_ml", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_directories(*dirs: Path) -> None:
    """Create directories if they do not exist.

    Args:
        *dirs: Directory paths.
    """
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, file_path: Path) -> None:
    """Serialize an object to pickle file.

    Args:
        obj: Python object to serialize.
        file_path: Destination path.
    """
    with file_path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path: Path) -> Any:
    """Load object from pickle file.

    Args:
        file_path: Source path.

    Returns:
        Deserialized Python object.
    """
    with file_path.open("rb") as f:
        return pickle.load(f)
