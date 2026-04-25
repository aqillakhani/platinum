"""Structured logging setup. Call configure_logging() once at process start."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


def configure_logging(
    log_file: Path | str | None = None,
    level: str = "INFO",
    fmt: str = "%(asctime)s %(levelname)-7s %(name)s — %(message)s",
) -> None:
    """Set up root logger with console + optional rotating file handler."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Wipe any prior handlers (re-config safe)
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Tame noisy libraries
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("urllib3").setLevel("WARNING")


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor matching `from .logger import get_logger`."""
    return logging.getLogger(name)
