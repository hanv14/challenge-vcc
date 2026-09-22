"""Logging that goes to the terminal and to the run's log file at once."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "vccp"
_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(log_file: Path | None = None, verbose: bool = False) -> logging.Logger:
    """Attach a console handler, and a file handler when a path is given.

    Calling this twice replaces the handlers rather than doubling every line.
    """
    logger = get_logger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def banner(title: str) -> None:
    """A visible divider between stages."""
    logger = get_logger()
    logger.info("")
    logger.info("=" * 72)
    logger.info(title)
    logger.info("=" * 72)
