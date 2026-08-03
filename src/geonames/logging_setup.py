"""One place to configure application logging for the whole project."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a console handler.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Raises:
        ValueError: If the level name is not a valid logging level.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
