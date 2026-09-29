"""Colored console + file logger factory for ANC-SIH."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"

_LEVEL_COLOURS: dict[int, str] = {
    logging.DEBUG: "\033[36m",      # cyan
    logging.INFO: "\033[36m",       # cyan
    logging.WARNING: "\033[33m",    # yellow
    logging.ERROR: "\033[31m",      # red
    logging.CRITICAL: "\033[35m",   # magenta
}

_LOG_FORMAT = "[%(asctime)s %(levelname)-8s %(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_RUNS_DIR = Path("runs")
_LOG_FILE = _RUNS_DIR / "anc.log"


class _ColorFormatter(logging.Formatter):
    """``logging.Formatter`` subclass that prepends ANSI colour codes.

    Colour is applied only when writing to a TTY so log files stay clean.
    """

    def __init__(self, use_colour: bool = True) -> None:
        super().__init__(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Return a formatted (optionally coloured) log line.

        Args:
            record: The :class:`logging.LogRecord` to format.

        Returns:
            Formatted string ready to be written to a stream.
        """
        formatted = super().format(record)
        if self._use_colour:
            colour = _LEVEL_COLOURS.get(record.levelno, "")
            formatted = f"{_BOLD}{colour}{formatted}{_RESET}"
        return formatted


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with console (coloured) and optional file handlers.

    Calling this function multiple times with the same *name* is safe — handlers
    are never duplicated.

    Args:
        name:  Logger name, typically ``__name__`` of the calling module.
        level: Logging level for both handlers (default :data:`logging.INFO`).

    Returns:
        Configured :class:`logging.Logger` instance.

    Example::

        log = get_logger(__name__)
        log.info("Pipeline started")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # don't bubble up to root logger

    # ------------------------------------------------------------------ #
    # Console handler — coloured when attached to a real TTY              #
    # ------------------------------------------------------------------ #
    is_tty: bool = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(_ColorFormatter(use_colour=is_tty))
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------ #
    # File handler — only when runs/ directory exists                     #
    # ------------------------------------------------------------------ #
    if _RUNS_DIR.exists():
        try:
            file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
            file_handler.setLevel(level)
            # Plain formatter (no ANSI) for the log file.
            file_handler.setFormatter(
                logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
            )
            logger.addHandler(file_handler)
        except OSError:
            # If we can't write the log file, just continue with console only.
            logger.warning("Could not open log file %s; logging to console only.", _LOG_FILE)

    return logger
