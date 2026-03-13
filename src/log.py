"""Shared logging setup for thesis-kb."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str, filename: str, level=logging.DEBUG) -> logging.Logger:
    """Create a file-based logger with rotation."""
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RotatingFileHandler(
            log_dir / filename, maxBytes=5_000_000, backupCount=3
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
