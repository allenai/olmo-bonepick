from datetime import datetime
import logging
from pathlib import Path
import sys

LOGGER = logging.getLogger(__package__)


def init_logger(cache_location: Path | str | None = None, logger: logging.Logger | None = None) -> logging.Logger:
    """Initialize a global logger writing to stderr and a log file.

    Args:
        cache_location: Directory where logs should be stored.
        run_name: Identifier for this run.
        logger: Logger to initialize. If None, the global logger will be used.
    """

    # use default logger if not provided
    logger = logger or LOGGER

    # Clear old handlers if re-initialized
    logger.handlers.clear()

    # stderr handler
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(stream_handler)

    if cache_location is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(cache_location) / f"run_{timestamp}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # file handler
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))

        # add file handler
        logger.addHandler(file_handler)

        # log message
        logger.info(f"Logging to {log_path}")

    logger.info("Logger initialized")
    return logger
