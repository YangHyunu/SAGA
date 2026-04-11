"""Centralised logging configuration for SAGA."""
import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
_LOG_DATE_FMT_CONSOLE = "%H:%M:%S"
_LOG_DATE_FMT_FILE = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(
    level: int = logging.INFO,
    log_dir: str = "logs",
) -> None:
    """Configure root logger with console + rotating file handler.

    Safe to call multiple times; only the first call takes effect.
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT_CONSOLE))
    root.addHandler(console)

    # Rotating file handler
    os.makedirs(log_dir, exist_ok=True)
    file_h = RotatingFileHandler(
        os.path.join(log_dir, "saga.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_h.setLevel(level)
    file_h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT_FILE))
    root.addHandler(file_h)

    # Quiet noisy third-party loggers
    for name in (
        "httpx",
        "httpcore",
        "chromadb",
        "sentence_transformers",
        "urllib3",
        "openai",
        "anthropic",
        "hpack",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Ensure saga loggers are at the requested level
    for name in ("saga", "saga.llm.client", "saga.system_stabilizer"):
        logging.getLogger(name).setLevel(level)
