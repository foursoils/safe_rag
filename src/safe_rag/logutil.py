"""Send third-party INFO logs to a file so tqdm can own the terminal."""

from __future__ import annotations

import logging
from pathlib import Path

LIBRARY_LOGGERS = (
    "",
    "lightrag",
    "nano-vectordb",
    "httpx",
    "openai",
    "httpcore",
    "asyncio",
)


def _is_console_handler(handler: logging.Handler) -> bool:
    if isinstance(handler, logging.FileHandler):
        return False
    return isinstance(handler, logging.StreamHandler)


def redirect_library_logs(path: Path | str, *, mode: str = "w") -> logging.Handler:
    """Strip console handlers from noisy loggers and write them to ``path``."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in LIBRARY_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if name:
            logger.propagate = False
        logger.handlers = [item for item in logger.handlers if not _is_console_handler(item)]
        if handler not in logger.handlers:
            logger.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return handler
