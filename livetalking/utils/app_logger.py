import logging
import sys
from pathlib import Path


def _reconfigure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_reconfigure_stdio()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "runtime" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _coerce_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    if level is None:
        return logging.INFO
    value = str(level).strip().upper()
    if not value:
        return logging.INFO
    resolved = logging.getLevelName(value)
    return resolved if isinstance(resolved, int) else logging.INFO


def configure_logging(level: str | int | None = None) -> None:
    resolved_level = _coerce_level(level)
    formatter = logging.Formatter(_LOG_FORMAT)

    logger.setLevel(resolved_level)
    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_DIR / "livetalking.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    for handler in logger.handlers:
        handler.setLevel(resolved_level)
        handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)


logger = logging.getLogger(__name__)
configure_logging(None)
