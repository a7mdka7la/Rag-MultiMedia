"""Small, phase-agnostic helpers used across the pipeline.

Currently:
- Logging setup via loguru.
- SHA-256 hashing over bytes or strings.
- Deterministic chunk-ID construction.

Later phases append here (bbox helpers in Phase 1, page-image cache key in Phase 5).
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from src.config import settings

_LOGGING_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Install a single loguru sink at the configured level. Idempotent.

    Args:
        level: Override log level (else uses `settings.log_level`).
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=(level or settings.log_level).upper(),
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> "
            "<level>{level:<7}</level> "
            "<cyan>{name}:{function}:{line}</cyan> "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    _LOGGING_CONFIGURED = True


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(data: str) -> str:
    """Return the hex SHA-256 digest of `data`, UTF-8 encoded."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_parts(*parts: str | bytes | None) -> str:
    """Hash a sequence of parts into one stable hex digest.

    `None` parts are encoded as the literal string "<none>" so they contribute a
    fixed marker rather than being skipped — important for chunk IDs where
    missing bbox must still produce a distinct input.

    Args:
        *parts: Strings, bytes, or None — mixed is fine.

    Returns:
        Hex SHA-256 digest.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(b"\x00")  # unambiguous separator between parts
        if p is None:
            h.update(b"<none>")
        elif isinstance(p, bytes):
            h.update(p)
        else:
            h.update(p.encode("utf-8"))
    return h.hexdigest()


def chunk_id(doc_id: str, page: int, bbox: Iterable[float] | None, content: str) -> str:
    """Build a deterministic chunk ID from `(doc_id, page, bbox, content)`.

    Args:
        doc_id: Stable document identifier (usually the PDF file hash).
        page: 1-indexed logical page number.
        bbox: Four-float bounding box, or None if the chunk has no spatial position.
        content: Raw chunk text / markdown / caption.

    Returns:
        Hex SHA-256 digest — collision-resistant, reproducible across runs.
    """
    bbox_str = (
        "-".join(f"{c:.3f}" for c in bbox) if bbox is not None else None
    )
    return sha256_parts(doc_id, str(page), bbox_str, content)


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) as a directory if missing; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
