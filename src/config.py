"""Typed, immutable settings loaded from `.env`.

`python-dotenv` hydrates `os.environ`; the frozen :class:`Settings` dataclass
then reads each value once, validates it, and freezes. Modules that need an
API key call :func:`require_key` at the point of use so this module stays
importable without secrets in scaffolding / CI contexts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Load .env once at import. `override=False` lets explicitly-set process env win.
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def _env_str(key: str, default: str = "") -> str:
    """Read an env var as string, trimming whitespace."""
    return os.environ.get(key, default).strip()


def _env_bool(key: str, default: bool) -> bool:
    """Read an env var as bool. Accepts 1/0, true/false, yes/no (case-insensitive)."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    match raw.strip().lower():
        case "1" | "true" | "yes" | "on":
            return True
        case "0" | "false" | "no" | "off":
            return False
        case _:
            raise ValueError(f"Cannot parse {key}={raw!r} as bool")


def _env_float(key: str, default: float) -> float:
    """Read an env var as float. Raises ValueError on unparseable input."""
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _resolve(path_str: str) -> Path:
    """Resolve a path (possibly relative) against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """All tunables and secrets for the system. Constructed once at import time."""

    # Secrets — empty string is allowed so the module loads without keys;
    # call require_key() at the point of use.
    google_api_key: str
    google_api_keys: tuple[str, ...]
    groq_api_key: str

    # Paths
    project_root: Path
    pdf_path: Path
    chroma_path: Path
    cache_dir: Path
    images_dir: Path
    chunks_path: Path

    # Models
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    gemini_model: str = "gemini-2.5-flash"
    answer_model: str = "llama-3.3-70b-versatile"
    router_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Retrieval tunables
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60

    # Generation tunables
    answer_temperature: float = 0.2

    # Feature flags
    use_router: bool = False

    # Eval tunables
    eval_inter_question_delay_s: float = 2.0

    # Logging
    log_level: str = "INFO"

    # Docling
    docling_tokenizer: str = "BAAI/bge-m3"
    docling_ocr: bool = False

    # Collection name
    chroma_collection_name: str = "imf_report"

    # Chroma store metadata
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shape (not presence of secrets — those are lazy)."""
        if self.final_top_k > self.rerank_top_k:
            raise ValueError(
                f"final_top_k ({self.final_top_k}) cannot exceed rerank_top_k ({self.rerank_top_k})"
            )
        if self.eval_inter_question_delay_s < 0:
            raise ValueError("eval_inter_question_delay_s must be non-negative")
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Unknown LOG_LEVEL: {self.log_level!r}")


def _load_google_api_keys() -> tuple[str, ...]:
    """Parse GOOGLE_API_KEYS (comma-separated) with GOOGLE_API_KEY as fallback.

    Gemini's free tier is 20 requests/day per key; to keep one-shot ingest
    feasible we support rotating across multiple keys. If GOOGLE_API_KEYS is
    unset, fall back to the single primary key so the test suite and older
    configs keep working.
    """
    multi = _env_str("GOOGLE_API_KEYS")
    if multi:
        keys = tuple(k.strip() for k in multi.split(",") if k.strip())
        if keys:
            return keys
    single = _env_str("GOOGLE_API_KEY")
    return (single,) if single else ()


def _load() -> Settings:
    """Build the process-wide Settings from environment variables."""
    return Settings(
        google_api_key=_env_str("GOOGLE_API_KEY"),
        google_api_keys=_load_google_api_keys(),
        groq_api_key=_env_str("GROQ_API_KEY"),
        project_root=_PROJECT_ROOT,
        pdf_path=_resolve(_env_str("PDF_PATH", "data/source.pdf")),
        chroma_path=_resolve(_env_str("CHROMA_PATH", "data/chroma_db")),
        cache_dir=_resolve(_env_str("CACHE_DIR", "data/cache")),
        images_dir=_resolve(_env_str("IMAGES_DIR", "data/images")),
        chunks_path=_resolve(_env_str("CHUNKS_PATH", "data/chunks.jsonl")),
        use_router=_env_bool("USE_ROUTER", default=False),
        eval_inter_question_delay_s=_env_float("EVAL_INTER_QUESTION_DELAY_S", default=2.0),
        log_level=_env_str("LOG_LEVEL", "INFO"),
    )


settings: Final[Settings] = _load()


def require_key(name: str) -> str:
    """Return the named API key, raising if unset.

    Args:
        name: One of "GOOGLE_API_KEY", "GROQ_API_KEY".

    Returns:
        The key value.

    Raises:
        RuntimeError: If the key is missing or empty.
    """
    match name:
        case "GOOGLE_API_KEY":
            value = settings.google_api_key
        case "GROQ_API_KEY":
            value = settings.groq_api_key
        case _:
            raise ValueError(f"Unknown API key name: {name!r}")
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to .env (see .env.example) before running this command."
        )
    return value
