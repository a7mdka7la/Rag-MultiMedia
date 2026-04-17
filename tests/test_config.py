"""Phase-0 smoke tests for src.config and src.utils."""

from __future__ import annotations

import dataclasses

import pytest

from src import config, utils


def test_settings_loads() -> None:
    """Settings is constructed at import time and is immutable."""
    assert config.settings is not None
    assert config.settings.project_root.exists()


def test_settings_is_frozen() -> None:
    """Frozen dataclass — mutation must raise FrozenInstanceError."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.settings.log_level = "DEBUG"  # type: ignore[misc]


def test_default_retrieval_tunables() -> None:
    """RRF / rerank defaults match the spec."""
    s = config.settings
    assert s.dense_top_k == 20
    assert s.bm25_top_k == 20
    assert s.rerank_top_k == 20
    assert s.final_top_k == 5
    assert s.rrf_k == 60
    assert s.final_top_k <= s.rerank_top_k


def test_default_models_match_spec() -> None:
    """Spec-pinned model identifiers — accidental renames would fail here."""
    s = config.settings
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert s.gemini_model == "gemini-2.5-flash"
    assert s.answer_model == "llama-3.3-70b-versatile"
    assert s.router_model == "llama-3.1-8b-instant"
    assert s.groq_base_url == "https://api.groq.com/openai/v1"


def test_paths_are_absolute() -> None:
    """All configured paths resolve to absolute paths."""
    s = config.settings
    for p in (s.pdf_path, s.chroma_path, s.cache_dir, s.images_dir, s.chunks_path):
        assert p.is_absolute(), f"{p} is not absolute"


def test_require_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """require_key() raises a helpful error when the key is empty."""
    # Rebind the module-level settings to a fresh instance with empty keys.
    # We clear the env first so _load() produces empty strings.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    empty_settings = config._load()
    monkeypatch.setattr(config, "settings", empty_settings)
    assert config.settings.groq_api_key == ""
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        config.require_key("GROQ_API_KEY")
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        config.require_key("GOOGLE_API_KEY")


def test_require_key_unknown() -> None:
    """Unknown key name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown API key name"):
        config.require_key("NOT_A_KEY")


def test_chunk_id_is_deterministic() -> None:
    """Same inputs → same ID; different inputs → different ID."""
    a = utils.chunk_id("doc1", 3, (0.1, 0.2, 0.3, 0.4), "hello")
    b = utils.chunk_id("doc1", 3, (0.1, 0.2, 0.3, 0.4), "hello")
    c = utils.chunk_id("doc1", 3, (0.1, 0.2, 0.3, 0.4), "hello!")
    d = utils.chunk_id("doc1", 3, None, "hello")
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 64  # sha256 hex


def test_sha256_parts_handles_none() -> None:
    """None is encoded distinctly — doesn't silently collide with empty string."""
    assert utils.sha256_parts("x", None) != utils.sha256_parts("x", "")


def test_setup_logging_is_idempotent() -> None:
    """Calling setup_logging twice adds only one sink."""
    utils.setup_logging()
    utils.setup_logging()  # must not raise or duplicate
