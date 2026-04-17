"""Tests for src.generate — citation regex, prompt shape, mocked end-to-end.

Groq is mocked: we only want to verify the parsing and plumbing around it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.chunk import Chunk
from src.generate import (
    CITATION_RE,
    Answer,
    answer,
    build_messages,
    parse_citations,
)
from src.retrieve import RetrievedChunk


# --------------------------- fixtures ---------------------------


def _mk(cid: str, page: int, modality: str, content: str, section: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=cid,
            content=content,
            modality=modality,
            page=page,
            section=section,
            bbox=None,
            image_path=None,
            extra={},
        ),
        score=1.0,
        rank=0,
        stage_scores={"dense": 0.0, "bm25": 0.0, "rrf": 0.0, "rerank": 1.0},
    )


@pytest.fixture()
def sample_chunks() -> list[RetrievedChunk]:
    return [
        _mk("a", 12, "text", "Inflation moderated to 23.3% in late 2024.", "Macro outlook"),
        _mk("b", 12, "table", "| Year | Inflation |\n|---|---|\n| 2024 | 23.3 |"),
        _mk("c", 7, "text", "Real GDP is projected to grow 4.2% in FY2024/25."),
    ]


# --------------------------- unit tests ---------------------------


def test_citation_regex_matches_all_shapes() -> None:
    text = "Inflation is 23.3% [p.12, text]. See also [p.12, table] and [p.7,  image]."
    cits = parse_citations(text)
    assert [c.page for c in cits] == [12, 12, 7]
    assert [c.modality for c in cits] == ["text", "table", "image"]
    # Non-modality modalities don't match.
    assert CITATION_RE.search("[p.5, chart]") is None


def test_build_messages_has_system_and_user_roles(sample_chunks: list[RetrievedChunk]) -> None:
    msgs = build_messages("What is inflation?", sample_chunks)
    assert [m["role"] for m in msgs] == ["system", "user"]
    # System prompt must contain the citation rule.
    assert "[p.<page>, <modality>]" in msgs[0]["content"]
    # User content carries all three context blocks and the question.
    user = msgs[1]["content"]
    assert "Inflation moderated to 23.3%" in user
    assert "| 2024 | 23.3 |" in user
    assert "Question: What is inflation?" in user
    # Page + modality metadata appears in each context header.
    assert "page=12" in user and "modality=table" in user


# --------------------------- end-to-end (mocked Groq) -------------


def _fake_stream(text: str):
    """Yield Groq-shaped chunks that drip the given text one token at a time."""
    for tok in text.split(" "):
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=tok + " "))]
        )


def test_answer_resolves_citations_back_to_chunks(
    sample_chunks: list[RetrievedChunk], mocker: pytest.MonkeyPatch
) -> None:
    """End-to-end: mocked Groq emits an answer, `answer()` resolves citations."""
    fake_output = "Inflation fell to 23.3% [p.12, text]. See also [p.12, table]."
    mock_client = mocker.MagicMock()  # type: ignore[attr-defined]
    mock_client.chat.completions.create.return_value = list(_fake_stream(fake_output))

    result = answer("What is inflation?", sample_chunks, client=mock_client)

    assert isinstance(result, Answer)
    # Stream accumulates exactly what the fake emitted (space joined).
    assert "23.3%" in result.text
    assert "[p.12, text]" in result.text
    # Both citations parsed.
    assert [(c.page, c.modality) for c in result.citations] == [(12, "text"), (12, "table")]
    # Chunk refs resolve to the exactly-matching chunks, deduped by id.
    ref_ids = [r.chunk.id for r in result.chunk_refs]
    assert ref_ids == ["a", "b"]
    # Client was called exactly once.
    mock_client.chat.completions.create.assert_called_once()


def test_answer_when_no_citations_returns_empty_refs(
    sample_chunks: list[RetrievedChunk], mocker: pytest.MonkeyPatch
) -> None:
    """Bare text (no citations) → empty chunk_refs, empty citations."""
    mock_client = mocker.MagicMock()  # type: ignore[attr-defined]
    mock_client.chat.completions.create.return_value = list(
        _fake_stream("The report does not contain information to answer this question.")
    )
    result = answer("unanswerable?", sample_chunks, client=mock_client)
    assert result.citations == []
    assert result.chunk_refs == []
    assert "does not contain" in result.text
