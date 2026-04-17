"""Tests for src.index — metadata flattening, tokenization, and end-to-end build.

The end-to-end test actually loads BGE-M3 (cached on disk from Phase 1) and
writes a real Chroma collection to a tmp path. It's the meaningful check:
unit-mocking the embedder would verify almost nothing about what can go wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.chunk import Chunk
from src.config import settings
from src.index import (
    IndexHandle,
    _flatten_metadata,
    build_index,
    load_index,
    tokenize,
)


# --------------------------- unit tests ---------------------------


def test_tokenize_is_lowercase_whitespace() -> None:
    """BM25 tokenization must be deterministic and case-insensitive."""
    assert tokenize("The Quick BROWN Fox") == ["the", "quick", "brown", "fox"]
    assert tokenize("  multi   spaces   ") == ["multi", "spaces"]
    # Matters for queries: the same normalization must apply at query time.
    assert tokenize("GDP forecast 2025") == tokenize("gdp   FORECAST 2025")


def test_flatten_metadata_for_chroma_is_scalars_only() -> None:
    """Chroma rejects tuples, dicts, and Nones in metadata."""
    c = Chunk(
        id="abc",
        content="hello",
        modality="image",
        page=4,
        section="Intro",
        bbox=(1.0, 2.0, 3.0, 4.0),
        image_path=Path("/tmp/x.png"),
        extra={"context": "surrounding", "width": 320},
    )
    md = _flatten_metadata(c)
    # No None, tuple, dict, or list values.
    for k, v in md.items():
        assert v is not None, f"metadata[{k}] is None — Chroma will reject"
        assert not isinstance(v, (tuple, list, dict)), f"metadata[{k}] is a container: {type(v)}"
    # bbox spread across four float keys
    assert md["bbox_x0"] == 1.0
    assert md["bbox_x1"] == 3.0
    # extra round-trips through JSON
    assert json.loads(md["extra_json"])["context"] == "surrounding"
    # image_path is a string
    assert isinstance(md["image_path"], str)


def test_flatten_metadata_handles_text_chunk_without_bbox() -> None:
    """Text chunks may have bbox=None and image_path=None — those must be handled."""
    c = Chunk(
        id="x",
        content="text",
        modality="text",
        page=1,
        section=None,
        bbox=None,
        image_path=None,
        extra={},
    )
    md = _flatten_metadata(c)
    assert "bbox_x0" not in md, "bbox keys should be omitted when bbox is None"
    assert md["image_path"] == ""
    assert md["section"] == ""


# --------------------------- end-to-end -----------------------------


def _synthetic_chunks_jsonl(path: Path) -> list[Chunk]:
    """Write a tiny synthetic chunks.jsonl with each modality represented.

    Returns the in-memory Chunk list for cross-checking.
    """
    chunks = [
        Chunk(
            id="t1",
            content="The GDP forecast for 2025 is expected to rise.",
            modality="text",
            page=1,
            section="Overview",
            bbox=(10.0, 20.0, 100.0, 50.0),
            image_path=None,
            extra={"chunk_index": 0},
        ),
        Chunk(
            id="t2",
            content="Inflation pressures remain moderate through the forecast horizon.",
            modality="text",
            page=2,
            section="Macroeconomic outlook",
            bbox=None,
            image_path=None,
            extra={"chunk_index": 1},
        ),
        Chunk(
            id="tab1",
            content="Table summarising fiscal balances.\n\n| Year | Balance |\n|---|---|\n| 2024 | -5.1 |",
            modality="table",
            page=3,
            section=None,
            bbox=(20.0, 30.0, 200.0, 150.0),
            image_path=None,
            extra={"raw_markdown": "| Year | Balance |"},
        ),
        Chunk(
            id="img1",
            content="Line chart showing real GDP growth between 2020 and 2028.",
            modality="image",
            page=4,
            section=None,
            bbox=(50.0, 60.0, 300.0, 250.0),
            image_path=Path("/tmp/nonexistent.png"),  # metadata only; we don't read it
            extra={"context": "Figure 1 caption"},
        ),
    ]
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.to_jsonl() + "\n")
    return chunks


@pytest.fixture(scope="module")
def built_index(tmp_path_factory: pytest.TempPathFactory) -> tuple[IndexHandle, Path, list[Chunk]]:
    """Build the index once per module — BGE-M3 load is ~5s, worth caching."""
    work = tmp_path_factory.mktemp("index")
    chunks_path = work / "chunks.jsonl"
    chunks = _synthetic_chunks_jsonl(chunks_path)
    out_dir = work / "chroma"
    handle = build_index(chunks_path, out_dir)
    return handle, out_dir, chunks


def test_build_index_populates_chroma_and_bm25(
    built_index: tuple[IndexHandle, Path, list[Chunk]],
) -> None:
    """Every chunk ends up in Chroma and in the BM25 corpus."""
    handle, _, chunks = built_index
    assert handle.collection.count() == len(chunks)
    assert handle.bm25_ids == [c.id for c in chunks]


def test_build_index_writes_manifest_and_bm25_files(
    built_index: tuple[IndexHandle, Path, list[Chunk]],
) -> None:
    """Manifest + BM25 pickle must land on disk so load_index works without rebuild."""
    _, out_dir, chunks = built_index
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["chunk_count"] == len(chunks)
    assert manifest["modality_counts"] == {"text": 2, "table": 1, "image": 1}
    assert manifest["embedding_model"] == settings.embedding_model
    assert (out_dir / "bm25.pkl").exists()


def test_load_index_round_trips_without_rebuild(
    built_index: tuple[IndexHandle, Path, list[Chunk]],
) -> None:
    """`load_index` returns the same IDs and chunk count."""
    _, out_dir, chunks = built_index
    reloaded = load_index(out_dir)
    assert reloaded.collection.count() == len(chunks)
    assert reloaded.bm25_ids == [c.id for c in chunks]


def test_rebuild_on_unchanged_input_is_a_noop(
    built_index: tuple[IndexHandle, Path, list[Chunk]],
) -> None:
    """A second build_index with same chunks.jsonl must short-circuit via hash match."""
    _, out_dir, chunks = built_index
    chunks_path = (out_dir.parent / "chunks.jsonl")
    handle2 = build_index(chunks_path, out_dir)
    assert handle2.collection.count() == len(chunks)


def test_bm25_ranks_relevant_text_chunk_for_query(
    built_index: tuple[IndexHandle, Path, list[Chunk]],
) -> None:
    """BM25 should put the GDP-forecast chunk at rank 0 for a GDP query."""
    handle, _, _ = built_index
    scores = handle.bm25.get_scores(tokenize("GDP forecast 2025"))
    top_idx = int(scores.argmax())
    assert handle.bm25_ids[top_idx] == "t1", (
        f"expected 't1' to rank first for 'GDP forecast 2025', got "
        f"{handle.bm25_ids[top_idx]!r}"
    )


def test_dense_search_returns_nonzero_results(
    built_index: tuple[IndexHandle, Path, list[Chunk]],
) -> None:
    """Dense query round-trip: embed a query, get ≥1 hit from Chroma."""
    handle, _, _ = built_index
    query_vec = handle.embedder.encode(
        ["What is the GDP forecast for 2025?"], normalize_embeddings=True
    ).tolist()
    results = handle.collection.query(query_embeddings=query_vec, n_results=3)
    assert results["ids"] and results["ids"][0], "Chroma returned no dense hits"
    # Top hit should plausibly be the GDP text chunk.
    assert "t1" in results["ids"][0][:3]
