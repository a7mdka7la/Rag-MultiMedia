"""Tests for src.retrieve — RRF fusion + end-to-end hybrid retrieval.

The end-to-end test loads both BGE-M3 and the cross-encoder reranker against
a synthetic chunks.jsonl — slow on a cold cache, but unit-mocking the reranker
would verify nothing about the thing that can actually go wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chunk import Chunk
from src.index import IndexHandle, build_index
from src.retrieve import retrieve, rrf_fuse

# --------------------------- unit: RRF ------------------------------


def test_rrf_fuse_gives_docs_in_both_rankings_a_boost() -> None:
    """A doc appearing in both rankings outscores docs only in one."""
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 10.0), ("d", 5.0), ("a", 4.0)]
    fused = rrf_fuse([dense, sparse], k=60)
    fused_ids = [cid for cid, _ in fused]
    # a (ranks 1 & 3) and b (ranks 2 & 1) both appear twice, so they outrank c and d.
    assert fused_ids.index("a") < fused_ids.index("c")
    assert fused_ids.index("b") < fused_ids.index("d")
    # Explicit score check for one entry.
    # a: 1/(60+1) + 1/(60+3) = ~0.0323
    assert fused[fused_ids.index("a")][1] == pytest.approx(1 / 61 + 1 / 63, rel=1e-6)


def test_rrf_fuse_single_ranking_is_reciprocal_rank() -> None:
    """With one input list, RRF degenerates to 1/(k+rank)."""
    fused = rrf_fuse([[("x", 0.5), ("y", 0.3)]], k=60)
    assert fused[0] == ("x", pytest.approx(1 / 61))
    assert fused[1] == ("y", pytest.approx(1 / 62))


# --------------------------- end-to-end -----------------------------


def _synthetic_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks = [
        Chunk(
            id="t1",
            content="The GDP forecast for 2025 is expected to rise by 4.2 percent.",
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
            content="Summary: fiscal balance.\n\n| Year | Balance |\n|---|---|\n| 2024 | -5.1 |",
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
            image_path=Path("/tmp/nonexistent.png"),
            extra={"context": "Figure 1 caption"},
        ),
    ]
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.to_jsonl() + "\n")
    return chunks


@pytest.fixture(scope="module")
def built_index(tmp_path_factory: pytest.TempPathFactory) -> IndexHandle:
    work = tmp_path_factory.mktemp("retrieve_idx")
    chunks_path = work / "chunks.jsonl"
    _synthetic_chunks_jsonl(chunks_path)
    return build_index(chunks_path, work / "chroma")


def test_retrieve_returns_at_most_final_top_k_ranked_chunks(built_index: IndexHandle) -> None:
    results = retrieve("GDP forecast for 2025", built_index, final_top_k=3)
    assert 1 <= len(results) <= 3
    # Ranks are 0-indexed and monotonically increasing.
    assert [r.rank for r in results] == list(range(len(results)))
    # Final scores should be sorted descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # Every result carries the four stage scores we advertised.
    for r in results:
        assert set(r.stage_scores) == {"dense", "bm25", "rrf", "rerank"}


def test_retrieve_puts_gdp_chunk_first(built_index: IndexHandle) -> None:
    """For a GDP-forecast query, the t1 chunk should clearly beat the table + image."""
    results = retrieve("What is the GDP forecast for 2025?", built_index, final_top_k=3)
    assert results[0].chunk.id == "t1", (
        f"expected 't1' at rank 1, got {results[0].chunk.id!r}"
    )
