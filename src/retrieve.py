"""Phase-3 hybrid retrieval: dense + BM25 → RRF → cross-encoder rerank.

Usage:

    uv run python -m src.retrieve "What is the GDP forecast for 2025?"
    uv run python -m src.retrieve "..." --debug   # per-stage top hits + timings
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from sentence_transformers import CrossEncoder

from src.chunk import Chunk
from src.config import settings
from src.index import IndexHandle, load_index, tokenize
from src.router import RetrievalConfig, classify
from src.utils import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)

# Cross-encoder is lazy — loaded on first retrieve() call and reused process-wide.
_RERANKER: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        logger.info(f"Loading reranker: {settings.reranker_model}")
        t0 = time.time()
        _RERANKER = CrossEncoder(settings.reranker_model, max_length=1024)
        logger.info(f"Reranker ready in {time.time() - t0:.1f}s")
    return _RERANKER


@dataclass(slots=True)
class RetrievedChunk:
    """One chunk plus per-stage scores — generation reads `.chunk`, debug reads the rest."""

    chunk: Chunk
    score: float                                      # final score (rerank logit)
    rank: int                                         # 0-indexed final rank
    stage_scores: dict[str, float] = field(default_factory=dict)


def _dense_topk(index: IndexHandle, query: str, k: int) -> list[tuple[str, float]]:
    """Query Chroma; return [(id, cosine_similarity), ...] in rank order."""
    qv = index.embedder.encode([query], normalize_embeddings=True).tolist()
    res = index.collection.query(query_embeddings=qv, n_results=k)
    ids = res["ids"][0]
    # hnsw:space=cosine → distances are (1 - cosine_similarity).
    distances = (res["distances"] or [[]])[0]
    return [(cid, 1.0 - float(d)) for cid, d in zip(ids, distances, strict=True)]


def _bm25_topk(index: IndexHandle, query: str, k: int) -> list[tuple[str, float]]:
    """Score every doc with BM25; return top-k (id, score) with score > 0."""
    scores = index.bm25.get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:k]
    return [(index.bm25_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]


def rrf_fuse(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion across rankings.

    RRF score for doc d = Σ_r 1 / (k + rank_r(d)), with rank_r 1-indexed.
    `k=60` is the canonical constant from Cormack et al. (2009).
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _score) in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def retrieve(
    query: str,
    index: IndexHandle,
    *,
    dense_top_k: int | None = None,
    bm25_top_k: int | None = None,
    rerank_top_k: int | None = None,
    final_top_k: int | None = None,
    use_router: bool | None = None,
    debug: bool = False,
) -> list[RetrievedChunk]:
    """Hybrid retrieval: dense + BM25 → RRF → (optional router boost) → rerank → top-k.

    Args:
        query: user question.
        index: loaded :class:`IndexHandle`.
        dense_top_k, bm25_top_k, rerank_top_k, final_top_k: per-stage cut-offs;
            default to the values in :mod:`src.config`.
        use_router: override :attr:`settings.use_router` for this call. When True
            the query is classified and modality-specific boosts multiply the
            fused RRF scores before the rerank slice.
        debug: if True, log per-stage top-5 results and timings.

    Returns:
        :class:`RetrievedChunk` objects in final rank order (highest score first).
    """
    dense_k = dense_top_k or settings.dense_top_k
    bm25_k = bm25_top_k or settings.bm25_top_k
    rerank_k = rerank_top_k or settings.rerank_top_k
    final_k = final_top_k or settings.final_top_k
    router_on = settings.use_router if use_router is None else use_router

    t0 = time.time()
    dense_hits = _dense_topk(index, query, dense_k)
    t_dense = time.time() - t0

    t0 = time.time()
    bm25_hits = _bm25_topk(index, query, bm25_k)
    t_bm25 = time.time() - t0

    dense_scores = dict(dense_hits)
    bm25_scores = dict(bm25_hits)

    fused = rrf_fuse([dense_hits, bm25_hits], k=settings.rrf_k)
    rrf_scores = dict(fused)
    candidate_ids = [cid for cid, _ in fused[:rerank_k]]

    router_config: RetrievalConfig | None = None
    if router_on:
        router_config = classify(query)

    # Filter out any candidates that somehow aren't in chunks_by_id (shouldn't happen,
    # but keeps us robust against a stale Chroma collection).
    candidates = [index.chunks_by_id[cid] for cid in candidate_ids if cid in index.chunks_by_id]
    if not candidates:
        return []

    # Force reranker load before timing — first call can download ~2GB of weights.
    reranker = _get_reranker()
    t0 = time.time()
    pairs: list[list[str]] = [[query, c.content] for c in candidates]
    raw = reranker.predict(pairs)
    rerank_scores = [float(s) for s in np.asarray(raw).tolist()]
    t_rerank = time.time() - t0

    # Router: nudge the final ordering toward the expected modality. We sort on a
    # boosted score but keep `rerank_scores[i]` as the reported score (boost is a
    # tie-breaker / ordering tweak, not a "real" relevance signal).
    boosts = router_config.modality_boosts if router_config else {}
    final_scores = [
        rerank_scores[i] * boosts.get(candidates[i].modality, 1.0)
        for i in range(len(candidates))
    ]
    order = sorted(range(len(candidates)), key=lambda i: final_scores[i], reverse=True)[:final_k]

    results: list[RetrievedChunk] = []
    for rank, i in enumerate(order):
        c = candidates[i]
        results.append(
            RetrievedChunk(
                chunk=c,
                score=rerank_scores[i],
                rank=rank,
                stage_scores={
                    "dense": dense_scores.get(c.id, 0.0),
                    "bm25": bm25_scores.get(c.id, 0.0),
                    "rrf": rrf_scores.get(c.id, 0.0),
                    "rerank": rerank_scores[i],
                },
            )
        )

    if debug:
        if router_config is not None:
            logger.info(
                f"router: class={router_config.query_class} boosts={router_config.modality_boosts}"
            )
        logger.info(
            f"timings — dense: {t_dense*1000:.0f}ms  "
            f"bm25: {t_bm25*1000:.0f}ms  rerank: {t_rerank*1000:.0f}ms"
        )
        _log_stage("dense", dense_hits[:5], index, "sim")
        _log_stage("bm25", bm25_hits[:5], index, "score")
        logger.info(f"rerank top-{len(results)}:")
        for r in results:
            logger.info(
                f"  #{r.rank+1}  p{r.chunk.page} {r.chunk.modality:<5}  "
                f"rerank={r.stage_scores['rerank']:+.2f}  rrf={r.stage_scores['rrf']:.3f}  "
                f":: {_snippet(r.chunk.content, 80)!r}"
            )

    return results


def _log_stage(
    name: str, hits: list[tuple[str, float]], index: IndexHandle, score_label: str
) -> None:
    logger.info(f"{name} top-{len(hits)}:")
    for i, (cid, s) in enumerate(hits):
        c = index.chunks_by_id.get(cid)
        page = c.page if c else "?"
        mod = c.modality if c else "?"
        snippet = _snippet(c.content, 60) if c else cid
        logger.info(f"  {i+1}. p{page} {mod:<5} {score_label}={s:.3f} :: {snippet!r}")


def _snippet(text: str, n: int) -> str:
    s = " ".join(text.split())
    return s[:n] + ("…" if len(s) > n else "")


@app.command()
def main(
    query: str = typer.Argument(..., help="Query string."),
    debug: bool = typer.Option(False, "--debug", help="Log per-stage top-5 + timings."),
    router: bool = typer.Option(
        None, "--router/--no-router", help="Force router on/off (else uses USE_ROUTER env)."
    ),
    index_dir: Path = typer.Option(
        None, "--index", help="Chroma directory (bm25.pkl + manifest.json). Defaults to CHROMA_PATH."
    ),
) -> None:
    """Retrieve top-k chunks for `query`."""
    setup_logging()
    out = (index_dir or settings.chroma_path).resolve()
    index = load_index(out)
    results = retrieve(query, index, use_router=router, debug=debug)
    print()
    for r in results:
        print(
            f"#{r.rank+1}  [p.{r.chunk.page}, {r.chunk.modality}]  "
            f"rerank={r.stage_scores['rerank']:+.2f}"
        )
        print(f"   {_snippet(r.chunk.content, 220)}")
        print()


if __name__ == "__main__":
    app()
