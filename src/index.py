"""Phase-2 CLI: embed chunks with BGE-M3, upsert to Chroma, build BM25, write manifest.

Usage:

    uv run python -m src.index --chunks data/chunks.jsonl --out data/chroma_db
    uv run python -m src.index --force   # re-embed even if chunks.jsonl hash matches

Re-running on an unchanged `chunks.jsonl` is a no-op (<5 s) — the manifest's
content hash matches and we skip straight to `load_index`.
"""

from __future__ import annotations

import json
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import chromadb
import typer
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.chunk import Chunk
from src.config import settings
from src.utils import ensure_dir, setup_logging, sha256_bytes

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

app = typer.Typer(add_completion=False, no_args_is_help=False)

_MANIFEST_NAME = "manifest.json"
_BM25_NAME = "bm25.pkl"


def tokenize(text: str) -> list[str]:
    """Lowercased whitespace tokenization shared by BM25 indexing and querying.

    `rank-bm25` doesn't do subwords; keeping it simple here and reusing the
    same function at query time guarantees the tokenizations match.
    """
    return text.lower().split()


def _flatten_metadata(c: Chunk) -> dict[str, Any]:
    """Produce Chroma-compatible metadata (scalars only).

    Chroma rejects tuples and dicts; `bbox` becomes four floats and `extra`
    becomes a JSON string. None values are omitted (Chroma rejects them).
    """
    md: dict[str, Any] = {
        "modality": c.modality,
        "page": c.page,
        "section": c.section or "",
        "image_path": str(c.image_path) if c.image_path else "",
        "extra_json": json.dumps(c.extra, ensure_ascii=False),
    }
    if c.bbox is not None:
        x0, y0, x1, y1 = c.bbox
        md["bbox_x0"] = float(x0)
        md["bbox_y0"] = float(y0)
        md["bbox_x1"] = float(x1)
        md["bbox_y1"] = float(y1)
    return md


def _load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(Chunk.from_jsonl(line))
    return chunks


@dataclass(slots=True)
class IndexHandle:
    """Everything a retriever needs: dense collection + sparse BM25 + shared embedder + chunks."""

    collection: "Collection"
    bm25: BM25Okapi
    bm25_ids: list[str]          # chunk ids aligned with bm25 corpus order
    embedder: SentenceTransformer
    chunks_by_id: dict[str, Chunk]  # id → full Chunk; used by retrieve to resolve hits


def build_index(
    chunks_path: Path,
    out_dir: Path,
    *,
    force: bool = False,
) -> IndexHandle:
    """Embed all chunks, upsert to Chroma, build BM25, write manifest.

    Args:
        chunks_path: JSONL produced by Phase 1 ingest.
        out_dir: destination dir — becomes the Chroma persistent path and also
            holds `bm25.pkl` and `manifest.json`.
        force: if True, rebuild even when the manifest's chunks-hash matches.

    Returns:
        :class:`IndexHandle` ready for retrieval.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")

    ensure_dir(out_dir)
    chunks_hash = sha256_bytes(chunks_path.read_bytes())
    manifest_path = out_dir / _MANIFEST_NAME
    if manifest_path.exists() and not force:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if manifest.get("chunks_hash") == chunks_hash:
            logger.info("chunks.jsonl unchanged (hash match) — loading existing index.")
            return load_index(out_dir)
        logger.info("chunks.jsonl changed — rebuilding index.")

    chunks = _load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"{chunks_path} has no chunks to index")

    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path.name}")
    texts = [c.content for c in chunks]
    ids = [c.id for c in chunks]

    # --- BGE-M3 dense embeddings ----------------------------------------
    logger.info(f"Loading embedder: {settings.embedding_model}")
    t0 = time.time()
    embedder = SentenceTransformer(settings.embedding_model)
    # BGE-M3 defaults to 8192 tokens — on Apple MPS padded batches can blow
    # through the MPS buffer cap. 1024 is plenty for our chunk sizes.
    embedder.max_seq_length = 1024
    logger.info(f"Embedder ready in {time.time() - t0:.1f}s")

    logger.info(f"Embedding {len(chunks)} chunks (batch_size=32)...")
    t0 = time.time()
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    logger.info(f"Embedding done in {time.time() - t0:.1f}s")

    # --- Chroma persistent collection (recreate so upsert semantics are clean) ---
    client = chromadb.PersistentClient(path=str(out_dir))
    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:  # noqa: BLE001  — collection may not exist yet; either is fine
        pass
    collection = client.create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},  # BGE-M3 embeddings are L2-normalized → cosine
    )
    # Chunked add (Chroma limits per-call record counts).
    batch = 1000
    for i in range(0, len(chunks), batch):
        collection.add(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch].tolist(),
            documents=texts[i : i + batch],
            metadatas=[_flatten_metadata(c) for c in chunks[i : i + batch]],
        )
    logger.info(
        f"Chroma collection '{settings.chroma_collection_name}' has {collection.count()} items"
    )

    # --- BM25 sparse ----------------------------------------------------
    logger.info("Building BM25 index...")
    corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus)
    with (out_dir / _BM25_NAME).open("wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "ids": ids,
                "tokenized_corpus": corpus,
                # Phase 3 needs full Chunk objects (bbox / image_path / section)
                # to resolve retrieval hits back to the UI.
                "chunks": chunks,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    logger.info(f"BM25 index saved to {_BM25_NAME}")

    # --- Manifest -------------------------------------------------------
    modality_counts = Counter(c.modality for c in chunks)
    manifest = {
        "chunk_count": len(chunks),
        "modality_counts": dict(modality_counts),
        "chunks_hash": chunks_hash,
        "embedding_model": settings.embedding_model,
        "collection": settings.chroma_collection_name,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        f"Indexed {manifest['chunk_count']} chunks "
        f"(text={modality_counts.get('text', 0)}, "
        f"table={modality_counts.get('table', 0)}, "
        f"image={modality_counts.get('image', 0)})"
    )

    return IndexHandle(
        collection=collection,
        bm25=bm25,
        bm25_ids=ids,
        embedder=embedder,
        chunks_by_id={c.id: c for c in chunks},
    )


def load_index(out_dir: Path) -> IndexHandle:
    """Load an existing index from disk. Does NOT rebuild.

    Raises FileNotFoundError if manifest or bm25 pickle is missing.
    """
    manifest_path = out_dir / _MANIFEST_NAME
    bm25_path = out_dir / _BM25_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found — run `make index` first."
        )
    if not bm25_path.exists():
        raise FileNotFoundError(
            f"{bm25_path} not found — run `make index` first."
        )

    client = chromadb.PersistentClient(path=str(out_dir))
    collection = client.get_collection(name=settings.chroma_collection_name)
    with bm25_path.open("rb") as f:
        bm25_data = pickle.load(f)
    chunks = bm25_data.get("chunks")
    if chunks is None:
        raise RuntimeError(
            "bm25.pkl is missing 'chunks' — rebuild with `make index` (Phase 3 format)."
        )
    embedder = SentenceTransformer(settings.embedding_model)
    embedder.max_seq_length = 1024
    return IndexHandle(
        collection=collection,
        bm25=bm25_data["bm25"],
        bm25_ids=bm25_data["ids"],
        embedder=embedder,
        chunks_by_id={c.id: c for c in chunks},
    )


@app.command()
def main(
    chunks: Path = typer.Option(
        None, "--chunks", help="Path to chunks.jsonl. Defaults to CHUNKS_PATH from .env."
    ),
    out: Path = typer.Option(
        None, "--out", help="Output dir (Chroma + bm25.pkl + manifest). Defaults to CHROMA_PATH."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-embed even if the manifest's chunks-hash matches."
    ),
) -> None:
    """CLI entrypoint: build (or reuse) the dense + BM25 index."""
    setup_logging()
    chunks_path = (chunks or settings.chunks_path).resolve()
    out_dir = (out or settings.chroma_path).resolve()
    handle = build_index(chunks_path, out_dir, force=force)
    # User-facing summary — `print` is allowed in CLI entry points.
    print(
        f"Indexed {handle.collection.count()} chunks into "
        f"{settings.chroma_collection_name!r} at {out_dir}"
    )


if __name__ == "__main__":
    app()
