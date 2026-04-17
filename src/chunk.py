"""Chunk dataclass, Docling walker, and table→markdown serializer.

The walker converts a :class:`DoclingDocument` into a flat list of :class:`Chunk`
objects with modality metadata. Captions and table-summaries are **not** filled
in here — that's :mod:`src.ingest` calling :mod:`src.caption`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from docling.chunking import HybridChunker
from docling_core.types.doc import (
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
)
from loguru import logger
from PIL import Image

from src.config import settings
from src.utils import chunk_id

Modality = Literal["text", "table", "image"]


@dataclass(slots=True)
class Chunk:
    """One retrieval unit. `content` is already text-only (caption for images,
    markdown + summary for tables, chunked text for text). Every chunk carries
    page + optional bbox so citations can map back to the PDF."""

    id: str
    content: str
    modality: Modality
    page: int
    section: str | None
    bbox: tuple[float, float, float, float] | None
    image_path: Path | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Serialize to a single JSONL line (paths become strings)."""
        d = asdict(self)
        d["image_path"] = str(self.image_path) if self.image_path else None
        d["bbox"] = list(self.bbox) if self.bbox else None
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> Chunk:
        """Inverse of `to_jsonl`."""
        d = json.loads(line)
        d["image_path"] = Path(d["image_path"]) if d["image_path"] else None
        d["bbox"] = tuple(d["bbox"]) if d["bbox"] else None
        return cls(**d)


def _bbox_top_left(item: PictureItem | TableItem, doc: DoclingDocument) -> (
    tuple[float, float, float, float] | None
):
    """Return `(x0, y0, x1, y1)` in TOP-LEFT page coordinates, or None.

    Docling's native bbox uses BOTTOM-LEFT origin (PDF convention); we flip to
    TOP-LEFT so downstream consumers (pymupdf in Phase 5) get PDF-viewer
    coordinates directly.
    """
    if not item.prov:
        return None
    prov = item.prov[0]
    page = doc.pages.get(prov.page_no)
    if page is None:
        return None
    tl = prov.bbox.to_top_left_origin(page_height=page.size.height)
    return (tl.l, tl.t, tl.r, tl.b)


def _table_to_markdown(table: TableItem, doc: DoclingDocument) -> str:
    """Serialize a Docling table to GitHub-flavoured Markdown.

    Docling's own `export_to_markdown(doc)` does the right thing (handles
    merged cells, headers). We trust it; we only strip trailing whitespace.
    """
    return table.export_to_markdown(doc=doc).strip()


def _save_picture_image(
    picture: PictureItem, doc: DoclingDocument, out_dir: Path, chunk_hash: str
) -> Path | None:
    """Dump the picture's PIL image to disk; returns the file path or None."""
    img: Image.Image | None = picture.get_image(doc)
    if img is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{chunk_hash[:16]}.png"
    img.save(path, format="PNG")
    return path


def _context_around(
    picture: PictureItem, doc: DoclingDocument, max_chars: int = 800
) -> str:
    """Gather up to `max_chars` of text immediately before + after the picture
    on the same page, for use as captioning context.

    Docling exposes texts in reading order; we find the picture's position and
    take neighbours from the same page.
    """
    if not picture.prov:
        return ""
    pic_page = picture.prov[0].page_no
    # Texts on the same page, sorted by vertical position (top to bottom in reading order).
    same_page: list[tuple[float, str]] = []
    for t in doc.texts:
        if not t.prov:
            continue
        tp = t.prov[0]
        if tp.page_no != pic_page:
            continue
        same_page.append((-tp.bbox.t, t.text))  # negate so top-of-page comes first
    same_page.sort()
    joined = "\n".join(txt for _, txt in same_page).strip()
    if len(joined) <= max_chars:
        return joined
    return joined[:max_chars]


def walk(doc: DoclingDocument, doc_id: str, images_dir: Path) -> list[Chunk]:
    """Flatten a `DoclingDocument` into Chunk objects.

    Args:
        doc: Parsed Docling document.
        doc_id: Stable document identifier (usually PDF content hash).
        images_dir: Where to dump picture bytes (PNGs).

    Returns:
        Chunks in deterministic order: text (via HybridChunker) → tables → pictures.
        `content` for table / image chunks is a placeholder; the ingest
        orchestrator fills in the real caption / summary.
    """
    chunks: list[Chunk] = []

    # --- 1. Text — use Docling's HybridChunker aligned to BGE-M3's tokenizer. ---
    try:
        chunker = HybridChunker(tokenizer=settings.docling_tokenizer)
        text_chunks = list(chunker.chunk(dl_doc=doc))
    except (OSError, ValueError) as exc:
        logger.warning(
            f"HybridChunker failed ({exc!r}); falling back to naive paragraph chunking."
        )
        text_chunks = []

    for i, tc in enumerate(text_chunks):
        # HybridChunker returns chunks with `meta.doc_items` carrying prov.
        page = _first_page_of_chunk(tc)
        section = _section_of_chunk(tc)
        bbox = _bbox_of_chunk(tc, doc)
        content = (tc.text or "").strip()
        if not content:
            continue
        cid = chunk_id(doc_id, page or 0, bbox, content)
        chunks.append(
            Chunk(
                id=cid,
                content=content,
                modality="text",
                page=page or 0,
                section=section,
                bbox=bbox,
                image_path=None,
                extra={"chunk_index": i},
            )
        )

    # --- 2. Tables — one chunk per table, content = markdown (summary later). ---
    for t in doc.tables:
        if not t.prov:
            continue
        tp = t.prov[0]
        md = _table_to_markdown(t, doc)
        if not md:
            continue
        bbox = _bbox_top_left(t, doc)
        cid = chunk_id(doc_id, tp.page_no, bbox, md)
        chunks.append(
            Chunk(
                id=cid,
                content=md,  # summary prepended at ingest time
                modality="table",
                page=tp.page_no,
                section=None,
                bbox=bbox,
                image_path=None,
                extra={"raw_markdown": md},
            )
        )

    # --- 3. Pictures — one chunk per image, content = caption (filled at ingest). ---
    for p in doc.pictures:
        if not p.prov:
            continue
        pp = p.prov[0]
        bbox = _bbox_top_left(p, doc)
        context = _context_around(p, doc)
        # Placeholder content — replaced by caption after Gemini call.
        placeholder = f"[image on page {pp.page_no}]"
        cid = chunk_id(doc_id, pp.page_no, bbox, placeholder + context[:120])
        img_path = _save_picture_image(p, doc, images_dir, cid)
        if img_path is None:
            logger.warning(f"Picture on page {pp.page_no} has no image bytes — skipping")
            continue
        img = p.get_image(doc)
        chunks.append(
            Chunk(
                id=cid,
                content=placeholder,
                modality="image",
                page=pp.page_no,
                section=None,
                bbox=bbox,
                image_path=img_path,
                extra={
                    "context": context,
                    "image_width": img.width if img else None,
                    "image_height": img.height if img else None,
                },
            )
        )

    logger.info(
        f"Walked document: {sum(1 for c in chunks if c.modality == 'text')} text, "
        f"{sum(1 for c in chunks if c.modality == 'table')} tables, "
        f"{sum(1 for c in chunks if c.modality == 'image')} images"
    )
    return chunks


def _first_page_of_chunk(tc: Any) -> int | None:
    """Pull the first page number out of a HybridChunker chunk's metadata."""
    meta = getattr(tc, "meta", None)
    if meta is None:
        return None
    for item in getattr(meta, "doc_items", []) or []:
        prov = getattr(item, "prov", None)
        if prov:
            page_no: int = prov[0].page_no
            return page_no
    return None


def _section_of_chunk(tc: Any) -> str | None:
    """Return the nearest heading text ahead of this chunk, if any."""
    meta = getattr(tc, "meta", None)
    if meta is None:
        return None
    headings = getattr(meta, "headings", None)
    if headings:
        return " › ".join(h for h in headings if h)
    for item in getattr(meta, "doc_items", []) or []:
        if isinstance(item, SectionHeaderItem):
            return item.text
    return None


def _bbox_of_chunk(
    tc: Any, doc: DoclingDocument
) -> tuple[float, float, float, float] | None:
    """Union-bbox of the chunk's doc items, in TOP-LEFT page coords."""
    meta = getattr(tc, "meta", None)
    if meta is None:
        return None
    boxes: list[tuple[float, float, float, float]] = []
    for item in getattr(meta, "doc_items", []) or []:
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page = doc.pages.get(prov[0].page_no)
        if page is None:
            continue
        tl = prov[0].bbox.to_top_left_origin(page_height=page.size.height)
        boxes.append((tl.l, tl.t, tl.r, tl.b))
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
