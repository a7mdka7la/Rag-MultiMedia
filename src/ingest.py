"""Phase-1 CLI: parse a PDF with Docling, caption pictures and summarise tables
with Gemini, then write a deterministic `chunks.jsonl` to disk.

Usage:

    uv run python -m src.ingest --pdf data/source.pdf --out data/
    uv run python -m src.ingest --force          # bust the Docling parse cache too

Re-running on an unchanged PDF is a no-op thanks to the caption cache +
parse-result pickle (< 10 s).
"""

from __future__ import annotations

import pickle
import time
from collections import Counter
from pathlib import Path

import typer
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument
from loguru import logger

from src.caption import Captioner
from src.chunk import Chunk, walk
from src.config import settings
from src.utils import ensure_dir, sha256_bytes, setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _parse_pdf(pdf_path: Path, cache_dir: Path, *, force: bool) -> DoclingDocument:
    """Parse the PDF once, pickle the resulting DoclingDocument for reuse.

    Docling's conversion is the slow step (~30 s for the IMF report); we cache
    the converted document keyed on the PDF's content hash so re-running
    ingest (without --force) reparses only when the PDF changes.
    """
    doc_hash = sha256_bytes(pdf_path.read_bytes())
    parse_cache = ensure_dir(cache_dir / "parse") / f"{doc_hash}.pkl"
    if parse_cache.exists() and not force:
        logger.info(f"Using cached Docling parse at {parse_cache.name}")
        with parse_cache.open("rb") as f:
            return pickle.load(f)

    logger.info(f"Parsing {pdf_path.name} with Docling (do_ocr=False, images_scale=2.0)...")
    opts = PdfPipelineOptions()
    opts.do_ocr = settings.docling_ocr
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
    )
    t0 = time.time()
    doc = converter.convert(pdf_path).document
    logger.info(f"Docling parse done in {time.time() - t0:.1f}s")
    with parse_cache.open("wb") as f:
        pickle.dump(doc, f, protocol=pickle.HIGHEST_PROTOCOL)
    return doc


def _fill_captions_and_summaries(
    chunks: list[Chunk], captioner: Captioner
) -> tuple[int, int, int]:
    """In-place: replace image chunks' content with Gemini captions and prepend
    table summaries to table chunks.

    Per-chunk failures (e.g. a 503 that didn't recover inside the retry budget)
    are logged and leave the chunk's placeholder content in place — the rest of
    the pipeline continues. Re-running ingest later will hit fresh chunks only
    via the cache, so transient outages cost at most one extra run.

    Returns: (cache_hits, fresh_calls, failures).
    """
    from google.genai import errors as genai_errors

    hits = fresh = failures = 0
    quota_exhausted = False
    for c in chunks:
        if quota_exhausted and c.modality in ("image", "table"):
            # Once every Gemini key has hit its daily limit, skip remaining
            # captions rather than re-raising N more times. Cache will pick
            # them up on the next run.
            c.extra["caption_error"] = "all Gemini keys exhausted (daily quota)"
            failures += 1
            continue
        try:
            if c.modality == "image":
                if c.image_path is None or not c.image_path.exists():
                    logger.warning(
                        f"Image chunk {c.id[:12]} missing image_path — skipping caption"
                    )
                    failures += 1
                    continue
                image_bytes = c.image_path.read_bytes()
                context = str(c.extra.get("context", ""))
                result = captioner.caption_image(image_bytes, context)
                c.content = result.text
                c.extra["caption_prompt_version"] = result.prompt_version
            elif c.modality == "table":
                md = c.content
                result = captioner.summarize_table(md)
                c.content = f"{result.text}\n\n{md}"
                c.extra["table_summary"] = result.text
                c.extra["summary_prompt_version"] = result.prompt_version
            else:
                continue
            if result.cache_hit:
                hits += 1
            else:
                fresh += 1
        except RuntimeError as exc:
            msg = str(exc)
            if "exhausted" in msg and "Gemini" in msg:
                logger.error(
                    f"{msg} Stopping caption pass; {len(chunks) - (hits + fresh + failures)} "
                    f"remaining chunks will keep placeholder content."
                )
                c.extra["caption_error"] = msg
                failures += 1
                quota_exhausted = True
                continue
            logger.error(
                f"Captioning failed for {c.modality} chunk {c.id[:12]} "
                f"on page {c.page}: {exc!r}. Re-run to retry — cache will skip "
                f"already-captioned items."
            )
            c.extra["caption_error"] = str(exc)
            failures += 1
        except genai_errors.APIError as exc:
            logger.error(
                f"Captioning failed for {c.modality} chunk {c.id[:12]} "
                f"on page {c.page}: {exc!r}. Re-run to retry — cache will skip "
                f"already-captioned items."
            )
            c.extra["caption_error"] = str(exc)
            failures += 1
    return hits, fresh, failures


def _write_jsonl(chunks: list[Chunk], path: Path) -> None:
    """Write chunks to a JSONL file (one chunk per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.to_jsonl())
            f.write("\n")


def ingest(pdf_path: Path, out_dir: Path, *, force: bool = False) -> list[Chunk]:
    """Run the Phase-1 ingestion pipeline end-to-end.

    Args:
        pdf_path: Absolute path to the source PDF.
        out_dir: Output root. Subpaths (images/, cache/, chunks.jsonl) resolve under
            this dir unless the corresponding .env path is already absolute.
        force: If True, bust the Docling parse cache (captions still hit their own cache).

    Returns:
        The final list of Chunk objects (also written to `out_dir/chunks.jsonl`).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Resolve output paths: if the configured path is under the default `data/`
    # root, redirect it to `out_dir`; otherwise honour whatever .env said.
    default_root = (settings.project_root / "data").resolve()
    def _rebase(p: Path) -> Path:
        try:
            return out_dir / p.relative_to(default_root)
        except ValueError:
            return p
    images_dir = ensure_dir(_rebase(settings.images_dir))
    cache_dir = ensure_dir(_rebase(settings.cache_dir))
    chunks_path = _rebase(settings.chunks_path)

    doc = _parse_pdf(pdf_path, cache_dir, force=force)
    doc_id = sha256_bytes(pdf_path.read_bytes())[:16]
    chunks = walk(doc, doc_id=doc_id, images_dir=images_dir)
    captioner = Captioner()
    t0 = time.time()
    hits, fresh, failures = _fill_captions_and_summaries(chunks, captioner)
    logger.info(
        f"Caption/summary pass: {hits} cache hits, {fresh} fresh calls, "
        f"{failures} failures in {time.time() - t0:.1f}s"
    )

    _write_jsonl(chunks, chunks_path)
    logger.info(f"Wrote {len(chunks)} chunks to {chunks_path}")
    if failures:
        logger.warning(
            f"{failures} chunk(s) were not captioned due to errors. "
            f"Re-run 'python -m src.ingest' later — cached successes will not be recomputed."
        )
    return chunks


@app.command()
def main(
    pdf: Path = typer.Option(
        None, "--pdf", help="Path to the source PDF. Defaults to PDF_PATH from .env."
    ),
    out: Path = typer.Option(
        None, "--out", help="Output root dir. Defaults to project data/."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-parse the PDF even if a cached parse exists."
    ),
) -> None:
    """CLI entrypoint: parse PDF + caption + write chunks.jsonl."""
    setup_logging()
    pdf_path = (pdf or settings.pdf_path).resolve()
    out_dir = (out or settings.project_root / "data").resolve()
    chunks = ingest(pdf_path, out_dir, force=force)
    modality = Counter(c.modality for c in chunks)
    # User-facing summary — `print` is allowed in CLI entry points.
    print(
        f"Parsed {len(chunks)} chunks: "
        f"{modality['text']} text, {modality['table']} tables, {modality['image']} images"
    )


if __name__ == "__main__":
    app()
