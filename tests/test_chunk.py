"""Tests for src.chunk — Chunk dataclass + Docling walker.

The page-index sanity test is the spec's explicit footgun: Docling's page
numbers must match the PDF viewer's 1-indexed pages, otherwise every citation
downstream will be off-by-one.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.chunk import Chunk, walk
from src.utils import sha256_bytes

# --------------------------- dataclass --------------------------


def test_chunk_jsonl_roundtrip(tmp_path: Path) -> None:
    """to_jsonl → from_jsonl round-trips every field, including Path + tuple."""
    img = tmp_path / "foo.png"
    img.write_bytes(b"fake")
    original = Chunk(
        id="abc123",
        content="hello",
        modality="image",
        page=4,
        section="Section 1 › Intro",
        bbox=(10.0, 20.0, 100.0, 200.0),
        image_path=img,
        extra={"context": "surrounding text", "caption_prompt_version": "image-v1"},
    )
    restored = Chunk.from_jsonl(original.to_jsonl())
    assert restored == original


def test_chunk_jsonl_handles_none_optional_fields() -> None:
    """bbox and image_path may be None (text chunks); the round-trip must preserve that."""
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
    restored = Chunk.from_jsonl(c.to_jsonl())
    assert restored == c
    # Sanity: JSON form puts None for bbox/image_path (not missing keys).
    payload = json.loads(c.to_jsonl())
    assert payload["bbox"] is None
    assert payload["image_path"] is None


# --------------------------- walker -----------------------------


@pytest.fixture(scope="module")
def walked_fixture(sample_pdf: Path, tmp_path_factory: pytest.TempPathFactory) -> list[Chunk]:
    """Parse the fixture PDF once per module and run `walk` over it.

    Cached on disk with a content-hash key so re-runs of the whole suite
    don't repeat the slow Docling parse.
    """
    cache_root = tmp_path_factory.mktemp("walker")
    doc_hash = sha256_bytes(sample_pdf.read_bytes())
    cache_file = cache_root / f"{doc_hash}.pkl"

    if cache_file.exists():
        with cache_file.open("rb") as f:
            doc = pickle.load(f)
    else:
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.generate_picture_images = True
        opts.images_scale = 2.0
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
        )
        doc = converter.convert(sample_pdf).document
        with cache_file.open("wb") as f:
            pickle.dump(doc, f)

    images_dir = cache_root / "images"
    return walk(doc, doc_id=doc_hash[:16], images_dir=images_dir)


def test_walk_produces_text_and_image_chunks(walked_fixture: list[Chunk]) -> None:
    """Fixture has two text pages + a synthetic figure on page 2."""
    modalities = {c.modality for c in walked_fixture}
    assert "text" in modalities
    assert "image" in modalities
    # The fixture has no tables, only text + picture.


def test_walk_image_chunks_have_bytes_on_disk(walked_fixture: list[Chunk]) -> None:
    """Every image chunk's image_path must point to an actual PNG written by the walker."""
    images = [c for c in walked_fixture if c.modality == "image"]
    assert images, "fixture should surface at least one picture"
    for c in images:
        assert c.image_path is not None
        assert c.image_path.exists()
        assert c.image_path.suffix == ".png"


def test_walk_chunk_ids_are_deterministic(sample_pdf: Path, tmp_path: Path) -> None:
    """Re-running walk on the same document must yield identical chunk IDs."""
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
    )
    doc = converter.convert(sample_pdf).document

    a = walk(doc, doc_id="deadbeefdeadbeef", images_dir=tmp_path / "a")
    b = walk(doc, doc_id="deadbeefdeadbeef", images_dir=tmp_path / "b")
    assert [c.id for c in a] == [c.id for c in b]


def test_page_index_matches_pdf(walked_fixture: list[Chunk]) -> None:
    """Spec's explicit footgun guard: Docling pages must match PDF viewer pages (1-indexed).

    The fixture's page 2 contains the caption "Figure 1. A blue rectangle standing
    in for a chart." (see scripts/make_fixture_pdf.py) and its enclosing heading
    is "Section 2: A figure". Any text chunk whose content/section references
    page-2 material must report page == 2. An off-by-one would surface here.
    """
    page_two_hits = [
        c
        for c in walked_fixture
        if c.modality == "text"
        and ("Figure 1." in c.content or (c.section and "Section 2" in c.section))
    ]
    assert page_two_hits, "expected at least one text chunk tied to page 2 of the fixture"
    for c in page_two_hits:
        assert c.page == 2, (
            f"chunk {c.id[:12]} (section={c.section!r}) looks like page-2 content "
            f"but was tagged page={c.page}"
        )

    page_one_hits = [
        c
        for c in walked_fixture
        if c.modality == "text" and (c.section and "Section 1" in c.section)
    ]
    assert page_one_hits, "expected at least one text chunk under 'Section 1' (page 1)"
    for c in page_one_hits:
        assert c.page == 1


def test_image_chunk_on_expected_page(walked_fixture: list[Chunk]) -> None:
    """Fixture's single figure is drawn on page 2."""
    images = [c for c in walked_fixture if c.modality == "image"]
    assert len(images) >= 1
    assert all(c.page == 2 for c in images)
