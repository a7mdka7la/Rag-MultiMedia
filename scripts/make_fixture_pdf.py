"""One-off generator for tests/fixtures/sample.pdf.

Run once, check the resulting PDF into git so tests don't depend on pymupdf's
output being byte-stable across versions:

    uv run python scripts/make_fixture_pdf.py

The file is a 2-page document with a heading, a paragraph, a tiny table
rendered as text (since we're not exercising table detection here — that's
Phase 1's job on the real PDF), and an embedded image so ingestion has
something picture-like to find.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf

OUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample.pdf"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _draw_page_one(page: fitz.Page) -> None:
    page.insert_text(
        (72, 90),
        "Sample Fixture Document",
        fontsize=18,
        fontname="helv",
    )
    page.insert_text(
        (72, 120),
        "Section 1: Overview",
        fontsize=14,
        fontname="helv",
    )
    para = (
        "This fixture exercises the ingestion pipeline with a tiny, deterministic "
        "PDF. It contains a heading, a paragraph, and on page two a figure. "
        "The embedded text layer is real so Docling parses it without OCR."
    )
    page.insert_textbox(
        fitz.Rect(72, 140, 540, 220),
        para,
        fontsize=11,
        fontname="helv",
    )


def _draw_page_two(page: fitz.Page) -> None:
    page.insert_text(
        (72, 90),
        "Section 2: A figure",
        fontsize=14,
        fontname="helv",
    )
    # Simple synthetic figure — a rectangle with a label underneath.
    rect = fitz.Rect(72, 120, 320, 260)
    page.draw_rect(rect, color=(0.2, 0.4, 0.9), fill=(0.85, 0.90, 0.98), width=1.5)
    page.insert_text(
        (96, 190),
        "Synthetic chart placeholder",
        fontsize=12,
        fontname="helv",
    )
    page.insert_text(
        (72, 280),
        "Figure 1. A blue rectangle standing in for a chart.",
        fontsize=10,
        fontname="helv",
    )


def main() -> None:
    doc = fitz.open()
    _draw_page_one(doc.new_page(width=612, height=792))
    _draw_page_two(doc.new_page(width=612, height=792))
    doc.save(str(OUT_PATH), deflate=True, clean=True)
    doc.close()
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
