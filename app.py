"""Streamlit chat UI for the multimodal IMF RAG.

Run: `uv run streamlit run app.py`

Lays out a single-pane chat. On each user turn:
  1. Retrieve top-5 chunks via `src.retrieve`.
  2. Stream an answer via `src.generate`.
  3. Render a Sources panel — one card per citation, with text snippet / table
     markdown / image as appropriate, plus the cited PDF page with the source
     region highlighted in red (PyMuPDF, cached to `data/cache/pages/`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz  # pymupdf
import streamlit as st

from src.config import settings
from src.generate import answer_stream, parse_citations, _resolve_chunk_refs
from src.index import load_index
from src.retrieve import RetrievedChunk, retrieve

# --------------------------- setup ---------------------------

st.set_page_config(
    page_title="IMF Article IV — Multimodal RAG",
    page_icon="🧭",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading index + models…")
def _get_index():
    """Single process-wide index handle (BGE-M3 + Chroma + BM25)."""
    return load_index(settings.chroma_path)


@st.cache_data(show_spinner=False)
def _render_page_with_bbox(
    pdf_path: str,
    page: int,
    bbox: tuple[float, float, float, float],
) -> bytes:
    """Render a single PDF page with a red rectangle over `bbox`.

    Cached on disk at `data/cache/pages/{page}_{bbox_hash}.png` so the same
    citation doesn't re-render on every rerun.
    """
    bbox_hash = hashlib.sha256(repr(bbox).encode()).hexdigest()[:12]
    cache_dir = settings.cache_dir / "pages"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"p{page}_{bbox_hash}.png"
    if cache_path.exists():
        return cache_path.read_bytes()

    doc = fitz.open(pdf_path)
    try:
        p = doc.load_page(page - 1)  # PDF pages are 1-indexed in our metadata
        rect = fitz.Rect(*bbox)
        p.draw_rect(rect, color=(1, 0, 0), width=2)
        pix = p.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2× zoom for clarity
        pix.save(str(cache_path))
    finally:
        doc.close()
    return cache_path.read_bytes()


# --------------------------- sidebar -------------------------

with st.sidebar:
    st.header("System")
    index = _get_index()
    st.metric("Chunks indexed", index.collection.count())
    st.caption(f"**Embedder**: `{settings.embedding_model}`")
    st.caption(f"**Reranker**: `{settings.reranker_model}`")
    st.caption(f"**Answer LLM**: `{settings.answer_model}` (Groq)")
    st.caption(f"**Captioner**: `{settings.gemini_model}` (Gemini)")
    st.divider()
    pdf_exists = settings.pdf_path.exists()
    st.caption(
        f"**PDF**: {'✅ `' + settings.pdf_path.name + '`' if pdf_exists else '⚠️ not found — page highlights disabled'}"
    )
    st.caption(f"**Router**: {'on' if settings.use_router else 'off (Phase 6)'}")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# --------------------------- main pane -----------------------

st.title("IMF Article IV — Multimodal RAG")
st.caption(
    "Ask about text, tables, or figures in the report. "
    "Every claim is cited back to the PDF with `[p.<page>, <modality>]`."
)


def _render_chunk_card(r: RetrievedChunk, i: int) -> None:
    c = r.chunk
    st.markdown(
        f"**{i}. \\[p.{c.page}, {c.modality}\\]**"
        + (f" — *{c.section}*" if c.section else "")
    )

    if c.modality == "image" and c.image_path and Path(c.image_path).exists():
        st.image(str(c.image_path), width=420)
        st.caption(c.content[:500])
    elif c.modality == "table":
        body = c.content if len(c.content) <= 1500 else c.content[:1500] + " …"
        st.markdown(body)
    else:
        snippet = c.content if len(c.content) <= 600 else c.content[:600] + " …"
        st.markdown(f"> {snippet}")

    # Bbox-highlighted PDF page — the mandatory multimodal grounding cue.
    if c.bbox and settings.pdf_path.exists():
        try:
            png = _render_page_with_bbox(str(settings.pdf_path), c.page, c.bbox)
            st.image(
                png,
                caption=f"Page {c.page} — cited region highlighted",
                width=560,
            )
        except Exception as exc:
            st.caption(f"(page render failed: {exc})")
    st.divider()


def _render_sources(refs: list[RetrievedChunk]) -> None:
    if not refs:
        return
    with st.expander(f"📄 Sources ({len(refs)})", expanded=False):
        for i, r in enumerate(refs, 1):
            _render_chunk_card(r, i)


# Chat history lives in session state so it persists across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])

prompt = st.chat_input("Ask a question about the report…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + reranking…"):
            retrieved = retrieve(prompt, index)

        placeholder = st.empty()
        parts: list[str] = []
        for delta in answer_stream(prompt, retrieved):
            parts.append(delta)
            placeholder.markdown("".join(parts) + "▌")
        full_text = "".join(parts)
        placeholder.markdown(full_text)

        cits = parse_citations(full_text)
        refs = _resolve_chunk_refs(cits, retrieved)
        _render_sources(refs)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_text, "sources": refs}
        )
