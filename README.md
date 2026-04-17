# IMF Article IV — Multimodal RAG

> DSAI 413 assignment — a question-answering system over a single complex IMF Article IV PDF that handles **text, tables, and images** with a unified retrieval pipeline, returns answers grounded in the document, and cites the page + modality for every claim.

> ⚠️ This README is a Phase-0 stub. The full version (architecture diagram, quickstart, eval results, limitations) lands in Phase 8.

## Quickstart (preview — not all targets work yet)

```bash
# 1. Install deps (uses uv — https://docs.astral.sh/uv/)
uv sync

# 2. Add secrets
cp .env.example .env
# edit .env and paste GOOGLE_API_KEY + GROQ_API_KEY

# 3. Drop the source PDF at data/source.pdf
# 4. Parse + index (Phase 1 + 2 — not yet implemented)
make ingest

# 5. Run the chat UI (Phase 5 — not yet implemented)
make run
```

## Status

| Phase | Scope | Status |
| :--- | :--- | :--- |
| 0 | Scaffold | ✅ |
| 1 | Ingestion (Docling + Gemini captioning) | ⬜ |
| 2 | Indexing (BGE-M3 + Chroma + BM25) | ⬜ |
| 3 | Retrieval (dense + BM25 + RRF + rerank) | ⬜ |
| 4 | Generation (Groq Llama 3.3 70B + citations) | ⬜ |
| 5 | Streamlit UI + bbox highlighting | ⬜ |
| 6 | Query router (innovation) | ⬜ |
| 7 | RAGAS evaluation | ⬜ |
| 8 | Polish + docs/report.md + docs/video_script.md | ⬜ |

See [CLAUDE.md](CLAUDE.md) for architecture decisions and working-context notes.
