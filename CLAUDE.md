# CLAUDE.md — Working context for this repo

> Read this first if you're picking the project up in a new session. It's the single source of truth; the full spec lives in the original prompt but you don't need it if you trust this file.

## Project summary

A multimodal RAG system over a single IMF Article IV PDF for the DSAI 413 assignment. Answers user questions about text, tables, and images in the report with mandatory page + modality citations. Single unified ChromaDB index, hybrid dense + BM25 retrieval with cross-encoder reranking, streaming Groq answers in a Streamlit chat UI. Ingestion is one-shot and cached.

## Tech stack (do not substitute)

| Component | Library / model | Identifier |
| :--- | :--- | :--- |
| PDF parsing | Docling | `docling` |
| Image captioning | Gemini 2.5 Flash | `gemini-2.5-flash` (**not** `gemini-2.0-flash`, retired March 2026) |
| Embeddings | BGE-M3 (multilingual) | `BAAI/bge-m3` via `sentence-transformers` |
| Reranker | BGE reranker v2 m3 | `BAAI/bge-reranker-v2-m3` via `sentence-transformers.CrossEncoder` |
| Vector store | ChromaDB | `chromadb` (persistent local) |
| Sparse retrieval | BM25 | `rank-bm25` (`BM25Okapi`) |
| Answer LLM | Llama 3.3 70B on Groq | `llama-3.3-70b-versatile`, via `groq` SDK |
| Router LLM | Llama 3.1 8B on Groq | `llama-3.1-8b-instant` |
| UI | Streamlit | `streamlit` |
| Eval | RAGAS | `ragas` (+ `langchain-core` transitive — see gotchas) |
| Page-image rendering | PyMuPDF | `pymupdf` |
| CLI | Typer | `typer` |
| Logging | Loguru | `loguru` |
| Config / secrets | python-dotenv + frozen `@dataclass` | `python-dotenv` |
| Retry | Tenacity | `tenacity` |
| Package mgmt | uv | Python 3.11–3.13 (pinned 3.12) |

**Never add:** LangChain chains / retrievers / agents in our code, LlamaIndex, Haystack, OpenAI CLIP, pypdf as a primary parser, or any agent framework.

## Architectural decisions (non-negotiable)

1. **Single embedding space.** Everything becomes text before embedding. Text chunks embed directly; tables → Markdown + 1-line LLM summary prepended; images → detailed Gemini-generated caption. One ChromaDB collection, one embedding model.
2. **Modality is metadata, not a separate index.** Every chunk carries `modality ∈ {"text","table","image"}`, `page`, `section`, `bbox`, and for images `image_path`. Retrieval is modality-agnostic; the UI uses modality to render.
3. **Hybrid retrieval with reranking.** Dense (BGE-M3) + BM25 fused by RRF (k=60) → top-20 → cross-encoder reranker → top-5 → LLM.
4. **Citations are mandatory.** The system prompt requires `[p.<page>, <modality>]` on every claim. The UI parses them back out and shows the source chunk (image / table / text) and — for chunks with a bbox — the cited PDF page with the bbox highlighted (PyMuPDF).
5. **Ingestion is one-shot and cached.** Re-running ingest on an unchanged PDF hits caches; nothing is re-captioned or re-embedded. Caption cache key = SHA-256(image bytes); table-summary cache key = SHA-256(markdown).

## Common commands

```bash
# Install / sync deps
uv sync

# Environment
cp .env.example .env        # then paste real API keys

# Full pipeline from clean state
make ingest                 # parse + index — produces chunks.jsonl, images/, chroma_db/, bm25.pkl
make run                    # launch Streamlit chat
make eval                   # run RAGAS over eval/questions.json → eval/results.md

# Sub-steps for iteration
make parse                  # src.ingest only — re-parse PDF
make index                  # src.index only — re-embed + re-BM25

# Quality gates
make test                   # pytest
make lint                   # ruff + mypy --strict src/
```

Per-module CLIs (useful for debugging a single stage):

```bash
uv run python -m src.ingest --pdf data/source.pdf --out data/
uv run python -m src.index --chunks data/chunks.jsonl --out data/chroma_db
uv run python -m src.retrieve "What is the GDP forecast for 2025?" --debug
uv run python -m src.generate "What does the report say about inflation?"
```

## Known gotchas

- **Working-directory trailing space.** The project path is `.../Multi media/assignment 1 ` (note the trailing space in `assignment 1 `). Always quote paths in shell commands; `cd` without quoting will fail.
- **RAGAS transitively depends on `langchain-core`.** That's fine — the "no LangChain" rule means no LangChain chains / agents in **our** code. RAGAS internally wraps LLMs through LangChain adapters; we configure it to point at Groq via `LangchainLLMWrapper(ChatOpenAI(base_url="https://api.groq.com/openai/v1", ...))`. This is not a code smell; it's how RAGAS is designed.
- **Groq free-tier rate limits.** 30 req/min on `llama-3.3-70b-versatile`. Evaluation runs questions **serially** with a configurable inter-question delay (`EVAL_INTER_QUESTION_DELAY_S`, default 2s) and wraps every Groq call in `tenacity.retry(wait_exponential, retry_if_exception_type(groq.RateLimitError))`.
- **Gemini SDK choice.** Use `google-genai` (the new unified SDK), **not** the deprecated `google-generativeai`.
- **Chroma scalar-only metadata.** Chroma won't accept tuples or dicts in metadata. We flatten `bbox` to four floats (`bbox_x0/y0/x1/y1`) and serialize `extra` as a JSON string in `extra_json`.
- **Docling flags.** Use `do_ocr=False` (the IMF PDF has an embedded text layer; OCR would slow us down and corrupt text). `generate_picture_images=True` is required so each picture element has bytes we can caption and display.
- **Page numbering.** Docling reports 1-indexed logical page numbers that match the PDF viewer. Tested end-to-end in `tests/test_chunk.py::test_page_index_matches_pdf` — don't remove it.
- **HybridChunker metadata layout.** Section headings land on `chunk.meta.headings`, not inside `chunk.text`. Our `_section_of_chunk` reads `meta.headings` first and only falls back to scanning `doc_items` for `SectionHeaderItem`. If a test searches for a heading in a chunk's content/text it will fail — search `section` instead (see `test_page_index_matches_pdf`).
- **Gemini free-tier daily quota = 20 RPD per key** on `gemini-2.5-flash`. The IMF PDF needs ~134 captions (56 pictures + 78 tables) — one key can't finish in a day. `Captioner` round-robins across a comma-separated `GOOGLE_API_KEYS` list in `.env`. A 429 whose message includes `PerDay` or `limit: 20` retires that key for the process; a per-minute 429 just rotates to the next live key and, if all are 429 in the same minute, sleeps 60s. When all keys hit daily quota, ingest finishes the current chunk, then skips the remaining uncaptioned chunks with a `caption_error` marker — re-run tomorrow and the caption cache picks up exactly where you left off.
- **Don't tenacity-retry 429.** Earlier `_is_retryable` treated 429 as transient and retried 5×; with daily-quota 429s that burns 5 of your 20 daily calls for one logical request. Retry is now 5xx-only (`errors.ServerError`); 429 is handled by the key-rotation layer above.
- **Loguru + tenacity.** Tenacity's default `before_sleep` callback uses stdlib `logging`, which loguru doesn't route. We pass a custom `_log_retry(state: RetryCallState)` so retry warnings actually appear in the loguru sink.
- **BGE-M3 `max_seq_length` on Apple MPS.** BGE-M3 defaults to 8192 tokens. On Apple Silicon, `sentence-transformers` routes to MPS by default and a padded batch can demand a buffer Chroma-sized (we hit `RuntimeError: Invalid buffer size: 128.00 GiB`). `src/index.py` pins `embedder.max_seq_length = 1024` right after load — covers every chunk in this project with huge headroom. Do the same if you ever instantiate BGE-M3 elsewhere.
- **Chroma cosine distance ≠ similarity.** We create the collection with `metadata={"hnsw:space": "cosine"}`, so `results["distances"][0]` returns `1 - cos_sim`, not `cos_sim` itself. `src/retrieve.py::_dense_topk` converts back via `1.0 - d`; don't use the raw distance as a similarity when logging or fusing.
- **`bm25.pkl` now persists full `Chunk` objects.** Phase 3 needs bbox / image_path / section for the Sources panel; `load_index` populates `IndexHandle.chunks_by_id` from the pickle. Old Phase-2 pickles without the `"chunks"` key will raise — rebuild with `make index`.
- **Reranker cold-start dominates the first retrieve call.** `BAAI/bge-reranker-v2-m3` is ~2.3 GB; the first `CrossEncoder(...)` can take minutes over a cold HF cache. `src/retrieve.py::retrieve` calls `_get_reranker()` *before* starting the rerank timer so logged timings reflect actual compute, not the one-off download.
- **Groq free-tier 12k TPM on `llama-3.3-70b-versatile`.** A single docling-produced table chunk can be >30 KB of text, which alone blows past the per-minute token budget (we saw `413 Payload Too Large — Requested 13402, Limit 12000`). `src/generate.py::_format_context` caps each context block to `_MAX_CHARS_PER_BLOCK = 2000` and appends `"… [truncated]"`. Paid tier lifts this but the truncation is cheap insurance.
- **Router boosts apply post-rerank, not post-RRF.** An earlier Phase-6 draft multiplied fused RRF scores by modality boosts before slicing the rerank pool; the effect was nil — the reranker dominates and the rerank candidate pool already contained the relevant tables. `src/retrieve.py` now multiplies the **rerank** score by `router_config.modality_boosts.get(modality, 1.0)` and sorts on that. The CE score stays the primary signal, the router only tilts the final top-k. A/B on "What is the inflation rate by year?" surfaces 3 tables in top-5 with router on vs 2 off (p.81 table replaces a p.60 text chunk).
- **RAGAS critic calls also bust Groq 12k TPM.** RAGAS's `faithfulness` and `context_precision` concatenate answer + all contexts in a single LLM request. Using full docling chunk content → 413 Payload Too Large (we saw 17k / 22k tokens requested). `eval/run_eval.py` truncates each context to `_MAX_CONTEXT_CHARS = 1500` before building the HF `Dataset`. Same logic as `src.generate._format_context`, just applied at the eval boundary.
- **RAGAS 0.4 deprecation warnings are cosmetic.** `LangchainLLMWrapper`, `LangchainEmbeddingsWrapper`, and the `from ragas.metrics import faithfulness` path all emit DeprecationWarnings that say "will be removed in v1.0." They still work in 0.4.3. The new `ragas.metrics.collections` namespace is a module, not a metric instance — migration would require a different `evaluate()` call shape. Leaving as-is for now.
- (more gotchas get appended here as I hit them)

## Phase status & conventions

- One commit per phase, message format `phase N: <title>`.
- Every phase ends with: running its checkpoint command, appending any new gotchas to this file, then committing.
- No `print` in `src/` — use `loguru`. `print` is OK in CLI entry points (i.e. inside Typer command bodies) when writing user-facing output.
- Chunk IDs are deterministic: SHA-256 over `(doc_id, page, bbox, content)`. Never random.
