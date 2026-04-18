# Multimodal RAG for IMF Article IV Reports — Technical Report

**Ahmed Kahla · DSAI 413 · AUC · Spring 2026**

## 1. Problem & goal

An IMF Article IV report bundles three kinds of information that are normally handled by different systems: prose assessments of the economy, tables of macroeconomic forecasts, and charts of time series. A conventional text-only RAG pipeline would silently drop two of the three modalities, and the reader has no way to tell from the answer which page or figure a claim came from. The goal of this project is a single question-answering system that (i) retrieves uniformly across text / table / image content, (ii) produces answers grounded in the document, and (iii) makes every claim verifiable by linking back to the cited page — including, for tables and figures, the exact region on the page.

The target document is the 2025 IMF Article IV Egypt report (~80 pages, ~134 non-text elements: 78 tables, 56 pictures).

## 2. Architecture

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| PDF parsing | Docling | Layout-aware, preserves tables and figures as structured elements with page + bounding-box metadata. |
| Image captions | Gemini 2.5 Flash | Long-context multimodal model; captions include surrounding paragraph so charts are described in domain terms. |
| Embeddings | BGE-M3 | Multilingual (Arabic + English in one model), 1024-dim dense vectors, widely benchmarked. |
| Reranker | `bge-reranker-v2-m3` | Same M3 family as the embedder; cross-encoder gives large precision gains on top-k. |
| Vector store | ChromaDB | Local, persistent, no server; cosine-space HNSW. |
| Sparse retrieval | BM25Okapi | Catches exact-match numbers and proper nouns that dense embeddings smooth away. |
| Answer LLM | Llama 3.3 70B on Groq | Strong reasoning, streaming support, free-tier available. |
| UI | Streamlit | Chat + image display + sidebar in one file. |
| Eval | RAGAS | Standard LLM-judged metrics for RAG (`faithfulness`, `answer_relevancy`, `context_precision`). |
| Page rendering | PyMuPDF | Draws the cited bounding box on a page raster for the Sources panel. |

### Three non-obvious decisions

**(a) Single embedding space, not one index per modality.** Every non-text element is turned into descriptive text first — tables become Markdown with a one-line Gemini summary prepended, images become a detailed Gemini caption that references the surrounding paragraph. All three modalities then share a single BGE-M3 embedding and a single ChromaDB collection. Modality is stored as a metadata field. This means cross-modal queries ("show me the chart about inflation") reach the right chunk without any orchestration layer, and the reranker makes the final relevance call using the same signal as it would for prose.

**(b) Hybrid retrieval with reciprocal rank fusion.** Dense retrieval alone misses exact numeric matches ("FY 2024/25", "23.3%") that BM25 catches easily. BM25 alone misses paraphrases. The pipeline runs both (top-20 each), fuses with RRF (k=60, the canonical Cormack constant), feeds the top-20 fused candidates to a cross-encoder, and emits the final top-5. Every stage is independently debuggable through `src.retrieve --debug`.

**(c) Query router as a tie-breaker, not a selector.** A one-call Llama 3.1 8B classifier labels the query as `{factual, table, chart, summary}` and returns modality boosts. Crucially, the boost multiplies the **rerank score**, not the pre-rerank fused score — the cross-encoder's semantic ranking stays the primary signal, and the router only nudges the final ordering when rerank scores are close. An A/B on *"What is the inflation rate by year?"* surfaces three table chunks in the top-5 with the router on versus two with it off (a page-81 table replaces a page-60 text chunk).

## 3. Per-modality implementation notes

- **Text** is chunked with Docling's `HybridChunker` using the BGE-M3 tokenizer as the boundary model, so no chunk ever gets truncated at embed time.
- **Tables** are serialized to Markdown, then a one-line Gemini summary is prepended (e.g. *"This IMF table projects key macroeconomic indicators… across 2020/21–2029/30"*). The markdown body stays intact so the LLM can answer numeric questions off the full row/column structure.
- **Images** are captioned by Gemini with the surrounding paragraph pasted in as context, so chart captions mention the specific lines (e.g. *"the blue line showing headline inflation peaks at ~38% in Sept 2023"*) rather than generic shape descriptions. Captions are SHA-256 cached so re-runs hit the cache.

## 4. Retrieval pipeline & timings

On warm caches, `src.retrieve` for a typical query:

```
dense:  280 ms   (ChromaDB HNSW, k=20)
bm25:     1 ms   (in-memory BM25Okapi, k=20)
rerank: 10 500 ms (BGE-reranker-v2-m3 on CPU, 20 pairs)
```

The cross-encoder dominates; it's the accuracy lever worth paying for. First-call reranker load (~8 s) is handled outside the rerank timer so A/B measurements aren't polluted by the one-off weights download.

## 5. Evaluation

15 questions (5 text, 5 table, 5 image, 1 in Arabic to exercise BGE-M3's multilingual capability) are scored with RAGAS pointed at Groq, **plus** an offline BGE-M3 cosine-similarity metric between the generated answer and the ground-truth reference — a deterministic signal that always completes regardless of API quota.

| Metric | Coverage | Mean |
| :--- | :--- | :--- |
| `semantic_similarity` (BGE-M3 cosine, offline) | 15 / 15 | **0.688** |
| `answer_relevancy` (RAGAS critic on Groq) | 4 / 15 | 0.673 |
| `context_precision` (RAGAS critic on Groq) | 2 / 15 | 0.850 |
| `faithfulness` (RAGAS critic on Groq) | 0 / 15 | — |

**Error analysis.** The eight questions with `semantic_similarity ≥ 0.70` all retrieve the right page and cite it correctly — these include the three image questions on pages 14–16, the Figure-6 Medium-Term Risk Assessment table, and the sovereign-bank-nexus text question. The three weakest answers ("report does not contain information") share a cause: the 8B fallback generator refuses to synthesize when the retrieved table chunk is schema-heavy (columns of headers with few numbers in context), whereas 70B would merge the retrieved row/column labels into a description. The Arabic question scores low (0.306) because the generator responded in English while the ground truth is Arabic — BGE-M3 is cross-lingual but not identity-preserving across languages.

**Honest caveats.** Two Groq free-tier behaviours gated the RAGAS critic. First, the 100 k tokens-per-day limit on `llama-3.3-70b-versatile` was exhausted by the 15 answer-generation calls before the critic pass ran; the answer model was therefore swapped to `llama-3.1-8b-instant` for this eval run (retrieval, reranking, and router stages are unchanged — the production system still ships 70B). Second, RAGAS issues `n=3` generation requests internally and Groq caps at `n=1`, returning HTTP 400 for several critic calls. Both are artefacts of the grading environment, not the retrieval pipeline. On a paid tier with `n=1` coerced the three RAGAS metrics complete in full; the offline `semantic_similarity` already does.

## 6. Limitations & future work

- **One document at a time.** The codebase is single-PDF by design. A multi-document store would need a `doc_id` metadata field and a `--doc` filter on every query.
- **OCR disabled.** Article IV reports have an embedded text layer, so Docling's OCR pass is off. A scanned PDF would have empty text chunks and only image captions — a real concern for older IMF publications.
- **Partial image captioning on free tier.** Gemini's 20 RPD per key × 3 live keys gave 8 of 56 image captions (the important charts on pages 14–16 covering GDP growth, PMI, exchange rates, current account). The remaining 48 retain a placeholder, which still carries page / section / bbox metadata but is not semantically retrievable. Re-running `make ingest` on the next day resumes from the caption cache and fills the gap.
- **Free-tier friction.** Groq's 12 k TPM, 30 rpm, and 100 k TPD, plus Gemini's 20 RPD per key, shaped several design decisions (context truncation, Gemini key rotation, 2 s eval inter-question delay, model-id override via `ANSWER_MODEL` env var). A paid tier removes all of them.
- **Router is a tie-breaker only.** Modality boosts move the needle only when the reranker is nearly tied between modalities. A more ambitious router would adjust dense/BM25 top-k or even swap the embedding model per class.

## References

- Cormack, Clarke, Büttcher (2009). *Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods.* SIGIR.
- Chen, Xiao, et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings through Self-Knowledge Distillation.*
- Livshits et al. (2024). *Docling: An Efficient Open-Source Toolkit for PDF Conversion.*
- Es, James, et al. (2024). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.*
