# Multimodal RAG for IMF Article IV Reports

Ahmed Kahla · DSAI 413 · AUC · Spring 2026

## 1. The problem

IMF Article IV reports are 80-page PDFs that mix three things a normal RAG pipeline can't really handle at the same time: long prose sections, macroeconomic tables, and time-series charts. If you only embed the text, you lose the tables and figures. If you build one index per modality, you need an orchestration layer to pick which index to hit, and cross-modal questions like "show me the chart about inflation" get awkward fast.

I wanted one system where a single query could return a mix of text, table, and figure chunks, every answer was grounded in the document, and the reader could click through to the cited page with the relevant region highlighted.

The test document is the 2025 Article IV for Egypt. Roughly 80 pages, 78 tables, 56 figures.

## 2. Stack and why

PDF parsing is Docling. Unlike pypdf or pdfplumber it keeps the full layout tree, so every table survives as a structured element, every picture ends up with page and bbox metadata, and I didn't have to write layout heuristics.

Image captions come from Gemini 2.5 Flash. I pass it the picture bytes plus the surrounding paragraph from the same page, so the caption actually references the specific labels and series in the chart instead of saying "a line chart shows a trend". That matters, because the caption is what ends up in the vector store.

Embeddings are BGE-M3. It handles Arabic and English in the same space (the assignment asks for an Arabic question, and M3 covers it without a second model), it's 1024-dim, and it's well benchmarked. The reranker is `bge-reranker-v2-m3`, same family, which moved the needle visibly on top-k precision versus just sorting by cosine distance.

Storage is ChromaDB: local, persistent, no server, fine for one document. BM25 is `rank-bm25`, pickled next to the Chroma folder.

The answer LLM is Llama 3.3 70B on Groq for streaming. I used the native Groq SDK rather than the OpenAI-compat shim because the typed errors and retry ergonomics were cleaner.

UI is Streamlit. Evaluation is RAGAS plus an offline cosine-similarity metric I added after RAGAS ran into Groq's daily limits mid-run (more on that in §5). Page-region highlighting is PyMuPDF: when an answer cites `[p.15, image]`, I render page 15 as a PNG, draw a red rectangle around the chunk's bbox, cache the result, and show it in the Sources panel.

## 3. Three decisions worth calling out

**One embedding space, not one per modality.** Everything becomes text before it gets embedded. Tables are serialised to Markdown with a one-line Gemini summary prepended. Images become Gemini captions. Text stays as text. Then a single BGE-M3 model embeds all three into one Chroma collection, with modality kept as metadata. This makes cross-modal retrieval a non-issue: the reranker decides what's relevant using the same signal it uses for prose, and the UI reads the modality field to decide how to render the source (markdown table vs image vs snippet).

**Hybrid retrieval with reciprocal rank fusion.** Dense embeddings blur over exact matches that BM25 catches instantly — "FY 2024/25", specific percentages, proper nouns. BM25 alone misses paraphrases. I run both at top-20, fuse with RRF at k=60 (Cormack's constant), feed the top-20 fused candidates to the cross-encoder, and hand the top-5 reranked chunks to the LLM. Every stage logs independently via `src.retrieve --debug`, which paid off several times during tuning.

**Where the router applies its boost.** The router is a one-call 8B classifier that labels the query as factual / table / chart / summary and returns modality multipliers. My first version multiplied the fused RRF scores, and I couldn't see any effect in the top-5 — the reranker dominated downstream and basically overrode whatever I was doing before it. Moving the multiplication to after the rerank (so boosted rerank score becomes the sort key, then top-5) made it actually influence results. On "What is the inflation rate by year?" the router-on run surfaces three tables in the top-5 instead of two, pulling in the page-81 table I wanted.

## 4. Per-modality notes

Text is chunked with Docling's HybridChunker wired to the BGE-M3 tokenizer, so no chunk gets truncated at embed time.

Tables ship as Markdown with a one-line Gemini summary on top. I kept the full Markdown body on purpose — the LLM often needs to read the row labels and the year columns to answer numeric questions, and a summary alone isn't enough for that.

Image captions use surrounding paragraphs as context. That single change is what made captions domain-aware: instead of "a line chart depicts a trend over time" you get captions that reference the specific lines and labels in the chart. Captions are SHA-256 cached on image bytes, so reruns hit the cache.

## 5. Retrieval timings and evaluation

Warm-cache timings for a typical query:

```
dense:    280 ms  (ChromaDB HNSW, k=20)
bm25:       1 ms  (in-memory, k=20)
rerank: 10 500 ms (bge-reranker-v2-m3, CPU, 20 pairs)
```

The reranker is the slow part and also the accuracy part. First-call reranker load is about 8 s while the weights come down from HF; I take that outside the timing so A/B measurements aren't polluted by a one-off download.

I wrote 15 questions: 5 text, 5 table, 5 image, one of them in Arabic. Scores are in `eval/results.md`:

| Metric | Coverage | Mean |
| :--- | :--- | :--- |
| `semantic_similarity` (BGE-M3 cosine, offline) | 15 / 15 | **0.688** |
| `answer_relevancy` (RAGAS, Groq) | 4 / 15 | 0.673 |
| `context_precision` (RAGAS, Groq) | 2 / 15 | 0.850 |
| `faithfulness` (RAGAS, Groq) | 0 / 15 | — |

Eight questions score ≥ 0.70 on similarity, and in each of those the citations point to the page the answer was actually taken from. The three image questions on pages 14–16 (quarterly GDP growth, the PMI, the exchange-rate series), the Figure-6 risk-assessment table on page 80, and the sovereign-bank-nexus text question on page 24 are the strongest.

Three questions came back with the prompt's refusal line ("the report does not contain information to answer this question"). All three are schema-heavy table questions — Table 3a and Table 4 — where the retrieved chunk is mostly column headers with sparse numbers in the narrow window the reranker picked. A 70B model stitches those labels together into a description; the 8B I had to fall back on gives up instead.

The Arabic question scores 0.306. The generator answered in English because the system prompt is English, and cross-lingual cosine between an English answer and an Arabic ground truth isn't zero but it's a lot lower than a same-language pair. A real fix is detecting the query language and echoing it in the response.

About the blank cells in the RAGAS columns: two Groq free-tier behaviours got in the way. The first is the 100 k tokens-per-day cap on `llama-3.3-70b-versatile`. The 15 answer-generation calls alone ate most of it, so by the time the critic pass ran I was out of tokens. I swapped the answer model to `llama-3.1-8b-instant` for the eval run (the retrieval stages, the reranker, and the router are unchanged; the Streamlit demo still ships 70B). The second is that RAGAS issues `n=3` generation requests and Groq's OpenAI-compat endpoint hard-caps at `n=1`, so a block of critic jobs came back `400 BadRequest`. Both are environment issues, not pipeline issues: on a paid tier the three RAGAS metrics complete fully, and the offline `semantic_similarity` already does.

## 6. Limitations and what I'd do next

*One document at a time.* The collection is keyed to one PDF by design. Multi-document would need a `doc_id` metadata field and a filter on every query. Not hard, just not in scope.

*OCR off.* Article IV PDFs have an embedded text layer, so Docling's OCR pass is disabled. A scanned PDF would come out with empty text chunks and would lean entirely on image captions.

*Partial captioning on the free tier.* Gemini's free tier is 20 RPD per API key. I had four keys. One was permanently denied by Google, and the other three hit daily limits before finishing all 56 figures. 8 figures ended up captioned (the important charts on pages 14–16); the rest still have placeholders that carry page / section / bbox metadata but aren't semantically retrievable. Running `make ingest` a day later resumes from the caption cache and fills the gap.

*Rate-limit friction.* Most of the unusual design choices in this project — 2 000-char context truncation, Gemini key rotation, a 2 s inter-question delay in the eval, an `ANSWER_MODEL` env override — are reactions to Groq and Gemini free-tier quotas. A paid plan deletes most of this complexity.

*Router is marginal.* Modality boosts only shift the final order when the reranker is close to a tie between modalities. A more ambitious router would tune `dense_top_k` and `rerank_top_k` per class, or even switch embedding models for chart-heavy queries.

## References

- Cormack, Clarke, and Büttcher (2009). *Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods.* SIGIR.
- Chen, Xiao et al. (2024). *BGE M3-Embedding.*
- Livshits et al. (2024). *Docling: An Efficient Open-Source Toolkit for PDF Conversion.*
- Es, James, et al. (2024). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.*
