# Demo video — shot-by-shot script (2–5 min)

Target length: ~3 min 30 s. Record on one monitor; second monitor (or overlay) for narration notes.

---

## 0:00 – 0:20 · Hook

**On screen:** full-screen terminal running `make run`; Streamlit tab opens showing the landing page (empty chat + sidebar metrics).

**Narrate:**
> "This is a multimodal RAG over an 80-page IMF Article IV report. Every answer is grounded in the document and cites the page and modality for every claim — and for tables and figures, you'll see the exact region highlighted on the PDF. Let me show you."

## 0:20 – 0:45 · Architecture

**On screen:** split the screen — left: the architecture diagram from `README.md` (rendered GitHub page at `github.com/a7mdka7la/Rag-MultiMedia`); right: the Streamlit app, sidebar visible.

**Narrate (one beat per box):**
> "Docling turns the PDF into chunks. Tables get a one-line Gemini summary; images get a detailed Gemini caption — so everything is text before we embed.
> Then one BGE-M3 collection for all three modalities, plus BM25 for exact-term matches. At query time: dense + BM25, reciprocal rank fusion, cross-encoder rerank. Top-5 goes to Llama 3.3 70B on Groq, streamed with mandatory citations."

## 0:45 – 1:15 · Text question

**On screen:** type into the chat input: *"What does the report say about inflation in Egypt?"*

**Narrate over the streaming answer:**
> "The answer streams token by token. Every factual sentence ends with a citation like `[p.16, text]`."

Once the answer finishes and the Sources panel appears, expand it.

**Narrate while clicking the first source:**
> "The Sources panel shows the actual retrieved chunk — text snippet with the section heading — and here, the cited page from the PDF with the paragraph highlighted in red."

## 1:15 – 1:50 · Table question

**On screen:** new chat turn: *"What is the real GDP growth forecast by fiscal year?"*

**Narrate:**
> "Same pipeline, different modality. The router classifies this as a table query and boosts table chunks on the final rerank. You can see three tables in the top-5 instead of two, and the answer cites `[p.7, table]`."

Expand Sources. Scroll to show the table rendered as markdown, then the page image with the table area boxed.

## 1:50 – 2:25 · Chart / image question

**On screen:** *"Show me the chart of inflation over time."*

**Narrate:**
> "An image query. The retriever returns the figure chunk whose Gemini caption describes the exact lines plotted. The Sources panel displays the extracted image itself, then the source page with the figure outlined."

## 2:25 – 3:00 · Router A/B

**On screen:** flip the `Query router` toggle **off** in the sidebar; ask *"What is the inflation rate by year?"* and scroll to the Sources. Count the tables.
Flip the toggle **on**; ask the same question; scroll to the Sources. Count the tables again.

**Narrate:**
> "With the router off, we get two tables in the top-5. With it on, the one-call Llama 3.1 8B classifier labels this as a table query and the modality boosts tilt the final ordering — three tables surface. The cross-encoder's semantic ranking is still the primary signal; the router is only a tie-breaker."

## 3:00 – 3:30 · Evaluation + one limitation

**On screen:** open `eval/results.md` in the IDE or GitHub.

**Narrate:**
> "15 questions — five text, five table, five image, with one question in Arabic to exercise BGE-M3's multilingual capability — scored with RAGAS for faithfulness, answer relevancy, and context precision. Highest scores on faithfulness, lowest on context precision — the top-5 still picks up chunks that are topically relevant but not strictly needed for the specific answer."

**On screen:** scroll to the Limitations section of `README.md`.

**Narrate one closing beat:**
> "Current scope is a single PDF and free-tier Groq, which caps context tokens. Next step: multi-document store and paid-tier batching."

## 3:30 – end · Closing card

**On screen:** black card with:

```
Ahmed Kahla — DSAI 413 — AUC 2026
github.com/a7mdka7la/Rag-MultiMedia
```

Hold for 3 s. Done.

---

## Recording checklist

- [ ] Increase terminal + browser font sizes (Cmd-+ a few times) so on-screen text is legible at 720 p.
- [ ] Pre-warm the reranker by running one query before hitting record — avoids the 8 s cold-start pause on the first chat turn.
- [ ] Pre-warm the index (`make run` already loaded) — first `make run` spins up Chroma + embedder + reranker.
- [ ] Have a fresh chat (Clear conversation button) before each new section so the video isn't cluttered.
- [ ] Keep Finder / notifications / dock hidden; Do Not Disturb on.
- [ ] Record one take per section; cut in post. Retake a section rather than re-recording the whole video.
