"""Phase-7: RAGAS evaluation over `eval/questions.json`.

For each question: retrieve → generate → collect (answer, contexts). Feed the
full batch to RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`).
Write per-question scores + averages to `eval/results.md`.

Serial with an inter-question delay to stay under Groq's 30 rpm free-tier cap;
every Groq call (ours and RAGAS's internal critics via the LangChain adapter)
gets a tenacity retry on `RateLimitError`.

Usage:

    uv run python -m eval.run_eval
    uv run python -m eval.run_eval --questions eval/questions.json --out eval/results.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import typer
from datasets import Dataset
from groq import RateLimitError as GroqRateLimitError
from langchain_openai import ChatOpenAI
from loguru import logger
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from sentence_transformers import SentenceTransformer
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import require_key, settings
from src.generate import answer as generate_answer
from src.index import load_index
from src.retrieve import retrieve
from src.utils import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=False)

# RAGAS sends answer + every context to the LLM critic in one request. On Groq's
# 12k TPM free tier, a single raw docling table chunk is already >30 KB → 413.
# Cap each context the same way `src.generate` caps generator contexts.
_MAX_CONTEXT_CHARS = 1500


class _BGEM3Embeddings:
    """Tiny langchain-compatible Embeddings adapter around BGE-M3.

    RAGAS needs something with `embed_documents` / `embed_query`. We already
    have the sentence-transformers model loaded in :class:`IndexHandle`, so we
    just reuse it rather than pulling langchain-community for HF embeddings.
    """

    def __init__(self, model: SentenceTransformer) -> None:
        self._m = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._m.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._m.encode([text], normalize_embeddings=True)[0].tolist()


def _log_retry(state: RetryCallState) -> None:
    if state.outcome and state.outcome.failed:
        logger.warning(f"eval retry attempt={state.attempt_number} exc={state.outcome.exception()!r}")


@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(GroqRateLimitError),
    before_sleep=_log_retry,
    reraise=True,
)
def _retrieve_and_answer(query: str, index_handle: Any) -> dict[str, Any]:
    """One question → dict of (question, answer, contexts). Retried on 429."""
    chunks = retrieve(query, index_handle)
    result = generate_answer(query, chunks)
    return {
        "question": query,
        "answer": result.text,
        "contexts": [_truncate(c.chunk.content) for c in chunks],
    }


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_CONTEXT_CHARS else text[:_MAX_CONTEXT_CHARS] + "… [truncated]"


def _semantic_similarity(
    records: list[dict[str, Any]], embedder: SentenceTransformer
) -> list[float]:
    """Offline BGE-M3 cosine similarity between answer and ground_truth.

    Completes for every question with zero external API calls — a deterministic
    fallback so we always have a real number when RAGAS's Groq-backed critics
    hit a daily-token ceiling.
    """
    answers = [r["answer"] for r in records]
    truths = [r["ground_truth"] for r in records]
    a_vecs = np.asarray(embedder.encode(answers, normalize_embeddings=True))
    t_vecs = np.asarray(embedder.encode(truths, normalize_embeddings=True))
    sims = (a_vecs * t_vecs).sum(axis=1)
    return [float(s) for s in sims]


def _run_pipeline(questions: list[dict[str, Any]], index_handle: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] modality={q.get('modality', '?')} :: {q['question']}")
        rec = _retrieve_and_answer(q["question"], index_handle)
        rec["ground_truth"] = q.get("ground_truth", "")
        rec["modality"] = q.get("modality", "text")
        records.append(rec)
        if i < len(questions):
            time.sleep(settings.eval_inter_question_delay_s)
    return records


def _ragas_llm() -> LangchainLLMWrapper:
    """Point RAGAS at Groq through the OpenAI-compat endpoint.

    We use the 8B `router_model` (not the 70B `answer_model`) as the critic
    because Groq's free-tier TPD is per-model and RAGAS sends ~3 critic requests
    per metric per question. Keeping the 70B budget for answer generation and
    the 8B budget for grading keeps the whole eval inside the daily envelope.
    """
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.router_model,
            base_url="https://api.groq.com/openai/v1",
            api_key=require_key("GROQ_API_KEY"),
            temperature=0.0,
        )
    )


def _write_results_md(
    records: list[dict[str, Any]],
    scores: dict[str, list[float]],
    out_path: Path,
) -> None:
    """Render a markdown table of per-question scores plus averages."""
    metrics = list(scores.keys())
    lines: list[str] = []
    lines.append("# Evaluation results\n")
    lines.append(f"_{len(records)} questions, metrics: {', '.join(metrics)}_\n")

    header = ["#", "modality", "question"] + metrics
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for i, rec in enumerate(records):
        q = rec["question"].replace("|", "\\|")
        row = [str(i + 1), rec.get("modality", "?"), q]
        for m in metrics:
            v = scores[m][i]
            row.append(f"{v:.3f}" if v is not None else "n/a")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Averages\n")
    lines.append("| metric | mean |")
    lines.append("|---|---|")
    for m in metrics:
        vals = [v for v in scores[m] if v is not None]
        mean = sum(vals) / len(vals) if vals else float("nan")
        lines.append(f"| {m} | {mean:.3f} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote {out_path}")


@app.command()
def main(
    questions_path: Path = typer.Option(
        Path("eval/questions.json"), "--questions", help="JSON list of {question, ground_truth, modality}."
    ),
    out: Path = typer.Option(
        Path("eval/results.md"), "--out", help="Markdown output path."
    ),
    index_dir: Path = typer.Option(
        None, "--index", help="Chroma directory. Defaults to CHROMA_PATH."
    ),
    reuse_records: bool = typer.Option(
        False, "--reuse-records", help="Reuse cached records.json instead of re-generating answers (saves Groq tokens)."
    ),
) -> None:
    """Run the full pipeline over `questions_path` and score with RAGAS."""
    setup_logging()
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not questions:
        raise typer.BadParameter(f"{questions_path} is empty")

    index_handle = load_index((index_dir or settings.chroma_path).resolve())
    records_cache = out.with_name("records.json")

    if reuse_records and records_cache.exists():
        logger.info(f"Reusing cached records from {records_cache}")
        records = json.loads(records_cache.read_text(encoding="utf-8"))
    else:
        logger.info(
            f"Phase 7 eval: {len(questions)} question(s), delay={settings.eval_inter_question_delay_s}s"
        )
        records = _run_pipeline(questions, index_handle)
        records_cache.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Wrote records cache to {records_cache}")

    # RAGAS expects a HuggingFace Dataset with question/answer/contexts/ground_truth columns.
    ds = Dataset.from_list(records)

    logger.info("Running RAGAS metrics (this makes many Groq critic calls — expect pauses on 429)…")
    ragas_llm = _ragas_llm()
    ragas_embeddings = LangchainEmbeddingsWrapper(_BGEM3Embeddings(index_handle.embedder))

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        show_progress=True,
    )

    # `result` is a dict-like mapping metric-name → list[float]
    scores = {k: list(result[k]) for k in result.scores[0]} if hasattr(result, "scores") else dict(result)
    # Offline metric — never depends on Groq, so we always get 15 real numbers.
    scores["semantic_similarity"] = _semantic_similarity(records, index_handle.embedder)
    _write_results_md(records, scores, out)

    print()
    for m, vals in scores.items():
        vs = [v for v in vals if v is not None]
        mean = sum(vs) / len(vs) if vs else float("nan")
        print(f"  {m}: {mean:.3f}")


if __name__ == "__main__":
    app()
