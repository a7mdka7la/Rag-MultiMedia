"""Phase-4 CLI: Groq-streamed answer generation with mandatory citations.

Usage:

    uv run python -m src.generate "What does the report say about inflation?"

The prompt forces the LLM to (a) answer only from the retrieved context and
(b) tag every claim with `[p.<page>, <modality>]`. After the stream finishes we
parse citations back out with a regex and resolve each one to the retrieved
chunk that produced it, so the UI can render the actual source.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from groq import Groq, RateLimitError
from loguru import logger
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import require_key, settings
from src.index import load_index
from src.retrieve import RetrievedChunk, retrieve
from src.utils import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)

# Every claim in the answer must end with [p.<page>, <modality>].
# The same regex is reused by the UI to turn cited text into links.
CITATION_RE: re.Pattern[str] = re.compile(r"\[p\.(\d+),\s*(text|table|image)\]")


SYSTEM_PROMPT = """You are an expert analyst answering questions about an IMF Article IV report.

Rules you MUST follow:
1. Answer ONLY from the provided context blocks. Never invent facts or cite pages that are not in the context.
2. Every factual claim MUST end with a citation of the form [p.<page>, <modality>],
   where <modality> is one of text, table, or image. Example: [p.12, table].
3. You may combine multiple citations for a single claim: [p.7, text] [p.8, table].
4. If the context does not contain the answer, reply exactly:
   "The report does not contain information to answer this question."
5. Be concise. Do not repeat the question. Do not add preamble or closing remarks.
"""


@dataclass(slots=True, frozen=True)
class Citation:
    """One `[p.<page>, <modality>]` parsed from the LLM output."""

    page: int
    modality: str
    raw: str


@dataclass(slots=True)
class Answer:
    """Full generation result with citations resolved back to retrieved chunks."""

    text: str
    citations: list[Citation]
    raw_llm_output: str
    chunk_refs: list[RetrievedChunk] = field(default_factory=list)


_MAX_CHARS_PER_BLOCK = 2000


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, r in enumerate(chunks, 1):
        c = r.chunk
        header = f"[Context {i}] page={c.page} modality={c.modality}"
        if c.section:
            header += f" section={c.section!r}"
        # Cap each block — Groq free-tier is 12k TPM; a single raw table chunk
        # can be >30KB, which alone blows past the limit.
        body = c.content
        if len(body) > _MAX_CHARS_PER_BLOCK:
            body = body[:_MAX_CHARS_PER_BLOCK] + "… [truncated]"
        blocks.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(blocks)


def build_messages(query: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    """Return OpenAI-style chat messages for Groq's chat completion API."""
    context = _format_context(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}"},
    ]


def parse_citations(text: str) -> list[Citation]:
    """Extract every `[p.N, modality]` match, in order of appearance."""
    return [
        Citation(page=int(m.group(1)), modality=m.group(2), raw=m.group(0))
        for m in CITATION_RE.finditer(text)
    ]


def _log_retry(state: RetryCallState) -> None:
    """Forward tenacity retry events into the loguru sink."""
    if state.outcome and state.outcome.failed:
        exc = state.outcome.exception()
        logger.warning(f"groq retry attempt={state.attempt_number} exc={exc!r}")


@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(RateLimitError),
    before_sleep=_log_retry,
    reraise=True,
)
def _open_stream(client: Groq, messages: list[dict[str, str]]) -> Any:
    """Start the chat completion stream. 429s are retried; other errors propagate."""
    return client.chat.completions.create(
        model=settings.answer_model,
        messages=messages,
        temperature=settings.answer_temperature,
        stream=True,
    )


def _client() -> Groq:
    return Groq(api_key=require_key("GROQ_API_KEY"))


def answer_stream(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    client: Groq | None = None,
) -> Iterator[str]:
    """Yield token deltas as they arrive from Groq. Caller accumulates into full text."""
    c = client or _client()
    messages = build_messages(query, chunks)
    stream = _open_stream(c, messages)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _resolve_chunk_refs(
    citations: list[Citation], chunks: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Map each citation to the retrieved chunk on that (page, modality)."""
    refs: list[RetrievedChunk] = []
    seen: set[str] = set()
    for cit in citations:
        for r in chunks:
            if r.chunk.page == cit.page and r.chunk.modality == cit.modality:
                if r.chunk.id not in seen:
                    refs.append(r)
                    seen.add(r.chunk.id)
                break
    return refs


def answer(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    client: Groq | None = None,
) -> Answer:
    """Non-streaming wrapper: drains the stream into a full :class:`Answer`."""
    raw = "".join(answer_stream(query, chunks, client=client))
    citations = parse_citations(raw)
    return Answer(
        text=raw,
        citations=citations,
        raw_llm_output=raw,
        chunk_refs=_resolve_chunk_refs(citations, chunks),
    )


@app.command()
def main(
    query: str = typer.Argument(..., help="Question to answer."),
    index_dir: Path = typer.Option(
        None, "--index", help="Chroma directory. Defaults to CHROMA_PATH."
    ),
) -> None:
    """Retrieve, generate, and stream the answer to stdout."""
    setup_logging()
    out = (index_dir or settings.chroma_path).resolve()
    index = load_index(out)
    chunks = retrieve(query, index)

    print()
    parts: list[str] = []
    for delta in answer_stream(query, chunks):
        print(delta, end="", flush=True)
        parts.append(delta)
    print("\n")

    raw = "".join(parts)
    cits = parse_citations(raw)
    refs = _resolve_chunk_refs(cits, chunks)
    print(f"--- {len(cits)} citation(s), {len(refs)} unique source chunk(s) ---")
    for r in refs:
        print(f"  • [p.{r.chunk.page}, {r.chunk.modality}] {r.chunk.section or ''}")


if __name__ == "__main__":
    app()
