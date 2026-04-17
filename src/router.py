"""Phase-6 query router: classify question → per-modality boosts.

One Groq `llama-3.1-8b-instant` call maps the query to one of four classes and
returns a :class:`RetrievalConfig` whose `modality_boosts` multiply post-RRF
fused scores so the reranker sees the right mix of text / table / image chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

from groq import Groq
from loguru import logger

from src.config import require_key, settings

QueryClass = Literal["factual", "table", "chart", "summary"]
_VALID: set[str] = {"factual", "table", "chart", "summary"}

_ROUTER_SYSTEM = """You classify user questions about an IMF country report into ONE category.

Categories:
- table: asks for specific numbers, comparisons, or data the report tabulates (e.g. "inflation in 2023", "fiscal balance vs GDP").
- chart: asks about trends, shapes, or what a figure shows (e.g. "how has debt evolved", "what does the GDP chart show").
- summary: asks for a high-level overview or synthesis (e.g. "summarize the report", "what are the main risks").
- factual: anything else — a single fact or short explanation from prose.

Respond with ONE lowercase word: factual, table, chart, or summary. No punctuation, no explanation."""


@dataclass(slots=True, frozen=True)
class RetrievalConfig:
    """Per-query retrieval overrides picked by the router."""

    query_class: QueryClass
    modality_boosts: dict[str, float] = field(default_factory=dict)


_CLASS_TO_BOOST: dict[str, dict[str, float]] = {
    "factual": {},
    "table": {"table": 1.5, "image": 0.8},
    "chart": {"image": 1.8, "table": 1.1},
    "summary": {},
}


def _client() -> Groq:
    return Groq(api_key=require_key("GROQ_API_KEY"))


def classify(query: str, *, client: Groq | None = None) -> RetrievalConfig:
    """Ask Groq 8B for the query class. Falls back to 'factual' on any parse issue."""
    c = client or _client()
    resp = c.chat.completions.create(
        model=settings.router_model,
        messages=[
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=4,
    )
    raw = (resp.choices[0].message.content or "").strip().lower()
    # The 8B sometimes prefixes a word — grab the first token.
    label = raw.split()[0] if raw else "factual"
    label = label.strip(".,!?:")
    if label not in _VALID:
        logger.warning(f"router: unparseable class {raw!r}; defaulting to factual")
        label = "factual"
    query_class = cast(QueryClass, label)
    return RetrievalConfig(query_class=query_class, modality_boosts=dict(_CLASS_TO_BOOST[label]))
