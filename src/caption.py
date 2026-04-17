"""Gemini 2.5 Flash captioner for images and table summaries.

Exposes one class, :class:`Captioner`, with two methods:
- :meth:`caption_image` — detailed ≥3-sentence caption of a figure with context.
- :meth:`summarize_table` — 1-line summary prepended to the table's markdown.

Both are cached on disk by SHA-256(prompt_version + content). A warm cache
makes re-running ingest on the same PDF a no-op (< 10 s).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from google import genai
from google.genai import errors, types
from loguru import logger
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.utils import ensure_dir, sha256_bytes, sha256_parts, sha256_text

# Prompt versions — bump these to invalidate the cache for a prompt change.
IMAGE_PROMPT_V1 = (
    "You are captioning a figure from an IMF Article IV report for a multimodal "
    "retrieval system. Describe the figure in 3–6 complete sentences so that a "
    "reader who has not seen it can still answer a question about it.\n\n"
    "Cover all of:\n"
    "1. The chart type or figure genre (e.g. line chart, bar chart, map, photograph).\n"
    "2. The x- and y-axes or regions, including units (% of GDP, year, etc.) where visible.\n"
    "3. The series, categories, or labels shown.\n"
    "4. The key trend(s) or comparison(s) the figure makes and any highlighted values.\n"
    "5. Any title, caption, source, or footnote text printed in the figure.\n\n"
    "Do NOT invent numbers that are not in the figure or in the surrounding context. "
    "If a value is illegible, say 'illegible'. Return only the caption text — no "
    "preamble, no bullet lists, no markdown fences."
)

TABLE_PROMPT_V1 = (
    "Write ONE complete sentence (max 30 words) summarising what this IMF table shows. "
    "Name the quantity measured, the units if visible, and the comparison axis "
    "(years, countries, categories). Do not list individual values. Return the "
    "sentence only — no preamble, no markdown."
)

_CACHE_DIR_IMAGE = settings.cache_dir / "captions"
_CACHE_DIR_TABLE = settings.cache_dir / "table_summaries"


def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient Gemini *server* failures only.

    429s are handled at a higher level by key rotation, not here — retrying a
    daily-quota 429 five times burns five daily budget slots for no gain.
    """
    return isinstance(exc, errors.ServerError)


def _log_retry(state: RetryCallState) -> None:
    """Loguru-friendly before_sleep callback (tenacity's built-in wants stdlib logging)."""
    exc = state.outcome.exception() if state.outcome else None
    sleep = getattr(state.next_action, "sleep", "?")
    logger.warning(
        f"Gemini call failed ({type(exc).__name__}: {exc}) — "
        f"retrying in {sleep}s (attempt {state.attempt_number})"
    )


_gemini_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
    reraise=True,
)


def _is_daily_quota_429(exc: errors.ClientError) -> bool:
    """True if a 429 is the per-day free-tier limit (not per-minute).

    Gemini's error message distinguishes them: `...PerDay...` vs `...PerMinute...`
    (quota id), and the headline mentions "limit: 20" for daily on 2.5-flash
    free tier vs "limit: 10" for per-minute. When unsure, treat as per-minute
    (don't retire the key prematurely).
    """
    msg = str(getattr(exc, "message", "")) + " " + str(exc)
    haystack = msg.lower()
    if "perday" in haystack.replace(" ", "") or "per day" in haystack:
        return True
    # Fallback: "limit: 20" on a 429 for 2.5-flash strongly implies daily.
    return "limit: 20" in haystack


@dataclass(slots=True)
class CaptionResult:
    """Caption text + provenance so we can tell cache hits apart from fresh calls."""

    text: str
    cache_hit: bool
    prompt_version: str


class Captioner:
    """Thin wrapper around `google.genai.Client` that caches results by hash.

    Supports round-robin rotation across multiple Gemini API keys so the
    free-tier 20-RPD per-key limit doesn't block ingest of the ~130-image IMF
    PDF in one day. Rotation happens per call (not just on 429) so each key
    stays well under its 10-RPM per-minute cap; a 429 classified as
    daily-quota retires that key for the remainder of the process.
    """

    def __init__(
        self,
        api_keys: list[str] | tuple[str, ...] | None = None,
        model: str | None = None,
        cache_dir_image: Path = _CACHE_DIR_IMAGE,
        cache_dir_table: Path = _CACHE_DIR_TABLE,
        *,
        per_minute_backoff_s: float = 60.0,
    ) -> None:
        keys = tuple(api_keys) if api_keys is not None else settings.google_api_keys
        if not keys:
            raise RuntimeError(
                "No Gemini API keys configured. Set GOOGLE_API_KEY or "
                "GOOGLE_API_KEYS (comma-separated) in .env."
            )
        self._keys: tuple[str, ...] = keys
        self._clients = [genai.Client(api_key=k) for k in self._keys]
        self._cursor = 0  # round-robin pointer; advanced before each call
        self._exhausted: set[int] = set()  # keys retired by a daily-quota 429
        self._per_minute_backoff_s = per_minute_backoff_s
        self._model = model or settings.gemini_model
        self._cache_dir_image = ensure_dir(cache_dir_image)
        self._cache_dir_table = ensure_dir(cache_dir_table)
        logger.info(
            f"Captioner ready with {len(self._keys)} Gemini key(s), model={self._model}"
        )

    # ------------------------- key rotation -------------------------

    def _next_live_key(self) -> int:
        """Advance the round-robin cursor to the next non-exhausted key.

        Returns the new cursor index, or -1 if every key has been retired.
        """
        n = len(self._clients)
        for i in range(1, n + 1):
            candidate = (self._cursor + i) % n
            if candidate not in self._exhausted:
                self._cursor = candidate
                return candidate
        return -1

    def _retire_current(self, reason: str) -> None:
        """Mark the current key as permanently exhausted for this process."""
        idx = self._cursor
        self._exhausted.add(idx)
        logger.warning(
            f"Gemini key #{idx} retired ({reason}). "
            f"{len(self._exhausted)}/{len(self._clients)} keys exhausted."
        )

    def _invoke_with_rotation(
        self, call: Callable[[genai.Client], str], label: str
    ) -> str:
        """Run `call(client)` against a rotating set of clients.

        Advances the cursor before every call so successive requests spread
        across all live keys (keeps each well under its 10-RPM per-minute cap).
        On daily-quota 429: retire that key, try the next. On per-minute 429:
        sleep and retry. On 5xx: let tenacity inside `call` handle it.
        """
        total_keys = len(self._clients)
        # Round-robin advance — pick the next live key for this call.
        if self._next_live_key() == -1:
            raise RuntimeError(
                f"All {total_keys} Gemini API keys exhausted (daily quota). "
                f"Captions already completed are cached on disk — re-run "
                f"ingest tomorrow to resume."
            )
        per_minute_retries = 0
        while True:
            try:
                return call(self._clients[self._cursor])
            except errors.ClientError as exc:
                code = getattr(exc, "code", None)
                # 401/403 are permanent key-level denials — retire the key so
                # subsequent rotations skip it instead of burning one call per chunk.
                if code in (401, 403):
                    self._retire_current(f"{code} on {label}: {str(exc)[:120]}")
                    if self._next_live_key() == -1:
                        raise RuntimeError(
                            f"All {total_keys} Gemini API keys unusable "
                            f"(denied or exhausted)."
                        ) from exc
                    continue
                if code != 429:
                    raise
                if _is_daily_quota_429(exc):
                    self._retire_current(f"daily-quota 429 on {label}")
                    if self._next_live_key() == -1:
                        raise RuntimeError(
                            f"All {total_keys} Gemini API keys exhausted (daily quota). "
                            f"Captions already completed are cached — re-run tomorrow."
                        ) from exc
                    continue
                # Per-minute 429: try the next live key first (cheap), then back off.
                per_minute_retries += 1
                if per_minute_retries <= total_keys:
                    logger.warning(
                        f"Gemini key #{self._cursor} hit per-minute 429 on {label}; "
                        f"rotating to next key (attempt {per_minute_retries}/{total_keys})."
                    )
                    if self._next_live_key() == -1:
                        raise RuntimeError(
                            f"All {total_keys} Gemini keys 429 — daily quotas likely exhausted."
                        ) from exc
                    continue
                logger.warning(
                    f"All live keys hit per-minute 429 on {label}; "
                    f"sleeping {self._per_minute_backoff_s}s before retry."
                )
                time.sleep(self._per_minute_backoff_s)
                per_minute_retries = 0

    # ------------------------- public API -------------------------

    def caption_image(
        self,
        image_bytes: bytes,
        context: str,
        mime_type: str = "image/png",
    ) -> CaptionResult:
        """Caption an image with surrounding-text context.

        Args:
            image_bytes: Raw PNG/JPEG bytes.
            context: Surrounding-page text (up to a few hundred chars). Empty is OK.
            mime_type: Image MIME type; PNG is the default from Docling.

        Returns:
            :class:`CaptionResult` with ≥3-sentence caption or cached value.
        """
        key = sha256_parts(
            IMAGE_PROMPT_V1,
            context,
            sha256_bytes(image_bytes),
        )
        path = self._cache_dir_image / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return CaptionResult(
                text=data["text"], cache_hit=True, prompt_version="image-v1"
            )
        text = self._invoke_with_rotation(
            lambda client: self._call_image(client, image_bytes, context, mime_type),
            label="image-caption",
        )
        path.write_text(
            json.dumps({"text": text, "prompt": "image-v1"}, ensure_ascii=False),
            encoding="utf-8",
        )
        return CaptionResult(text=text, cache_hit=False, prompt_version="image-v1")

    def summarize_table(self, markdown: str) -> CaptionResult:
        """Return a 1-sentence description of a Markdown-serialized table."""
        key = sha256_parts(TABLE_PROMPT_V1, sha256_text(markdown))
        path = self._cache_dir_table / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return CaptionResult(
                text=data["text"], cache_hit=True, prompt_version="table-v1"
            )
        text = self._invoke_with_rotation(
            lambda client: self._call_table(client, markdown),
            label="table-summary",
        )
        path.write_text(
            json.dumps({"text": text, "prompt": "table-v1"}, ensure_ascii=False),
            encoding="utf-8",
        )
        return CaptionResult(text=text, cache_hit=False, prompt_version="table-v1")

    # ------------------------- internal calls -------------------------

    @_gemini_retry
    def _call_image(
        self,
        client: genai.Client,
        image_bytes: bytes,
        context: str,
        mime_type: str,
    ) -> str:
        """Single Gemini call for an image caption (with 5xx retry)."""
        user_text = IMAGE_PROMPT_V1
        if context:
            user_text += f"\n\nSurrounding-page text for context (may include the figure caption):\n---\n{context}\n---"
        response = client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                user_text,
            ],
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty caption")
        return text

    @_gemini_retry
    def _call_table(self, client: genai.Client, markdown: str) -> str:
        """Single Gemini call for a table summary (with 5xx retry)."""
        prompt = TABLE_PROMPT_V1 + "\n\nTable (Markdown):\n" + markdown
        response = client.models.generate_content(
            model=self._model,
            contents=[prompt],
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty table summary")
        return text
