"""Tests for src.router — classifier parsing + fallback behaviour.

Groq is mocked; we only verify parsing and the class-to-boost mapping.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.router import RetrievalConfig, classify


def _fake_groq(content: str):
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )
            )
        )
    )
    return client


@pytest.mark.parametrize(
    "raw, expected_class",
    [
        ("table", "table"),
        ("chart", "chart"),
        ("factual", "factual"),
        ("summary", "summary"),
        ("  TABLE.\n", "table"),
        ("table — the question asks for figures", "table"),
    ],
)
def test_classify_parses_label(raw: str, expected_class: str) -> None:
    config = classify("irrelevant query", client=_fake_groq(raw))
    assert isinstance(config, RetrievalConfig)
    assert config.query_class == expected_class


def test_table_class_boosts_tables_most() -> None:
    config = classify("q", client=_fake_groq("table"))
    assert config.modality_boosts["table"] > config.modality_boosts.get("image", 1.0)


def test_chart_class_boosts_images_most() -> None:
    config = classify("q", client=_fake_groq("chart"))
    assert config.modality_boosts["image"] > config.modality_boosts.get("table", 1.0)


def test_factual_has_no_boosts() -> None:
    config = classify("q", client=_fake_groq("factual"))
    assert config.modality_boosts == {}


def test_classify_unparseable_falls_back_to_factual() -> None:
    config = classify("weird", client=_fake_groq("banana"))
    assert config.query_class == "factual"
    assert config.modality_boosts == {}


def test_classify_empty_response_falls_back_to_factual() -> None:
    config = classify("weird", client=_fake_groq(""))
    assert config.query_class == "factual"
