"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    """Absolute path to the tiny 2-page fixture PDF used by downstream tests."""
    path = FIXTURES / "sample.pdf"
    if not path.exists():
        pytest.skip(
            f"fixture PDF missing at {path} — regenerate with "
            "'uv run python scripts/make_fixture_pdf.py' (created in Phase 1)"
        )
    return path

