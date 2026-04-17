# Makefile — thin wrappers around the project CLIs.
# Phase-0 targets that aren't implemented yet echo TODO; they'll be filled in at the phase they belong to.

.PHONY: help parse index ingest run eval test lint fmt clean

help:
	@echo "Available targets:"
	@echo "  parse   — run src.ingest only (parse PDF → chunks.jsonl + images/)        [Phase 1]"
	@echo "  index   — run src.index only (BGE-M3 + Chroma + BM25)                      [Phase 2]"
	@echo "  ingest  — parse + index in sequence                                         [Phase 2]"
	@echo "  run     — launch Streamlit chat UI                                          [Phase 5]"
	@echo "  eval    — run RAGAS evaluation over eval/questions.json                     [Phase 7]"
	@echo "  test    — run pytest"
	@echo "  lint    — run ruff + mypy --strict src/"
	@echo "  fmt     — run ruff format"
	@echo "  clean   — remove generated data (chroma_db, chunks.jsonl, cache)"

parse:
	uv run python -m src.ingest --pdf data/source.pdf --out data/

index:
	uv run python -m src.index --chunks data/chunks.jsonl --out data/chroma_db

ingest: parse index

run:
	uv run streamlit run app.py

eval:
	@echo "TODO: implement in Phase 7 — 'uv run python -m eval.run_eval'"

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run mypy --strict src/

fmt:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf data/chroma_db data/chunks.jsonl data/cache data/images
