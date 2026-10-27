# Docs Q&A

RAG question answering over library docs (FastAPI first), built as a learning project for AI
engineering. Nick reads the code to learn the architecture, so keep it easy to follow.
See README.md for the architecture and run steps.

## Code style

Follow the style guide for every change, and when refactoring older code:

@docs/CODE_STYLE.md

Python specifics: booleans and boolean-returning functions use `is_` / `has_` prefixes
(snake_case), and the formatter is Ruff with its default settings.

## Commands

- Install: `uv sync`
- Tests: `uv run pytest`
- Tests including the pgvector round trip (needs `docker compose up -d`):
  `TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/docs_qa_test uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`

Before reporting tests as passing, check the pytest summary for skipped tests:
`tests/test_db_integration.py` skips unless `TEST_DATABASE_URL` is set.
