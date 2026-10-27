"""Ingestion CLI: fetch a corpus at its pinned ref, chunk it, embed the chunks, and store them.

Run with:  uv run docs-qa-ingest --corpus fastapi
"""

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

from docs_qa import db
from docs_qa.chunking import Chunk, chunk_markdown
from docs_qa.config import Settings, get_settings
from docs_qa.corpora import CORPORA, Corpus
from docs_qa.embeddings import get_embedder
from docs_qa.git_checkout import fetch_corpus
from docs_qa.includes import resolve_includes
from docs_qa.logging_setup import configure_logging

log = logging.getLogger(__name__)

DEFAULT_CORPUS = "fastapi"
HEADING_PATH_SEPARATOR = " > "
INDEX_PAGE_NAME = "index"


def build_page_url(base_url: str, source_path: str, anchor: str | None) -> str:
    """Map a docs file to its URL on an mkdocs site.

    "tutorial/first-steps.md" -> "<base_url>/tutorial/first-steps/", "index.md" -> "<base_url>/".
    """
    path = source_path.removesuffix(".md")
    is_index_page = path == INDEX_PAGE_NAME or path.endswith(f"/{INDEX_PAGE_NAME}")
    if is_index_page:
        path = path.removesuffix(INDEX_PAGE_NAME)
    else:
        path += "/"

    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    return f"{url}#{anchor}" if anchor else url


def read_pages(corpus: Corpus, repo_dir: Path) -> Iterator[tuple[str, str]]:
    """Yield (path inside docs_dir, Markdown with includes resolved) for each page."""
    docs_root = repo_dir / corpus.docs_dir
    # FastAPI's include paths are relative to the folder above docs_dir, where mkdocs.yml lives.
    include_root = docs_root.parent

    for file in sorted(docs_root.rglob("*.md")):
        source_path = file.relative_to(docs_root).as_posix()
        if source_path in corpus.exclude:
            continue

        markdown = file.read_text(encoding="utf-8")
        yield source_path, resolve_includes(markdown, include_root)


def build_chunk_row(corpus: Corpus, chunk: Chunk) -> db.ChunkRow:
    heading_path = HEADING_PATH_SEPARATOR.join(chunk.heading_path)
    return db.ChunkRow(
        corpus=corpus.name,
        source_path=chunk.source_path,
        heading_path=heading_path or chunk.source_path,
        url=build_page_url(corpus.base_url, chunk.source_path, chunk.anchor),
        content=chunk.text,
        token_count=chunk.token_count,
    )


def build_embedding_text(row: db.ChunkRow) -> str:
    """Put the heading path in front of the text so the vector knows what the section is about."""
    return f"{row.heading_path}\n\n{row.content}"


def build_chunk_rows(corpus: Corpus, repo_dir: Path, max_tokens: int) -> list[db.ChunkRow]:
    rows: list[db.ChunkRow] = []
    for source_path, markdown in read_pages(corpus, repo_dir):
        chunks = chunk_markdown(markdown, source_path, max_tokens)
        rows.extend(build_chunk_row(corpus, chunk) for chunk in chunks)
    return rows


def embed_and_store(settings: Settings, corpus: Corpus, rows: list[db.ChunkRow]) -> None:
    log.info("Embedding with %s", settings.embedding_model)
    embedder = get_embedder(settings)
    texts = [build_embedding_text(row) for row in rows]
    embeddings = embedder.embed_documents(texts)

    with db.connect(settings.database_url) as conn:
        db.create_schema(conn, embedder.dimension)
        db.replace_corpus(conn, corpus.name, rows, embeddings)


def ingest(corpus_name: str) -> int:
    settings = get_settings()
    corpus = CORPORA[corpus_name]

    log.info("Fetching %s at %s", corpus.repo_url, corpus.ref)
    repo_dir = fetch_corpus(corpus, settings.data_dir)

    rows = build_chunk_rows(corpus, repo_dir, settings.chunk_max_tokens)
    page_count = len({row.source_path for row in rows})
    log.info("Chunked %d pages into %d chunks", page_count, len(rows))

    embed_and_store(settings, corpus, rows)
    log.info("Stored %d chunks for corpus %r", len(rows), corpus.name)
    return len(rows)


def main() -> None:
    description = __doc__.splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--corpus", choices=sorted(CORPORA), default=DEFAULT_CORPUS)
    args = parser.parse_args()

    configure_logging()
    ingest(args.corpus)


if __name__ == "__main__":
    main()
