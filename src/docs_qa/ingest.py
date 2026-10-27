"""Ingestion CLI: fetch a corpus at its pinned ref, chunk it, embed the chunks, and store them.

Run with:  uv run docs-qa-ingest --corpus fastapi
"""

import argparse
import logging
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from docs_qa import db
from docs_qa.chunking import Chunk, chunk_markdown
from docs_qa.config import get_settings
from docs_qa.corpora import CORPORA, Corpus
from docs_qa.embeddings import get_embedder

log = logging.getLogger(__name__)

# FastAPI's docs pull code samples in from files: "{* ../../docs_src/app.py hl[3] *}" renders as a
# code block, and "{!../../docs_src/app.py!}" pastes the file inside an existing code fence.
CODE_BLOCK_INCLUDE_RE = re.compile(r"\{\*\s*(\S+)[^*]*\*\}")
RAW_INCLUDE_RE = re.compile(r"\{!\s*(\S+?)\s*!\}")


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def fetch_corpus(corpus: Corpus, data_dir: Path) -> Path:
    """Sparse, shallow checkout of just the docs folders at the pinned ref. Safe to re-run."""
    repo_dir = data_dir / corpus.name
    if not (repo_dir / ".git").exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        _git("init", "--quiet", cwd=repo_dir)
        _git("remote", "add", "origin", corpus.repo_url, cwd=repo_dir)
    _git("sparse-checkout", "set", corpus.docs_dir, *corpus.extra_dirs, cwd=repo_dir)
    _git("fetch", "--quiet", "--depth", "1", "origin", corpus.ref, cwd=repo_dir)
    _git("checkout", "--quiet", "--force", "FETCH_HEAD", cwd=repo_dir)
    return repo_dir


def resolve_includes(markdown: str, include_root: Path) -> str:
    """Inline the code files that a page includes by reference, so their code gets indexed too."""

    def read(path: str) -> str | None:
        file = include_root / path
        return file.read_text(encoding="utf-8").rstrip() if file.is_file() else None

    def code_block(match: re.Match) -> str:
        code = read(match.group(1))
        return f"```python\n{code}\n```" if code is not None else f"[code sample: {match.group(1)}]"

    def raw(match: re.Match) -> str:
        code = read(match.group(1))
        return code if code is not None else f"# code sample: {match.group(1)}"

    return RAW_INCLUDE_RE.sub(raw, CODE_BLOCK_INCLUDE_RE.sub(code_block, markdown))


def page_url(base_url: str, source_path: str, anchor: str | None) -> str:
    """Map a docs file to its URL on an mkdocs site.

    "tutorial/first-steps.md" -> "<base_url>/tutorial/first-steps/", "index.md" -> "<base_url>/".
    """
    path = source_path.removesuffix(".md")
    if path == "index" or path.endswith("/index"):
        path = path.removesuffix("index")
    else:
        path += "/"
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    return f"{url}#{anchor}" if anchor else url


def iter_pages(corpus: Corpus, repo_dir: Path) -> Iterator[tuple[str, str]]:
    """Yield (path inside docs_dir, Markdown with includes resolved) for each page."""
    docs_root = repo_dir / corpus.docs_dir
    # FastAPI's include paths are relative to the folder above docs_dir, where mkdocs.yml lives.
    include_root = docs_root.parent
    for file in sorted(docs_root.rglob("*.md")):
        source_path = file.relative_to(docs_root).as_posix()
        if source_path in corpus.exclude:
            continue
        yield source_path, resolve_includes(file.read_text(encoding="utf-8"), include_root)


def to_row(corpus: Corpus, chunk: Chunk) -> db.ChunkRow:
    return db.ChunkRow(
        corpus=corpus.name,
        source_path=chunk.source_path,
        heading_path=" > ".join(chunk.heading_path) or chunk.source_path,
        url=page_url(corpus.base_url, chunk.source_path, chunk.anchor),
        content=chunk.text,
        token_count=chunk.token_count,
    )


def embedding_text(row: db.ChunkRow) -> str:
    """Put the heading path in front of the text so the vector knows what the section is about."""
    return f"{row.heading_path}\n\n{row.content}"


def ingest(corpus_name: str) -> int:
    settings = get_settings()
    corpus = CORPORA[corpus_name]

    log.info("Fetching %s at %s", corpus.repo_url, corpus.ref)
    repo_dir = fetch_corpus(corpus, settings.data_dir)

    rows = [
        to_row(corpus, chunk)
        for source_path, markdown in iter_pages(corpus, repo_dir)
        for chunk in chunk_markdown(markdown, source_path, settings.chunk_max_tokens)
    ]
    log.info("Chunked %d pages into %d chunks", len({r.source_path for r in rows}), len(rows))

    log.info("Embedding with %s", settings.embedding_model)
    embedder = get_embedder(settings)
    embeddings = embedder.embed_documents([embedding_text(row) for row in rows])

    with db.connect(settings.database_url) as conn:
        db.create_schema(conn, embedder.dimension)
        db.replace_corpus(conn, corpus.name, rows, embeddings)
    log.info("Stored %d chunks for corpus %r", len(rows), corpus.name)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", choices=sorted(CORPORA), default="fastapi")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest(args.corpus)


if __name__ == "__main__":
    main()
