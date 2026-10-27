"""Round trip through a real Postgres + pgvector. Skipped unless TEST_DATABASE_URL is set.

Point it at a throwaway database: the test drops and recreates the chunks table.
"""

import os

import pytest

from docs_qa import db
from docs_qa.retrieval import Retriever

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL not set")


def test_store_and_search(fake_embedder):
    rows = [
        db.ChunkRow("test", "a.md", "Path Parameters", "u/a", "path parameters in the url", 5),
        db.ChunkRow("test", "b.md", "Query Parameters", "u/b", "query parameters after the ?", 5),
        db.ChunkRow("test", "c.md", "Background Tasks", "u/c", "run work after the response", 5),
    ]
    with db.connect(DATABASE_URL) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
        db.create_schema(conn, fake_embedder.dimension)
        db.replace_corpus(
            conn, "test", rows, fake_embedder.embed_documents([r.content for r in rows])
        )
        # Re-ingesting replaces rather than duplicates.
        db.replace_corpus(
            conn, "test", rows, fake_embedder.embed_documents([r.content for r in rows])
        )
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 3

    pool = db.create_pool(DATABASE_URL)
    try:
        results = Retriever(pool, fake_embedder).search("query parameters after the ?", k=2)
    finally:
        pool.close()

    assert len(results) == 2
    assert results[0].source_path == "b.md"
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score >= results[1].score


def test_dimension_mismatch_is_reported(fake_embedder):
    with db.connect(DATABASE_URL) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
        db.create_schema(conn, fake_embedder.dimension)
        with pytest.raises(RuntimeError, match="dimensions"):
            db.create_schema(conn, fake_embedder.dimension + 1)
