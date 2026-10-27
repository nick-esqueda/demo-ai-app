"""Postgres access: connections, the chunks table, and bulk writes.

The query side lives in retrieval.py, next to the code that uses it.
"""

from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

DEFAULT_POOL_TIMEOUT_SECONDS = 10.0

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    corpus       text NOT NULL,
    source_path  text NOT NULL,   -- file path inside the docs folder
    heading_path text NOT NULL,   -- e.g. "Tutorial - User Guide > First Steps"
    url          text NOT NULL,   -- link to the section on the public docs site
    content      text NOT NULL,
    token_count  integer NOT NULL,
    embedding    vector({dimension}) NOT NULL
);
-- HNSW is an approximate nearest-neighbour index; vector_cosine_ops makes it serve
-- the cosine distance operator (<=>) used at query time.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_corpus_idx ON chunks (corpus);
"""

# atttypmod holds a vector column's dimension.
EMBEDDING_DIMENSION_SQL = """
SELECT atttypmod FROM pg_attribute
WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'
"""

DELETE_CORPUS_SQL = "DELETE FROM chunks WHERE corpus = %s"

INSERT_CHUNK_SQL = """
INSERT INTO chunks (corpus, source_path, heading_path, url, content, token_count, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


@dataclass(frozen=True)
class ChunkRow:
    """One row of the chunks table, minus the id and the embedding."""

    corpus: str
    source_path: str
    heading_path: str
    url: str
    content: str
    token_count: int


def register_vector_type(conn: psycopg.Connection) -> None:
    """Teach psycopg to send and receive pgvector's vector type as numpy arrays."""
    register_vector(conn)


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()

    register_vector_type(conn)
    return conn


def create_pool(database_url: str, timeout: float = DEFAULT_POOL_TIMEOUT_SECONDS) -> ConnectionPool:
    """Connection pool for the API. Expects the schema to exist (run ingestion first).

    Waits up to `timeout` seconds for the first connection, so a database that is down or
    misconfigured fails at startup instead of on the first request.
    """
    pool = ConnectionPool(
        database_url,
        configure=register_vector_type,
        kwargs={"connect_timeout": max(1, int(timeout))},
        timeout=timeout,  # how long a request waits for a free connection
        open=True,
    )

    try:
        pool.wait(timeout=timeout)
    except Exception:
        pool.close()
        raise
    return pool


def read_embedding_dimension(conn: psycopg.Connection) -> int | None:
    """The dimension of the existing chunks.embedding column, or None if there's no table."""
    row = conn.execute(EMBEDDING_DIMENSION_SQL).fetchone()
    return row[0] if row else None


def create_schema(conn: psycopg.Connection, dimension: int) -> None:
    existing_dimension = read_embedding_dimension(conn)
    is_dimension_mismatch = existing_dimension is not None and existing_dimension != dimension
    if is_dimension_mismatch:
        raise RuntimeError(
            f"chunks.embedding has {existing_dimension} dimensions but the embedder produces "
            f"{dimension}. Drop the table (DROP TABLE chunks) and re-run ingestion."
        )

    conn.execute(SCHEMA_SQL.format(dimension=dimension))
    conn.commit()


def build_insert_params(row: ChunkRow, embedding: list[float]) -> tuple:
    """Values for INSERT_CHUNK_SQL, in column order."""
    vector = np.array(embedding, dtype=np.float32)
    return (
        row.corpus,
        row.source_path,
        row.heading_path,
        row.url,
        row.content,
        row.token_count,
        vector,
    )


def replace_corpus(
    conn: psycopg.Connection,
    corpus: str,
    rows: list[ChunkRow],
    embeddings: list[list[float]],
) -> None:
    """Delete a corpus's chunks and insert the new ones in a single transaction."""
    rows_with_embeddings = zip(rows, embeddings, strict=True)
    insert_params = [build_insert_params(row, embedding) for row, embedding in rows_with_embeddings]

    with conn.transaction():
        conn.execute(DELETE_CORPUS_SQL, (corpus,))
        with conn.cursor() as cur:
            cur.executemany(INSERT_CHUNK_SQL, insert_params)
