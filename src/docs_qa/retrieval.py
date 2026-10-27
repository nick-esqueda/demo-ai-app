"""Retrieval: embed the question and fetch the nearest chunks by cosine distance."""

from dataclasses import dataclass

import numpy as np
from psycopg_pool import ConnectionPool

from docs_qa.embeddings import Embedder

# <=> is pgvector's cosine distance operator (0 = same direction, 2 = opposite).
# Ordering by it lets Postgres use the HNSW index created in db.py.
SEARCH_SQL = """
SELECT id, source_path, heading_path, url, content, embedding <=> %(query)s AS distance
FROM chunks
ORDER BY embedding <=> %(query)s
LIMIT %(k)s
"""


@dataclass(frozen=True)
class RetrievedChunk:
    id: int
    source_path: str
    heading_path: str
    url: str
    content: str
    # Cosine similarity, 1 - distance. Higher is more relevant.
    score: float


class Retriever:
    def __init__(self, pool: ConnectionPool, embedder: Embedder):
        self._pool = pool
        self._embedder = embedder

    def search(self, question: str, k: int) -> list[RetrievedChunk]:
        query = np.array(self._embedder.embed_query(question), dtype=np.float32)
        with self._pool.connection() as conn:
            rows = conn.execute(SEARCH_SQL, {"query": query, "k": k}).fetchall()
        return [
            RetrievedChunk(
                id=id_,
                source_path=source_path,
                heading_path=heading_path,
                url=url,
                content=content,
                score=1 - distance,
            )
            for id_, source_path, heading_path, url, content, distance in rows
        ]
