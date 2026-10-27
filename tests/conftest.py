import hashlib

import pytest

from docs_qa.retrieval import RetrievedChunk


class FakeEmbedder:
    """Deterministic stand-in for a real embedding model: hashes words into a small vector."""

    dimension = 16

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in text.lower().split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            id=1,
            source_path="tutorial/path-params.md",
            heading_path="Path Parameters",
            url="https://fastapi.tiangolo.com/tutorial/path-params/#path-parameters",
            content="Declare path parameters with the same syntax as Python format strings.",
            score=0.82,
        ),
        RetrievedChunk(
            id=2,
            source_path="tutorial/query-params.md",
            heading_path="Query Parameters",
            url="https://fastapi.tiangolo.com/tutorial/query-params/#query-parameters",
            content="Function parameters that are not path parameters are query parameters.",
            score=0.74,
        ),
    ]
