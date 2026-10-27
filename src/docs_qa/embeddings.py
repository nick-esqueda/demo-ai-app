"""Text embedding behind a small interface, so the provider can be swapped (e.g. for Voyage).

Queries and documents are embedded through separate methods because many retrieval
models (BGE, Voyage, E5) encode them differently.
"""

from typing import Protocol

from docs_qa.config import Settings


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Runs a local sentence-transformers model. Downloads it from Hugging Face on first use."""

    # BGE models expect this prefix on queries (not on documents) for retrieval.
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, batch_size: int = 32):
        # Imported here so the rest of the app (and the tests) don't need torch loaded.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._query_prefix = self.BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(self._query_prefix + text, normalize_embeddings=True).tolist()


def get_embedder(settings: Settings) -> Embedder:
    """Build the embedder named in settings. Add new providers here."""
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider!r}")
