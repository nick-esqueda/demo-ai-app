"""Text embedding behind a small interface, so the provider can be swapped (e.g. for Voyage).

Queries and documents are embedded through separate methods because many retrieval
models (BGE, Voyage, E5) encode them differently.
"""

import os
from collections.abc import Callable
from typing import Protocol

from docs_qa.config import Settings

# Hugging Face tokenizers spin up their own thread pool, which is known to deadlock when mixed
# with other threads or forked processes (e.g. uvicorn --reload). Turn it off unless set.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_BATCH_SIZE = 32
# Only show a progress bar for big jobs like ingestion, not for single queries.
PROGRESS_BAR_MIN_TEXTS = 100


# Strategy pattern: retrieval and ingestion only depend on this interface, never on a
# specific embedding library. To add a provider, write a class with these members and
# register it in EMBEDDER_BUILDERS below.
class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Runs a local sentence-transformers model. Downloads it from Hugging Face on first use."""

    # BGE models expect this prefix on queries (not on documents) for retrieval.
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self, model_name: str, device: str | None = None, batch_size: int = DEFAULT_BATCH_SIZE
    ):
        # Imported here so the rest of the app (and the tests) don't need torch loaded.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size

        is_bge_model = "bge" in model_name.lower()
        self._query_prefix = self.BGE_QUERY_PREFIX if is_bge_model else ""

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > PROGRESS_BAR_MIN_TEXTS,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed_query = self._query_prefix + text
        vector = self._model.encode(prefixed_query, normalize_embeddings=True)
        return vector.tolist()


def build_sentence_transformer_embedder(settings: Settings) -> Embedder:
    return SentenceTransformerEmbedder(settings.embedding_model, device=settings.embedding_device)


# Registry of embedding providers, keyed by the EMBEDDING_PROVIDER setting.
EMBEDDER_BUILDERS: dict[str, Callable[[Settings], Embedder]] = {
    "sentence-transformers": build_sentence_transformer_embedder,
}


def get_embedder(settings: Settings) -> Embedder:
    """Build the embedder named in settings.embedding_provider."""
    build_embedder = EMBEDDER_BUILDERS.get(settings.embedding_provider)
    if build_embedder is None:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider!r}")
    return build_embedder(settings)
