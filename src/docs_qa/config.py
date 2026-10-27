"""App configuration, read from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres with the pgvector extension (see docker-compose.yml).
    database_url: str = "postgresql://postgres:postgres@localhost:5432/docs_qa"

    # Chat model. "openai-compatible" works with any OpenAI-style API; the default points at
    # Google Gemini's free tier. Set LLM_PROVIDER=anthropic to use Claude instead.
    llm_provider: str = "openai-compatible"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: SecretStr | None = None
    llm_model: str = "gemini-flash-latest"
    # Upper bound on the answer length. Thinking models count their reasoning against it too.
    llm_max_tokens: int = 8192

    # Only used when llm_provider is "anthropic".
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    # How much Claude thinks before answering: low | medium | high | xhigh | max.
    anthropic_effort: str = "medium"

    # Embeddings. Changing the model usually changes the vector size, so re-ingest after.
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Retrieval and chunking.
    top_k: int = 5
    chunk_max_tokens: int = 500

    # Where ingestion clones the source docs.
    data_dir: Path = Path("data")


@lru_cache
def get_settings() -> Settings:
    return Settings()
