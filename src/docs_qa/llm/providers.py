"""Pick the LLM client named by the LLM_PROVIDER setting."""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import SecretStr

from docs_qa.config import Settings
from docs_qa.llm.anthropic_client import AnthropicLLM
from docs_qa.llm.base import LLMClient
from docs_qa.llm.openai_compatible_client import OpenAICompatibleLLM


def read_secret(secret: SecretStr | None) -> str | None:
    return secret.get_secret_value() if secret else None


def build_openai_compatible_llm(settings: Settings) -> LLMClient:
    return OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=read_secret(settings.llm_api_key),
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )


def build_anthropic_llm(settings: Settings) -> LLMClient:
    return AnthropicLLM(
        api_key=read_secret(settings.anthropic_api_key),
        model=settings.anthropic_model,
        effort=settings.anthropic_effort,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )


@dataclass(frozen=True)
class LLMProvider:
    build: Callable[[Settings], LLMClient]
    # One line for the startup log saying which model this provider will call.
    describe: Callable[[Settings], str]


# Registry pattern: maps each LLM_PROVIDER value to how to build and describe its client.
# To add a provider, write a client class (see base.py), then add an entry here.
LLM_PROVIDERS: dict[str, LLMProvider] = {
    "openai-compatible": LLMProvider(
        build=build_openai_compatible_llm,
        describe=lambda settings: f"{settings.llm_model} at {settings.llm_base_url}",
    ),
    "anthropic": LLMProvider(
        build=build_anthropic_llm,
        describe=lambda settings: settings.anthropic_model,
    ),
}


def _look_up_provider(settings: Settings) -> LLMProvider:
    provider = LLM_PROVIDERS.get(settings.llm_provider)
    if provider is None:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")
    return provider


def get_llm(settings: Settings) -> LLMClient:
    """Build the client named in settings.llm_provider."""
    return _look_up_provider(settings).build(settings)


def describe_llm(settings: Settings) -> str:
    return _look_up_provider(settings).describe(settings)
