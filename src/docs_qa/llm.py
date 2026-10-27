"""Chat model clients behind one small interface, so the provider is a config choice.

- OpenAICompatibleLLM covers any server that speaks the OpenAI Chat Completions API:
  Google Gemini (free tier, the default), Groq, Ollama, OpenAI itself, and others.
- AnthropicLLM calls Claude through the Anthropic SDK.
"""

from dataclasses import dataclass
from typing import Protocol

import anthropic
import openai

from docs_qa.config import Settings


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    # True when the provider declined to answer (safety filter or refusal).
    refused: bool = False


class LLMError(Exception):
    """The provider returned an error or couldn't be reached."""


class LLMRateLimitError(LLMError):
    """The provider rejected the request for rate limit or quota reasons (common on free tiers)."""


class LLMUnavailableError(LLMError):
    """The provider is overloaded or down (HTTP 503/529). Usually clears up on its own."""


def _error_message(exc: openai.APIError | anthropic.APIError) -> str:
    """Pull the provider's own error message out of an SDK exception.

    The SDKs' default message can be just "Error code: 503", e.g. because Gemini wraps its
    error JSON in a list, so dig the message out of the body.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("message"):
            return f"{error['message']} (status {getattr(exc, 'status_code', '?')})"
    return str(exc)


def _to_llm_error(exc: openai.APIError | anthropic.APIError) -> LLMError:
    message = _error_message(exc)
    status = getattr(exc, "status_code", None)
    if status == 429:
        return LLMRateLimitError(message)
    if status in (503, 529):
        return LLMUnavailableError(message)
    if isinstance(exc, openai.APITimeoutError | anthropic.APITimeoutError):
        return LLMUnavailableError(f"No response from the LLM provider in time: {message}")
    return LLMError(message)


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> Completion: ...


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        max_tokens: int,
        timeout: float = 30.0,
    ):
        # Local servers like Ollama ignore the key, but the SDK requires one to be set.
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
            max_retries=1,
        )
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> Completion:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                # The newer name for max_tokens, which the OpenAI SDK marks as deprecated.
                max_completion_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except openai.APIError as exc:
            raise _to_llm_error(exc) from exc

        choice = response.choices[0]
        usage = response.usage
        return Completion(
            text=(choice.message.content or "").strip(),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model or self._model,
            refused=choice.finish_reason == "content_filter",
        )


class AnthropicLLM:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        effort: str,
        max_tokens: int,
        timeout: float = 30.0,
    ):
        # With no key passed, the SDK reads the ANTHROPIC_API_KEY env var.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=1)
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> Completion:
        try:
            # The beta endpoint is only needed for `fallbacks`: if the model's safety classifiers
            # decline a request, the API retries it on a recommended fallback model in the same
            # call.
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                # Effort controls how much the model thinks before answering.
                output_config={"effort": self._effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIError as exc:
            raise _to_llm_error(exc) from exc

        # The response can also hold thinking blocks; the answer is the text blocks.
        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(
            text=text.strip(),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            refused=response.stop_reason == "refusal",
        )


def get_llm(settings: Settings) -> LLMClient:
    """Build the client named in settings.llm_provider."""
    if settings.llm_provider == "openai-compatible":
        key = settings.llm_api_key
        return OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=key.get_secret_value() if key else None,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider == "anthropic":
        key = settings.anthropic_api_key
        return AnthropicLLM(
            api_key=key.get_secret_value() if key else None,
            model=settings.anthropic_model,
            effort=settings.anthropic_effort,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")
