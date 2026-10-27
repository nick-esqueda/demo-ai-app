"""The LLM clients, with the provider SDKs mocked out. No network calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from docs_qa.config import Settings
from docs_qa.llm import (
    AnthropicLLM,
    LLMRateLimitError,
    LLMUnavailableError,
    OpenAICompatibleLLM,
    get_llm,
)


def test_openai_compatible_request_and_response():
    llm = OpenAICompatibleLLM("https://example.test/v1", "key", model="gemini-x", max_tokens=100)
    llm._client = MagicMock()
    llm._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=" Answer [1] "), finish_reason="stop")
        ],
        usage=SimpleNamespace(prompt_tokens=300, completion_tokens=12),
        model="gemini-x-001",
    )

    result = llm.complete("system text", "user text")

    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gemini-x"
    assert kwargs["max_tokens"] == 100
    assert kwargs["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert (result.text, result.input_tokens, result.output_tokens) == ("Answer [1]", 300, 12)
    assert result.model == "gemini-x-001"
    assert not result.refused


def test_openai_compatible_maps_rate_limits():
    llm = OpenAICompatibleLLM("https://example.test/v1", None, model="m", max_tokens=100)
    llm._client = MagicMock()
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    llm._client.chat.completions.create.side_effect = openai.RateLimitError(
        "quota exceeded", response=httpx.Response(429, request=request), body=None
    )

    with pytest.raises(LLMRateLimitError):
        llm.complete("s", "u")


def test_openai_compatible_surfaces_gemini_overloaded_message():
    """Gemini's OpenAI-compatible endpoint wraps its error JSON in a list."""
    llm = OpenAICompatibleLLM("https://example.test/v1", "key", model="m", max_tokens=100)
    llm._client = MagicMock()
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    body = [
        {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}
    ]
    llm._client.chat.completions.create.side_effect = openai.InternalServerError(
        "Error code: 503", response=httpx.Response(503, request=request), body=body
    )

    with pytest.raises(LLMUnavailableError, match="The model is overloaded"):
        llm.complete("s", "u")


def test_clients_use_a_short_timeout_and_one_retry():
    llm = OpenAICompatibleLLM("https://example.test/v1", "key", model="m", max_tokens=100)
    assert (llm._client.timeout, llm._client.max_retries) == (30.0, 1)
    claude = AnthropicLLM("key", model="m", effort="low", max_tokens=100, timeout=12)
    assert (claude._client.timeout, claude._client.max_retries) == (12, 1)


def test_anthropic_request_and_response():
    llm = AnthropicLLM("key", model="claude-opus-5-5", effort="medium", max_tokens=100)
    llm._client = MagicMock()
    llm._client.beta.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Answer [2]"),
        ],
        usage=SimpleNamespace(input_tokens=400, output_tokens=30),
        model="claude-opus-5-5",
        stop_reason="end_turn",
    )

    result = llm.complete("system text", "user text")

    kwargs = llm._client.beta.messages.create.call_args.kwargs
    assert kwargs["system"] == "system text"
    assert kwargs["messages"] == [{"role": "user", "content": "user text"}]
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["fallbacks"] == "default"
    assert (result.text, result.input_tokens, result.output_tokens) == ("Answer [2]", 400, 30)
    assert not result.refused


def test_get_llm_picks_provider():
    assert isinstance(get_llm(Settings(_env_file=None)), OpenAICompatibleLLM)
    anthropic_settings = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="k")
    assert isinstance(get_llm(anthropic_settings), AnthropicLLM)
    with pytest.raises(ValueError):
        get_llm(Settings(_env_file=None, llm_provider="nope"))
