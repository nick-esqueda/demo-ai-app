"""Chat model clients behind one small interface, so the provider is a config choice.

- base.py: the LLMClient interface and the Completion it returns
- errors.py: provider-independent errors, and the translation from SDK errors
- openai_compatible_client.py: any OpenAI-style API (Gemini, Groq, Ollama, OpenAI, ...)
- anthropic_client.py: Claude through the Anthropic SDK
- providers.py: the registry that picks a client from settings
"""

from docs_qa.llm.anthropic_client import AnthropicLLM
from docs_qa.llm.base import Completion, LLMClient
from docs_qa.llm.errors import LLMError, LLMRateLimitError, LLMUnavailableError
from docs_qa.llm.openai_compatible_client import OpenAICompatibleLLM
from docs_qa.llm.providers import describe_llm, get_llm

__all__ = [
    "AnthropicLLM",
    "Completion",
    "LLMClient",
    "LLMError",
    "LLMRateLimitError",
    "LLMUnavailableError",
    "OpenAICompatibleLLM",
    "describe_llm",
    "get_llm",
]
