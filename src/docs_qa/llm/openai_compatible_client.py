"""Client for any server that speaks the OpenAI Chat Completions API.

That covers Google Gemini (free tier, the default), Groq, Ollama, OpenAI itself, and others.
"""

import openai
from openai.types.chat import ChatCompletion

from docs_qa.llm.base import LLM_MAX_RETRIES, Completion
from docs_qa.llm.errors import convert_to_llm_error

# Local servers like Ollama ignore the key, but the SDK requires one to be set.
PLACEHOLDER_API_KEY = "not-needed"
CONTENT_FILTER_FINISH_REASON = "content_filter"


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        max_tokens: int,
        timeout: float,
    ):
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key or PLACEHOLDER_API_KEY,
            timeout=timeout,
            max_retries=LLM_MAX_RETRIES,
        )
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> Completion:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                # The newer name for max_tokens, which the OpenAI SDK marks as deprecated.
                max_completion_tokens=self._max_tokens,
                messages=messages,
            )
        except openai.APIError as exc:
            raise convert_to_llm_error(exc) from exc

        return self._build_completion(response)

    def _build_completion(self, response: ChatCompletion) -> Completion:
        choice = response.choices[0]
        usage = response.usage
        answer_text = choice.message.content or ""

        return Completion(
            text=answer_text.strip(),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model or self._model,
            is_refused=choice.finish_reason == CONTENT_FILTER_FINISH_REASON,
        )
