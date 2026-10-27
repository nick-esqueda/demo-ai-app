"""Client for Claude through the Anthropic SDK."""

import anthropic

from docs_qa.llm.base import LLM_MAX_RETRIES, Completion
from docs_qa.llm.errors import convert_to_llm_error

# With `fallbacks`, a request that the model's safety classifiers decline is retried on a
# recommended fallback model in the same call. It needs the beta endpoint and this header.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODE = "default"
REFUSAL_STOP_REASON = "refusal"


def join_text_blocks(content: list) -> str:
    """The response can also hold thinking blocks; the answer is the text blocks."""
    text_blocks = [block.text for block in content if block.type == "text"]
    return "".join(text_blocks)


class AnthropicLLM:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        effort: str,
        max_tokens: int,
        timeout: float,
    ):
        # With no key passed, the SDK reads the ANTHROPIC_API_KEY env var.
        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout, max_retries=LLM_MAX_RETRIES
        )
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> Completion:
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                # Effort controls how much the model thinks before answering.
                output_config={"effort": self._effort},
                betas=[FALLBACK_BETA],
                fallbacks=FALLBACK_MODE,
            )
        except anthropic.APIError as exc:
            raise convert_to_llm_error(exc) from exc

        answer_text = join_text_blocks(response.content)
        return Completion(
            text=answer_text.strip(),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            is_refused=response.stop_reason == REFUSAL_STOP_REASON,
        )
