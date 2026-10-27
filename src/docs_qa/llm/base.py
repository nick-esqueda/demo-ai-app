"""The interface every chat model client implements."""

from dataclasses import dataclass
from typing import Protocol

# Each failed or timed-out call is retried this many times by the provider SDK.
LLM_MAX_RETRIES = 1


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    # True when the provider declined to answer (safety filter or refusal).
    is_refused: bool = False


# Strategy pattern: generation only depends on this interface, never on a provider SDK.
# To add a provider, write a class with a `complete` method like this one and register it
# in providers.py.
class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> Completion: ...
