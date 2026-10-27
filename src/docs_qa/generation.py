"""Generation: send the question and sources to the LLM, then extract the citations."""

import re
from dataclasses import dataclass, field

from docs_qa.llm import LLMClient
from docs_qa.prompts import SYSTEM_PROMPT, build_user_message
from docs_qa.retrieval import RetrievedChunk

CITATION_RE = re.compile(r"\[(\d+)\]")

NO_SOURCES_ANSWER = (
    "I couldn't find anything relevant in the indexed documentation. "
    "Has ingestion been run? (uv run docs-qa-ingest)"
)
REFUSAL_ANSWER = "I can't help with that request."


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Answer:
    text: str
    # Source numbers (1-based, matching the prompt) that the answer cites, in order of first use.
    cited: list[int] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str | None = None


def extract_citations(text: str, num_sources: int) -> list[int]:
    """Find [n] markers that point at a real source, without duplicates."""
    cited: list[int] = []
    for match in CITATION_RE.finditer(text):
        number = int(match.group(1))
        if 1 <= number <= num_sources and number not in cited:
            cited.append(number)
    return cited


class AnswerGenerator:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        if not chunks:
            return Answer(text=NO_SOURCES_ANSWER)

        completion = self._llm.complete(SYSTEM_PROMPT, build_user_message(question, chunks))
        usage = Usage(completion.input_tokens, completion.output_tokens)

        if completion.refused:
            return Answer(text=REFUSAL_ANSWER, usage=usage, model=completion.model)
        return Answer(
            text=completion.text,
            cited=extract_citations(completion.text, len(chunks)),
            usage=usage,
            model=completion.model,
        )
