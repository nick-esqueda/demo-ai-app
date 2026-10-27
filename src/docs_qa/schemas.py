"""Request and response bodies for the HTTP API, and how to build them."""

from pydantic import BaseModel, Field

from docs_qa.generation import Answer
from docs_qa.retrieval import RetrievedChunk

MAX_QUESTION_LENGTH = 2000
MAX_TOP_K = 20
SCORE_DECIMALS = 4


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)


class Source(BaseModel):
    number: int
    cited: bool
    heading_path: str
    url: str
    source_path: str
    score: float
    content: str


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AskResponse(BaseModel):
    answer: str
    # Every retrieved chunk, numbered as in the prompt. `cited` marks the ones the answer used.
    sources: list[Source]
    usage: TokenUsage
    model: str | None


def build_source(number: int, chunk: RetrievedChunk, is_cited: bool) -> Source:
    return Source(
        number=number,
        cited=is_cited,
        heading_path=chunk.heading_path,
        url=chunk.url,
        source_path=chunk.source_path,
        score=round(chunk.score, SCORE_DECIMALS),
        content=chunk.content,
    )


def build_ask_response(answer: Answer, chunks: list[RetrievedChunk]) -> AskResponse:
    sources = [
        build_source(number, chunk, is_cited=number in answer.cited_numbers)
        for number, chunk in enumerate(chunks, start=1)
    ]
    usage = TokenUsage(
        input_tokens=answer.usage.input_tokens,
        output_tokens=answer.usage.output_tokens,
    )
    return AskResponse(answer=answer.text, sources=sources, usage=usage, model=answer.model)
