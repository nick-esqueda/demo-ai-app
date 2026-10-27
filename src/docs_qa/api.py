"""HTTP API. Run with:  uv run uvicorn docs_qa.api:app --reload"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from docs_qa import db
from docs_qa.config import Settings, get_settings
from docs_qa.embeddings import get_embedder
from docs_qa.generation import AnswerGenerator
from docs_qa.llm import LLMError, LLMRateLimitError, get_llm
from docs_qa.retrieval import Retriever


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the long-lived pieces once at startup: DB pool, embedding model, LLM client."""
    settings = get_settings()
    pool = db.create_pool(settings.database_url)
    app.state.retriever = Retriever(pool, get_embedder(settings))
    app.state.generator = AnswerGenerator(get_llm(settings))
    yield
    pool.close()


app = FastAPI(title="Docs Q&A", lifespan=lifespan)


# Dependencies. Tests swap these out through app.dependency_overrides.
def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_generator(request: Request) -> AnswerGenerator:
    return request.app.state.generator


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# A plain `def` (not `async def`) because embedding, the DB driver, and the LLM client used
# here are all blocking. FastAPI runs sync endpoints in a thread pool so they don't stall
# the server.
@app.post("/ask")
def ask(
    body: AskRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
    generator: Annotated[AnswerGenerator, Depends(get_generator)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AskResponse:
    chunks = retriever.search(body.question, k=body.top_k or settings.top_k)

    try:
        answer = generator.generate(body.question, chunks)
    except LLMRateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"LLM rate limit or quota hit: {exc}") from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc

    return AskResponse(
        answer=answer.text,
        sources=[
            Source(
                number=number,
                cited=number in answer.cited,
                heading_path=chunk.heading_path,
                url=chunk.url,
                source_path=chunk.source_path,
                score=round(chunk.score, 4),
                content=chunk.content,
            )
            for number, chunk in enumerate(chunks, start=1)
        ],
        usage=TokenUsage(
            input_tokens=answer.usage.input_tokens,
            output_tokens=answer.usage.output_tokens,
        ),
        model=answer.model,
    )
