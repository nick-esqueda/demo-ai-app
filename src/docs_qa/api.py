"""HTTP API. Run with:  uv run uvicorn docs_qa.api:app --reload"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from docs_qa import db
from docs_qa.config import Settings, get_settings
from docs_qa.embeddings import get_embedder
from docs_qa.generation import AnswerGenerator
from docs_qa.llm import LLMError, LLMRateLimitError, LLMUnavailableError, get_llm
from docs_qa.retrieval import Retriever

# Uvicorn only sets up logging for its own loggers; this makes the app's log lines show too.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("docs_qa.api")


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the long-lived pieces once at startup: DB pool, embedding model, LLM client."""
    settings = get_settings()

    start = time.perf_counter()
    pool = db.create_pool(settings.database_url, timeout=settings.db_timeout_seconds)
    with pool.connection() as conn:
        count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    log.info("Connected to Postgres in %d ms; %d chunks indexed", _elapsed_ms(start), count)

    start = time.perf_counter()
    embedder = get_embedder(settings)
    # Embed once now, on a worker thread like a request would, so the first request isn't slow
    # and a model that can't run from a thread shows up here rather than as a hung request.
    await asyncio.to_thread(embedder.embed_query, "warm up")
    log.info(
        "Loaded embedding model %s on %s in %d ms",
        settings.embedding_model,
        settings.embedding_device,
        _elapsed_ms(start),
    )

    app.state.retriever = Retriever(pool, embedder)
    app.state.generator = AnswerGenerator(get_llm(settings))
    log.info("LLM: %s via %s", _llm_name(settings), settings.llm_provider)
    yield
    pool.close()


def _llm_name(settings: Settings) -> str:
    if settings.llm_provider == "anthropic":
        return settings.anthropic_model
    return f"{settings.llm_model} at {settings.llm_base_url}"


app = FastAPI(title="Docs Q&A", lifespan=lifespan)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log when a request arrives, not only when it finishes, so a slow request is visible."""
    start = time.perf_counter()
    log.info("-> %s %s", request.method, request.url.path)
    response = await call_next(request)
    log.info(
        "<- %s %s %d in %d ms",
        request.method,
        request.url.path,
        response.status_code,
        _elapsed_ms(start),
    )
    return response


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
    start = time.perf_counter()
    chunks = retriever.search(body.question, k=body.top_k or settings.top_k)
    log.info("Retrieved %d chunks in %d ms", len(chunks), _elapsed_ms(start))

    start = time.perf_counter()
    try:
        answer = generator.generate(body.question, chunks)
    except LLMRateLimitError as exc:
        log.warning("LLM rate limited after %d ms: %s", _elapsed_ms(start), exc)
        raise HTTPException(status_code=429, detail=f"LLM rate limit or quota hit: {exc}") from exc
    except LLMUnavailableError as exc:
        log.warning("LLM unavailable after %d ms: %s", _elapsed_ms(start), exc)
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider is overloaded or not responding, try again shortly: {exc}",
        ) from exc
    except LLMError as exc:
        log.warning("LLM failed after %d ms: %s", _elapsed_ms(start), exc)
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc
    log.info(
        "LLM answered in %d ms (%s, %d in / %d out tokens)",
        _elapsed_ms(start),
        answer.model,
        answer.usage.input_tokens,
        answer.usage.output_tokens,
    )

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
