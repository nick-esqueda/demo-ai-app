"""HTTP API. Run with:  uv run uvicorn docs_qa.api:app --reload"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from psycopg_pool import ConnectionPool

from docs_qa import db
from docs_qa.config import Settings, get_settings
from docs_qa.embeddings import Embedder, get_embedder
from docs_qa.generation import Answer, AnswerGenerator
from docs_qa.llm import LLMError, LLMRateLimitError, LLMUnavailableError, describe_llm, get_llm
from docs_qa.logging_setup import configure_logging
from docs_qa.retrieval import RetrievedChunk, Retriever
from docs_qa.schemas import AskRequest, AskResponse, build_ask_response
from docs_qa.timing import Stopwatch

configure_logging()
log = logging.getLogger("docs_qa.api")

COUNT_CHUNKS_SQL = "SELECT count(*) FROM chunks"
WARM_UP_QUERY = "warm up"

# How each LLM error is reported to the API caller: (HTTP status, message prefix).
# The first entry that matches the error's type wins, so keep subclasses above LLMError.
LLM_ERROR_RESPONSES: list[tuple[type[LLMError], int, str]] = [
    (LLMRateLimitError, 429, "LLM rate limit or quota hit"),
    (LLMUnavailableError, 503, "LLM provider is overloaded or not responding, try again shortly"),
    (LLMError, 502, "LLM provider error"),
]


def open_database_pool(settings: Settings) -> ConnectionPool:
    stopwatch = Stopwatch()
    pool = db.create_pool(settings.database_url, timeout=settings.db_timeout_seconds)

    with pool.connection() as conn:
        chunk_count = conn.execute(COUNT_CHUNKS_SQL).fetchone()[0]
    log.info("Connected to Postgres in %d ms; %d chunks indexed", stopwatch.elapsed_ms, chunk_count)
    return pool


async def load_embedder(settings: Settings) -> Embedder:
    stopwatch = Stopwatch()
    embedder = get_embedder(settings)

    # Embed once now, on a worker thread like a request would, so the first request isn't slow
    # and a model that can't run from a thread shows up here rather than as a hung request.
    await asyncio.to_thread(embedder.embed_query, WARM_UP_QUERY)

    log.info(
        "Loaded embedding model %s on %s in %d ms",
        settings.embedding_model,
        settings.embedding_device,
        stopwatch.elapsed_ms,
    )
    return embedder


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the long-lived pieces once at startup: DB pool, embedding model, LLM client."""
    settings = get_settings()
    pool = open_database_pool(settings)
    embedder = await load_embedder(settings)

    app.state.retriever = Retriever(pool, embedder)
    app.state.generator = AnswerGenerator(get_llm(settings))
    log.info("LLM: %s via %s", describe_llm(settings), settings.llm_provider)

    yield
    pool.close()


app = FastAPI(title="Docs Q&A", lifespan=lifespan)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log when a request arrives, not only when it finishes, so a slow request is visible."""
    stopwatch = Stopwatch()
    method, path = request.method, request.url.path
    log.info("-> %s %s", method, path)

    response = await call_next(request)
    log.info("<- %s %s %d in %d ms", method, path, response.status_code, stopwatch.elapsed_ms)
    return response


# Dependencies. Tests swap these out through app.dependency_overrides.
def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_generator(request: Request) -> AnswerGenerator:
    return request.app.state.generator


def build_llm_http_error(exc: LLMError) -> HTTPException:
    for error_type, status_code, message_prefix in LLM_ERROR_RESPONSES:
        if isinstance(exc, error_type):
            return HTTPException(status_code=status_code, detail=f"{message_prefix}: {exc}")
    raise TypeError(f"No HTTP response configured for {type(exc).__name__}")


def retrieve_chunks(retriever: Retriever, question: str, k: int) -> list[RetrievedChunk]:
    stopwatch = Stopwatch()
    chunks = retriever.search(question, k=k)
    log.info("Retrieved %d chunks in %d ms", len(chunks), stopwatch.elapsed_ms)
    return chunks


def generate_answer(
    generator: AnswerGenerator, question: str, chunks: list[RetrievedChunk]
) -> Answer:
    stopwatch = Stopwatch()
    try:
        answer = generator.generate(question, chunks)
    except LLMError as exc:
        log.warning("LLM failed after %d ms: %s", stopwatch.elapsed_ms, exc)
        raise build_llm_http_error(exc) from exc

    usage = answer.usage
    log.info(
        "LLM answered in %d ms (%s, %d in / %d out tokens)",
        stopwatch.elapsed_ms,
        answer.model,
        usage.input_tokens,
        usage.output_tokens,
    )
    return answer


@app.get("/health")
def check_health() -> dict[str, str]:
    return {"status": "ok"}


# A plain `def` (not `async def`) because embedding, the DB driver, and the LLM client used
# here are all blocking. FastAPI runs sync endpoints in a thread pool so they don't stall
# the server.
@app.post("/ask")
def answer_question(
    body: AskRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
    generator: Annotated[AnswerGenerator, Depends(get_generator)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AskResponse:
    k = body.top_k or settings.top_k
    chunks = retrieve_chunks(retriever, body.question, k)
    answer = generate_answer(generator, body.question, chunks)
    return build_ask_response(answer, chunks)
