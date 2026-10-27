# Docs Q&A

Ask questions about the FastAPI documentation and get answers that cite their sources.
It is a retrieval-augmented generation (RAG) app, built as a hands-on way to learn
AI engineering: embeddings, vector search, prompting, and later evals, tracing, and agents.

## How it works

```
                 ingestion (offline, once per docs version)
  FastAPI repo ──► Markdown pages ──► chunks ──► embeddings ──► Postgres + pgvector
  (pinned tag)     (code samples       (split by     (local
                    inlined)            headings)     model)

                 question answering (POST /ask)
  question ──► embed ──► top-k nearest chunks ──► prompt with numbered sources ──► LLM
                         (cosine distance,                                          │
                          HNSW index)            answer + cited sources + usage ◄───┘
```

| Module | What it does |
| --- | --- |
| `config.py` | All settings, from env vars or `.env` |
| `corpora.py` | Which docs to ingest: repo, pinned ref, docs folder, public URL |
| `chunking.py` | Splits Markdown by headings, then by paragraphs to stay under ~500 tokens |
| `embeddings.py` | `Embedder` interface plus a local sentence-transformers implementation |
| `db.py` | Postgres connection, `chunks` table with an HNSW index, bulk writes |
| `ingest.py` | CLI: fetch, chunk, embed, store |
| `retrieval.py` | Embeds the question and runs the nearest-neighbour SQL query |
| `prompts.py` | System prompt and the numbered-sources user message |
| `llm.py` | `LLMClient` interface: any OpenAI-compatible API (Gemini, Groq, Ollama), or Claude |
| `generation.py` | Calls the LLM and extracts `[n]` citations from the answer |
| `api.py` | FastAPI app with `POST /ask` |

## Run it

You need [uv](https://docs.astral.sh/uv/), Docker, and git.

1. **Get a free LLM key.** The default is Google Gemini's free tier: create a key at
   <https://aistudio.google.com/apikey>. (Groq and local Ollama also work, see below.)

2. **Configure.**
   ```bash
   cp .env.example .env
   # then set LLM_API_KEY in .env
   ```

3. **Install and start Postgres.**
   ```bash
   uv sync
   docker compose up -d
   ```

4. **Ingest the FastAPI docs.** The first run downloads the embedding model (~130 MB) and
   the docs, then embeds a few thousand chunks, which takes a few minutes on a laptop CPU.
   ```bash
   uv run docs-qa-ingest --corpus fastapi
   ```

5. **Start the API and ask something.**
   ```bash
   uv run uvicorn docs_qa.api:app --reload
   ```
   ```bash
   curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "How do I declare an optional query parameter?"}' | jq
   ```
   Or use the interactive docs at <http://localhost:8000/docs>.

The response has the `answer`, every retrieved chunk under `sources` (numbered as in the
prompt, with `cited: true` on the ones the answer used), and token `usage`.

## Choosing the LLM

Generation goes through `LLMClient` in `llm.py`. Switch providers with env vars, no code changes:

| Provider | Cost | Settings |
| --- | --- | --- |
| Google Gemini (default) | Free tier with rate limits | `LLM_API_KEY` from [AI Studio](https://aistudio.google.com/apikey). Model `gemini-flash-latest`. |
| Groq | Free tier with rate limits | `LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_API_KEY` from [Groq console](https://console.groq.com/keys), `LLM_MODEL` from [their model list](https://console.groq.com/docs/models) |
| Ollama (local) | Free, runs on your machine | `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=llama3.2` (after `ollama pull llama3.2`), no key |
| Claude | Paid | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY` from the [Claude Console](https://platform.claude.com/) |

Any other OpenAI-compatible server works the same way through `LLM_BASE_URL`.
Model names change often, so if a request fails with "model not found", check the provider's
current model list. Free tiers have per-minute and per-day limits; hitting one returns a 429
from `/ask`. Also note that Google may use prompts sent on the Gemini free tier to improve its
products, which is fine for public docs but worth knowing.

## Tests

```bash
uv run pytest
```

The tests mock the LLM and the embedding model, so they need no API key or network.
`tests/test_db_integration.py` also runs a real round trip through pgvector when you point it at
the throwaway database that Docker Compose creates:

```bash
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/docs_qa_test uv run pytest
```

## Notes

- **Pinned docs.** `corpora.py` pins FastAPI to a release tag so results are reproducible
  (this matters once there are evals). Bump the `ref` to update, then re-run ingestion, which
  replaces that corpus's rows.
- **Changing the embedding model** usually changes the vector size. Ingestion refuses to mix
  sizes; drop the table (`docker compose exec db psql -U postgres -d docs_qa -c 'DROP TABLE chunks'`)
  and re-ingest.
- **Torch download.** The local embedding model pulls in PyTorch. On Linux without a GPU, uv
  installs the CUDA build by default, which is large; it still works on CPU.
- **Adding a corpus.** Add an entry to `CORPORA`. Pydantic's docs are Markdown and should work
  as is. SQLAlchemy's are reStructuredText and need a parser first.
