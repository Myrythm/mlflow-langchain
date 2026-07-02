# backend

Thin FastAPI wrapper around the `support_rag` library. No retrieval/generation logic
lives here — endpoints validate input, call the library, and translate errors to HTTP.

## Run

From the **repo root** (requires `.env` with `OPENAI_API_KEY`, a built knowledge base
via `uv run python build_kb.py`, and the MLflow server on :5000):

```bash
uv run uvicorn backend.main:app --port 8000
```

Interactive OpenAPI docs: http://127.0.0.1:8000/docs

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/config` | Retrieval modes, sparse models, k range, defaults, example questions |
| `GET /api/health` | API status + MLflow reachability |
| `POST /api/chat` | Streaming answer over SSE: `sources` → `token`… → `done` (or `error`) |

## Tests

```bash
uv run pytest tests/test_api.py
```

The agent is monkeypatched in tests — no OpenAI key, MLflow server, or Chroma index
needed.
