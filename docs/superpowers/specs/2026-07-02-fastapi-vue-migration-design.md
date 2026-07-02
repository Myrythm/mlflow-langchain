# FastAPI + Vue Migration Design

**Date:** 2026-07-02
**Status:** Approved

## Goal

Remove the Gradio UI entirely and replace it with a two-part app:

- `/backend` — a thin FastAPI REST API that wraps the existing `support_rag` library
  (no changes to core logic).
- `/frontend` — a separate Vue 3 + Tailwind CSS single-page app with a deliberate
  "minimal-technical" visual design (developer-tool aesthetic, not generic AI-product UI).

Feature scope is parity with the current Gradio UI: streaming chat with selectable
retrieval strategy (mode / sparse model / top-k), a live retrieved-sources panel, example
questions, and clear-conversation. Eval (`run_eval.py`, `compare_retrievers.py`) and KB
building (`build_kb.py`) stay CLI-only.

## Architecture

Approach chosen: **thin backend, library stays put**. `support_rag/` remains at the repo
root unchanged; `backend/` is a small Python package that imports it. One
`pyproject.toml` / uv environment serves both. The frontend is an independent Vite app
that talks to the API over HTTP.

```
mlflow-llm/
  support_rag/          # core library — unchanged (ui.py deleted)
  backend/              # NEW: FastAPI wrapper
    main.py             # app, lifespan (setup_tracing), CORS, endpoints
    schemas.py          # Pydantic request/response models
  frontend/             # NEW: Vue 3 + Vite + Tailwind SPA
  main.py, build_kb.py, run_eval.py, compare_retrievers.py   # unchanged
  tests/                # existing tests unchanged + new tests/test_api.py
```

Deleted: `app.py`, `support_rag/ui.py`, the `gradio` dependency.
New Python dependencies: `fastapi`, `uvicorn`, `sse-starlette`.
Updated docs: `CLAUDE.md`, root `README.md`, plus short `backend/README.md` and
`frontend/README.md`.

## Backend (FastAPI)

Run with `uv run uvicorn backend.main:app --port 8000` from the repo root.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/config` | Available retrieval modes, sparse models, k range (1–8), defaults, and `EXAMPLE_QUESTIONS` — the frontend populates its controls from this, so configuration is never duplicated. |
| `POST /api/chat` | Streaming chat over Server-Sent Events (see below). |
| `GET /api/health` | Server status plus whether the MLflow tracking server is reachable. |

### `POST /api/chat`

Request body (Pydantic):

```python
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    question: str                    # min_length=1 (after strip)
    history: list[ChatMessage] = []
    mode: Literal["dense", "sparse", "hybrid"] = config.DEFAULT_MODE
    sparse_model: str = config.DEFAULT_SPARSE   # validated against config.SPARSE_MODELS
    k: int = config.DEFAULT_K        # ge=1, le=8
```

Response: `sse-starlette.EventSourceResponse` emitting, in order:

1. `event: sources` — JSON array of `{title, category, snippet}` from the retrieved
   documents, sent **before the first token** (retrieval completes first in
   `build_stream_agent`, matching current Gradio behavior).
2. `event: token` — `{"text": "..."}` per answer fragment from the LLM.
3. `event: done` — terminator.

Browser-native `EventSource` is GET-only, so the frontend consumes the stream via
`fetch()` + `ReadableStream` with a small SSE line parser.

### Implementation notes

- The endpoint only calls the existing `build_stream_agent(HybridRetriever(...))`; no
  core logic changes.
- Agent instances are cached in a dict keyed `(mode, sparse_model, k)` (the `_get_agent`
  pattern from the old `ui.py` moves into the backend), so chains are not rebuilt per
  request.
- `setup_tracing()` runs once in the FastAPI lifespan handler, so every request is traced
  into the `customer-support-rag` MLflow experiment exactly as before.
- Conversations are **stateless**: the frontend sends `history` with each request (the
  Gradio UI already kept history client-side); truncation to
  `config.MAX_HISTORY_MESSAGES` already happens in `generation.py`. No server-side
  session store.
- CORS: `CORSMiddleware` allowing the Vite dev origins `http://localhost:5173` and
  `http://127.0.0.1:5173`.

### Error handling

- Request validation → automatic 422 from FastAPI/Pydantic (including `sparse_model` not
  in `config.SPARSE_MODELS`).
- Failures before the stream starts → `HTTPException` with a `detail` message that names
  the cause and the fix: **503** for missing prerequisites (Chroma index not built →
  "run `uv run python build_kb.py` first"; `OPENAI_API_KEY` not set), **500** for
  unexpected retrieval/setup errors.
- Failures mid-stream (HTTP status already sent) → `event: error` with a message; the
  stream then ends. The frontend shows the error inline and restores the question to the
  composer (preserving current Gradio behavior).

## Frontend (Vue 3 + Tailwind)

Stack: Vite + Vue 3 (`<script setup>`, Composition API) + Tailwind CSS v4. No component
library (styling is hand-written to carry the design direction). No Pinia — state lives
in one composable. The Vite dev server proxies `/api` → `http://localhost:8000`.

### Visual design — "minimal-technical"

- **Typography is the primary design element.** Two fonts, bundled locally via
  Fontsource (no CDN): **IBM Plex Sans** for conversation content, **IBM Plex Mono** for
  everything "machine": retrieval mode labels, k values, document category tags, stage
  status, latency readouts, panel headers. The sans/mono split *is* the hierarchy. Type
  scale is limited and decisive (12/14/16/22px), not a soft gradient of sizes.
- **Color:** warm off-white paper `#FAFAF7`, near-black ink `#1A1A17`, and a single
  international-orange accent `#E5581B` used sparingly: the Send button, the streaming
  cursor, the active stage indicator, source numbers. Warm grays for borders and
  secondary text. No gradients; no drop shadows — panels are separated by crisp 1px
  borders.
- **Layout reflects the tool's function** (a RAG exploration instrument, not a ChatGPT
  clone). Two asymmetric columns:
  - **Left (main, ~65ch reading measure):** the conversation. User turns right-aligned
    with a small mono `you` label, no chat bubbles; assistant turns left-aligned at a
    comfortable measure. Composer fixed at the bottom of the column.
  - **Right (inspector rail, ~320px):** retrieval controls (mode / sparse model / k,
    populated from `GET /api/config`) on top; below, the retrieved sources for the
    latest turn — accent-colored number, title, mono category tag, snippet. Each
    assistant message stores its own sources, so clicking an older message re-shows that
    turn's sources.
  - Thin header: app name in mono (`nimbus/support`) and a small status dot driven by
    `GET /api/health` (API + MLflow reachability).

### Interaction and loading states

- After Send, a mono status line appears in the answer slot: `retrieving…` → when the
  `sources` event arrives, the inspector fills and the status becomes `generating…` →
  tokens stream in with an orange block cursor at the text edge. Each stage shows its
  elapsed time (e.g. `retrieving · 412ms`). No generic spinner.
- Empty state: the clickable `EXAMPLE_QUESTIONS` list (from `/api/config`), not an
  illustration.
- Mid-stream `error` event → inline error message in the conversation; the question is
  restored to the composer.
- Assistant answers are rendered as Markdown (`marked` + `DOMPurify` sanitization),
  since the agent gives step-by-step answers.

### File structure

```
frontend/
  index.html, vite.config.js, package.json
  src/
    main.js, App.vue, style.css      # Tailwind v4 config lives in CSS
    api.js                           # /api/config, /api/health, POST /api/chat + SSE parser
    composables/useChat.js           # conversation state, stage lifecycle, streaming
    components/
      MessageList.vue, Message.vue, Composer.vue
      InspectorPanel.vue, RetrievalControls.vue, SourceCard.vue
      StatusLine.vue                 # retrieving/generating indicator + latency
```

## Testing

- **Backend** — `tests/test_api.py` (pytest + FastAPI `TestClient`). The agent factory is
  monkeypatched so tests run without an OpenAI key, MLflow server, or Chroma index.
  Covered: `/api/config` response shape; `/api/chat` SSE event order
  (`sources` → `token`… → `done`); validation errors → 422; pre-stream failure → 5xx
  with clear detail; mid-stream failure → `error` event.
- **Frontend** — Vitest for the failure-prone logic only: the SSE parser in `api.js` and
  the `useChat` state transitions. No component tests (YAGNI at this size).
- Existing `uv run pytest` and `uv run ruff check .` stay green.

## Run instructions (goes into READMEs)

```bash
# Terminal 1 — MLflow tracking server (unchanged)
uv run mlflow server --host 127.0.0.1 --port 5000

# Terminal 2 — backend API
uv run uvicorn backend.main:app --port 8000

# Terminal 3 — frontend dev server
cd frontend && npm install && npm run dev    # http://localhost:5173
```

## Out of scope

- API endpoints for eval or KB building (stay CLI-only).
- Server-side sessions or persistence of conversations.
- Dark mode (one polished light theme).
- Production deployment setup (Docker, static hosting, reverse proxy).
