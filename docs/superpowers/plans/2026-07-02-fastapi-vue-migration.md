# FastAPI + Vue Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Gradio UI with a FastAPI backend (`/backend`) and a separate Vue 3 + Tailwind frontend (`/frontend`), keeping the `support_rag` core library untouched.

**Architecture:** `/backend` is a thin FastAPI package at the repo root that imports the existing `support_rag` library — three endpoints (`GET /api/config`, `GET /api/health`, `POST /api/chat` streaming Server-Sent Events). `/frontend` is an independent Vite + Vue 3 + Tailwind v4 SPA ("minimal-technical" design: IBM Plex Sans/Mono, warm paper + one orange accent, conversation column + inspector rail) that consumes the API via a fetch-based SSE reader. Spec: `docs/superpowers/specs/2026-07-02-fastapi-vue-migration-design.md`.

**Tech Stack:** Python 3.12 / uv / FastAPI / uvicorn / pytest + FastAPI TestClient (httpx); Node 20+ / Vite / Vue 3 (`<script setup>`) / Tailwind CSS v4 / Vitest / marked + DOMPurify / @fontsource IBM Plex.

## Global Constraints

- Python `>=3.12`; ruff line-length **95** with rules `E,F,I,B,UP` — run `uv run ruff check .` before every commit; import order matters (rule `I`).
- Do NOT modify anything in `support_rag/` except deleting `support_rag/ui.py`. Core logic must not change.
- All Python run through `uv run ...` from the repo root. All npm commands run inside `frontend/`.
- Backend tests must pass **without** `OPENAI_API_KEY`, a running MLflow server, or a built Chroma index (mock at the `_get_agent` seam).
- SSE event order on `/api/chat` is fixed: `sources` → `token`(×N) → `done`; mid-stream failure replaces the tail with a single `error` event.
- Colors: paper `#FAFAF7`, ink `#1A1A17`, accent `#E5581B`. Fonts: IBM Plex Sans (content) / IBM Plex Mono (machine data), bundled via Fontsource — no CDN.
- Git commits: conventional-commit style, **no Co-Authored-By lines of any kind**.
- The dev environment is Windows (PowerShell); all commands below are cross-platform unless noted.

---

### Task 1: Backend dependencies + Pydantic schemas

**Files:**
- Modify: `pyproject.toml` (via `uv add`)
- Create: `backend/__init__.py`
- Create: `backend/schemas.py`
- Test: `tests/test_api.py` (schema section)

**Interfaces:**
- Consumes: `support_rag.config` (`DEFAULT_MODE`, `DEFAULT_SPARSE`, `DEFAULT_K`, `SPARSE_MODELS`).
- Produces (used by Tasks 2–3):
  - `backend.schemas.K_MIN: int = 1`, `backend.schemas.K_MAX: int = 8`
  - `ChatMessage(role: Literal["user","assistant"], content: str)`
  - `ChatRequest(question: str, history: list[ChatMessage] = [], mode: Literal["dense","sparse","hybrid"] = config.DEFAULT_MODE, sparse_model: str = config.DEFAULT_SPARSE, k: int = config.DEFAULT_K)` — rejects blank `question`, unknown `sparse_model`, `k` outside `[K_MIN, K_MAX]`; strips `question`.
  - `SourceDoc(title: str, category: str, snippet: str)`
  - `ConfigResponse(modes: list[str], sparse_models: list[str], k_min: int, k_max: int, default_mode: str, default_sparse: str, default_k: int, example_questions: list[str])`
  - `HealthResponse(status: str, mlflow_reachable: bool)`

- [ ] **Step 1: Add dependencies**

```bash
uv add fastapi uvicorn
uv add --dev httpx
```

Expected: `pyproject.toml` gains `fastapi>=...`, `uvicorn>=...` under `dependencies` and `httpx>=...` under the `dev` group; `uv.lock` updates.

- [ ] **Step 2: Create the backend package marker**

Create `backend/__init__.py`:

```python
"""FastAPI wrapper around the support_rag library (thin HTTP layer, no core logic)."""
```

- [ ] **Step 3: Write the failing schema tests**

Create `tests/test_api.py`:

```python
"""Tests for the FastAPI wrapper (backend/) — no OpenAI, MLflow, or Chroma needed."""

import pytest
from pydantic import ValidationError

from backend.schemas import ChatRequest


def test_chat_request_defaults_come_from_config():
    req = ChatRequest(question="How much is Pro?")
    assert req.mode == "hybrid"
    assert req.sparse_model == "bm25"
    assert req.k == 4
    assert req.history == []


def test_chat_request_strips_and_rejects_blank_question():
    assert ChatRequest(question="  hi  ").question == "hi"
    with pytest.raises(ValidationError):
        ChatRequest(question="   ")


def test_chat_request_rejects_bad_values():
    with pytest.raises(ValidationError):
        ChatRequest(question="q", mode="cosine")
    with pytest.raises(ValidationError):
        ChatRequest(question="q", sparse_model="splade")
    with pytest.raises(ValidationError):
        ChatRequest(question="q", k=0)
    with pytest.raises(ValidationError):
        ChatRequest(question="q", k=9)


def test_chat_request_accepts_history_roles():
    req = ChatRequest(
        question="And Business?",
        history=[
            {"role": "user", "content": "How much does Pro cost?"},
            {"role": "assistant", "content": "$12."},
        ],
    )
    assert [m.role for m in req.history] == ["user", "assistant"]
    with pytest.raises(ValidationError):
        ChatRequest(question="q", history=[{"role": "system", "content": "x"}])
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.schemas'` (collection error).

- [ ] **Step 5: Implement the schemas**

Create `backend/schemas.py`:

```python
"""Pydantic request/response schemas for the support-RAG API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from support_rag import config

K_MIN = 1
K_MAX = 8


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)
    mode: Literal["dense", "sparse", "hybrid"] = config.DEFAULT_MODE
    sparse_model: str = config.DEFAULT_SPARSE
    k: int = Field(default=config.DEFAULT_K, ge=K_MIN, le=K_MAX)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    @field_validator("sparse_model")
    @classmethod
    def _sparse_model_known(cls, value: str) -> str:
        if value not in config.SPARSE_MODELS:
            raise ValueError(
                f"unknown sparse model {value!r}; expected one of "
                f"{sorted(config.SPARSE_MODELS)}"
            )
        return value


class SourceDoc(BaseModel):
    title: str
    category: str
    snippet: str


class ConfigResponse(BaseModel):
    modes: list[str]
    sparse_models: list[str]
    k_min: int
    k_max: int
    default_mode: str
    default_sparse: str
    default_k: int
    example_questions: list[str]


class HealthResponse(BaseModel):
    status: str
    mlflow_reachable: bool
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: 4 passed.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check .
git add pyproject.toml uv.lock backend/__init__.py backend/schemas.py tests/test_api.py
git commit -m "feat: add FastAPI deps and API schemas for the backend"
```

---

### Task 2: FastAPI app skeleton — `/api/config` and `/api/health`

**Files:**
- Create: `backend/main.py`
- Test: `tests/test_api.py` (append)

**Interfaces:**
- Consumes: `backend.schemas` (Task 1), `support_rag.config`, `support_rag.tracing.setup_tracing`.
- Produces (used by Task 3 and the frontend):
  - `backend.main.app` — the FastAPI instance (uvicorn target `backend.main:app`).
  - `GET /api/config` → `ConfigResponse` JSON.
  - `GET /api/health` → `{"status": "ok", "mlflow_reachable": bool}`.
  - `backend.main._mlflow_reachable() -> bool` (monkeypatch seam for tests).
  - `setup_tracing` re-exported at module level (monkeypatch seam: `monkeypatch.setattr(main, "setup_tracing", ...)`).
  - CORS for `http://localhost:5173` and `http://127.0.0.1:5173`.

- [ ] **Step 1: Write the failing endpoint tests**

Append to `tests/test_api.py` (add imports to the top of the file, keeping ruff import order):

```python
from fastapi.testclient import TestClient

from backend import main
```

and the tests + fixture at the bottom:

```python
@pytest.fixture()
def client(monkeypatch):
    # Lifespan calls setup_tracing(), which needs a running MLflow server — stub it out.
    monkeypatch.setattr(main, "setup_tracing", lambda: None)
    with TestClient(main.app) as test_client:
        yield test_client


def test_config_endpoint_reports_options_and_defaults(client):
    body = client.get("/api/config").json()
    assert body["modes"] == ["dense", "sparse", "hybrid"]
    assert body["sparse_models"] == ["bm25"]
    assert (body["k_min"], body["k_max"]) == (1, 8)
    assert body["default_mode"] == "hybrid"
    assert body["default_sparse"] == "bm25"
    assert body["default_k"] == 4
    assert len(body["example_questions"]) >= 1


def test_health_reports_mlflow_reachability(client, monkeypatch):
    monkeypatch.setattr(main, "_mlflow_reachable", lambda: False)
    assert client.get("/api/health").json() == {
        "status": "ok",
        "mlflow_reachable": False,
    }


def test_cors_allows_vite_dev_origin(client):
    resp = client.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL — `ImportError: cannot import name 'main' from 'backend'` (collection error).

- [ ] **Step 3: Implement the app skeleton**

Create `backend/main.py`:

```python
"""FastAPI wrapper around the support_rag library.

Thin by design: endpoints validate input, call the existing library functions, and
translate results/errors to HTTP. Run from the repo root:

    uv run uvicorn backend.main:app --port 8000
"""

import logging
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from support_rag import config
from support_rag.tracing import setup_tracing

from .schemas import K_MAX, K_MIN, ConfigResponse, HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing()  # traces every request into the customer-support-rag experiment
    yield


app = FastAPI(title="Nimbus Support API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mlflow_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{config.TRACKING_URI}/health", timeout=2) as resp:
            return resp.status == 200
    except OSError:
        return False


@app.get("/api/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    return ConfigResponse(
        modes=["dense", "sparse", "hybrid"],
        sparse_models=sorted(config.SPARSE_MODELS),
        k_min=K_MIN,
        k_max=K_MAX,
        default_mode=config.DEFAULT_MODE,
        default_sparse=config.DEFAULT_SPARSE,
        default_k=config.DEFAULT_K,
        example_questions=list(config.EXAMPLE_QUESTIONS),
    )


@app.get("/api/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(status="ok", mlflow_reachable=_mlflow_reachable())
```

Note: the `lifespan` body resolves `setup_tracing` from module globals at call time, which is what makes `monkeypatch.setattr(main, "setup_tracing", ...)` work.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: 7 passed. (`test_cors_allows_vite_dev_origin` passes because CORSMiddleware answers preflight for any registered route path — `/api/chat` arrives in Task 3, but preflight does not require the route to exist.)

If `test_cors_allows_vite_dev_origin` fails with 405 because no route matches: change the preflight URL to `/api/config` — the assertion about the header is what matters, not the path.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check .
git add backend/main.py tests/test_api.py
git commit -m "feat: FastAPI app with /api/config and /api/health"
```

---

### Task 3: `POST /api/chat` — SSE streaming endpoint

**Files:**
- Modify: `backend/main.py`
- Test: `tests/test_api.py` (append)

**Interfaces:**
- Consumes: `support_rag.generation.build_stream_agent`, `support_rag.retrieval.HybridRetriever`, `backend.schemas.{ChatRequest, SourceDoc}`.
- Produces (used by the frontend, Task 6):
  - `POST /api/chat` accepting `ChatRequest` JSON; responds `text/event-stream` with events, in order: `sources` (data = JSON array of `{title, category, snippet}`), `token` (data = `{"text": str}`) repeated, `done` (data = `{}`). Mid-stream failure → single `error` event (data = `{"message": str}`) and the stream ends.
  - Pre-stream failure → HTTP 503 (detail names OPENAI_API_KEY) or 500 (generic, points at build_kb.py and logs).
  - `backend.main._get_agent(mode, sparse_model, k)` — cached agent factory (monkeypatch seam for tests).

- [ ] **Step 1: Write the failing chat tests**

Append to `tests/test_api.py`. Add `import json` to the imports at the top of the file. Then:

```python
def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse a full SSE body into [(event, parsed_json_data), ...]."""
    events = []
    for block in text.strip().split("\n\n"):
        event, data = "message", None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        if data is not None:
            events.append((event, data))
    return events


def _install_fake_agent(
    monkeypatch, tokens=("the ", "answer"), docs=None, error=None
):
    """Replace backend.main._get_agent with a stub; returns the capture dict."""
    captured = {}

    def get_agent(mode, sparse_model, k):
        captured["agent_key"] = (mode, sparse_model, k)

        def agent(question, history=None):
            captured["question"] = question
            captured["history"] = history
            if error is not None:
                raise error
            return iter(tokens), list(docs or [])

        return agent

    monkeypatch.setattr(main, "_get_agent", get_agent)
    return captured


def _doc(content="Pro costs $12 per member per month.", title="Pricing", cat="billing"):
    from langchain_core.documents import Document

    return Document(
        page_content=content,
        metadata={"chunk_id": "pricing#chunk0", "title": title, "category": cat},
    )


def test_chat_streams_sources_then_tokens_then_done(client, monkeypatch):
    _install_fake_agent(monkeypatch, tokens=("A", "B", "C"), docs=[_doc()])

    resp = client.post("/api/chat", json={"question": "How much is Pro?"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    assert [e for e, _ in events] == ["sources", "token", "token", "token", "done"]
    assert events[0][1] == [
        {
            "title": "Pricing",
            "category": "billing",
            "snippet": "Pro costs $12 per member per month.",
        }
    ]
    assert "".join(d["text"] for e, d in events if e == "token") == "ABC"


def test_chat_passes_history_and_settings_to_agent(client, monkeypatch):
    captured = _install_fake_agent(monkeypatch)
    prior = [
        {"role": "user", "content": "How much does Pro cost?"},
        {"role": "assistant", "content": "$12."},
    ]

    client.post(
        "/api/chat",
        json={
            "question": "And Business?",
            "history": prior,
            "mode": "dense",
            "sparse_model": "bm25",
            "k": 2,
        },
    )

    assert captured["agent_key"] == ("dense", "bm25", 2)
    assert captured["question"] == "And Business?"
    assert captured["history"] == prior


def test_chat_truncates_long_snippets_with_ellipsis(client, monkeypatch):
    _install_fake_agent(monkeypatch, docs=[_doc(content="x" * 300)])

    events = _parse_sse(client.post("/api/chat", json={"question": "q"}).text)

    snippet = events[0][1][0]["snippet"]
    assert snippet == "x" * 160 + "…"


def test_chat_validation_errors_are_422(client):
    assert client.post("/api/chat", json={"question": "  "}).status_code == 422
    assert (
        client.post("/api/chat", json={"question": "q", "mode": "cosine"}).status_code
        == 422
    )
    assert client.post("/api/chat", json={"question": "q", "k": 99}).status_code == 422


def test_chat_missing_api_key_maps_to_503(client, monkeypatch):
    _install_fake_agent(
        monkeypatch, error=RuntimeError("OPENAI_API_KEY environment variable not set")
    )

    resp = client.post("/api/chat", json={"question": "q"})

    assert resp.status_code == 503
    assert "OPENAI_API_KEY" in resp.json()["detail"]


def test_chat_other_pre_stream_errors_map_to_500(client, monkeypatch):
    _install_fake_agent(monkeypatch, error=RuntimeError("chroma exploded"))

    resp = client.post("/api/chat", json={"question": "q"})

    assert resp.status_code == 500
    assert "build_kb.py" in resp.json()["detail"]
    assert "chroma exploded" not in resp.json()["detail"]  # no internals leaked


def test_chat_midstream_error_emits_error_event(client, monkeypatch):
    def broken_tokens():
        yield "partial "
        raise RuntimeError("stream died")

    def get_agent(mode, sparse_model, k):
        def agent(question, history=None):
            return broken_tokens(), []

        return agent

    monkeypatch.setattr(main, "_get_agent", get_agent)

    events = _parse_sse(client.post("/api/chat", json={"question": "q"}).text)

    assert [e for e, _ in events] == ["sources", "token", "error"]
    assert "sorry" in events[-1][1]["message"].lower()
    assert "stream died" not in events[-1][1]["message"]


def test_get_agent_caches_per_configuration(monkeypatch):
    built = []
    monkeypatch.setattr(
        main, "build_stream_agent", lambda retriever: built.append(retriever) or object()
    )
    main._agents.clear()

    first = main._get_agent("hybrid", "bm25", 4)
    second = main._get_agent("hybrid", "bm25", 4)
    third = main._get_agent("dense", "bm25", 4)

    assert first is second
    assert first is not third
    assert len(built) == 2
    main._agents.clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: the new tests FAIL — `AttributeError: <module 'backend.main'> has no attribute '_get_agent'` and 404s for `/api/chat`. Tasks 1–2 tests still pass.

- [ ] **Step 3: Implement the chat endpoint**

Modify `backend/main.py`. Merge the following into the existing imports — replace the current `from fastapi import FastAPI` line and the current `from .schemas import ...` line rather than duplicating them (keep ruff `I` ordering):

```python
import json
from collections.abc import Callable, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from support_rag.generation import build_stream_agent
from support_rag.retrieval import HybridRetriever

from .schemas import K_MAX, K_MIN, ChatRequest, ConfigResponse, HealthResponse, SourceDoc
```

Add after `logger = logging.getLogger(__name__)`:

```python
_agents: dict[tuple[str, str, int], Callable] = {}

_ERROR_REPLY = (
    "Sorry — something went wrong while generating an answer. Please try again in a "
    "moment; your question is still in the input box."
)
_SNIPPET_CHARS = 160
```

Add below `_mlflow_reachable`:

```python
def _get_agent(mode: str, sparse_model: str, k: int) -> Callable:
    """Build (once) and cache the streaming agent for a retrieval configuration."""
    key = (mode, sparse_model, k)
    if key not in _agents:
        _agents[key] = build_stream_agent(
            HybridRetriever(mode=mode, sparse_model=sparse_model, k=k)
        )
    return _agents[key]


def _snippet(text: str) -> str:
    head = text[:_SNIPPET_CHARS].strip().replace("\n", " ")
    return head + ("…" if len(text) > _SNIPPET_CHARS else "")


def _pre_stream_error(exc: Exception) -> HTTPException:
    if "OPENAI_API_KEY" in str(exc) or type(exc).__name__ == "AuthenticationError":
        return HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is missing or invalid — set it in .env (copy from "
            ".env.example) and restart the backend.",
        )
    return HTTPException(
        status_code=500,
        detail="Retrieval failed before generation started — check that the knowledge "
        "base is built (`uv run python build_kb.py`) and see the backend logs.",
    )


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"
```

Add the endpoint after `get_health`:

```python
@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    agent = _get_agent(req.mode, req.sparse_model, req.k)
    try:
        tokens, docs = agent(req.question, history=[m.model_dump() for m in req.history])
    except Exception as exc:
        logger.exception("Support agent failed to answer %r", req.question)
        raise _pre_stream_error(exc) from exc

    sources = [
        SourceDoc(
            title=d.metadata.get("title", ""),
            category=d.metadata.get("category", ""),
            snippet=_snippet(d.page_content),
        ).model_dump()
        for d in docs
    ]

    def event_stream() -> Iterator[str]:
        yield _sse("sources", json.dumps(sources))
        try:
            for token in tokens:
                yield _sse("token", json.dumps({"text": token}))
        except Exception:
            logger.exception("Answer stream failed for %r", req.question)
            yield _sse("error", json.dumps({"message": _ERROR_REPLY}))
            return
        yield _sse("done", "{}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

Notes for the implementer:
- The route is deliberately **sync** (`def`, not `async def`): FastAPI runs it in a threadpool, so the blocking retrieval call (`agent(...)` embeds the query and hits Chroma) never blocks the event loop. Starlette iterates the sync generator in a threadpool too.
- Retrieval happens eagerly inside `agent(...)` (see `build_stream_agent` in `support_rag/generation.py`), which is why retrieval errors can still become proper HTTP status codes — the stream has not started yet.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: 15 passed.

- [ ] **Step 5: Run the full suite, lint, commit**

Run: `uv run pytest` — expected: all pass (the old `tests/test_ui.py` still exists and still passes; it is removed in Task 4).
Run: `uv run ruff check .` — expected: no errors.

```bash
git add backend/main.py tests/test_api.py
git commit -m "feat: POST /api/chat streaming SSE endpoint"
```

---

### Task 4: Remove Gradio; update docs (CLAUDE.md, README.md, backend/README.md)

**Files:**
- Delete: `app.py`, `support_rag/ui.py`, `tests/test_ui.py`
- Modify: `pyproject.toml` (via `uv remove gradio`), `CLAUDE.md`, `README.md`
- Create: `backend/README.md`

**Interfaces:**
- Consumes: nothing from other tasks (Tasks 1–3 must be committed first so the replacement UI path exists).
- Produces: a repo with no Gradio anywhere (`grep -ri gradio` finds only historical docs under `docs/superpowers/`).

- [ ] **Step 1: Delete the Gradio code and dependency**

```bash
git rm app.py support_rag/ui.py tests/test_ui.py
uv remove gradio
```

- [ ] **Step 2: Verify nothing still imports gradio**

Run: `uv run pytest`
Expected: all tests pass (test count drops by 5 — `tests/test_ui.py` is gone).

Run: `git grep -li gradio -- "*.py"`
Expected: no output (exit code 1 — no tracked Python file mentions gradio).

- [ ] **Step 3: Update CLAUDE.md**

In `CLAUDE.md`, replace the block:

```markdown
# Gradio chat UI with a retrieval-strategy selector (dense/sparse/hybrid + top-k)
uv run python app.py   # http://127.0.0.1:7860
```

with:

```markdown
# Backend REST API (FastAPI) — http://127.0.0.1:8000, OpenAPI docs at /docs
uv run uvicorn backend.main:app --port 8000

# Frontend dev server (Vue 3 + Vite) — http://localhost:5173, proxies /api to :8000
cd frontend && npm install && npm run dev

# Frontend unit tests (SSE parser + chat composable)
cd frontend && npm test
```

Replace the sentence:

```markdown
running before `main.py`, `app.py`, `run_eval.py`, or `compare_retrievers.py` will work —
```

with:

```markdown
running before `main.py`, the backend (`backend.main:app`), `run_eval.py`, or
`compare_retrievers.py` will work —
```

Replace the architecture bullet:

```markdown
- `ui.py` — `build_demo()`, the Gradio chat interface used by `app.py`.
```

with:

```markdown
- The web UI lives outside the package: `backend/` is a thin FastAPI wrapper
  (`/api/config`, `/api/health`, and `POST /api/chat` streaming SSE:
  `sources` → `token`… → `done`), and `frontend/` is a separate Vue 3 + Vite +
  Tailwind SPA that consumes it. Neither contains retrieval/generation logic.
```

Replace the sentence fragment:

```markdown
`app.py`, or eval scripts.
```

with:

```markdown
the backend, or eval scripts.
```

- [ ] **Step 4: Update README.md**

Replace the section:

```markdown
## Chat UI (Gradio)

A chat interface with a retrieval-strategy selector (dense / sparse / hybrid + top-k) and a
live panel of the retrieved knowledge-base articles:

```bash
uv run python app.py
```

Opens at http://127.0.0.1:7860. Each turn is traced into the `customer-support-rag`
experiment.
```

with:

```markdown
## Web UI (FastAPI + Vue)

A streaming chat interface with a retrieval-strategy selector (dense / sparse / hybrid +
top-k) and a live panel of the retrieved knowledge-base passages. Two processes:

```bash
# Backend (FastAPI) — http://127.0.0.1:8000, OpenAPI docs at /docs
uv run uvicorn backend.main:app --port 8000

# Frontend (Vue 3 + Vite) — http://localhost:5173
cd frontend
npm install
npm run dev
```

Each turn is traced into the `customer-support-rag` experiment. Answers stream over
Server-Sent Events (`sources` → `token`… → `done`).
```

In the "Project structure" tree, remove the line:

```
  ui.py               # Gradio chat interface (build_demo())
```

and replace the line:

```
app.py                 # CLI: launch the Gradio chat UI
```

with:

```
backend/               # FastAPI wrapper: /api/config, /api/health, /api/chat (SSE)
frontend/              # Vue 3 + Vite + Tailwind chat UI (see frontend/README.md)
```

- [ ] **Step 5: Create backend/README.md**

```markdown
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
```

- [ ] **Step 6: Verify and commit**

Run: `uv run pytest` — expected: all pass.
Run: `uv run ruff check .` — expected: no errors.

```bash
git add -A
git commit -m "feat!: remove Gradio UI in favor of the FastAPI backend"
```

---

### Task 5: Frontend scaffold — Vite + Vue 3 + Tailwind v4 builds

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.js`, `frontend/index.html`, `frontend/.gitignore`
- Create: `frontend/src/main.js`, `frontend/src/style.css`, `frontend/src/App.vue` (placeholder)
- Modify: `.gitignore` (root)

**Interfaces:**
- Consumes: nothing (independent of backend tasks).
- Produces (used by Tasks 6–8): a building Vite app; design tokens as Tailwind theme values `bg-paper`, `text-ink`, `text-accent`/`bg-accent`, `border-line`, `text-muted`, `font-sans` (IBM Plex Sans), `font-mono` (IBM Plex Mono); dev proxy `/api` → `http://127.0.0.1:8000`; `npm test` wired to Vitest.

- [ ] **Step 1: Create the app skeleton files**

`frontend/package.json`:

```json
{
  "name": "nimbus-support-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "@fontsource/ibm-plex-mono": "^5.2.5",
    "@fontsource/ibm-plex-sans": "^5.2.5",
    "dompurify": "^3.2.4",
    "marked": "^15.0.7",
    "vue": "^3.5.13"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.1.3",
    "@vitejs/plugin-vue": "^5.2.1",
    "tailwindcss": "^4.1.3",
    "vite": "^6.2.0",
    "vitest": "^3.1.1"
  }
}
```

`frontend/vite.config.js`:

```js
import tailwindcss from "@tailwindcss/vite";
import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  test: {
    environment: "node",
  },
});
```

`frontend/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>nimbus/support</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>
```

`frontend/.gitignore`:

```
node_modules/
dist/
```

`frontend/src/main.js`:

```js
import { createApp } from "vue";

import App from "./App.vue";
import "./style.css";

createApp(App).mount("#app");
```

`frontend/src/style.css` (Tailwind v4 — theme lives in CSS, no tailwind.config.js):

```css
@import "tailwindcss";
@import "@fontsource/ibm-plex-sans/400.css";
@import "@fontsource/ibm-plex-sans/500.css";
@import "@fontsource/ibm-plex-sans/600.css";
@import "@fontsource/ibm-plex-mono/400.css";
@import "@fontsource/ibm-plex-mono/500.css";

@theme {
  --color-paper: #fafaf7;
  --color-ink: #1a1a17;
  --color-accent: #e5581b;
  --color-line: #e3e1da;
  --color-muted: #6e6b62;
  --font-sans: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", monospace;
}

html,
body,
#app {
  height: 100%;
}

body {
  @apply bg-paper font-sans text-ink antialiased;
}

/* Markdown answers (rendered via marked + DOMPurify into .answer) */
.answer :is(p, ul, ol) {
  @apply my-2 leading-relaxed;
}
.answer ol {
  @apply list-decimal pl-5;
}
.answer ul {
  @apply list-disc pl-5;
}
.answer code {
  @apply rounded-sm bg-ink/5 px-1 py-0.5 font-mono text-[13px];
}
.answer a {
  @apply text-accent underline;
}

/* Orange block cursor at the streaming text edge */
.answer.streaming > :last-child::after {
  content: "▍";
  @apply text-accent;
}
```

`frontend/src/App.vue` (placeholder, replaced in Task 8):

```vue
<template>
  <div class="p-6 font-mono text-sm">
    nimbus<span class="text-accent">/</span>support — scaffold OK
  </div>
</template>
```

- [ ] **Step 2: Install and build**

```bash
cd frontend
npm install
npm run build
```

Expected: `npm install` resolves without errors; `vite build` outputs `dist/` with no warnings about unknown classes. (If `@apply` on custom theme colors errors, ensure the `@theme` block appears before the `@apply` usages — it does in the file above.)

- [ ] **Step 3: Add frontend artifacts to the root .gitignore**

Append to the root `.gitignore`:

```
# Frontend (Node)
node_modules/
frontend/dist/
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore frontend
git commit -m "feat: scaffold Vue 3 + Vite + Tailwind v4 frontend with design tokens"
```

---

### Task 6: `api.js` — fetch client + SSE parser (Vitest)

**Files:**
- Create: `frontend/src/api.js`
- Test: `frontend/src/api.test.js`

**Interfaces:**
- Consumes: backend endpoints (Task 2–3 shapes) at runtime; nothing at test time (pure functions + mocked fetch).
- Produces (used by Task 7–8):
  - `fetchConfig() -> Promise<object>` — GET `/api/config`, throws on non-2xx.
  - `fetchHealth() -> Promise<object>` — GET `/api/health`, throws on non-2xx.
  - `createSSEParser(onEvent: (event: string, data: string) => void) -> (chunk: string) => void` — incremental SSE parser, safe across chunk boundaries.
  - `streamChat(payload, {onSources, onToken, onError, onDone}) -> Promise<void>` — POST `/api/chat`; throws `Error(detail)` on non-2xx before streaming.

- [ ] **Step 1: Write the failing parser tests**

`frontend/src/api.test.js`:

```js
import { describe, expect, it, vi } from "vitest";

import { createSSEParser } from "./api";

function collect() {
  const events = [];
  const feed = createSSEParser((event, data) => events.push([event, data]));
  return { events, feed };
}

describe("createSSEParser", () => {
  it("parses a complete event", () => {
    const { events, feed } = collect();
    feed('event: token\ndata: {"text":"A"}\n\n');
    expect(events).toEqual([["token", '{"text":"A"}']]);
  });

  it("handles events split across chunk boundaries", () => {
    const { events, feed } = collect();
    feed("event: tok");
    feed('en\ndata: {"te');
    feed('xt":"AB"}\n\n');
    expect(events).toEqual([["token", '{"text":"AB"}']]);
  });

  it("handles multiple events in one chunk, in order", () => {
    const { events, feed } = collect();
    feed(
      'event: sources\ndata: []\n\nevent: token\ndata: {"text":"A"}\n\nevent: done\ndata: {}\n\n',
    );
    expect(events.map(([e]) => e)).toEqual(["sources", "token", "done"]);
  });

  it("ignores comments and blocks without data", () => {
    const { events, feed } = collect();
    feed(": ping\n\nevent: ghost\n\ndata: {}\n\n");
    expect(events).toEqual([["message", "{}"]]);
  });
});

describe("streamChat", () => {
  it("routes events to the right callbacks", async () => {
    const body = [
      'event: sources\ndata: [{"title":"Pricing","category":"billing","snippet":"s"}]\n\n',
      'event: token\ndata: {"text":"A"}\n\nevent: token\ndata: {"text":"B"}\n\n',
      "event: done\ndata: {}\n\n",
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        body: new ReadableStream({
          start(controller) {
            const enc = new TextEncoder();
            body.forEach((chunk) => controller.enqueue(enc.encode(chunk)));
            controller.close();
          },
        }),
      }),
    );
    const { streamChat } = await import("./api");
    const got = { sources: null, text: "", done: false };

    await streamChat(
      { question: "q" },
      {
        onSources: (s) => (got.sources = s),
        onToken: (t) => (got.text += t),
        onError: () => {},
        onDone: () => (got.done = true),
      },
    );

    expect(got.sources).toEqual([
      { title: "Pricing", category: "billing", snippet: "s" },
    ]);
    expect(got.text).toBe("AB");
    expect(got.done).toBe(true);
    vi.unstubAllGlobals();
  });

  it("throws the backend detail message on a non-2xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        json: async () => ({ detail: "OPENAI_API_KEY is missing" }),
      }),
    );
    const { streamChat } = await import("./api");

    await expect(
      streamChat({ question: "q" }, { onSources() {}, onToken() {}, onError() {}, onDone() {} }),
    ).rejects.toThrow("OPENAI_API_KEY is missing");
    vi.unstubAllGlobals();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run (in `frontend/`): `npm test`
Expected: FAIL — cannot resolve `./api`.

- [ ] **Step 3: Implement api.js**

`frontend/src/api.js`:

```js
// HTTP client for the FastAPI backend. The Vite dev server proxies /api to :8000.

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export function fetchConfig() {
  return getJSON("/api/config");
}

export function fetchHealth() {
  return getJSON("/api/health");
}

// Incremental text/event-stream parser. Returns feed(chunk); calls
// onEvent(eventName, rawData) once per complete SSE message. Buffers partial
// messages across chunks, ignores comment lines (":") and messages without data.
export function createSSEParser(onEvent) {
  let buffer = "";
  return function feed(chunk) {
    buffer += chunk;
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message";
      const dataLines = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (dataLines.length) onEvent(event, dataLines.join("\n"));
    }
  };
}

export async function streamChat(payload, { onSources, onToken, onError, onDone }) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = `chat request failed: ${res.status}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      // keep the generic message when the error body is not JSON
    }
    throw new Error(detail);
  }

  const feed = createSSEParser((event, data) => {
    if (event === "sources") onSources(JSON.parse(data));
    else if (event === "token") onToken(JSON.parse(data).text);
    else if (event === "error") onError(JSON.parse(data).message);
    else if (event === "done") onDone();
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    feed(decoder.decode(value, { stream: true }));
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run (in `frontend/`): `npm test`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.js frontend/src/api.test.js
git commit -m "feat: frontend API client with incremental SSE parser"
```

---

### Task 7: `useChat` composable (Vitest)

**Files:**
- Create: `frontend/src/composables/useChat.js`
- Test: `frontend/src/composables/useChat.test.js`

**Interfaces:**
- Consumes: `streamChat` from `../api` (Task 6 signature).
- Produces (used by Task 8):
  - `useChat() -> { messages, stage, draft, selectedIndex, activeSources, send, clear }` where
    - `messages: Ref<Array<{role, content, sources?, timings?, error?}>>`
    - `stage: Ref<"idle"|"retrieving"|"generating">`
    - `draft: Ref<string>` — the composer text (restored on failure for easy retry)
    - `selectedIndex: Ref<number|null>` — which assistant message the inspector shows
    - `activeSources: ComputedRef<Array>` — sources of the selected (or latest) assistant message
    - `send(settings: {mode, sparse_model, k}) -> Promise<void>`
    - `clear() -> void`

- [ ] **Step 1: Write the failing composable tests**

`frontend/src/composables/useChat.test.js`:

```js
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api", () => ({ streamChat: vi.fn() }));

import { streamChat } from "../api";
import { useChat } from "./useChat";

const SETTINGS = { mode: "hybrid", sparse_model: "bm25", k: 4 };
const SOURCES = [{ title: "Pricing", category: "billing", snippet: "s" }];

beforeEach(() => {
  streamChat.mockReset();
});

function happyStream() {
  streamChat.mockImplementation(async (payload, cb) => {
    cb.onSources(SOURCES);
    cb.onToken("A");
    cb.onToken("B");
    cb.onDone();
  });
}

describe("useChat.send", () => {
  it("appends the turn, stores sources and streamed content", async () => {
    happyStream();
    const chat = useChat();
    chat.draft.value = "How much is Pro?";

    await chat.send(SETTINGS);

    expect(chat.messages.value.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(chat.messages.value[1].content).toBe("AB");
    expect(chat.messages.value[1].sources).toEqual(SOURCES);
    expect(chat.stage.value).toBe("idle");
    expect(chat.draft.value).toBe("");
  });

  it("sends prior turns as history and the retrieval settings", async () => {
    happyStream();
    const chat = useChat();
    chat.draft.value = "first";
    await chat.send(SETTINGS);
    chat.draft.value = "second";

    await chat.send({ mode: "dense", sparse_model: "bm25", k: 2 });

    const payload = streamChat.mock.calls[1][0];
    expect(payload.history).toEqual([
      { role: "user", content: "first" },
      { role: "assistant", content: "AB" },
    ]);
    expect(payload).toMatchObject({ question: "second", mode: "dense", k: 2 });
  });

  it("ignores blank drafts and does not call the API", async () => {
    const chat = useChat();
    chat.draft.value = "   ";
    await chat.send(SETTINGS);
    expect(streamChat).not.toHaveBeenCalled();
    expect(chat.messages.value).toEqual([]);
  });

  it("replaces a partial answer with the error and restores the draft", async () => {
    streamChat.mockImplementation(async (payload, cb) => {
      cb.onSources(SOURCES);
      cb.onToken("partial ");
      cb.onError("stream broke politely");
    });
    const chat = useChat();
    chat.draft.value = "hello";

    await chat.send(SETTINGS);

    const reply = chat.messages.value[1];
    expect(reply.error).toBe(true);
    expect(reply.content).toBe("stream broke politely");
    expect(reply.content).not.toContain("partial");
    expect(chat.draft.value).toBe("hello");
    expect(chat.stage.value).toBe("idle");
  });

  it("turns a rejected request (pre-stream HTTP error) into an error turn", async () => {
    streamChat.mockRejectedValue(new Error("OPENAI_API_KEY is missing"));
    const chat = useChat();
    chat.draft.value = "hello";

    await chat.send(SETTINGS);

    expect(chat.messages.value[1].error).toBe(true);
    expect(chat.messages.value[1].content).toContain("OPENAI_API_KEY");
    expect(chat.draft.value).toBe("hello");
  });

  it("excludes error turns from subsequent history", async () => {
    streamChat.mockRejectedValueOnce(new Error("boom"));
    const chat = useChat();
    chat.draft.value = "hello";
    await chat.send(SETTINGS);

    happyStream();
    chat.draft.value = "hello again";
    await chat.send(SETTINGS);

    const payload = streamChat.mock.calls[1][0];
    expect(payload.history).toEqual([{ role: "user", content: "hello" }]);
  });
});

describe("stage + sources selection", () => {
  it("walks idle -> retrieving -> generating -> idle", async () => {
    const seen = [];
    const chat = useChat();
    streamChat.mockImplementation(async (payload, cb) => {
      seen.push(chat.stage.value); // retrieving
      cb.onSources(SOURCES);
      seen.push(chat.stage.value); // generating
      cb.onDone();
    });
    chat.draft.value = "q";

    await chat.send(SETTINGS);

    expect(seen).toEqual(["retrieving", "generating"]);
    expect(chat.stage.value).toBe("idle");
  });

  it("activeSources follows the latest turn unless one is selected", async () => {
    happyStream();
    const chat = useChat();
    chat.draft.value = "q1";
    await chat.send(SETTINGS);
    expect(chat.activeSources.value).toEqual(SOURCES);

    chat.selectedIndex.value = 0; // a user message -> no sources
    expect(chat.activeSources.value).toEqual([]);
  });

  it("clear resets everything", async () => {
    happyStream();
    const chat = useChat();
    chat.draft.value = "q";
    await chat.send(SETTINGS);

    chat.clear();

    expect(chat.messages.value).toEqual([]);
    expect(chat.stage.value).toBe("idle");
    expect(chat.selectedIndex.value).toBe(null);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run (in `frontend/`): `npm test`
Expected: FAIL — cannot resolve `./useChat`. The Task 6 tests still pass.

- [ ] **Step 3: Implement the composable**

`frontend/src/composables/useChat.js`:

```js
import { computed, ref } from "vue";

import { streamChat } from "../api";

const ERROR_REPLY =
  "Sorry — something went wrong while generating an answer. Please try again in a " +
  "moment; your question is still in the input box.";

// Conversation state for one chat session. Stateless server: prior turns are
// replayed as `history` on every request (error turns excluded).
export function useChat() {
  const messages = ref([]);
  const stage = ref("idle"); // idle | retrieving | generating
  const draft = ref("");
  const selectedIndex = ref(null);

  const activeSources = computed(() => {
    const pick =
      selectedIndex.value !== null
        ? messages.value[selectedIndex.value]
        : [...messages.value].reverse().find((m) => m.role === "assistant");
    return pick?.sources ?? [];
  });

  function clear() {
    messages.value = [];
    stage.value = "idle";
    selectedIndex.value = null;
  }

  async function send(settings) {
    const question = draft.value.trim();
    if (!question || stage.value !== "idle") return;

    const history = messages.value
      .filter((m) => !m.error)
      .map(({ role, content }) => ({ role, content }));

    draft.value = "";
    selectedIndex.value = null;
    messages.value.push({ role: "user", content: question });
    messages.value.push({ role: "assistant", content: "", sources: [], timings: {} });
    const reply = messages.value[messages.value.length - 1];

    stage.value = "retrieving";
    const t0 = performance.now();

    const fail = (message) => {
      reply.content = message || ERROR_REPLY;
      reply.error = true;
      draft.value = question; // easy retry, like the old UI
      stage.value = "idle";
    };

    try {
      await streamChat(
        { question, history, ...settings },
        {
          onSources(sources) {
            reply.sources = sources;
            reply.timings.retrieval = Math.round(performance.now() - t0);
            stage.value = "generating";
          },
          onToken(text) {
            reply.content += text;
          },
          onError(message) {
            fail(message);
          },
          onDone() {
            reply.timings.generation =
              Math.round(performance.now() - t0) - (reply.timings.retrieval ?? 0);
            stage.value = "idle";
          },
        },
      );
    } catch (err) {
      fail(err.message);
    }
    if (stage.value !== "idle") stage.value = "idle"; // stream ended without done/error
  }

  return { messages, stage, draft, selectedIndex, activeSources, send, clear };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run (in `frontend/`): `npm test`
Expected: 15 passed (6 from Task 6 + 9 new).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/composables
git commit -m "feat: useChat composable — staged streaming state and history replay"
```

---

### Task 8: UI components and App layout

**Files:**
- Create: `frontend/src/components/Message.vue`, `StatusLine.vue`, `MessageList.vue`, `Composer.vue`, `RetrievalControls.vue`, `SourceCard.vue`, `InspectorPanel.vue`
- Modify: `frontend/src/App.vue` (replace the placeholder)

**Interfaces:**
- Consumes: `useChat` (Task 7), `fetchConfig`/`fetchHealth` (Task 6), design tokens (Task 5), `GET /api/config` response shape (Task 2).
- Produces: the complete UI. No unit tests (presentation only); verified visually in Step 3.

- [ ] **Step 1: Create the components**

`frontend/src/components/Message.vue`:

```vue
<script setup>
import DOMPurify from "dompurify";
import { marked } from "marked";
import { computed } from "vue";

const props = defineProps({
  message: { type: Object, required: true },
  selected: { type: Boolean, default: false },
  streaming: { type: Boolean, default: false },
});
defineEmits(["select"]);

const html = computed(() =>
  DOMPurify.sanitize(marked.parse(props.message.content ?? "")),
);
</script>

<template>
  <div v-if="message.role === 'user'" class="mt-10 flex flex-col items-end">
    <span class="font-mono text-[11px] tracking-widest text-muted uppercase">you</span>
    <p class="mt-1 max-w-[52ch] text-right text-[15px] font-medium">
      {{ message.content }}
    </p>
  </div>

  <div v-else class="mt-4">
    <button
      class="font-mono text-[11px] tracking-widest uppercase"
      :class="selected ? 'text-accent' : 'text-muted hover:text-accent'"
      @click="$emit('select')"
    >
      agent<span v-if="message.sources?.length"> · {{ message.sources.length }} src</span>
    </button>
    <p
      v-if="message.error"
      class="mt-1 border-l-2 border-accent pl-3 text-[15px] text-muted"
    >
      {{ message.content }}
    </p>
    <div v-else class="answer mt-1 text-[15px]" :class="{ streaming }" v-html="html"></div>
  </div>
</template>
```

`frontend/src/components/StatusLine.vue`:

```vue
<script setup>
defineProps({
  stage: { type: String, required: true },
  timings: { type: Object, default: () => ({}) },
});
</script>

<template>
  <p class="mt-2 flex gap-4 font-mono text-[11px] text-muted">
    <span v-if="stage === 'retrieving'" class="text-accent">retrieving…</span>
    <span v-else-if="timings.retrieval != null">retrieving · {{ timings.retrieval }}ms</span>
    <span v-if="stage === 'generating'" class="text-accent">generating…</span>
    <span v-else-if="timings.generation != null">generating · {{ timings.generation }}ms</span>
  </p>
</template>
```

`frontend/src/components/MessageList.vue`:

```vue
<script setup>
import { nextTick, ref, watch } from "vue";

import Message from "./Message.vue";
import StatusLine from "./StatusLine.vue";

const props = defineProps({
  messages: { type: Array, required: true },
  stage: { type: String, required: true },
  selectedIndex: { type: Number, default: null },
  examples: { type: Array, default: () => [] },
});
defineEmits(["select", "example"]);

const scroller = ref(null);
watch(
  () => [props.messages.length, props.messages.at(-1)?.content],
  async () => {
    await nextTick();
    scroller.value?.scrollTo({ top: scroller.value.scrollHeight });
  },
);
</script>

<template>
  <div ref="scroller" class="overflow-y-auto">
    <div class="mx-auto w-full max-w-[65ch] px-6 pt-4 pb-10">
      <div v-if="messages.length === 0" class="pt-16">
        <p class="font-mono text-[11px] tracking-widest text-muted uppercase">
          try one of these
        </p>
        <ul class="mt-4 divide-y divide-line border-y border-line">
          <li v-for="q in examples" :key="q">
            <button
              class="w-full py-3 text-left text-[15px] hover:text-accent"
              @click="$emit('example', q)"
            >
              {{ q }}
            </button>
          </li>
        </ul>
      </div>

      <template v-for="(msg, i) in messages" :key="i">
        <Message
          :message="msg"
          :selected="i === selectedIndex"
          :streaming="i === messages.length - 1 && stage === 'generating'"
          @select="$emit('select', i)"
        />
        <StatusLine
          v-if="i === messages.length - 1 && msg.role === 'assistant' && !msg.error"
          :stage="stage"
          :timings="msg.timings ?? {}"
        />
      </template>
    </div>
  </div>
</template>
```

`frontend/src/components/Composer.vue`:

```vue
<script setup>
const props = defineProps({
  modelValue: { type: String, required: true },
  busy: { type: Boolean, default: false },
});
const emit = defineEmits(["update:modelValue", "send", "clear"]);

function onKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!props.busy) emit("send");
  }
}
</script>

<template>
  <div class="shrink-0 border-t border-line">
    <div class="mx-auto w-full max-w-[65ch] px-6 py-4">
      <div class="flex items-end gap-3">
        <textarea
          :value="modelValue"
          rows="2"
          placeholder="e.g. How much does the Pro plan cost?"
          class="min-h-[3.25rem] flex-1 resize-none border border-line bg-transparent px-3
            py-2 text-[15px] placeholder:text-muted focus:border-ink focus:outline-none"
          @input="$emit('update:modelValue', $event.target.value)"
          @keydown="onKeydown"
        ></textarea>
        <button
          class="h-[3.25rem] bg-accent px-5 font-mono text-sm font-medium text-paper
            disabled:opacity-40"
          :disabled="busy || !modelValue.trim()"
          @click="$emit('send')"
        >
          send
        </button>
      </div>
      <button
        class="mt-2 font-mono text-[11px] tracking-widest text-muted uppercase
          hover:text-accent"
        @click="$emit('clear')"
      >
        clear conversation
      </button>
    </div>
  </div>
</template>
```

`frontend/src/components/RetrievalControls.vue`:

```vue
<script setup>
const props = defineProps({
  config: { type: Object, required: true },
  settings: { type: Object, required: true },
});
const emit = defineEmits(["update:settings"]);

function set(key, value) {
  emit("update:settings", { ...props.settings, [key]: value });
}
</script>

<template>
  <div class="space-y-3 font-mono text-xs">
    <label class="block">
      <span class="text-muted">mode</span>
      <select
        :value="settings.mode"
        class="mt-1 w-full border border-line bg-transparent px-2 py-1.5
          focus:border-ink focus:outline-none"
        @change="set('mode', $event.target.value)"
      >
        <option v-for="m in config.modes" :key="m" :value="m">{{ m }}</option>
      </select>
    </label>

    <label class="block">
      <span class="text-muted">sparse model</span>
      <select
        :value="settings.sparse_model"
        class="mt-1 w-full border border-line bg-transparent px-2 py-1.5
          focus:border-ink focus:outline-none"
        @change="set('sparse_model', $event.target.value)"
      >
        <option v-for="m in config.sparse_models" :key="m" :value="m">{{ m }}</option>
      </select>
    </label>

    <label class="block">
      <span class="text-muted">top-k · {{ settings.k }}</span>
      <input
        type="range"
        :min="config.k_min"
        :max="config.k_max"
        :value="settings.k"
        class="mt-1 w-full accent-accent"
        @input="set('k', Number($event.target.value))"
      />
    </label>
  </div>
</template>
```

`frontend/src/components/SourceCard.vue`:

```vue
<script setup>
defineProps({
  index: { type: Number, required: true },
  source: { type: Object, required: true },
});
</script>

<template>
  <li class="border-t border-line pt-3 first:border-t-0 first:pt-0">
    <div class="flex items-baseline gap-2">
      <span class="font-mono text-xs font-medium text-accent">
        {{ String(index).padStart(2, "0") }}
      </span>
      <span class="text-sm font-medium">{{ source.title }}</span>
    </div>
    <span class="mt-0.5 inline-block font-mono text-[11px] text-muted">
      {{ source.category }}
    </span>
    <p class="mt-1 text-[13px] leading-relaxed text-muted">{{ source.snippet }}</p>
  </li>
</template>
```

`frontend/src/components/InspectorPanel.vue`:

```vue
<script setup>
import RetrievalControls from "./RetrievalControls.vue";
import SourceCard from "./SourceCard.vue";

defineProps({
  config: { type: Object, default: null },
  settings: { type: Object, required: true },
  sources: { type: Array, default: () => [] },
});
defineEmits(["update:settings"]);
</script>

<template>
  <aside class="flex min-h-0 flex-col overflow-y-auto border-l border-line">
    <section class="border-b border-line p-4">
      <h2 class="font-mono text-[11px] tracking-widest text-muted uppercase">retrieval</h2>
      <RetrievalControls
        v-if="config"
        class="mt-3"
        :config="config"
        :settings="settings"
        @update:settings="$emit('update:settings', $event)"
      />
      <p v-else class="mt-3 font-mono text-xs text-muted">backend unreachable</p>
    </section>

    <section class="p-4">
      <h2 class="font-mono text-[11px] tracking-widest text-muted uppercase">
        sources<span v-if="sources.length"> · {{ sources.length }}</span>
      </h2>
      <p v-if="!sources.length" class="mt-3 text-sm text-muted">
        Ask a question to see which knowledge-base passages the answer is grounded in.
      </p>
      <ol v-else class="mt-3">
        <SourceCard v-for="(src, i) in sources" :key="i" :index="i + 1" :source="src" />
      </ol>
    </section>
  </aside>
</template>
```

- [ ] **Step 2: Replace App.vue**

`frontend/src/App.vue`:

```vue
<script setup>
import { onMounted, ref } from "vue";

import { fetchConfig, fetchHealth } from "./api";
import Composer from "./components/Composer.vue";
import InspectorPanel from "./components/InspectorPanel.vue";
import MessageList from "./components/MessageList.vue";
import { useChat } from "./composables/useChat";

const { messages, stage, draft, selectedIndex, activeSources, send, clear } = useChat();

const config = ref(null);
const health = ref(null);
const settings = ref({ mode: "hybrid", sparse_model: "bm25", k: 4 });

function ask(question) {
  draft.value = question;
  send(settings.value);
}

onMounted(async () => {
  try {
    config.value = await fetchConfig();
    settings.value = {
      mode: config.value.default_mode,
      sparse_model: config.value.default_sparse,
      k: config.value.default_k,
    };
  } catch {
    config.value = null; // inspector shows "backend unreachable"
  }
  try {
    health.value = await fetchHealth();
  } catch {
    health.value = null;
  }
});
</script>

<template>
  <div class="flex h-full flex-col">
    <header
      class="flex h-11 shrink-0 items-center justify-between border-b border-line px-4"
    >
      <span class="font-mono text-sm font-medium">
        nimbus<span class="text-accent">/</span>support
      </span>
      <span class="flex items-center gap-2 font-mono text-[11px] text-muted">
        <span
          class="inline-block h-2 w-2 rounded-full"
          :class="health?.status === 'ok' ? 'bg-accent' : 'bg-line'"
        ></span>
        {{
          health === null
            ? "backend offline"
            : health.mlflow_reachable
              ? "api + mlflow"
              : "api up · mlflow down"
        }}
      </span>
    </header>

    <div class="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_320px]">
      <main class="flex min-h-0 flex-col">
        <MessageList
          class="min-h-0 flex-1"
          :messages="messages"
          :stage="stage"
          :selected-index="selectedIndex"
          :examples="config?.example_questions ?? []"
          @select="selectedIndex = $event"
          @example="ask"
        />
        <Composer
          v-model="draft"
          :busy="stage !== 'idle'"
          @send="send(settings)"
          @clear="clear"
        />
      </main>
      <InspectorPanel
        :config="config"
        :settings="settings"
        :sources="activeSources"
        @update:settings="settings = $event"
      />
    </div>
  </div>
</template>
```

- [ ] **Step 3: Verify — build, tests, and a live smoke check**

Run (in `frontend/`): `npm test` — expected: 15 passed.
Run (in `frontend/`): `npm run build` — expected: builds clean.

Live smoke check (needs `.env` with `OPENAI_API_KEY`, built KB, and MLflow on :5000):

```bash
# terminal 1 (repo root)
uv run uvicorn backend.main:app --port 8000
# terminal 2
cd frontend && npm run dev
```

Open http://localhost:5173 and verify: header status dot; example questions clickable; asking a question shows `retrieving…` → sources fill the inspector → `generating…` → streaming text with the orange cursor → final per-stage timings; switching mode/k changes subsequent requests; `clear conversation` resets; killing the backend and asking again shows an inline error with the question restored in the composer.

- [ ] **Step 4: Commit**

```bash
git add frontend/src
git commit -m "feat: minimal-technical chat UI — conversation column + inspector rail"
```

---

### Task 9: Frontend README + final verification

**Files:**
- Create: `frontend/README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the finished migration.

- [ ] **Step 1: Create frontend/README.md**

```markdown
# frontend

Vue 3 + Vite + Tailwind CSS v4 chat UI for the Nimbus support RAG agent.
Design direction: minimal-technical — IBM Plex Sans/Mono, warm paper background,
one orange accent, a conversation column plus a retrieval "inspector" rail.

## Run

Requires Node 20+. The backend must be running on :8000 (see `../backend/README.md`);
the dev server proxies `/api` there.

```bash
npm install
npm run dev        # http://localhost:5173
```

## Other commands

```bash
npm test           # Vitest: SSE parser + useChat composable
npm run build      # production build to dist/
```

## Layout

- `src/api.js` — fetch client + incremental SSE parser for `POST /api/chat`
- `src/composables/useChat.js` — conversation state, stage lifecycle (retrieving →
  generating), history replay, error recovery
- `src/components/` — presentation only
- `src/style.css` — Tailwind v4 theme tokens (colors, fonts) and markdown styles
```

- [ ] **Step 2: Full verification pass**

```bash
uv run pytest            # expected: all pass
uv run ruff check .      # expected: no errors
cd frontend
npm test                 # expected: 15 passed
npm run build            # expected: clean build
```

- [ ] **Step 3: Verify no Gradio remnants**

Run (PowerShell): `Select-String -Path CLAUDE.md, README.md -Pattern gradio`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add frontend/README.md
git commit -m "docs: frontend README with run instructions"
```
