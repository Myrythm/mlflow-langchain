"""FastAPI wrapper around the support_rag library.

Thin by design: endpoints validate input, call the existing library functions, and
translate results/errors to HTTP. Run from the repo root:

    uv run uvicorn backend.main:app --port 8000
"""

import json
import logging
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from support_rag import config
from support_rag.generation import build_stream_agent
from support_rag.retrieval import HybridRetriever
from support_rag.tracing import setup_tracing

from .schemas import K_MAX, K_MIN, ChatRequest, ConfigResponse, HealthResponse, SourceDoc

logger = logging.getLogger(__name__)

_agents: dict[tuple[str, str, int], Callable] = {}

_ERROR_REPLY = (
    "Sorry — something went wrong while generating an answer. Please try again in a "
    "moment; your question is still in the input box."
)
_SNIPPET_CHARS = 160


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
