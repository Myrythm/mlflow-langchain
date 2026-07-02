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
