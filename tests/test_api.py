"""Tests for the FastAPI wrapper (backend/) — no OpenAI, MLflow, or Chroma needed."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import main
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
