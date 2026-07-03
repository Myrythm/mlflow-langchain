"""Tests for the FastAPI wrapper (backend/) — no OpenAI, MLflow, or Chroma needed."""

import json

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


def test_chat_agent_build_failure_maps_like_pre_stream_errors(client, monkeypatch):
    def broken_get_agent(mode, sparse_model, k):
        raise RuntimeError(
            "Missing credentials — set the OPENAI_API_KEY environment variable"
        )

    monkeypatch.setattr(main, "_get_agent", broken_get_agent)

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
