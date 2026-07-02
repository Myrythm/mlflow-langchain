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
