"""Tests for the generation chain: lazy LLM init and conversation history."""

import importlib
import sys

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from support_rag import config


def test_generation_import_does_not_require_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sys.modules.pop("support_rag.generation", None)

    importlib.import_module("support_rag.generation")  # must not raise


def test_prompt_includes_history_messages():
    from support_rag import generation

    messages = generation._prompt.invoke(
        {
            "context": "ctx",
            "question": "And on the Business plan?",
            "history": [
                {"role": "user", "content": "How much does the Pro plan cost?"},
                {"role": "assistant", "content": "$12 per member/month."},
            ],
        }
    ).to_messages()

    contents = [m.content for m in messages]
    assert any("How much does the Pro plan cost?" in c for c in contents)
    assert any("$12 per member/month." in c for c in contents)
    assert messages[-1].content == "And on the Business plan?"


class _FakeRetriever:
    def invoke(self, question):
        return [Document(page_content="Pro costs $12.", metadata={"chunk_id": "x#chunk0"})]


def test_agent_passes_history_to_the_llm(monkeypatch):
    from support_rag import generation

    # Fake LLM that echoes the fully-formatted prompt back as its answer.
    echo = RunnableLambda(lambda prompt: AIMessage(content=str(prompt.to_messages())))
    monkeypatch.setattr(generation, "_get_llm", lambda: echo)

    agent = generation.build_agent(_FakeRetriever())
    answer, docs = agent(
        "And on the Business plan?",
        history=[
            {"role": "user", "content": "How much does the Pro plan cost?"},
            {"role": "assistant", "content": "$12 per member/month."},
        ],
    )

    assert "How much does the Pro plan cost?" in answer
    assert "$12 per member/month." in answer
    assert docs[0].page_content == "Pro costs $12."


def test_stream_agent_yields_tokens_and_docs(monkeypatch):
    from support_rag import generation

    echo = RunnableLambda(lambda prompt: AIMessage(content="streamed answer"))
    monkeypatch.setattr(generation, "_get_llm", lambda: echo)

    agent = generation.build_stream_agent(_FakeRetriever())
    tokens, docs = agent("How much is Pro?", history=[])

    assert "".join(tokens) == "streamed answer"
    assert docs[0].page_content == "Pro costs $12."


def test_agent_caps_history_to_configured_window(monkeypatch):
    from support_rag import generation

    echo = RunnableLambda(lambda prompt: AIMessage(content=str(prompt.to_messages())))
    monkeypatch.setattr(generation, "_get_llm", lambda: echo)

    cap = config.MAX_HISTORY_MESSAGES
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i}"}
        for i in range(cap + 2)
    ]

    agent = generation.build_agent(_FakeRetriever())
    answer, _ = agent("latest question", history=history)

    assert "turn-0" not in answer  # oldest turns beyond the window are dropped
    assert f"turn-{cap + 1}" in answer  # most recent turn is kept
