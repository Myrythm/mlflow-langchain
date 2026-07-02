"""Tests for the Gradio callback logic in support_rag.ui (no browser needed)."""

from langchain_core.documents import Document

from support_rag import ui


def _fake_agent_factory(captured, tokens=("the ", "answer"), docs=None, error=None):
    def get_agent(mode, sparse_model, k):
        def agent(question, history=None):
            captured["question"] = question
            captured["history"] = history
            if error is not None:
                raise error
            return iter(tokens), docs or []

        return agent

    return get_agent


def test_respond_passes_history_to_agent(monkeypatch):
    captured = {}
    monkeypatch.setattr(ui, "_get_agent", _fake_agent_factory(captured))
    prior = [
        {"role": "user", "content": "How much does Pro cost?"},
        {"role": "assistant", "content": "$12."},
    ]

    history, textbox, _ = list(ui._respond("And Business?", prior, "hybrid", "bm25", 4))[-1]

    assert captured["history"] == prior
    assert history[-2:] == [
        {"role": "user", "content": "And Business?"},
        {"role": "assistant", "content": "the answer"},
    ]
    assert textbox == ""


def test_respond_streams_incremental_answers(monkeypatch):
    monkeypatch.setattr(ui, "_get_agent", _fake_agent_factory({}, tokens=("A", "B", "C")))

    partials = [
        history[-1]["content"]
        for history, _, _ in ui._respond("q", [], "hybrid", "bm25", 4)
        if history and history[-1]["role"] == "assistant"
    ]

    assert partials[-1] == "ABC"
    assert "A" in partials and "AB" in partials  # grows token by token


def test_respond_shows_friendly_message_on_agent_error(monkeypatch):
    monkeypatch.setattr(
        ui, "_get_agent", _fake_agent_factory({}, error=RuntimeError("API down"))
    )

    history, textbox, sources = list(ui._respond("hello", [], "hybrid", "bm25", 4))[-1]

    assert history[-1]["role"] == "assistant"
    assert "sorry" in history[-1]["content"].lower()
    assert "API down" not in history[-1]["content"]  # no raw internals leaked to chat
    assert textbox == "hello"  # question kept in the box for an easy retry


def test_respond_replaces_partial_answer_on_midstream_error(monkeypatch):
    def broken_tokens():
        yield "partial "
        raise RuntimeError("stream died")

    def get_agent(mode, sparse_model, k):
        def agent(question, history=None):
            return broken_tokens(), []

        return agent

    monkeypatch.setattr(ui, "_get_agent", get_agent)

    history, textbox, _ = list(ui._respond("hello", [], "hybrid", "bm25", 4))[-1]

    assert "sorry" in history[-1]["content"].lower()
    assert "partial" not in history[-1]["content"]
    assert textbox == "hello"


def test_format_sources_only_adds_ellipsis_when_truncated():
    short = Document(
        page_content="Short passage.",
        metadata={"chunk_id": "a#chunk0", "title": "A", "category": "test"},
    )
    long = Document(
        page_content="x" * 300,
        metadata={"chunk_id": "b#chunk0", "title": "B", "category": "test"},
    )

    rendered = ui._format_sources([short, long])

    assert "Short passage." in rendered
    assert "Short passage.…" not in rendered
    assert "x" * 160 + "…" in rendered
