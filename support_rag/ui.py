"""Gradio chat UI for the customer-support RAG agent.

A chat interface with selectable retrieval strategy (dense / sparse / hybrid + sparse
model + top-k) and a live panel showing which knowledge-base articles were retrieved.
Every turn is traced into MLflow (experiment `customer-support-rag`).
"""

import logging

import gradio as gr

from . import config
from .generation import build_stream_agent
from .retrieval import HybridRetriever
from .tracing import setup_tracing

logger = logging.getLogger(__name__)

_agents: dict[tuple, object] = {}

_ERROR_REPLY = (
    "Sorry — something went wrong while generating an answer. Please try again in a "
    "moment; your question is still in the input box."
)


def _get_agent(mode: str, sparse_model: str, k: int):
    key = (mode, sparse_model, int(k))
    if key not in _agents:
        _agents[key] = build_stream_agent(
            HybridRetriever(mode=mode, sparse_model=sparse_model, k=int(k))
        )
    return _agents[key]


def _format_sources(docs) -> str:
    if not docs:
        return "_No sources retrieved._"
    lines = ["### Retrieved sources"]
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        snippet = doc.page_content[:160].strip().replace("\n", " ")
        suffix = "…" if len(doc.page_content) > 160 else ""
        lines.append(
            f"**{i}. {meta.get('title')}** "
            f"_(category: {meta.get('category')})_\n\n{snippet}{suffix}"
        )
    return "\n\n".join(lines)


def _respond(message, history, mode, sparse_model, k):
    """Streaming chat callback: yields (chat history, textbox, sources panel) updates."""
    history = history or []
    if not message or not message.strip():
        yield history, "", "_Ask a question to see sources._"
        return
    try:
        tokens, docs = _get_agent(mode, sparse_model, int(k))(message, history=history)
    except Exception:
        logger.exception("Support agent failed to answer %r", message)
        # Keep the question in the textbox so the user can just hit Send again.
        yield (
            history
            + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": _ERROR_REPLY},
            ],
            message,
            "_No sources retrieved._",
        )
        return

    convo = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    sources = _format_sources(docs)
    yield convo, "", sources  # show the user turn + sources before the first token

    answer = ""
    try:
        for token in tokens:
            answer += token
            convo = convo[:-1] + [{"role": "assistant", "content": answer}]
            yield convo, "", sources
    except Exception:
        logger.exception("Answer stream failed for %r", message)
        convo = convo[:-1] + [{"role": "assistant", "content": _ERROR_REPLY}]
        yield convo, message, sources


def build_demo() -> gr.Blocks:
    setup_tracing()
    with gr.Blocks(title="Nimbus Support Assistant", fill_height=True) as demo:
        gr.Markdown(
            "# 🎧 Nimbus Support Assistant\n"
            "Hybrid RAG over the Nimbus support knowledge base "
            "(LangChain + OpenAI, retrieval on Chroma). Every turn is traced in MLflow."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=440, label="Conversation")
                message = gr.Textbox(
                    placeholder="e.g. How much does the Pro plan cost?",
                    label="Your question",
                    autofocus=True,
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
                gr.Examples(
                    examples=[[q] for q in config.EXAMPLE_QUESTIONS],
                    inputs=[message],
                    label="Try one of these",
                )
            with gr.Column(scale=2):
                with gr.Accordion("Retrieval settings", open=True):
                    mode = gr.Dropdown(
                        ["dense", "sparse", "hybrid"],
                        value=config.DEFAULT_MODE,
                        label="Mode",
                    )
                    sparse_model = gr.Dropdown(
                        list(config.SPARSE_MODELS),
                        value=config.DEFAULT_SPARSE,
                        label="Sparse model (used by sparse / hybrid)",
                    )
                    top_k = gr.Slider(
                        1, 8, value=config.DEFAULT_K, step=1, label="Top-k"
                    )
                sources = gr.Markdown("_Ask a question to see sources._")

        inputs = [message, chatbot, mode, sparse_model, top_k]
        outputs = [chatbot, message, sources]
        send.click(_respond, inputs, outputs)
        message.submit(_respond, inputs, outputs)
        clear.click(
            lambda: ([], "", "_Ask a question to see sources._"),
            None,
            [chatbot, message, sources],
        )
    return demo
