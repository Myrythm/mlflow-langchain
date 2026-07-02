# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A customer-support RAG agent (LangChain + OpenAI) demonstrating hybrid retrieval on Chroma,
MLflow Tracing, and MLflow GenAI evaluation. The knowledge base is a hand-authored Markdown
corpus (8 help-center articles) for a fictional SaaS product, **Nimbus**. Everything —
retrieval, generation, and eval — is grounded against `docs/nimbus-fact-sheet.md`.

## Commands

Uses `uv` for dependency management (`uv.lock` is checked in).

```bash
# One-time: start the MLflow tracking server (must stay running for tracing/eval)
uv run mlflow server --host 127.0.0.1 --port 5000

# One-time (or after editing knowledge-base/*.md): build dense (Chroma) + sparse (BM25) indexes
uv run python build_kb.py

# Run the default (hybrid) agent once, traced to MLflow
uv run python main.py

# Gradio chat UI with a retrieval-strategy selector (dense/sparse/hybrid + top-k)
uv run python app.py   # http://127.0.0.1:7860

# Evaluate the default agent (hybrid + BM25) against the 16-question eval set
uv run python run_eval.py 2     # smoke-test on first 2 questions
uv run python run_eval.py       # full eval set

# Benchmark dense / BM25 / hybrid side by side
uv run python compare_retrievers.py 2   # smoke-test
uv run python compare_retrievers.py     # full eval set

# Tests
uv run pytest
uv run pytest tests/test_knowledge_base.py       # single file
uv run pytest tests/test_config.py::test_only_bm25_sparse_model   # single test

# Lint (ruff: pycodestyle, pyflakes, import sorting, bugbear, pyupgrade)
uv run ruff check .
uv run ruff check --fix .
```

`OPENAI_API_KEY` must be set in `.env` (copy from `.env.example`). The MLflow server must be
running before `main.py`, `app.py`, `run_eval.py`, or `compare_retrievers.py` will work —
they all call `setup_tracing()`, which points at `http://127.0.0.1:5000`.

## Architecture

The library lives in `support_rag/`; everything at the repo root is a thin CLI entry point
over it.

- `config.py` — single source of truth for every tunable: model names, paths, MLflow
  target, chunking, retrieval defaults (mode/k/weights/RRF constant), and the system/guideline
  prompts. Change behavior here, not by hardcoding values elsewhere.
- `knowledge_base.py` — loads the Markdown corpus, chunks it, and builds/loads two parallel
  indexes: a persistent Chroma collection (`nimbus_support`, OpenAI dense embeddings) and an
  in-process sparse BM25 index (via `fastembed`), cached to disk under `data/`.
- `retrieval.py` — `HybridRetriever`, a `langchain_core.BaseRetriever` subclass supporting
  three modes: `dense` (Chroma similarity), `sparse` (BM25 dot product), `hybrid` (Reciprocal
  Rank Fusion of both rankings, weighted per `config.HYBRID_WEIGHTS`). Being a real
  `BaseRetriever` means `mlflow.langchain.autolog()` traces it as a RETRIEVER span for free,
  and MLflow's retrieval scorers work unchanged.

  Important local constraint: Chroma's native sparse/hybrid indexing is Chroma-Cloud-only —
  the local `PersistentClient` rejects it. That's why sparse scoring and RRF fusion happen
  in-process here rather than inside Chroma.
- `generation.py` — assembles the RAG chain (retriever → support-agent prompt → `gpt-4o-mini`).
  `get_default_answer()` returns the default hybrid+BM25 pipeline; `build_answer_fn(retriever)`
  builds one for any `HybridRetriever` configuration.
- `eval_dataset.py` — the 16 hand-written eval questions, each carrying `expected_facts` and a
  `source_doc`, both grounded in the fact sheet. Every knowledge-base article has at least
  one question (enforced by a test).
- `evaluation.py` — `build_scorers()` (Correctness, RelevanceToQuery, RetrievalGroundedness,
  RetrievalRelevance, RetrievalSufficiency, a custom `SUPPORT_GUIDELINE` judge, Safety),
  `evaluate_config()`, and `compare_configs()`, all built on `mlflow.genai.evaluate()`.
- `tracing.py` — `setup_tracing()`: sets the MLflow tracking URI/experiment
  (`customer-support-rag`) and enables `mlflow.langchain.autolog()`. Call this before touching
  the agent, retriever, or eval in any new script.
- `ui.py` — `build_demo()`, the Gradio chat interface used by `app.py`.

When changing retrieval or chunking behavior, `build_kb.py` must be re-run to rebuild the
Chroma collection and BM25 cache under `data/` before the change takes effect in `main.py`,
`app.py`, or eval scripts.
