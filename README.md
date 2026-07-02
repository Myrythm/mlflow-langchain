# mlflow-llm

A customer-support **RAG** agent (LangChain + OpenAI) with **hybrid retrieval** on
**Chroma**, **MLflow Tracing**, and **MLflow GenAI evaluation**.

- **Knowledge base:** a **hand-authored Markdown corpus** — 8 help-center articles for a
  fictional SaaS project-management product, **Nimbus** (`getting-started`,
  `pricing-and-billing`, `refunds-and-cancellation`, `integrations`, `security-and-sso`,
  `account-and-teams`, `features-overview`, `troubleshooting`). Each `.md` file in
  `knowledge-base/` carries a small `title` / `category` frontmatter block and is chunked
  into passages (~39 total). The canonical facts every article draws from live in
  [`docs/nimbus-fact-sheet.md`](docs/nimbus-fact-sheet.md), which is the eval ground truth.
  The **dense** index (OpenAI `text-embedding-3-small`) lives in a persistent **Chroma**
  collection (`nimbus_support`); the **sparse** vectors (**BM25** via `fastembed`) are
  computed in-process and cached.
- **Retrieval:** dense (semantic), sparse (BM25), or **hybrid** (Reciprocal Rank Fusion of
  dense + BM25). Default agent uses **hybrid (dense + BM25)**.

  > Note: Chroma's _native_ sparse/hybrid indexing is a Chroma Cloud feature — the local
  > `PersistentClient` rejects it. Locally we keep Chroma for the dense index and fuse with
  > in-process `fastembed` BM25 scores via RRF in `retrieval.py`.

- **Agent:** LangChain retriever → support-agent prompt → `gpt-4o-mini`, traced via
  `mlflow.langchain.autolog()`.
- **Eval:** `mlflow.genai.evaluate()` over a **hand-written set of 16 grounded questions**
  (each with `expected_facts` + the `source_doc` it's grounded in, drawn from the Nimbus
  fact sheet) — Correctness, RelevanceToQuery, RetrievalGroundedness, RetrievalRelevance,
  RetrievalSufficiency, a custom support Guidelines judge, and Safety.

## Setup

1. Put your OpenAI key in `.env` (copy from `.env.example`): `OPENAI_API_KEY=sk-...`
2. Start the MLflow tracking server (leave it running):

   ```bash
   uv run mlflow server --host 127.0.0.1 --port 5000
   ```

## Build the knowledge base (once)

Chunks the Markdown articles in `knowledge-base/`, builds the dense (Chroma) + sparse
(BM25) indexes, and caches the eval set. (Downloads the small BM25 model on first run.)

```bash
uv run python build_kb.py
```

## Run the support agent

```bash
uv run python main.py
```

Open http://127.0.0.1:5000 → **customer-support-rag** → **Traces**.

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

## Compare retrieval strategies

Runs the same MLflow eval across dense / BM25 / hybrid and prints a side-by-side scorer
table plus per-config run URLs:

```bash
uv run python compare_retrievers.py 2     # smoke-test on 2 questions first
uv run python compare_retrievers.py       # full eval set
```

## Evaluate a single config

```bash
uv run python run_eval.py 2     # smoke-test
uv run python run_eval.py       # full eval set (default = hybrid + BM25)
```

### Results (default config, regenerated 2026-06-30 on the then-14-question eval set)

> The eval set has since grown to 16 questions (2 new `features-overview` questions);
> re-run `run_eval.py` to refresh these numbers.

Default agent = **hybrid (dense + BM25)**, `k=4`:

| metric                 | hybrid (dense + BM25) |
| ---------------------- | --------------------- |
| correctness            | 0.929                 |
| relevance_to_query     | 1.000                 |
| retrieval_groundedness | 1.000                 |
| retrieval_sufficiency  | 1.000                 |
| retrieval_relevance    | 0.661                 |
| safety                 | 1.000                 |
| support_quality        | 0.071                 |

**Takeaways:** the agent answers 13/14 questions correctly with **perfect groundedness and
sufficiency** — the hand-authored corpus and grounded eval set make the answers cleanly
traceable to source.

_Caveats:_ `retrieval_relevance` (~0.66) is the precision penalty of `k=4` on a small
(~39-passage) corpus — retrieving 4 chunks when 1–2 suffice — not a quality problem.
`support_quality` is low because the strict custom guideline expects the agent to suggest
contacting a human even when it _can_ answer; it's a guideline artifact, not a retrieval
issue. Run `compare_retrievers.py` to benchmark dense / BM25 / hybrid side by side.

## Project structure

The library lives in the `support_rag` package; the root scripts are thin CLI entry points.

```
support_rag/
  config.py           # all constants: models, paths, MLflow target, retrieval defaults, prompts
  tracing.py          # setup_tracing() — MLflow URI + experiment + LangChain autolog
  knowledge_base.py   # markdown corpus loader + dense Chroma index + sparse (BM25) index
  retrieval.py        # HybridRetriever — dense / sparse / hybrid (RRF)
  generation.py       # RAG chain: build_answer_fn() / default_retriever() / get_default_answer()
  eval_dataset.py     # hand-written questions with expected facts + source docs
  evaluation.py       # build_scorers() / evaluate_config() / compare_configs()

build_kb.py            # CLI: build the dense + sparse indexes
main.py                # CLI: demo the default (hybrid) agent
backend/               # FastAPI wrapper: /api/config, /api/health, /api/chat (SSE)
frontend/              # Vue 3 + Vite + Tailwind chat UI (see frontend/README.md)
run_eval.py            # CLI: evaluate the default agent
compare_retrievers.py  # CLI: benchmark dense / BM25 / hybrid

knowledge-base/        # the Markdown knowledge base (one article per .md file)
docs/nimbus-fact-sheet.md  # canonical facts the articles + eval set draw from
```

Library usage:

```python
from support_rag import setup_tracing, get_default_answer, HybridRetriever, build_answer_fn

setup_tracing()
answer = get_default_answer()              # hybrid + BM25
answer("How much does the Pro plan cost?")

# or a specific retrieval strategy:
build_answer_fn(HybridRetriever(mode="dense", k=3))("...")
```
