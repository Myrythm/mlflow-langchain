# Nimbus Support KB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the doc2dial corpus with a hand-authored Markdown knowledge base for a fictional SaaS ("Nimbus"), keeping hybrid (dense + BM25) retrieval, MLflow tracing, and the GenAI eval suite.

**Architecture:** The `support_rag` pipeline shape is unchanged (load → chunk → index dense+sparse → retrieve → generate → eval). Only the data source changes: a new Markdown loader reads `knowledge-base/*.md` (YAML-ish frontmatter for metadata) instead of downloading/parsing doc2dial. SPLADE is removed so the only sparse model is BM25. The eval set becomes a hand-written Python list grounded in a canonical Nimbus fact sheet.

**Tech Stack:** Python 3.12, uv, LangChain, langchain-text-splitters, Chroma (OpenAI `text-embedding-3-small`), fastembed (BM25), MLflow GenAI eval, Gradio, pytest (new dev dep).

## Global Constraints

- Python `>=3.12`; manage everything with `uv` (`uv run …`, `uv add …`).
- Retrieval is **dense + BM25 only**. SPLADE must be fully removed — no `splade` keys, configs, or UI options remain.
- Chunking: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`.
- Markdown frontmatter format is exactly:
  ```
  ---
  title: <Human Title>
  category: <slug>
  ---
  ```
- Document metadata keys are exactly: `chunk_id`, `doc_id`, `title`, `category`. (`category` replaces the old `domain`.)
- KB content must be realistic and **internally consistent**, all facts drawn from the canonical fact sheet in Task 3. No `{{placeholder}}` tokens.
- The Chroma dense collection is named `nimbus_support`.
- Commit after every task with the message shown in its final step.

---

### Task 1: Add pytest and rewrite `config.py`

**Files:**
- Modify: `pyproject.toml` (add pytest dev dependency)
- Modify: `support_rag/config.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.KB_DIR: Path` (the `knowledge-base/` folder), `config.SPARSE_MODELS == {"bm25": "Qdrant/bm25"}`, `config.DEFAULT_SPARSE == "bm25"`, `config.DENSE_COLLECTION == "nimbus_support"`. Removes `DOC2DIAL_URL`, `DOC2DIAL_DIR`, `DOC2DIAL_DOMAINS`, `EVAL_SIZE`, `EVAL_SEED`.

- [ ] **Step 1: Add pytest as a dev dependency**

Run: `uv add --dev pytest`
Expected: `pyproject.toml` gains a `[dependency-groups]` (or `[tool.uv]` dev) entry for pytest; `uv.lock` updates.

- [ ] **Step 2: Create the empty test package marker**

Create `tests/__init__.py` with no content (empty file).

- [ ] **Step 3: Write the failing config test**

Create `tests/test_config.py`:

```python
from support_rag import config


def test_kb_dir_points_at_knowledge_base_folder():
    assert config.KB_DIR.name == "knowledge-base"


def test_only_bm25_sparse_model():
    assert config.SPARSE_MODELS == {"bm25": "Qdrant/bm25"}
    assert config.DEFAULT_SPARSE == "bm25"


def test_dense_collection_renamed():
    assert config.DENSE_COLLECTION == "nimbus_support"


def test_doc2dial_constants_removed():
    for name in ("DOC2DIAL_URL", "DOC2DIAL_DIR", "DOC2DIAL_DOMAINS", "EVAL_SIZE", "EVAL_SEED"):
        assert not hasattr(config, name), f"{name} should be removed"
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL (e.g. `AttributeError: module ... has no attribute 'KB_DIR'` and/or the doc2dial-removed assertions fail).

- [ ] **Step 5: Edit `config.py`**

In the "Knowledge base" section, replace the doc2dial block. The section should read:

```python
# --- Knowledge base (hand-authored Markdown corpus) ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PERSIST_DIR = DATA_DIR / "chroma"
KB_DIR = Path(__file__).resolve().parent.parent / "knowledge-base"
DENSE_COLLECTION = "nimbus_support"
SPARSE_MODELS = {
    "bm25": "Qdrant/bm25",
}
```

Delete these lines entirely: `DOC2DIAL_URL`, `DOC2DIAL_DIR`, `DOC2DIAL_DOMAINS`, and the old `DENSE_COLLECTION = "doc2dial_support"` and `SPARSE_MODELS` splade entry.

In the "Evaluation set" section, delete `EVAL_SIZE` and `EVAL_SEED` (eval is now hand-written). You may delete the whole `# --- Evaluation set ... ---` comment block.

In "Retrieval defaults", change `DEFAULT_SPARSE = "splade"` to:

```python
DEFAULT_SPARSE = "bm25"     # the only sparse model
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock support_rag/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: point config at markdown KB, drop doc2dial + SPLADE, add pytest"
```

---

### Task 2: Rewrite `knowledge_base.py` as a Markdown loader

**Files:**
- Modify: `support_rag/knowledge_base.py`
- Test: `tests/test_knowledge_base.py`

**Interfaces:**
- Consumes: `config.KB_DIR`, `config.CHUNK_SIZE`, `config.CHUNK_OVERLAP`, `config.DATA_DIR`, `config.SPARSE_MODELS`, `config.DENSE_COLLECTION`, `config.PERSIST_DIR`, `config.DENSE_MODEL`.
- Produces (names retrieval.py depends on, unchanged signatures):
  - `build_documents() -> list[Document]` — each Document has metadata `chunk_id`, `doc_id`, `title`, `category`.
  - `doc_by_id(chunk_id: str) -> Document`
  - `get_dense_collection(rebuild: bool = False)`
  - `get_sparse_index(model: str) -> list[dict]`
  - `embed_query_sparse(model: str, query: str) -> dict[int, float]`
  - New helper: `_parse_frontmatter(text: str) -> tuple[dict, str]`

- [ ] **Step 1: Write the failing loader test**

Create `tests/test_knowledge_base.py`:

```python
import importlib

from support_rag import config, knowledge_base


def _write_md(dirpath, name, frontmatter, body):
    (dirpath / name).write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")


def test_build_documents_reads_frontmatter_and_chunks(tmp_path, monkeypatch):
    _write_md(tmp_path, "pricing.md", "title: Pricing & Plans\ncategory: billing",
              "The Pro plan costs $12 per member per month.")
    monkeypatch.setattr(config, "KB_DIR", tmp_path)
    knowledge_base.build_documents.cache_clear()

    docs = knowledge_base.build_documents()

    assert len(docs) >= 1
    meta = docs[0].metadata
    assert meta["doc_id"] == "pricing"
    assert meta["title"] == "Pricing & Plans"
    assert meta["category"] == "billing"
    assert meta["chunk_id"] == "pricing#chunk0"
    assert "frontmatter" not in docs[0].page_content.lower()
    assert "$12" in docs[0].page_content


def test_parse_frontmatter_without_block_returns_empty_meta():
    meta, body = knowledge_base._parse_frontmatter("# Just a heading\nbody text")
    assert meta == {}
    assert body.startswith("# Just a heading")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_knowledge_base.py -v`
Expected: FAIL (`_parse_frontmatter` doesn't exist; `build_documents` still reads doc2dial).

- [ ] **Step 3: Rewrite `knowledge_base.py`**

Replace the entire module with:

```python
"""Knowledge base built from a hand-authored Markdown corpus.

Each `.md` file in `config.KB_DIR` is one knowledge-base article with a small YAML-style
frontmatter block (`title`, `category`). The body is chunked into passages; the dense index
lives in Chroma (OpenAI embeddings) and the sparse (BM25) vectors are computed with fastembed
and cached to disk. Fusion lives in `retrieval.py`.
"""

import functools
import pickle

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

_client = None
_sparse_models: dict[str, object] = {}
_sparse_index: dict[str, list[dict]] = {}


# --- Markdown corpus -> chunked Documents ----------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a leading `--- ... ---` frontmatter block from the body.

    Returns ({key: value}, body). If there is no frontmatter block, returns ({}, text).
    Values are parsed as plain strings (no nested YAML).
    """
    meta: dict[str, str] = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            block = text[3:end].strip()
            body = text[end + 4 :].lstrip("\n")
            for line in block.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip()
            return meta, body
    return meta, text


@functools.cache
def build_documents() -> list[Document]:
    """Chunk every Markdown article in `config.KB_DIR` into passages."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    docs: list[Document] = []
    for path in sorted(config.KB_DIR.glob("*.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        body = body.strip()
        if not body:
            continue
        doc_id = path.stem
        title = meta.get("title", doc_id)
        category = meta.get("category", "general")
        for i, chunk in enumerate(splitter.split_text(body)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "chunk_id": f"{doc_id}#chunk{i}",
                        "doc_id": doc_id,
                        "title": title,
                        "category": category,
                    },
                )
            )
    return docs


@functools.cache
def _doc_map() -> dict[str, Document]:
    return {d.metadata["chunk_id"]: d for d in build_documents()}


def doc_by_id(chunk_id: str) -> Document:
    return _doc_map()[chunk_id]


# --- Dense (Chroma) ---------------------------------------------------------------

def _get_client():
    global _client
    if _client is None:
        config.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.PERSIST_DIR))
    return _client


def _dense_ef() -> OpenAIEmbeddingFunction:
    return OpenAIEmbeddingFunction(
        api_key_env_var="OPENAI_API_KEY", model_name=config.DENSE_MODEL
    )


def get_dense_collection(rebuild: bool = False):
    client = _get_client()
    if rebuild:
        try:
            client.delete_collection(config.DENSE_COLLECTION)
        except Exception:
            pass

    if config.DENSE_COLLECTION in {c.name for c in client.list_collections()}:
        return client.get_collection(
            config.DENSE_COLLECTION, embedding_function=_dense_ef()
        )

    collection = client.create_collection(
        name=config.DENSE_COLLECTION, embedding_function=_dense_ef()
    )
    docs = build_documents()
    batch = 256  # batch the embedding calls
    for start in range(0, len(docs), batch):
        part = docs[start : start + batch]
        collection.add(
            ids=[d.metadata["chunk_id"] for d in part],
            documents=[d.page_content for d in part],
            metadatas=[
                {k: d.metadata[k] for k in ("doc_id", "title", "category")} for d in part
            ],
        )
    print(f"Built dense collection '{collection.name}': {collection.count()} passages.")
    return collection


# --- Sparse (fastembed: BM25) -----------------------------------------------------

def _get_sparse_model(model: str):
    if model not in config.SPARSE_MODELS:
        raise ValueError(f"sparse model must be one of {list(config.SPARSE_MODELS)}")
    if model not in _sparse_models:
        from fastembed import SparseTextEmbedding

        _sparse_models[model] = SparseTextEmbedding(
            model_name=config.SPARSE_MODELS[model]
        )
    return _sparse_models[model]


def _sparse_cache_path(model: str):
    return config.DATA_DIR / f"sparse_{model}.pkl"


def get_sparse_index(model: str) -> list[dict]:
    """Sparse doc vectors [{id, vec}], cached in memory and on disk."""
    if model in _sparse_index:
        return _sparse_index[model]

    cache = _sparse_cache_path(model)
    if cache.exists():
        _sparse_index[model] = pickle.loads(cache.read_bytes())
        return _sparse_index[model]

    docs = build_documents()
    vectors = _get_sparse_model(model).embed([d.page_content for d in docs])
    index = [
        {
            "id": doc.metadata["chunk_id"],
            "vec": dict(zip(vec.indices.tolist(), vec.values.tolist())),
        }
        for doc, vec in zip(docs, vectors)
    ]
    _sparse_index[model] = index
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(index))
    return index


def embed_query_sparse(model: str, query: str) -> dict[int, float]:
    vec = next(iter(_get_sparse_model(model).query_embed([query])))
    return dict(zip(vec.indices.tolist(), vec.values.tolist()))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_knowledge_base.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add support_rag/knowledge_base.py tests/test_knowledge_base.py
git commit -m "feat: load knowledge base from markdown corpus instead of doc2dial"
```

---

### Task 3: Author the Nimbus fact sheet and 8 Markdown articles

This task is delegated to a **Sonnet 4.6 subagent** (the user requested this). The deliverable is content, validated by re-running the Task 2 loader test against the real folder.

**Files:**
- Create: `docs/nimbus-fact-sheet.md` (the canonical source of truth — NOT indexed, lives outside `knowledge-base/`)
- Create: `knowledge-base/getting-started.md`
- Create: `knowledge-base/pricing-and-billing.md`
- Create: `knowledge-base/refunds-and-cancellation.md`
- Create: `knowledge-base/integrations.md`
- Create: `knowledge-base/security-and-sso.md`
- Create: `knowledge-base/account-and-teams.md`
- Create: `knowledge-base/features-overview.md`
- Create: `knowledge-base/troubleshooting.md`
- Test: `tests/test_corpus.py`

**Canonical Nimbus facts** (the subagent must use these verbatim; expand prose around them, never contradict them):

- **Product:** Nimbus — cloud project-management tool (boards, timeline view, automation, reporting).
- **Plans & price (per member/month):**
  - **Free** — up to 3 members, 2 boards, 100 automation runs/month, community support.
  - **Pro** — $12 billed annually / $14 billed monthly; unlimited boards, timeline view, 1,000 automation runs/month, 250 MB file uploads, priority email support, all integrations.
  - **Business** — $22 billed annually; advanced automation (10,000 runs/month), reporting dashboards, Google & Microsoft SSO, guest access, 5 GB file uploads.
  - **Enterprise** — custom pricing; SAML SSO + SCIM provisioning, audit logs, data residency (US or EU), dedicated customer success manager, 99.9% uptime SLA.
- **Billing:** monthly or annual (annual saves ~15%); invoiced via Stripe; upgrades are prorated.
- **Refunds:** 14-day money-back guarantee on first purchase; cancellation stops auto-renewal and access continues until the end of the current billing period.
- **Roles:** Owner, Admin, Member, Guest.
- **Integrations:** Slack, GitHub, GitLab, Google Drive, Figma, Zapier, plus outgoing webhooks. Connect via **Settings → Integrations** using OAuth.
- **Security:** AES-256 encryption at rest, TLS 1.2+ in transit, TOTP-based 2FA, SOC 2 Type II; SSO (Google/Microsoft) on Business, SAML SSO + SCIM on Enterprise.
- **Support tiers:** community (Free), email (Pro), priority (Business), dedicated CSM (Enterprise).

**Category per file:** `getting-started` → `getting-started`; `pricing-and-billing` → `billing`; `refunds-and-cancellation` → `billing`; `integrations` → `integrations`; `security-and-sso` → `security`; `account-and-teams` → `account`; `features-overview` → `features`; `troubleshooting` → `troubleshooting`.

- [ ] **Step 1: Dispatch the Sonnet 4.6 authoring subagent**

Dispatch a subagent (model: sonnet) with this prompt:

> You are writing realistic customer-support documentation for a fictional SaaS project-management product called **Nimbus**. First write `docs/nimbus-fact-sheet.md` capturing the canonical facts below as a structured reference. Then write 8 help-center articles into `knowledge-base/`, each beginning with frontmatter exactly in this form:
> ```
> ---
> title: <Human Title>
> category: <slug>
> ---
> ```
> followed by Markdown with clear headings, numbered steps where procedural, and a professional help-center tone modeled on Linear/Notion/Asana docs. Every fact (plan names, prices, limits, roles, integrations, security terms) MUST match the fact sheet exactly and stay consistent across all articles. Do NOT use `{{placeholder}}` tokens. Each article ~250–500 words.
> [paste the "Canonical Nimbus facts" block above]
> [paste the file list with their category slugs above]

- [ ] **Step 2: Write the corpus validation test**

Create `tests/test_corpus.py`:

```python
from support_rag import config, knowledge_base

EXPECTED_DOCS = {
    "getting-started", "pricing-and-billing", "refunds-and-cancellation",
    "integrations", "security-and-sso", "account-and-teams",
    "features-overview", "troubleshooting",
}


def test_all_articles_present():
    slugs = {p.stem for p in config.KB_DIR.glob("*.md")}
    assert EXPECTED_DOCS <= slugs


def test_every_article_has_frontmatter_and_no_placeholders():
    for path in config.KB_DIR.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        meta, body = knowledge_base._parse_frontmatter(text)
        assert "title" in meta, f"{path.name} missing title"
        assert "category" in meta, f"{path.name} missing category"
        assert body.strip(), f"{path.name} has empty body"
        assert "{{" not in text, f"{path.name} contains a placeholder token"


def test_corpus_chunks_build():
    knowledge_base.build_documents.cache_clear()
    docs = knowledge_base.build_documents()
    assert len(docs) >= 8
```

- [ ] **Step 3: Run the corpus test**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: PASS (3 tests). If it fails, the subagent's output is incomplete — fix the offending file(s) and re-run.

- [ ] **Step 4: Human review of content**

Read the 8 articles and the fact sheet. Confirm prices/roles/integrations are consistent across articles and read like real help-center docs. Fix any drift directly.

- [ ] **Step 5: Commit**

```bash
git add docs/nimbus-fact-sheet.md knowledge-base/*.md tests/test_corpus.py
git commit -m "feat: author Nimbus fact sheet and 8 help-center articles"
```

---

### Task 4: Rewrite `eval_dataset.py` as a hand-written eval set

**Files:**
- Modify: `support_rag/eval_dataset.py`
- Test: `tests/test_eval_dataset.py`

**Interfaces:**
- Consumes: `config.KB_DIR` (to validate `source_doc` references).
- Produces: `EVAL_DATASET: list[dict]` where each entry is `{"inputs": {"question": str}, "expectations": {"expected_facts": list[str], "source_doc": str}}`. Consumed unchanged by `evaluation.py`, `run_eval.py`, `compare_retrievers.py`.

- [ ] **Step 1: Write the failing eval-dataset test**

Create `tests/test_eval_dataset.py`:

```python
from support_rag import config
from support_rag.eval_dataset import EVAL_DATASET


def test_eval_set_size():
    assert 12 <= len(EVAL_DATASET) <= 15


def test_entry_shape_and_grounding():
    valid_docs = {p.stem for p in config.KB_DIR.glob("*.md")}
    for row in EVAL_DATASET:
        assert row["inputs"]["question"].strip()
        facts = row["expectations"]["expected_facts"]
        assert isinstance(facts, list) and facts and all(f.strip() for f in facts)
        assert row["expectations"]["source_doc"] in valid_docs
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_eval_dataset.py -v`
Expected: FAIL (the current `EVAL_DATASET` is built from doc2dial and entries lack `expected_facts`/`source_doc`).

- [ ] **Step 3: Rewrite `eval_dataset.py`**

Replace the entire module with (curate/adjust facts to match the authored articles — these are grounded in the Task 3 fact sheet):

```python
"""Hand-written evaluation set for the Nimbus support KB.

Each record is shaped for ``mlflow.genai.evaluate``:
  - ``inputs``        -> {"question": <user question>}
  - ``expectations``  -> {"expected_facts": [<key facts a correct answer must contain>],
                          "source_doc": <doc_id the answer is grounded in>}

Facts are grounded in ``docs/nimbus-fact-sheet.md``. Keep this in sync with the articles.
"""

EVAL_DATASET = [
    {
        "inputs": {"question": "How much does the Pro plan cost per member?"},
        "expectations": {
            "expected_facts": [
                "$12 per member/month billed annually",
                "$14 per member/month billed monthly",
            ],
            "source_doc": "pricing-and-billing",
        },
    },
    {
        "inputs": {"question": "How many members can I have on the Free plan?"},
        "expectations": {
            "expected_facts": ["The Free plan supports up to 3 members"],
            "source_doc": "pricing-and-billing",
        },
    },
    {
        "inputs": {"question": "What is the file upload size limit on each plan?"},
        "expectations": {
            "expected_facts": ["250 MB on Pro", "5 GB on Business"],
            "source_doc": "pricing-and-billing",
        },
    },
    {
        "inputs": {"question": "How many automation runs does the Business plan include?"},
        "expectations": {
            "expected_facts": ["10,000 automation runs per month"],
            "source_doc": "pricing-and-billing",
        },
    },
    {
        "inputs": {"question": "Can I get a refund after subscribing?"},
        "expectations": {
            "expected_facts": [
                "14-day money-back guarantee on the first purchase",
                "the refund must be requested within 14 days",
            ],
            "source_doc": "refunds-and-cancellation",
        },
    },
    {
        "inputs": {"question": "What happens to my data when I cancel my subscription?"},
        "expectations": {
            "expected_facts": [
                "cancellation stops auto-renewal",
                "access continues until the end of the current billing period",
            ],
            "source_doc": "refunds-and-cancellation",
        },
    },
    {
        "inputs": {"question": "Does Nimbus support single sign-on?"},
        "expectations": {
            "expected_facts": [
                "Google and Microsoft SSO on the Business plan",
                "SAML SSO on the Enterprise plan",
            ],
            "source_doc": "security-and-sso",
        },
    },
    {
        "inputs": {"question": "How do I turn on two-factor authentication?"},
        "expectations": {
            "expected_facts": [
                "2FA uses a TOTP authenticator app",
                "enable it from account security settings",
            ],
            "source_doc": "security-and-sso",
        },
    },
    {
        "inputs": {"question": "Which apps can I integrate with Nimbus?"},
        "expectations": {
            "expected_facts": ["Slack", "GitHub", "Google Drive"],
            "source_doc": "integrations",
        },
    },
    {
        "inputs": {"question": "How do I connect Slack to Nimbus?"},
        "expectations": {
            "expected_facts": [
                "go to Settings then Integrations",
                "authorize the connection via OAuth",
            ],
            "source_doc": "integrations",
        },
    },
    {
        "inputs": {"question": "How do I invite a teammate to my workspace?"},
        "expectations": {
            "expected_facts": [
                "invite members by email from the workspace members settings",
                "assign them a role",
            ],
            "source_doc": "account-and-teams",
        },
    },
    {
        "inputs": {"question": "What member roles are available?"},
        "expectations": {
            "expected_facts": ["Owner", "Admin", "Member", "Guest"],
            "source_doc": "account-and-teams",
        },
    },
    {
        "inputs": {"question": "How do I create my first board?"},
        "expectations": {
            "expected_facts": [
                "create a workspace first",
                "add a board from the dashboard",
            ],
            "source_doc": "getting-started",
        },
    },
    {
        "inputs": {"question": "My notifications aren't coming through. What should I do?"},
        "expectations": {
            "expected_facts": [
                "check your notification settings",
                "verify email deliverability / check spam",
            ],
            "source_doc": "troubleshooting",
        },
    },
]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_eval_dataset.py -v`
Expected: PASS (2 tests). If `source_doc` assertions fail, a `source_doc` doesn't match an authored filename — reconcile names.

- [ ] **Step 5: Commit**

```bash
git add support_rag/eval_dataset.py tests/test_eval_dataset.py
git commit -m "feat: hand-written Nimbus eval set grounded in the fact sheet"
```

---

### Task 5: Update consumers (UI, demos, comparison, docstrings)

**Files:**
- Modify: `support_rag/ui.py`
- Modify: `support_rag/generation.py:57` (docstring only)
- Modify: `main.py`
- Modify: `compare_retrievers.py`
- Modify: `build_kb.py` (docstring only)
- Test: `tests/test_smoke_imports.py`

**Interfaces:**
- Consumes: existing `HybridRetriever`, `build_agent`, `config.SPARSE_MODELS`, `config.DEFAULT_MODE/SPARSE/K`.
- Produces: nothing new; removes stale `intent`/`splade`/"Bitext"/doc2dial references.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_smoke_imports.py`:

```python
import compare_retrievers


def test_compare_configs_have_no_splade():
    names = set(compare_retrievers.CONFIGS)
    assert names == {"dense", "sparse_bm25", "hybrid_bm25"}
    assert not any("splade" in n for n in names)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_smoke_imports.py -v`
Expected: FAIL (`CONFIGS` still contains `sparse_splade` and `hybrid_splade`).

- [ ] **Step 3: Update `compare_retrievers.py`**

Replace the `CONFIGS` dict (lines 19-25) with:

```python
CONFIGS = {
    "dense": lambda: HybridRetriever(mode="dense", k=4),
    "sparse_bm25": lambda: HybridRetriever(mode="sparse", sparse_model="bm25", k=4),
    "hybrid_bm25": lambda: HybridRetriever(mode="hybrid", sparse_model="bm25", k=4),
}
```

Update the module docstring line 1 from `(dense / BM25 / SPLADE / hybrid)` to `(dense / BM25 / hybrid)`.

- [ ] **Step 4: Update `ui.py`**

In `_format_sources` (lines 33-37), replace the `meta.get('intent')` reference. The block becomes:

```python
        snippet = doc.page_content[:160].strip().replace("\n", " ")
        lines.append(
            f"**{i}. {meta.get('title')}** "
            f"_(category: {meta.get('category')})_\n\n{snippet}…"
        )
```

In `build_demo` (lines 57-60), update the header Markdown text — replace `"Bitext customer-support knowledge base"` with `"Nimbus support knowledge base"`. Update the textbox placeholder (line 65) from `"e.g. How do I cancel my order?"` to `"e.g. How much does the Pro plan cost?"`. Update the `gr.Blocks(title=...)` and heading from "Customer Support Assistant" to "Nimbus Support Assistant".

- [ ] **Step 5: Update `main.py`**

Replace `SAMPLE_QUESTIONS` (lines 12-17) with:

```python
# In-domain questions for the Nimbus support KB.
SAMPLE_QUESTIONS = [
    "How much does the Pro plan cost per member?",
    "Does Nimbus support single sign-on?",
    "How do I connect Slack to Nimbus?",
]
```

- [ ] **Step 6: Update docstrings**

In `support_rag/generation.py:57`, change `"""The default retrieval config (hybrid + SPLADE)."""` to `"""The default retrieval config (hybrid + BM25)."""`.

In `build_kb.py` line 1-5 docstring, replace the doc2dial reference: change `"""Build the knowledge base from doc2dial: download + chunk docs, ...` to `"""Build the knowledge base from the markdown corpus: chunk docs, build the dense Chroma index and the sparse (BM25) index, and build the eval set.`

- [ ] **Step 7: Run the smoke test to verify it passes**

Run: `uv run pytest tests/test_smoke_imports.py -v`
Expected: PASS (1 test).

- [ ] **Step 8: Run the full offline test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS (config, knowledge_base, corpus, eval_dataset, smoke = 12 tests).

- [ ] **Step 9: Commit**

```bash
git add support_rag/ui.py support_rag/generation.py main.py compare_retrievers.py build_kb.py tests/test_smoke_imports.py
git commit -m "refactor: update UI, demos, and comparison for Nimbus + BM25-only"
```

---

### Task 6: End-to-end build + eval sanity check, refresh README

This task hits OpenAI (embeddings + chat) and needs the MLflow server running, so it is a manual integration check, not an automated test.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Start the MLflow tracking server (separate terminal)**

Run: `uv run mlflow server --host 127.0.0.1 --port 5000`
Expected: server logs "Listening at: http://127.0.0.1:5000". Leave it running.

- [ ] **Step 2: Build the indexes from the markdown corpus**

Run: `uv run python build_kb.py`
Expected: prints chunked-corpus passage count (≥8), `nimbus_support: N passages (dense)`, `sparse 'bm25': indexed N passages`, and `eval set: 14 grounded examples`. No doc2dial download occurs.

- [ ] **Step 3: Smoke-test the agent**

Run: `uv run python main.py`
Expected: three Nimbus answers print, each grounded in the KB (correct price, SSO, Slack steps). Confirm a trace appears at http://127.0.0.1:5000 → `customer-support-rag` → Traces.

- [ ] **Step 4: Smoke-test the eval (2 rows)**

Run: `uv run python run_eval.py 2`
Expected: aggregate metrics print (correctness, retrieval_*, safety, support_quality) and an eval-run URL. No errors about missing `expected_response` / scorer inputs.

- [ ] **Step 5: Run the full eval**

Run: `uv run python run_eval.py`
Expected: metrics over all 14 examples; note the correctness/retrieval scores.

- [ ] **Step 6: Refresh the README**

Update `README.md`: replace the doc2dial description (lines 6-26) with the Nimbus markdown-KB description (hand-authored corpus in `knowledge-base/`, hybrid dense + BM25, hand-written eval set). Remove SPLADE from the retrieval/UI sections. Replace the stale results table (lines 76-96) with the numbers from Step 5 (or mark it "regenerated 2026-06-30" with the new figures). Update the `build_kb.py` description (no download step) and the project-structure note for `eval_dataset.py` ("curated questions" still holds).

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for the Nimbus markdown KB and refresh eval results"
```

---

## Notes / Out of Scope

- The `datasets>=5.0.0` dependency becomes unused once doc2dial is gone. Removing it is **out of scope** here (avoids a lockfile re-sync churn); leave it for a later cleanup.
- Existing `data/chroma` and `data/doc2dial` directories from prior doc2dial runs can be deleted manually; `build_kb.py` rebuilds the dense collection regardless. The old `sparse_*.pkl` cache is keyed by the new `data/` path and will be regenerated.
- No automated tests hit OpenAI/Chroma/MLflow — those stay as the Task 6 manual integration checks to keep the suite fast and offline.
