# Nimbus Support KB — Design Spec

**Date:** 2026-06-30
**Status:** Approved (pending final review)
**Goal:** Replace the doc2dial corpus with a hand-authored Markdown knowledge base for a
fictional SaaS product ("Nimbus"), to learn the end-to-end "build a KB from scratch" pipeline.

## 1. Motivation & Goal

The existing `support_rag` package is a customer-support RAG agent whose knowledge base is the
IBM **doc2dial** corpus (US public-service helpdesk docs). doc2dial is foreign-domain and its
loader is bound to a downloaded zip + grounded-dialogue eval set.

The goal here is **learning the KB-from-scratch pipeline** with data we fully control: author
our own Markdown docs, ingest → chunk → index → retrieve → generate → evaluate. The product
domain (customer support) is kept because the surrounding scaffolding (system prompt, support
guideline, MLflow scorers) is already tuned for it.

**Non-goals:** keeping doc2dial as a parallel/comparison source; multi-source ingestion;
SPLADE; LLM-generated eval data.

## 2. Key Decisions

| Decision | Choice |
|---|---|
| Position vs. doc2dial | **Replace entirely.** Remove all doc2dial code paths. |
| Retrieval | **Hybrid (dense + BM25 via RRF).** Drop SPLADE. |
| Dense model | Keep OpenAI `text-embedding-3-small` on Chroma. |
| Content domain | Fictional SaaS PM tool **"Nimbus"** — support/FAQ docs. |
| Eval set | **Hand-written ~12–15 Q&A** with expected facts + source doc. |
| Content quality | Must read like **real-world** SaaS help-center docs (see §6). |
| Authoring | Delegated to a **Sonnet 4.6 subagent**, fed a canonical fact sheet. |

## 3. Architecture & Code Changes

Pipeline shape is unchanged (load → chunk → index dense+sparse → retrieve → generate → eval).
Only the **data source** changes and **SPLADE is removed**.

| File | Change |
|---|---|
| `support_rag/knowledge_base.py` | **Rewrite the loader.** Remove all doc2dial code (zip download, JSON parse, `load_dialogues`, `_read_member`, `_ensure_zip`). Add a Markdown loader: read `knowledge-base/*.md`, parse YAML frontmatter, chunk the body. The Chroma dense index and the fastembed sparse index logic stay — they just point at the new documents. |
| `support_rag/config.py` | Remove `DOC2DIAL_URL`, `DOC2DIAL_DIR`, `DOC2DIAL_DOMAINS`. Add `KB_DIR` (the `knowledge-base/` folder). `SPARSE_MODELS = {"bm25": "Qdrant/bm25"}` only. `DEFAULT_SPARSE = "bm25"`. Rename `DENSE_COLLECTION` → `"nimbus_support"`. Keep prompts product-agnostic ("customer-support assistant"). |
| `support_rag/eval_dataset.py` | **Rewrite.** Remove derivation from doc2dial validation dialogues. Replace with a hand-written list of Q&A entries (see §7). |
| `build_kb.py` | Remove doc2dial download step. Flow becomes: load md → build dense collection → build BM25 index → build eval set. |
| `support_rag/retrieval.py` | Default sparse model `bm25`. Hybrid RRF logic unchanged. |
| `support_rag/ui.py` | Retrieval-strategy selector: remove the SPLADE option (keep dense / bm25 / hybrid). |
| `support_rag/generation.py`, `support_rag/evaluation.py` | Effectively unchanged (config-driven). Verify no lingering doc2dial references. |

### Markdown document format

Each `.md` file carries YAML frontmatter for metadata:

```markdown
---
title: Billing & Pricing
category: billing
---
# Billing & Pricing

Normal markdown body...
```

- `doc_id` = file slug (filename without extension).
- `category` (from frontmatter) replaces the old `domain` metadata field.
- `title` from frontmatter; fall back to the slug if missing.
- `chunk_id` = `f"{doc_id}#chunk{i}"`.
- Chunking: keep `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`.
- Frontmatter is stripped before chunking (metadata only, not indexed text).

## 4. Document Set (~8 files in `knowledge-base/`)

Chosen for topic variety so retrieval is exercised across distinct areas.

| File | Contents |
|---|---|
| `getting-started.md` | Onboarding; creating the first workspace/board. |
| `pricing-and-billing.md` | Plans (Free / Pro / Enterprise), billing cycle, changing plans. |
| `refunds-and-cancellation.md` | Refund policy, how to cancel, downgrade. |
| `integrations.md` | Slack, GitHub, Google Drive, etc. + how to connect. |
| `security-and-sso.md` | SSO/SAML, 2FA, encryption, data residency. |
| `account-and-teams.md` | Inviting members, roles/permissions, transferring ownership. |
| `features-overview.md` | Boards, timeline, automation, reporting. |
| `troubleshooting.md` | Login failures, sync errors, missing notifications. |

## 5. Build & Verify Flow (executed later)

1. Sonnet 4.6 subagent authors the 8 `.md` files + a draft eval set, all drawing from one
   canonical Nimbus fact sheet (see §6).
2. User reviews the authored content.
3. Implement the code changes (loader, config, eval, build script, UI).
4. Restart MLflow (`uv run mlflow server …`), then `uv run python build_kb.py` →
   `uv run python run_eval.py` as a sanity check.

## 6. Content Quality Requirements (real-world fidelity)

The fiction must read like genuine SaaS help-center documentation.

- **Plausible numbers & policies:** concrete pricing (e.g. Free / Pro $12 per user/month /
  Enterprise custom), real billing cycles, a 14-day refund window — **no** `{{placeholder}}`
  tokens.
- **Cross-document consistency:** plan names, roles, and feature names referenced in
  `pricing-and-billing` must match exactly when they reappear in `refunds-and-cancellation`
  or `account-and-teams`. No contradictions.
- **Technically sensible detail:** integration steps (Slack/GitHub) must describe a believable
  flow; security terms used correctly (IdP, SAML, SCIM, etc.).
- **Help-center voice:** clear headings, numbered steps, professional support tone — modeled
  on Linear/Notion/Asana docs.
- **Mechanism:** the subagent is given a **canonical Nimbus fact sheet** (plans, prices, roles,
  integrations, limits) first. Every document — and the eval set — draws facts from this sheet,
  guaranteeing consistency and realism.

## 7. Eval Set (~12–15 hand-written Q&A)

Each entry contains:

- `question` — a realistic support question.
- `expected_facts` — the key points a correct answer must contain (grounded in the fact sheet).
- `source_doc` — the `doc_id` the answer is grounded in.

Spread across ~6–8 documents so retrieval is tested across topics. The entry shape matches
what `evaluation.py` and the MLflow scorers already consume, so `run_eval.py` and
`compare_retrievers.py` run unchanged. The README results table is regenerated after the
first full eval run (the old doc2dial numbers no longer apply).

## 8. Out of Scope

- Keeping doc2dial available for comparison.
- SPLADE or any sparse model other than BM25.
- Multi-source / mixed-corpus ingestion.
- Automated (LLM-generated) eval data.
