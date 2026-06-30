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
