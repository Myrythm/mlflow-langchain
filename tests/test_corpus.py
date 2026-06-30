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
