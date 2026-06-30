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
