"""Tests for the sparse (BM25) index cache: JSON format and rebuild behavior."""

import json

import pytest

from support_rag import config, knowledge_base


class _FakeArray(list):
    def tolist(self):
        return list(self)


class _FakeSparseVector:
    def __init__(self, indices, values):
        self.indices = _FakeArray(indices)
        self.values = _FakeArray(values)


class _FakeSparseModel:
    """Deterministic stand-in for fastembed's SparseTextEmbedding."""

    def embed(self, texts):
        return [_FakeSparseVector([i, i + 1], [1.0, 0.5]) for i in range(len(texts))]

    def query_embed(self, queries):
        return [_FakeSparseVector([0], [1.0]) for _ in queries]


def _setup_kb(tmp_path, monkeypatch, articles):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True)
    for name, body in articles.items():
        (kb_dir / f"{name}.md").write_text(
            f"---\ntitle: {name}\ncategory: test\n---\n{body}", encoding="utf-8"
        )
    monkeypatch.setattr(config, "KB_DIR", kb_dir)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setitem(knowledge_base._sparse_models, "bm25", _FakeSparseModel())
    knowledge_base.build_documents.cache_clear()
    knowledge_base._doc_map.cache_clear()
    knowledge_base._sparse_index.clear()


def test_sparse_cache_is_json_not_pickle(tmp_path, monkeypatch):
    _setup_kb(tmp_path, monkeypatch, {"pricing": "The Pro plan costs $12."})

    knowledge_base.get_sparse_index("bm25")

    cache_files = list((tmp_path / "data").glob("sparse_bm25*"))
    assert len(cache_files) == 1
    assert cache_files[0].suffix == ".json"
    json.loads(cache_files[0].read_text(encoding="utf-8"))  # must parse as JSON


def test_sparse_index_loaded_from_disk_has_int_keys(tmp_path, monkeypatch):
    _setup_kb(tmp_path, monkeypatch, {"pricing": "The Pro plan costs $12."})

    knowledge_base.get_sparse_index("bm25")  # build + write cache
    knowledge_base._sparse_index.clear()  # force reload from disk
    index = knowledge_base.get_sparse_index("bm25")

    entry = index[0]
    assert entry["id"] == "pricing#chunk0"
    assert all(isinstance(k, int) for k in entry["vec"])
    assert all(isinstance(v, float) for v in entry["vec"].values())


def test_get_sparse_index_rebuild_ignores_stale_cache(tmp_path, monkeypatch):
    _setup_kb(tmp_path, monkeypatch, {"old-article": "Old corpus content."})
    knowledge_base.get_sparse_index("bm25")  # cache now holds the old corpus

    # The corpus changes (as after editing knowledge-base/*.md)...
    _setup_kb(tmp_path / "v2", monkeypatch, {"new-article": "New corpus content."})
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")  # same cache location

    # ...and rebuild=True must re-index instead of returning the stale cache.
    index = knowledge_base.get_sparse_index("bm25", rebuild=True)

    assert [e["id"] for e in index] == ["new-article#chunk0"]


class _FakeCollection:
    name = "nimbus_support"

    def count(self):
        return 0


def test_build_kb_rebuilds_sparse_index(monkeypatch):
    import build_kb

    seen = {}

    def fake_sparse(model, rebuild=False):
        seen[model] = rebuild
        return []

    monkeypatch.setattr(build_kb, "build_documents", lambda: [])
    monkeypatch.setattr(
        build_kb, "get_dense_collection", lambda rebuild=False: _FakeCollection()
    )
    monkeypatch.setattr(build_kb, "get_sparse_index", fake_sparse)

    build_kb.main()

    assert seen == {"bm25": True}


def test_rebuild_propagates_unexpected_delete_errors(monkeypatch):
    class _BrokenClient:
        def delete_collection(self, name):
            raise RuntimeError("connection lost")

        def list_collections(self):
            return []

        def create_collection(self, name, embedding_function=None):
            return _FakeCollection()

    monkeypatch.setattr(knowledge_base, "_get_client", lambda: _BrokenClient())

    with pytest.raises(RuntimeError, match="connection lost"):
        knowledge_base.get_dense_collection(rebuild=True)
