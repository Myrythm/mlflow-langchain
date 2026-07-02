"""Tests for HybridRetriever: RRF fusion logic and dense-query clamping."""

from support_rag import retrieval
from support_rag.retrieval import HybridRetriever


def test_rrf_consensus_doc_ranks_first():
    order = HybridRetriever._rrf([["a", "b"], ["a", "c"]], [1.0, 1.0])
    assert order[0] == "a"  # top of both rankings
    assert set(order) == {"a", "b", "c"}  # docs from only one ranking still included


def test_rrf_weights_break_rank_ties():
    assert HybridRetriever._rrf([["a"], ["b"]], [0.7, 0.3])[0] == "a"
    assert HybridRetriever._rrf([["a"], ["b"]], [0.3, 0.7])[0] == "b"


def test_rrf_higher_rank_beats_lower_rank():
    order = HybridRetriever._rrf([["a", "b", "c"], ["b", "a", "c"]], [1.0, 1.0])
    assert order[-1] == "c"  # bottom of both rankings stays last


class _FakeCollection:
    def __init__(self, n):
        self._n = n
        self.requested = None

    def count(self):
        return self._n

    def query(self, query_texts, n_results):
        self.requested = n_results
        return {"ids": [["d#chunk0"]]}


def test_dense_ranking_clamps_n_results_to_collection_size(monkeypatch):
    fake = _FakeCollection(5)
    monkeypatch.setattr(retrieval, "get_dense_collection", lambda: fake)

    HybridRetriever(mode="dense", k=2, candidate_limit=50)._dense_ranking("q")

    assert fake.requested == 5  # not 50 — avoids Chroma's over-fetch warning
