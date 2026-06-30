"""Configurable retriever: dense / sparse / hybrid.

- dense  -> Chroma (OpenAI embeddings) similarity search
- sparse -> fastembed BM25 scored in-process (dot product over sparse vectors)
- hybrid -> Reciprocal Rank Fusion of the dense and sparse rankings

A `langchain_core` BaseRetriever subclass, so `mlflow.langchain.autolog()` traces it as a
RETRIEVER span and MLflow's retrieval scorers work unchanged.
"""

from typing import Literal

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from . import config
from .knowledge_base import (
    doc_by_id,
    embed_query_sparse,
    get_dense_collection,
    get_sparse_index,
)

Mode = Literal["dense", "sparse", "hybrid"]


class HybridRetriever(BaseRetriever):
    mode: Mode = config.DEFAULT_MODE
    sparse_model: str = config.DEFAULT_SPARSE
    k: int = config.DEFAULT_K
    candidate_limit: int = config.CANDIDATE_LIMIT
    weights: tuple[float, float] = config.HYBRID_WEIGHTS

    def _dense_ranking(self, query: str) -> list[str]:
        result = get_dense_collection().query(
            query_texts=[query], n_results=self.candidate_limit
        )
        return result["ids"][0]

    def _sparse_ranking(self, query: str) -> list[str]:
        q_vec = embed_query_sparse(self.sparse_model, query)
        scored = [
            (
                entry["id"],
                sum(val * entry["vec"].get(idx, 0.0) for idx, val in q_vec.items()),
            )
            for entry in get_sparse_index(self.sparse_model)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored[: self.candidate_limit]]

    @staticmethod
    def _rrf(rankings: list[list[str]], weights: list[float]) -> list[str]:
        scores: dict[str, float] = {}
        for ranking, weight in zip(rankings, weights):
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (
                    config.RRF_K + rank + 1
                )
        return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        if self.mode == "dense":
            order = self._dense_ranking(query)
        elif self.mode == "sparse":
            order = self._sparse_ranking(query)
        else:
            order = self._rrf(
                [self._dense_ranking(query), self._sparse_ranking(query)],
                list(self.weights),
            )
        return [doc_by_id(doc_id) for doc_id in order[: self.k]]
