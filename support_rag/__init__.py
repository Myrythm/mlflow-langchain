"""Customer-support RAG package: hybrid retrieval (dense + sparse) on Chroma, with
MLflow tracing and evaluation."""

from .generation import build_answer_fn, default_retriever, get_default_answer
from .retrieval import HybridRetriever
from .tracing import setup_tracing

__all__ = [
    "setup_tracing",
    "HybridRetriever",
    "build_answer_fn",
    "default_retriever",
    "get_default_answer",
]
