"""Build the knowledge base from the markdown corpus: chunk docs, build the dense Chroma index and the sparse (BM25) index, and build the eval set.

    uv run python build_kb.py
"""

from support_rag.config import SPARSE_MODELS
from support_rag.knowledge_base import build_documents, get_dense_collection, get_sparse_index


def main() -> None:
    passages = build_documents()
    print(f"Chunked corpus: {len(passages)} passages")

    collection = get_dense_collection(rebuild=True)
    print(f"{collection.name}: {collection.count()} passages (dense)")

    for model in SPARSE_MODELS:
        index = get_sparse_index(model)
        print(f"sparse '{model}': indexed {len(index)} passages")

    # Build + cache the eval set (import triggers load_or_build_eval()).
    from support_rag.eval_dataset import EVAL_DATASET

    print(f"eval set: {len(EVAL_DATASET)} grounded examples")


if __name__ == "__main__":
    main()
