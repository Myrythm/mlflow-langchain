"""Knowledge base built from a hand-authored Markdown corpus.

Each `.md` file in `config.KB_DIR` is one knowledge-base article with a small YAML-style
frontmatter block (`title`, `category`). The body is chunked into passages; the dense index
lives in Chroma (OpenAI embeddings) and the sparse (BM25) vectors are computed with fastembed
and cached to disk. Fusion lives in `retrieval.py`.
"""

import functools
import json

import chromadb
from chromadb.errors import NotFoundError
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
        except NotFoundError:
            pass  # nothing to delete on a first build

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
    return config.DATA_DIR / f"sparse_{model}.json"


def get_sparse_index(model: str, rebuild: bool = False) -> list[dict]:
    """Sparse doc vectors [{id, vec}], cached in memory and on disk (JSON).

    `rebuild=True` drops both caches and re-indexes the current corpus — required after
    editing the knowledge base, or the sparse side keeps serving stale chunk ids.
    """
    if rebuild:
        _sparse_index.pop(model, None)
        _sparse_cache_path(model).unlink(missing_ok=True)

    if model in _sparse_index:
        return _sparse_index[model]

    cache = _sparse_cache_path(model)
    if cache.exists():
        # vec is stored as [[index, value], ...] pairs; keys back to int for scoring.
        _sparse_index[model] = [
            {"id": e["id"], "vec": {int(i): float(v) for i, v in e["vec"]}}
            for e in json.loads(cache.read_text(encoding="utf-8"))
        ]
        return _sparse_index[model]

    docs = build_documents()
    vectors = _get_sparse_model(model).embed([d.page_content for d in docs])
    index = [
        {
            "id": doc.metadata["chunk_id"],
            "vec": dict(zip(vec.indices.tolist(), vec.values.tolist(), strict=True)),
        }
        for doc, vec in zip(docs, vectors, strict=True)
    ]
    _sparse_index[model] = index
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            [{"id": e["id"], "vec": list(e["vec"].items())} for e in index]
        ),
        encoding="utf-8",
    )
    return index


def embed_query_sparse(model: str, query: str) -> dict[int, float]:
    vec = next(iter(_get_sparse_model(model).query_embed([query])))
    return dict(zip(vec.indices.tolist(), vec.values.tolist(), strict=True))
