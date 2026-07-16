"""Knowledge base built from a hand-authored Markdown corpus.

Each `.md` file in `config.KB_DIR` is one knowledge-base article with a small YAML-style
frontmatter block (`title`, `category`). The body is chunked into passages; the dense index
lives in Chroma (OpenAI embeddings) and the sparse (BM25) vectors are computed with fastembed
and cached to disk. Fusion lives in `retrieval.py`.

Classes
-------
KnowledgeBase
    Parses the Markdown corpus and chunks articles into ``Document`` passages.
DenseIndex
    Manages the Chroma (OpenAI embeddings) dense vector index.
SparseIndex
    Manages the fastembed BM25 sparse vector index with disk caching.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


# =============================================================================
# KnowledgeBase — corpus parsing & chunking
# =============================================================================


class KnowledgeBase:
    """Parses and chunks a directory of Markdown articles into ``Document`` passages.

    Parameters
    ----------
    kb_dir : Path, optional
        Directory containing ``*.md`` knowledge-base articles.
        Defaults to ``config.KB_DIR``.
    chunk_size : int, optional
        Maximum chunk size in characters.  Defaults to ``config.CHUNK_SIZE``.
    chunk_overlap : int, optional
        Overlap between consecutive chunks.  Defaults to ``config.CHUNK_OVERLAP``.
    """

    def __init__(
        self,
        kb_dir: Path | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._kb_dir = kb_dir or config.KB_DIR
        self._chunk_size = chunk_size or config.CHUNK_SIZE
        self._chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self._documents: list[Document] | None = None
        self._doc_map: dict[str, Document] | None = None

    # -- public ---------------------------------------------------------------

    def build_documents(self) -> list[Document]:
        """Chunk every Markdown article in the knowledge-base directory into passages."""
        if self._documents is not None:
            return self._documents

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
        )

        docs: list[Document] = []
        for path in sorted(self._kb_dir.glob("*.md")):
            meta, body = self._parse_frontmatter(path.read_text(encoding="utf-8"))
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

        self._documents = docs
        return docs

    def doc_by_id(self, chunk_id: str) -> Document:
        """Look up a single chunk ``Document`` by its ``chunk_id``."""
        return self._get_doc_map()[chunk_id]

    def invalidate_cache(self) -> None:
        """Drop cached documents so the next call rebuilds from disk."""
        self._documents = None
        self._doc_map = None

    # -- private --------------------------------------------------------------

    def _get_doc_map(self) -> dict[str, Document]:
        if self._doc_map is None:
            self._doc_map = {
                d.metadata["chunk_id"]: d for d in self.build_documents()
            }
        return self._doc_map

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        """Split a leading ``--- ... ---`` frontmatter block from the body.

        Returns ``({key: value}, body)``.  If there is no frontmatter block,
        returns ``({}, text)``.  Values are parsed as plain strings (no nested YAML).
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


# =============================================================================
# DenseIndex — Chroma wrapper
# =============================================================================


class DenseIndex:
    """Manages a Chroma persistent collection for dense (OpenAI embeddings) retrieval.

    Parameters
    ----------
    kb : KnowledgeBase
        The knowledge base supplying chunked passages.
    persist_dir : Path, optional
        Directory for the Chroma persistent store.  Defaults to ``config.PERSIST_DIR``.
    collection_name : str, optional
        Name of the Chroma collection.  Defaults to ``config.DENSE_COLLECTION``.
    model_name : str, optional
        OpenAI embedding model name.  Defaults to ``config.DENSE_MODEL``.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self._kb = kb
        self._persist_dir = persist_dir or config.PERSIST_DIR
        self._collection_name = collection_name or config.DENSE_COLLECTION
        self._model_name = model_name or config.DENSE_MODEL
        self._client: chromadb.PersistentClient | None = None

    # -- public ---------------------------------------------------------------

    def get_collection(self, rebuild: bool = False):
        """Return the Chroma collection, building it if necessary.

        Parameters
        ----------
        rebuild : bool
            If *True*, delete the existing collection and re-index from scratch.
        """
        client = self._get_client()
        if rebuild:
            try:
                client.delete_collection(self._collection_name)
            except NotFoundError:
                pass  # nothing to delete on a first build

        if self._collection_name in {c.name for c in client.list_collections()}:
            return client.get_collection(
                self._collection_name, embedding_function=self._embedding_function()
            )

        collection = client.create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function(),
        )
        docs = self._kb.build_documents()
        batch = 256  # batch the embedding calls
        for start in range(0, len(docs), batch):
            part = docs[start : start + batch]
            collection.add(
                ids=[d.metadata["chunk_id"] for d in part],
                documents=[d.page_content for d in part],
                metadatas=[
                    {k: d.metadata[k] for k in ("doc_id", "title", "category")}
                    for d in part
                ],
            )
        print(
            f"Built dense collection '{collection.name}': "
            f"{collection.count()} passages."
        )
        return collection

    # -- private --------------------------------------------------------------

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        return self._client

    def _embedding_function(self) -> OpenAIEmbeddingFunction:
        return OpenAIEmbeddingFunction(
            api_key_env_var="OPENAI_API_KEY", model_name=self._model_name
        )


# =============================================================================
# SparseIndex — fastembed BM25 wrapper
# =============================================================================


class SparseIndex:
    """Manages fastembed BM25 sparse vectors with disk-based JSON caching.

    Parameters
    ----------
    kb : KnowledgeBase
        The knowledge base supplying chunked passages.
    data_dir : Path, optional
        Directory for the sparse cache files.  Defaults to ``config.DATA_DIR``.
    models_config : dict, optional
        Mapping of model short-names to fastembed model identifiers.
        Defaults to ``config.SPARSE_MODELS``.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        data_dir: Path | None = None,
        models_config: dict[str, str] | None = None,
    ) -> None:
        self._kb = kb
        self._data_dir = data_dir or config.DATA_DIR
        self._models_config = models_config or config.SPARSE_MODELS
        self._models: dict[str, object] = {}
        self._index: dict[str, list[dict]] = {}

    # -- public ---------------------------------------------------------------

    def get_index(self, model: str, rebuild: bool = False) -> list[dict]:
        """Sparse doc vectors ``[{id, vec}]``, cached in memory and on disk (JSON).

        ``rebuild=True`` drops both caches and re-indexes the current corpus — required
        after editing the knowledge base, or the sparse side keeps serving stale chunk
        ids.
        """
        if rebuild:
            self._index.pop(model, None)
            self._cache_path(model).unlink(missing_ok=True)

        if model in self._index:
            return self._index[model]

        cache = self._cache_path(model)
        if cache.exists():
            # vec is stored as [[index, value], ...] pairs; keys back to int for scoring.
            self._index[model] = [
                {"id": e["id"], "vec": {int(i): float(v) for i, v in e["vec"]}}
                for e in json.loads(cache.read_text(encoding="utf-8"))
            ]
            return self._index[model]

        docs = self._kb.build_documents()
        vectors = self._get_model(model).embed([d.page_content for d in docs])
        index = [
            {
                "id": doc.metadata["chunk_id"],
                "vec": dict(
                    zip(vec.indices.tolist(), vec.values.tolist(), strict=True)
                ),
            }
            for doc, vec in zip(docs, vectors, strict=True)
        ]
        self._index[model] = index
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                [{"id": e["id"], "vec": list(e["vec"].items())} for e in index]
            ),
            encoding="utf-8",
        )
        return index

    def embed_query(self, model: str, query: str) -> dict[int, float]:
        """Embed a single query string into a sparse vector."""
        vec = next(iter(self._get_model(model).query_embed([query])))
        return dict(zip(vec.indices.tolist(), vec.values.tolist(), strict=True))

    # -- private --------------------------------------------------------------

    def _get_model(self, model: str):
        if model not in self._models_config:
            raise ValueError(
                f"sparse model must be one of {list(self._models_config)}"
            )
        if model not in self._models:
            from fastembed import SparseTextEmbedding

            self._models[model] = SparseTextEmbedding(
                model_name=self._models_config[model]
            )
        return self._models[model]

    def _cache_path(self, model: str) -> Path:
        return self._data_dir / f"sparse_{model}.json"


# =============================================================================
# Backward-compatible module-level API
# =============================================================================
#
# These shim functions delegate to lazily-created singleton instances so that
# existing consumers (retrieval.py, build_kb.py, tests) keep working without
# any changes.

_default_kb: KnowledgeBase | None = None
_default_dense: DenseIndex | None = None
_default_sparse: SparseIndex | None = None


def _get_default_kb() -> KnowledgeBase:
    global _default_kb
    if _default_kb is None:
        _default_kb = KnowledgeBase()
    return _default_kb


def _get_default_dense() -> DenseIndex:
    global _default_dense
    if _default_dense is None:
        _default_dense = DenseIndex(_get_default_kb())
    return _default_dense


def _get_default_sparse() -> SparseIndex:
    global _default_sparse
    if _default_sparse is None:
        _default_sparse = SparseIndex(_get_default_kb())
    return _default_sparse


# --- shim: KnowledgeBase -----------------------------------------------------


def _reset_singletons() -> None:
    """Destroy all singleton instances so the next access re-creates them
    with whatever config values are current.  This is critical for tests that
    monkeypatch ``config.KB_DIR``, ``config.DATA_DIR``, etc."""
    global _default_kb, _default_dense, _default_sparse
    _default_kb = None
    _default_dense = None
    _default_sparse = None


class _BuildDocumentsShim:
    """Wrapper that makes ``build_documents`` callable *and* exposes
    ``.cache_clear()`` so existing tests that call
    ``knowledge_base.build_documents.cache_clear()`` keep working."""

    def __call__(self) -> list[Document]:
        return _get_default_kb().build_documents()

    def cache_clear(self) -> None:
        _reset_singletons()


build_documents = _BuildDocumentsShim()


class _DocMapShim:
    """Exposes ``.cache_clear()`` for ``knowledge_base._doc_map.cache_clear()``."""

    def __call__(self) -> dict[str, Document]:
        return _get_default_kb()._get_doc_map()

    def cache_clear(self) -> None:
        _reset_singletons()


_doc_map = _DocMapShim()


def doc_by_id(chunk_id: str) -> Document:
    return _get_default_kb().doc_by_id(chunk_id)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    return KnowledgeBase._parse_frontmatter(text)


# --- shim: DenseIndex ---------------------------------------------------------


def _get_client():
    return _get_default_dense()._get_client()


def get_dense_collection(rebuild: bool = False):
    return _get_default_dense().get_collection(rebuild)


# --- shim: SparseIndex --------------------------------------------------------

# Expose the internal dicts so tests that do
#   ``monkeypatch.setitem(knowledge_base._sparse_models, ...)``
# and ``knowledge_base._sparse_index.clear()`` still work.

_sparse_models: dict[str, object] = {}
_sparse_index: dict[str, list[dict]] = {}


def _sync_sparse_state() -> None:
    """Push module-level ``_sparse_models`` / ``_sparse_index`` into the default
    ``SparseIndex`` instance so monkey-patching at the module level propagates."""
    si = _get_default_sparse()
    si._models = _sparse_models
    si._index = _sparse_index


def get_sparse_index(model: str, rebuild: bool = False) -> list[dict]:
    _sync_sparse_state()
    result = _get_default_sparse().get_index(model, rebuild)
    # Keep module-level dicts in sync after potential mutations.
    _sparse_index.update(_get_default_sparse()._index)
    return result


def embed_query_sparse(model: str, query: str) -> dict[int, float]:
    _sync_sparse_state()
    return _get_default_sparse().embed_query(model, query)
