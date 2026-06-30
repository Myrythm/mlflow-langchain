"""The RAG generation chain: retriever -> support prompt -> LLM."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from . import config
from .retrieval import HybridRetriever

_llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)
_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", config.SYSTEM_PROMPT + "\n\nContext:\n{context}"),
        ("human", "{question}"),
    ]
)


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_answer_fn(retriever):
    """Build an `answer(question) -> str` function bound to the given retriever."""
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _prompt
        | _llm
        | StrOutputParser()
    )

    def answer(question: str) -> str:
        return chain.invoke(question)

    return answer


def build_agent(retriever):
    """Build `agent(question) -> (answer, source_docs)` — retrieves once, then generates.

    Returns the retrieved Documents alongside the answer so a UI can show sources.
    """
    generate = _prompt | _llm | StrOutputParser()

    def agent(question: str):
        docs = retriever.invoke(question)
        answer = generate.invoke(
            {"context": _format_docs(docs), "question": question}
        )
        return answer, docs

    return agent


def default_retriever() -> HybridRetriever:
    """The default retrieval config (hybrid + BM25)."""
    return HybridRetriever(
        mode=config.DEFAULT_MODE,
        sparse_model=config.DEFAULT_SPARSE,
        k=config.DEFAULT_K,
    )


def get_default_answer():
    """The default support agent: hybrid retrieval + the generation chain."""
    return build_answer_fn(default_retriever())
