"""Pydantic request/response schemas for the support-RAG API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from support_rag import config

K_MIN = 1
K_MAX = 8


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)
    mode: Literal["dense", "sparse", "hybrid"] = config.DEFAULT_MODE
    sparse_model: str = config.DEFAULT_SPARSE
    k: int = Field(default=config.DEFAULT_K, ge=K_MIN, le=K_MAX)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    @field_validator("sparse_model")
    @classmethod
    def _sparse_model_known(cls, value: str) -> str:
        if value not in config.SPARSE_MODELS:
            raise ValueError(
                f"unknown sparse model {value!r}; expected one of "
                f"{sorted(config.SPARSE_MODELS)}"
            )
        return value


class SourceDoc(BaseModel):
    title: str
    category: str
    snippet: str


class ConfigResponse(BaseModel):
    modes: list[str]
    sparse_models: list[str]
    k_min: int
    k_max: int
    default_mode: str
    default_sparse: str
    default_k: int
    example_questions: list[str]


class HealthResponse(BaseModel):
    status: str
    mlflow_reachable: bool
