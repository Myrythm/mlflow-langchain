"""MLflow GenAI evaluation: scorer suite + helpers to evaluate / compare configs."""

import mlflow
from mlflow import MlflowClient
from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
    Safety,
)

from . import config
from .eval_dataset import EVAL_DATASET
from .generation import build_answer_fn


def build_scorers() -> list:
    judge = config.JUDGE_MODEL
    return [
        Correctness(model=judge),
        RelevanceToQuery(model=judge),
        RetrievalGroundedness(model=judge),
        RetrievalRelevance(model=judge),
        RetrievalSufficiency(model=judge),
        Guidelines(name="support_quality", guidelines=config.SUPPORT_GUIDELINE, model=judge),
        Safety(model=judge),
    ]


def evaluate_config(retriever, data=None, scorers=None, tag: str | None = None):
    """Run the eval for one retriever; optionally tag the run with a config name."""
    data = EVAL_DATASET if data is None else data
    scorers = build_scorers() if scorers is None else scorers
    result = mlflow.genai.evaluate(
        data=data, predict_fn=build_answer_fn(retriever), scorers=scorers
    )
    run_id = getattr(result, "run_id", None)
    if tag and run_id:
        MlflowClient().set_tag(run_id, "retrieval_config", tag)
    return result


def compare_configs(configs: dict, limit: int | None = None):
    """Evaluate each named config (name -> retriever factory). Returns (metrics, run_ids)."""
    data = EVAL_DATASET[:limit] if limit else EVAL_DATASET
    scorers = build_scorers()
    metrics: dict[str, dict] = {}
    run_ids: dict[str, str | None] = {}
    for name, make_retriever in configs.items():
        print(f"\n=== Evaluating config: {name} ({len(data)} examples) ===")
        result = evaluate_config(make_retriever(), data=data, scorers=scorers, tag=name)
        metrics[name] = result.metrics or {}
        run_ids[name] = getattr(result, "run_id", None)
    return metrics, run_ids
