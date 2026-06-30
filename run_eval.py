"""Evaluate the default (hybrid) support agent with MLflow's GenAI eval suite.

    uv run python run_eval.py        # full eval set
    uv run python run_eval.py 2      # smoke-test on the first 2 rows
"""

import sys

import mlflow

from support_rag.config import EXPERIMENT
from support_rag.eval_dataset import EVAL_DATASET
from support_rag.evaluation import evaluate_config
from support_rag.generation import default_retriever
from support_rag.tracing import setup_tracing


def main(limit: int | None = None) -> None:
    setup_tracing()
    data = EVAL_DATASET[:limit] if limit else EVAL_DATASET
    print(f"Evaluating {len(data)} example(s) on the default (hybrid) agent...")

    result = evaluate_config(default_retriever(), data=data, tag="default")

    print("\n=== Aggregate metrics ===")
    for name, value in sorted((result.metrics or {}).items()):
        print(f"  {name}: {value}")

    run_id = getattr(result, "run_id", None)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if run_id and exp:
        print(
            f"\nEval run: http://127.0.0.1:5000/#/experiments/{exp.experiment_id}/runs/{run_id}"
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
