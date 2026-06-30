"""Compare retrieval strategies (dense / BM25 / hybrid) with the MLflow eval.

    uv run python compare_retrievers.py        # full eval set per config
    uv run python compare_retrievers.py 2      # smoke-test: first 2 questions

Each config runs as its own MLflow run (tagged `retrieval_config`); results print side
by side.
"""

import sys

import mlflow

from support_rag.config import EXPERIMENT
from support_rag.evaluation import compare_configs
from support_rag.retrieval import HybridRetriever
from support_rag.tracing import setup_tracing

CONFIGS = {
    "dense": lambda: HybridRetriever(mode="dense", k=4),
    "sparse_bm25": lambda: HybridRetriever(mode="sparse", sparse_model="bm25", k=4),
    "hybrid_bm25": lambda: HybridRetriever(mode="hybrid", sparse_model="bm25", k=4),
}

REPORT_METRICS = [
    "correctness/mean",
    "retrieval_relevance/mean",
    "retrieval_groundedness/mean",
    "retrieval_sufficiency/mean",
    "relevance_to_query/mean",
    "support_quality/mean",
    "safety/mean",
]


def _print_table(metrics: dict[str, dict]) -> None:
    name_w = max(len(m) for m in REPORT_METRICS) + 2
    col_w = 14
    header = "metric".ljust(name_w) + "".join(c.rjust(col_w) for c in CONFIGS)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for metric in REPORT_METRICS:
        line = metric.ljust(name_w)
        for name in CONFIGS:
            val = metrics[name].get(metric)
            line += (f"{val:.3f}" if isinstance(val, (int, float)) else "-").rjust(col_w)
        print(line)
    print("=" * len(header))


def main(limit: int | None = None) -> None:
    setup_tracing()
    metrics, run_ids = compare_configs(CONFIGS, limit=limit)
    _print_table(metrics)

    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp:
        print("\nRuns:")
        for name, run_id in run_ids.items():
            if run_id:
                print(
                    f"  {name}: "
                    f"http://127.0.0.1:5000/#/experiments/{exp.experiment_id}/runs/{run_id}"
                )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
