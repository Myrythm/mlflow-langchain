"""MLflow tracing setup, as an explicit call (not an import side effect)."""

import mlflow

from . import config

_configured = False


def setup_tracing() -> None:
    """Point MLflow at the local server, select the experiment, enable LangChain autolog.

    Idempotent — safe to call from every entry point.
    """
    global _configured
    if _configured:
        return
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT)
    mlflow.langchain.autolog()
    _configured = True
