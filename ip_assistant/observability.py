import os
import time
import json
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - optional dependency handling
    mlflow = None  # type: ignore


def _init_mlflow() -> bool:
    """
      - `MLFLOW_TRACKING_URI` (defaults to file:./mlruns)
      - `MLFLOW_EXPERIMENT`   (defaults to IP-Assistant)
    """
    if mlflow is None:
        return False
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        experiment = os.getenv("MLFLOW_EXPERIMENT", "IP-Assistant")

        # Be resilient to misconfigured HTTP URIs (e.g., localhost inside container)
        if isinstance(tracking_uri, str) and tracking_uri.startswith("http"):
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", os.getenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "5"))
            try:
                mlflow.set_tracking_uri(tracking_uri)
                # Probe connectivity quickly; fall back to local file store if it fails
                MlflowClient().list_experiments(max_results=1)
            except Exception:
                tracking_uri = "file:./mlruns"
                mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment)
        return True
    except Exception:
        return False


class RunLogger:
    """Lightweight wrapper around MLflow for per-request logging.

    Safe to use even if MLflow isn't configured; calls become no-ops.
    """

    def __init__(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        self._enabled = _init_mlflow()
        self._active = False
        self._run = None
        self._tags = tags or {}
        self._run_name = run_name

    def __enter__(self):
        if self._enabled:
            try:
                self._run = mlflow.start_run(run_name=self._run_name)
                if self._tags:
                    mlflow.set_tags(self._tags)
                self._active = True
            except Exception:
                self._enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._enabled and self._active:
            try:
                mlflow.end_run(status="FAILED" if exc else "FINISHED")
            except Exception:
                pass
        self._active = False

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._enabled and self._active:
            try:
                mlflow.log_params({k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in params.items()})
            except Exception:
                pass

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if self._enabled and self._active:
            try:
                mlflow.log_metrics(metrics)
            except Exception:
                pass

    def log_text(self, name: str, text: str) -> None:
        if self._enabled and self._active:
            try:
                mlflow.log_text(text, artifact_file=name)
            except Exception:
                pass

    def log_json(self, name: str, data: Any) -> None:
        if self._enabled and self._active:
            try:
                mlflow.log_text(json.dumps(data, ensure_ascii=False, indent=2), artifact_file=name)
            except Exception:
                pass

    @contextmanager
    def timeit(self, metric_name: str):
        """Context manager to time a block and log `<metric_name>_ms`."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.log_metrics({f"{metric_name}_ms": elapsed_ms})


def summarize_chunks(chunks: Iterable[Dict[str, Any]], limit: int = 10) -> Dict[str, Any]:
    """Create a compact serializable summary for retrieved chunks."""
    out = []
    count = 0
    for c in chunks or []:
        count += 1
        if len(out) < limit:
            out.append({
                "publication_number": c.get("publication_number"),
                "section": c.get("section"),
                "decision": c.get("decision"),
                "score": float(c.get("score", 0.0) or 0.0),
                "text_preview": (c.get("text", "") or "")[:300],
            })
    return {"total": count, "samples": out}
