"""Prometheus metrics for model serving."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from app.model_loader import ModelBundle


prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total prediction HTTP requests.",
    ("endpoint", "model_version"),
)
prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total failed prediction HTTP requests.",
    ("endpoint", "model_version", "status"),
)
prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "End-to-end prediction request latency in seconds.",
    ("endpoint", "model_version"),
)
model_version_info = Gauge(
    "model_version_info",
    "Loaded model versions and algorithms.",
    ("model_version", "algorithm"),
)
predictions_by_class_total = Counter(
    "predictions_by_class_total",
    "Total individual predictions by class.",
    ("model_version", "prediction_class"),
)


def register_model_info(bundle: ModelBundle) -> None:
    model_version_info.labels(
        model_version=bundle.version,
        algorithm=bundle.metadata["algorithm"],
    ).set(1)


def record_prediction_metrics(
    *,
    endpoint: str,
    model_version: str | None,
    latency_seconds: float,
    prediction: int | list[int] | None,
    error_status: int | None,
) -> None:
    version = model_version or "unknown"
    prediction_requests_total.labels(
        endpoint=endpoint, model_version=version
    ).inc()
    prediction_latency_seconds.labels(
        endpoint=endpoint, model_version=version
    ).observe(latency_seconds)

    if error_status is not None:
        prediction_errors_total.labels(
            endpoint=endpoint,
            model_version=version,
            status=str(error_status),
        ).inc()
        return

    if prediction is None:
        return
    predictions = prediction if isinstance(prediction, list) else [prediction]
    for predicted_class in predictions:
        predictions_by_class_total.labels(
            model_version=version,
            prediction_class=str(predicted_class),
        ).inc()
