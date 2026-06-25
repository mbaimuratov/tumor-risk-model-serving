"""FastAPI service for online and batch tumor-risk inference."""

from __future__ import annotations

import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.metrics import record_prediction_metrics, register_model_info
from app.model_loader import (
    ModelArtifactMissingError,
    ModelBundle,
    ModelLoadError,
    ModelRegistry,
)
from app.request_logging import (
    get_request_id,
    log_prediction_request,
    utc_timestamp,
)
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    version = os.getenv("MODEL_VERSION", "v1")
    app.state.model_registry = ModelRegistry(version)
    register_model_info(app.state.model_registry.get())
    yield
    del app.state.model_registry


app = FastAPI(title="Tumor Risk Model API", version="1.0.0", lifespan=lifespan)
PREDICTION_PATHS = {"/predict", "/batch-predict"}


@app.middleware("http")
async def log_prediction_requests(request: Request, call_next):
    if request.url.path not in PREDICTION_PATHS:
        return await call_next(request)

    started_at = perf_counter()
    timestamp = utc_timestamp()
    request_id = get_request_id(request.headers.get("X-Request-ID"))
    registry = getattr(request.app.state, "model_registry", None)
    request.state.model_version = request.query_params.get("model_version") or getattr(
        registry, "default_version", None
    )
    request.state.prediction = None
    request.state.probability = None

    try:
        response = await call_next(request)
    except Exception:
        latency_seconds = perf_counter() - started_at
        log_prediction_request(
            timestamp=timestamp,
            request_id=request_id,
            model_version=request.state.model_version,
            latency_ms=latency_seconds * 1_000,
            prediction=request.state.prediction,
            probability=request.state.probability,
            error_status=500,
        )
        record_prediction_metrics(
            endpoint=request.url.path,
            model_version=request.state.model_version,
            latency_seconds=latency_seconds,
            prediction=request.state.prediction,
            error_status=500,
        )
        raise

    latency_seconds = perf_counter() - started_at
    error_status = response.status_code if response.status_code >= 400 else None
    response.headers["X-Request-ID"] = request_id
    log_prediction_request(
        timestamp=timestamp,
        request_id=request_id,
        model_version=request.state.model_version,
        latency_ms=latency_seconds * 1_000,
        prediction=request.state.prediction,
        probability=request.state.probability,
        error_status=error_status,
    )
    record_prediction_metrics(
        endpoint=request.url.path,
        model_version=request.state.model_version,
        latency_seconds=latency_seconds,
        prediction=request.state.prediction,
        error_status=error_status,
    )
    return response


ModelVersionQuery = Annotated[
    str | None,
    Query(pattern=r"^v[1-9][0-9]*$", description="Override the configured model version."),
]


def get_model_bundle(
    request: Request, model_version: str | None = None
) -> ModelBundle:
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    request.state.model_version = model_version or registry.default_version
    try:
        bundle = registry.get(model_version)
    except ModelArtifactMissingError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    request.state.model_version = bundle.version
    register_model_info(bundle)
    return bundle


def prepare_samples(
    samples: list[PredictionRequest], bundle: ModelBundle
) -> list[list[float]]:
    expected = set(bundle.features)
    prepared: list[list[float]] = []

    for index, sample in enumerate(samples):
        received = set(sample.features)
        if received != expected:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"Feature names do not match for sample {index}.",
                    "missing_features": sorted(expected - received),
                    "unexpected_features": sorted(received - expected),
                },
            )

        values = [sample.features[name] for name in bundle.features]
        if not all(math.isfinite(value) for value in values):
            raise HTTPException(
                status_code=422,
                detail=f"All feature values must be finite for sample {index}.",
            )
        prepared.append(values)

    return prepared


def predict_samples(
    samples: list[PredictionRequest], bundle: ModelBundle
) -> list[PredictionResult]:
    values = prepare_samples(samples, bundle)
    predictions = bundle.model.predict(values)
    probabilities = bundle.model.predict_proba(values)[:, 1]

    return [
        PredictionResult(
            prediction=int(prediction),
            risk="high" if prediction == 1 else "low",
            malignant_probability=float(probability),
        )
        for prediction, probability in zip(predictions, probabilities, strict=True)
    ]


@app.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    bundle = get_model_bundle(request)
    return {"status": "ok", "model_loaded": True, "model_version": bundle.version}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info")
def model_info(
    request: Request, model_version: ModelVersionQuery = None
) -> dict[str, object]:
    bundle = get_model_bundle(request, model_version)
    metadata = bundle.metadata
    return {
        "model_version": bundle.version,
        "algorithm": metadata["algorithm"],
        "features": bundle.features,
        "metrics": {
            "accuracy": metadata["accuracy"],
            "precision": metadata["precision"],
            "recall": metadata["recall"],
            "roc_auc": metadata["roc_auc"],
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
    model_version: ModelVersionQuery = None,
) -> PredictionResponse:
    bundle = get_model_bundle(request, model_version)
    result = predict_samples([payload], bundle)[0]
    request.state.prediction = result.prediction
    request.state.probability = result.malignant_probability
    return PredictionResponse(model_version=bundle.version, **result.model_dump())


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(
    payload: BatchPredictionRequest,
    request: Request,
    model_version: ModelVersionQuery = None,
) -> BatchPredictionResponse:
    bundle = get_model_bundle(request, model_version)
    results = predict_samples(payload.samples, bundle)
    request.state.prediction = [result.prediction for result in results]
    request.state.probability = [
        result.malignant_probability for result in results
    ]
    return BatchPredictionResponse(
        model_version=bundle.version,
        predictions=results,
    )
