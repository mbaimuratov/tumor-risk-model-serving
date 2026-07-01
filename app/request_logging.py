"""Structured, input-safe logging for prediction requests."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import TypeAlias
from uuid import uuid4


PredictionValue: TypeAlias = int | list[int] | None
ProbabilityValue: TypeAlias = float | list[float] | None
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")

prediction_logger = logging.getLogger("tumor_risk.predictions")
prediction_logger.setLevel(logging.INFO)
prediction_logger.propagate = False
if not prediction_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    prediction_logger.addHandler(handler)

shadow_logger = logging.getLogger("tumor_risk.shadow_predictions")
shadow_logger.setLevel(logging.INFO)
shadow_logger.propagate = False
if not shadow_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    shadow_logger.addHandler(handler)


def get_request_id(header_value: str | None) -> str:
    if header_value and REQUEST_ID_PATTERN.fullmatch(header_value):
        return header_value
    return str(uuid4())


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_prediction_request(
    *,
    timestamp: str,
    request_id: str,
    model_version: str | None,
    latency_ms: float,
    prediction: PredictionValue,
    probability: ProbabilityValue,
    error_status: int | None,
) -> None:
    event = {
        "timestamp": timestamp,
        "request_id": request_id,
        "model_version": model_version,
        "latency_ms": round(latency_ms, 3),
        "prediction": prediction,
        "probability": probability,
        "error_status": error_status,
    }
    prediction_logger.info(json.dumps(event, separators=(",", ":")))


def log_shadow_prediction(
    *,
    timestamp: str,
    request_id: str,
    champion_model_version: str,
    candidate_model_version: str | None,
    champion_prediction: int,
    candidate_prediction: int | None,
    champion_probability: float,
    candidate_probability: float | None,
    disagreement: bool | None,
    probability_delta: float | None,
    latency_champion_ms: float,
    latency_candidate_ms: float | None,
    candidate_error: str | None = None,
) -> None:
    event = {
        "timestamp": timestamp,
        "request_id": request_id,
        "champion_model_version": champion_model_version,
        "candidate_model_version": candidate_model_version,
        "champion_prediction": champion_prediction,
        "candidate_prediction": candidate_prediction,
        "champion_probability": champion_probability,
        "candidate_probability": candidate_probability,
        "disagreement": disagreement,
        "probability_delta": probability_delta,
        "latency_champion": round(latency_champion_ms, 3),
        "latency_candidate": (
            round(latency_candidate_ms, 3)
            if latency_candidate_ms is not None
            else None
        ),
        "candidate_error": candidate_error,
    }
    shadow_logger.info(json.dumps(event, separators=(",", ":")))
