import json
import logging
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app
import app.model_loader as model_loader
from app.model_loader import ModelLoadError, load_model
from app.request_logging import prediction_logger


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MODEL_VERSION", "v1")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample() -> dict[str, dict[str, float]]:
    dataset = load_breast_cancer()
    return {
        "features": dict(
            zip(dataset.feature_names.tolist(), dataset.data[0].tolist(), strict=True)
        )
    }


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "v1",
    }


def test_model_info_exposes_contract_and_metrics(client: TestClient) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "v1"
    assert body["algorithm"] == "LogisticRegression"
    assert len(body["features"]) == 30
    assert set(body["metrics"]) == {"accuracy", "precision", "recall", "roc_auc"}


def test_predict_returns_one_prediction(
    client: TestClient, sample: dict[str, dict[str, float]]
) -> None:
    response = client.post("/predict", json=sample)

    assert response.status_code == 200
    assert response.json()["prediction"] in {0, 1}
    assert 0 <= response.json()["malignant_probability"] <= 1


def test_predict_can_override_model_version(
    client: TestClient, sample: dict[str, dict[str, float]]
) -> None:
    response = client.post("/predict?model_version=v2", json=sample)

    assert response.status_code == 200
    assert response.json()["model_version"] == "v2"


def test_batch_predict_returns_all_predictions(
    client: TestClient, sample: dict[str, dict[str, float]]
) -> None:
    response = client.post("/batch-predict", json={"samples": [sample, sample]})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_predict_rejects_feature_mismatch(
    client: TestClient, sample: dict[str, dict[str, float]]
) -> None:
    sample["features"].pop("mean radius")

    response = client.post("/predict", json=sample)

    assert response.status_code == 422
    assert response.json()["detail"]["missing_features"] == ["mean radius"]


def test_prediction_log_contains_operational_fields_only(
    client: TestClient,
    sample: dict[str, dict[str, float]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prediction_logger, "propagate", True)
    caplog.set_level(logging.INFO, logger=prediction_logger.name)

    response = client.post(
        "/predict?model_version=v2",
        json=sample,
        headers={"X-Request-ID": "test-request-123"},
    )

    event = json.loads(caplog.records[-1].message)
    assert response.headers["X-Request-ID"] == "test-request-123"
    assert set(event) == {
        "timestamp",
        "request_id",
        "model_version",
        "latency_ms",
        "prediction",
        "probability",
        "error_status",
    }
    assert event["request_id"] == "test-request-123"
    assert event["model_version"] == "v2"
    assert event["error_status"] is None
    assert "features" not in event


def test_failed_prediction_is_logged_without_input(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prediction_logger, "propagate", True)
    caplog.set_level(logging.INFO, logger=prediction_logger.name)

    response = client.post("/predict", json={"features": {}})

    event = json.loads(caplog.records[-1].message)
    assert response.status_code == 422
    assert event["prediction"] is None
    assert event["probability"] is None
    assert event["error_status"] == 422


def test_prometheus_metrics_cover_predictions_and_errors(
    client: TestClient, sample: dict[str, dict[str, float]]
) -> None:
    successful_response = client.post("/predict?model_version=v2", json=sample)
    error_response = client.post("/predict", json={"features": {}})

    response = client.get("/metrics")

    assert successful_response.status_code == 200
    assert error_response.status_code == 422
    assert response.status_code == 200
    assert "prediction_requests_total" in response.text
    assert "prediction_errors_total" in response.text
    assert "prediction_latency_seconds" in response.text
    assert 'model_version_info{algorithm="RandomForestClassifier",model_version="v2"} 1.0' in response.text
    assert 'predictions_by_class_total{model_version="v2"' in response.text


def test_loader_fails_clearly_when_artifact_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="missing.*model.joblib"):
        load_model("v1", models_root=tmp_path)


def test_model_version_config_selects_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_VERSION", "v2")

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_version"] == "v2"


def test_model_uri_config_loads_mlflow_model(
    monkeypatch: pytest.MonkeyPatch,
    sample: dict[str, dict[str, float]],
) -> None:
    features = list(sample["features"])

    class FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str):
            assert name == "TumorRiskClassifier"
            assert alias == "champion"
            return SimpleNamespace(version="7", run_id="run-123")

        def get_run(self, run_id: str):
            assert run_id == "run-123"
            return SimpleNamespace(
                data=SimpleNamespace(
                    params={"model_type": "LogisticRegression"},
                    metrics={
                        "accuracy": 0.98,
                        "precision": 0.97,
                        "recall": 0.96,
                        "roc_auc": 0.99,
                    },
                )
            )

    class FakeSignatureInput:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeMlflowModel:
        metadata = {"algorithm": "LogisticRegression"}
        signature = SimpleNamespace(
            inputs=SimpleNamespace(
                inputs=[FakeSignatureInput(feature) for feature in features]
            )
        )

    class FakeSklearnModel:
        def predict(self, frame):
            return np.ones(len(frame), dtype=int)

        def predict_proba(self, frame):
            return np.array([[0.1, 0.9]] * len(frame))

    monkeypatch.setenv("MODEL_URI", "models:/TumorRiskClassifier@champion")
    monkeypatch.setattr(model_loader, "MlflowClient", FakeClient)
    monkeypatch.setattr(model_loader.Model, "load", lambda uri: FakeMlflowModel())
    monkeypatch.setattr(
        model_loader.mlflow.sklearn,
        "load_model",
        lambda uri: FakeSklearnModel(),
    )

    with TestClient(app) as test_client:
        health_response = test_client.get("/health")
        info_response = test_client.get("/model-info")
        predict_response = test_client.post("/predict", json=sample)

    assert health_response.status_code == 200
    assert health_response.json()["model_version"] == "champion"
    assert info_response.status_code == 200
    assert info_response.json()["metrics"]["recall"] == 0.96
    assert predict_response.status_code == 200
    assert predict_response.json()["model_version"] == "champion"
