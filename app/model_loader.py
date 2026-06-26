"""Versioned model loading and serving-contract validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import mlflow.sklearn
from mlflow.models import Model
from mlflow.tracking import MlflowClient


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPOSITORY_ROOT / "models"
VERSION_PATTERN = re.compile(r"v[1-9][0-9]*")
MLFLOW_MODEL_URI_PATTERN = re.compile(r"^models:/([^/@]+)(?:@([^/]+)|/([^/]+))$")
REQUIRED_METADATA_FIELDS = {
    "model_version",
    "training_date",
    "algorithm",
    "features",
    "accuracy",
    "precision",
    "recall",
    "roc_auc",
}


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded safely for serving."""


class ModelArtifactMissingError(ModelLoadError):
    """Raised when a selected model version has missing files."""


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    @property
    def features(self) -> list[str]:
        return self.metadata["features"]

    @property
    def source(self) -> str:
        return self.metadata.get("model_source", "local")


def load_model(
    version: str, models_root: Path = MODELS_ROOT
) -> ModelBundle:
    if not VERSION_PATTERN.fullmatch(version):
        raise ModelLoadError(
            f"Invalid model version {version!r}; expected a value such as 'v1'."
        )

    version_directory = models_root / version
    model_path = version_directory / "model.joblib"
    metadata_path = version_directory / "metadata.json"

    missing = [path for path in (model_path, metadata_path) if not path.is_file()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise ModelArtifactMissingError(
            f"Model version {version!r} is incomplete; missing: {missing_paths}"
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"Could not read metadata at {metadata_path}: {exc}") from exc

    missing_fields = sorted(REQUIRED_METADATA_FIELDS - metadata.keys())
    if missing_fields:
        raise ModelLoadError(
            f"Metadata for {version!r} is missing fields: {', '.join(missing_fields)}"
        )
    if metadata["model_version"] != version:
        raise ModelLoadError(
            f"Selected version {version!r} does not match metadata version "
            f"{metadata['model_version']!r}."
        )

    features = metadata["features"]
    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(name, str) and name for name in features)
        or len(features) != len(set(features))
    ):
        raise ModelLoadError("Metadata features must be a non-empty list of unique names.")

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ModelLoadError(f"Could not load model artifact at {model_path}: {exc}") from exc

    artifact_features = getattr(model, "serving_feature_names_", None)
    if artifact_features is None:
        raise ModelLoadError(
            "Model artifact has no serving feature contract; retrain it with the current "
            "training script."
        )
    if list(artifact_features) != features:
        raise ModelLoadError(
            "Feature names in model.joblib do not match metadata.json, including order."
        )
    if getattr(model, "n_features_in_", len(features)) != len(features):
        raise ModelLoadError("Model feature count does not match metadata.json.")
    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise ModelLoadError("Model must implement predict() and predict_proba().")

    return ModelBundle(model=model, metadata=metadata)


def parse_registered_model_uri(model_uri: str) -> tuple[str, str | None, str | None]:
    match = MLFLOW_MODEL_URI_PATTERN.fullmatch(model_uri)
    if not match:
        raise ModelLoadError(
            "MODEL_URI must use a registered MLflow model URI such as "
            "'models:/TumorRiskClassifier@champion' or 'models:/TumorRiskClassifier/2'."
        )
    return match.group(1), match.group(2), match.group(3)


def registered_model_metadata(model_uri: str) -> dict[str, Any]:
    model_name, alias, version = parse_registered_model_uri(model_uri)
    client = MlflowClient()
    try:
        model_version = (
            client.get_model_version_by_alias(model_name, alias)
            if alias is not None
            else client.get_model_version(model_name, version)
        )
    except Exception as exc:
        raise ModelLoadError(f"Could not resolve MLflow model URI {model_uri!r}: {exc}") from exc

    run = None
    if model_version.run_id:
        try:
            run = client.get_run(model_version.run_id)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load MLflow run metadata for {model_uri!r}: {exc}"
            ) from exc

    return {
        "registered_model_name": model_name,
        "registered_model_alias": alias,
        "registered_model_version": model_version.version,
        "run_id": model_version.run_id,
        "algorithm": (run.data.params.get("model_type") if run is not None else None),
        "metrics": run.data.metrics if run is not None else {},
        "training_timestamp": run.info.start_time if run is not None else None,
    }


def mlflow_signature_to_dict(mlflow_model: Model) -> dict[str, Any]:
    signature = mlflow_model.signature
    if signature is None:
        return {}
    return signature.to_dict()


def load_mlflow_model(model_uri: str) -> ModelBundle:
    registry_metadata = registered_model_metadata(model_uri)
    try:
        mlflow_model = Model.load(model_uri)
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        raise ModelLoadError(f"Could not load MLflow model at {model_uri!r}: {exc}") from exc

    if mlflow_model.signature is None or mlflow_model.signature.inputs is None:
        raise ModelLoadError("MLflow model must include an input signature.")

    features = [column.name for column in mlflow_model.signature.inputs.inputs]
    if (
        not features
        or not all(isinstance(name, str) and name for name in features)
        or len(features) != len(set(features))
    ):
        raise ModelLoadError("MLflow model signature must include unique feature names.")

    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise ModelLoadError("MLflow model must implement predict() and predict_proba().")

    metrics = registry_metadata["metrics"]
    metadata = {
        "model_source": "mlflow",
        "model_uri": model_uri,
        "model_version": (
            registry_metadata["registered_model_alias"]
            or registry_metadata["registered_model_version"]
        ),
        "registered_model_name": registry_metadata["registered_model_name"],
        "registered_model_alias": registry_metadata["registered_model_alias"],
        "registered_model_version": registry_metadata["registered_model_version"],
        "run_id": registry_metadata["run_id"],
        "training_timestamp": registry_metadata["training_timestamp"],
        "algorithm": (
            registry_metadata["algorithm"]
            or (mlflow_model.metadata or {}).get("algorithm")
            or "unknown"
        ),
        "features": features,
        "signature": mlflow_signature_to_dict(mlflow_model),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "roc_auc": metrics.get("roc_auc"),
    }
    return ModelBundle(model=model, metadata=metadata)


class ModelRegistry:
    """Load the configured model eagerly and cache query-selected versions."""

    def __init__(
        self,
        default_version: str,
        models_root: Path = MODELS_ROOT,
        model_uri: str | None = None,
    ) -> None:
        self.default_version = default_version
        self.models_root = models_root
        self.model_uri = model_uri
        self._lock = Lock()
        if model_uri:
            self._models = {model_uri: load_mlflow_model(model_uri)}
        else:
            self._models = {
                default_version: load_model(default_version, models_root=models_root)
            }

    def get(self, version: str | None = None) -> ModelBundle:
        if self.model_uri:
            if version is not None:
                raise ModelLoadError(
                    "model_version query overrides are only supported for local models; "
                    "set MODEL_URI to choose the MLflow model served by this API."
                )
            return self._models[self.model_uri]

        selected_version = version or self.default_version
        with self._lock:
            if selected_version not in self._models:
                self._models[selected_version] = load_model(
                    selected_version, models_root=self.models_root
                )
            return self._models[selected_version]
