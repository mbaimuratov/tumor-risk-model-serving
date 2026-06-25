"""Versioned model loading and serving-contract validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import joblib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPOSITORY_ROOT / "models"
VERSION_PATTERN = re.compile(r"v[1-9][0-9]*")
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


class ModelRegistry:
    """Load the configured model eagerly and cache query-selected versions."""

    def __init__(self, default_version: str, models_root: Path = MODELS_ROOT) -> None:
        self.default_version = default_version
        self.models_root = models_root
        self._lock = Lock()
        self._models = {
            default_version: load_model(default_version, models_root=models_root)
        }

    def get(self, version: str | None = None) -> ModelBundle:
        selected_version = version or self.default_version
        with self._lock:
            if selected_version not in self._models:
                self._models[selected_version] = load_model(
                    selected_version, models_root=self.models_root
                )
            return self._models[selected_version]
