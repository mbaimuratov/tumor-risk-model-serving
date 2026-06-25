"""Train and persist a versioned tumor-risk classifier."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_VERSIONS = ("v1", "v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-version",
        choices=SUPPORTED_VERSIONS,
        default=os.getenv("MODEL_VERSION", "v1"),
        help="Version to train; defaults to MODEL_VERSION or v1.",
    )
    return parser.parse_args()


def build_model(version: str) -> tuple[Pipeline, str]:
    if version == "v1":
        return (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1_000, random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
            "LogisticRegression",
        )

    return (
        Pipeline(
            steps=[
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                )
            ]
        ),
        "RandomForestClassifier",
    )


def main() -> None:
    version = parse_args().model_version
    model_directory = REPOSITORY_ROOT / "models" / version
    dataset = load_breast_cancer()
    features = dataset.feature_names.tolist()
    labels = (dataset.target == 0).astype(int)  # Positive means malignant/high risk.
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )

    model, algorithm = build_model(version)
    model.fit(x_train, y_train)
    model.serving_feature_names_ = features

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metadata = {
        "model_version": version,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "features": features,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / "model.joblib"
    metadata_path = model_directory / "metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")
    print(
        "Metrics: "
        f"accuracy={metadata['accuracy']:.4f}, "
        f"precision={metadata['precision']:.4f}, "
        f"recall={metadata['recall']:.4f}, "
        f"roc_auc={metadata['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
