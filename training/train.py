"""Train and persist a versioned tumor-risk classifier."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REGISTERED_MODEL_NAME = "tumor-risk-classifier"
SUPPORTED_VERSIONS = ("v1", "v2")

mlflow.set_experiment("tumor-risk-classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-version",
        choices=SUPPORTED_VERSIONS,
        default=os.getenv("MODEL_VERSION", "v1"),
        help="Version to train; defaults to MODEL_VERSION or v1.",
    )
    parser.add_argument(
        "--search-random-forest",
        action="store_true",
        help="Search for a random forest model instead of the default logistic regression.",
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


def training_params(model: Pipeline, algorithm: str) -> dict[str, object]:
    classifier = model.named_steps["classifier"]
    return {
        "model_type": algorithm,
        "test_size": TEST_SIZE,
        "random_state": getattr(classifier, "random_state", "none"),
        "scaler_used": "scaler" in model.named_steps,
        "class_weight": getattr(classifier, "class_weight", None) or "none",
        "max_depth": getattr(classifier, "max_depth", None) or "none",
        "n_estimators": getattr(classifier, "n_estimators", None) or "none",
        "solver": getattr(classifier, "solver", None) or "none",
        "C": getattr(classifier, "C", None) or "none",
    }


def save_confusion_matrix(
    y_true: list[int],
    predictions: list[int],
    output_path: Path,
) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["low risk", "high risk"],
        cmap="Blues",
    )
    display.ax_.set_title("Tumor Risk Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(display.figure_)


def save_roc_curve(
    y_true: list[int],
    probabilities: list[float],
    output_path: Path,
) -> None:
    display = RocCurveDisplay.from_predictions(y_true, probabilities)
    display.ax_.set_title("Tumor Risk ROC Curve")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(display.figure_)


def write_artifacts(
    artifact_directory: Path,
    y_test: list[int],
    predictions: list[int],
    probabilities: list[float],
    features: list[str],
    metadata: dict[str, object],
) -> list[Path]:
    artifact_directory.mkdir(parents=True, exist_ok=True)

    confusion_matrix_path = artifact_directory / "confusion_matrix.png"
    roc_curve_path = artifact_directory / "roc_curve.png"
    classification_report_path = artifact_directory / "classification_report.txt"
    feature_names_path = artifact_directory / "feature_names.json"
    metadata_path = artifact_directory / "metadata.json"
    requirements_path = REPOSITORY_ROOT / "requirements.txt"

    save_confusion_matrix(y_test, predictions, confusion_matrix_path)
    save_roc_curve(y_test, probabilities, roc_curve_path)
    classification_report_path.write_text(
        classification_report(
            y_test,
            predictions,
            target_names=["low risk", "high risk"],
        ),
        encoding="utf-8",
    )
    feature_names_path.write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    artifacts = [
        confusion_matrix_path,
        roc_curve_path,
        classification_report_path,
        feature_names_path,
        metadata_path,
    ]
    if requirements_path.is_file():
        artifacts.append(requirements_path)
    return artifacts


def model_input_frame(values, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, columns=features)


search_space = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": None},
]


def run_random_forest_search() -> None:
    dataset = load_breast_cancer()
    features = dataset.feature_names.tolist()
    labels = (dataset.target == 0).astype(int)  # Positive means malignant/high risk.
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )
    x_train_frame = model_input_frame(x_train, features)
    x_test_frame = model_input_frame(x_test, features)

    best_score = -1
    best_run_id = None
    best_config = None
    best_metrics = None

    with mlflow.start_run(run_name="random_forest_manual_grid_search"):
        mlflow.set_tags(
            {
                "search_type": "manual_grid_search",
                "model_type": "RandomForestClassifier",
                "task": "tumor-risk-classification",
            }
        )
        mlflow.log_param("candidate_count", len(search_space))

        for config in search_space:
            with mlflow.start_run(
                run_name=f"rf_n{config['n_estimators']}_depth{config['max_depth']}",
                nested=True,
            ) as child_run:
                model = Pipeline(
                    steps=[
                        (
                            "classifier",
                            RandomForestClassifier(
                                n_estimators=config["n_estimators"],
                                max_depth=config["max_depth"],
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                            ),
                        )
                    ]
                )
                model.fit(x_train_frame, y_train)
                predictions = model.predict(x_test_frame)
                probabilities = model.predict_proba(x_test_frame)[:, 1]
                metrics = {
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(y_test, predictions),
                    "recall": recall_score(y_test, predictions),
                    "roc_auc": roc_auc_score(y_test, probabilities),
                    "f1_score": f1_score(y_test, predictions),
                }
                metrics["false_negative_rate"] = 1 - metrics["recall"]
                metrics["false_positive_rate"] = 1 - metrics["precision"]

                mlflow.log_params(
                    {
                        "model_type": "RandomForestClassifier",
                        "n_estimators": config["n_estimators"],
                        "max_depth": config["max_depth"] or "none",
                        "test_size": TEST_SIZE,
                        "random_state": RANDOM_STATE,
                    }
                )
                mlflow.log_metrics(metrics)

                is_better = metrics["recall"] > best_score or (
                    metrics["recall"] == best_score
                    and (
                        best_metrics is None
                        or metrics["roc_auc"] > best_metrics["roc_auc"]
                    )
                )
                if is_better:
                    best_score = metrics["recall"]
                    best_run_id = child_run.info.run_id
                    best_config = config
                    best_metrics = metrics

        if best_config is None or best_metrics is None:
            raise RuntimeError("Random forest search did not evaluate any candidates.")

        mlflow.log_params(
            {
                "best_n_estimators": best_config["n_estimators"],
                "best_max_depth": best_config["max_depth"] or "none",
            }
        )
        mlflow.log_metrics(
            {
                "best_recall": best_metrics["recall"],
                "best_roc_auc": best_metrics["roc_auc"],
                "best_false_negative_rate": best_metrics["false_negative_rate"],
            }
        )
        mlflow.set_tag("best_child_run_id", best_run_id)

    print(f"Best Params: {best_config}, Best Score: {best_score:.4f}")


def main() -> None:
    args = parse_args()

    if args.search_random_forest:
        run_random_forest_search()
        return

    version = args.model_version
    model_directory = REPOSITORY_ROOT / "models" / version
    dataset = load_breast_cancer()
    features = dataset.feature_names.tolist()
    labels = (dataset.target == 0).astype(int)  # Positive means malignant/high risk.
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )
    x_train_frame = model_input_frame(x_train, features)
    x_test_frame = model_input_frame(x_test, features)

    model, algorithm = build_model(version)
    with mlflow.start_run(run_name=f"{version}-{algorithm}"):
        mlflow.log_params(training_params(model, algorithm))
        mlflow.set_tags(
            {
                "model_version": version,
                "task": "tumor-risk-classification",
            }
        )
        model.fit(x_train_frame, y_train)
        model.serving_feature_names_ = features

        predictions = model.predict(x_test_frame)
        probabilities = model.predict_proba(x_test_frame)[:, 1]
        input_example = x_test_frame.head(1)
        signature = infer_signature(
            input_example,
            {
                "prediction": model.predict(input_example),
                "malignant_probability": model.predict_proba(input_example)[:, 1],
            },
        )
        metadata = {
            "model_version": version,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "algorithm": algorithm,
            "features": features,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "f1_score": f1_score(y_test, predictions),
            "false_negative_rate": 1 - recall_score(y_test, predictions),
            "false_positive_rate": 1 - precision_score(y_test, predictions),
        }
        mlflow.log_metrics(
            {
                "accuracy": metadata["accuracy"],
                "precision": metadata["precision"],
                "recall": metadata["recall"],
                "roc_auc": metadata["roc_auc"],
                "f1_score": metadata["f1_score"],
                "false_negative_rate": metadata["false_negative_rate"],
                "false_positive_rate": metadata["false_positive_rate"],
            }
        )

        model_directory.mkdir(parents=True, exist_ok=True)
        artifact_directory = model_directory / "artifacts"
        model_path = model_directory / "model.joblib"
        metadata_path = model_directory / "metadata.json"
        joblib.dump(model, model_path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        artifact_paths = write_artifacts(
            artifact_directory=artifact_directory,
            y_test=y_test,
            predictions=predictions,
            probabilities=probabilities,
            features=features,
            metadata=metadata,
        )
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
            metadata={
                "model_version": version,
                "algorithm": algorithm,
            },
        )

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
