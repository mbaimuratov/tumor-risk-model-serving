#!/usr/bin/env sh
set -eu

MODEL_VERSION="${TRAIN_MODEL_VERSION:-v1}"

python training/train.py --model-version "${MODEL_VERSION}"
python training/train.py --register-best-model

CANDIDATE_VERSION="$(
python - <<'PY'
import mlflow

from training.train import REGISTERED_MODEL_NAME

client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
if not versions:
    raise SystemExit(f"No registered versions found for {REGISTERED_MODEL_NAME}.")
print(max(versions, key=lambda version: int(version.version)).version)
PY
)"

python training/train.py --set-candidate-version "${CANDIDATE_VERSION}"
python training/train.py --promote-candidate
