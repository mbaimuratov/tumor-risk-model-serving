# Tumor Risk Model Serving

## MLflow lifecycle

This project uses MLflow to track model training, register validated models, and
serve the current `champion` model through FastAPI.

1. Train a model.

   ```bash
   docker compose up -d mlflow
   docker compose --profile training run --rm trainer
   ```

2. Track the run in MLflow.

   The training container uses `MLFLOW_TRACKING_URI=http://mlflow:5000`, so runs
   are logged to the Compose MLflow tracking server. Open the UI at
   `http://127.0.0.1:5001`.

3. Log parameters, metrics, and artifacts.

   `training/train.py` logs training parameters, metrics such as `recall`,
   `roc_auc`, and `false_negative_rate`, model artifacts, the input example, and
   the model signature.

4. Register the model.

   The containerized training workflow registers the selected run as
   `TumorRiskClassifier`.

5. Assign the `candidate` alias.

   The newest registered model version is assigned to
   `TumorRiskClassifier@candidate`.

6. Validate the candidate.

   Pre-deployment validation checks that the candidate model loads, has a
   signature, has an input example, does not regress on `recall` or
   `false_negative_rate` versus the current `champion`, and passes a `/predict`
   smoke test.

7. Promote to `champion`.

   If validation passes, the workflow reassigns
   `TumorRiskClassifier@champion` to the candidate version. If validation fails,
   the existing `champion` alias is left unchanged.

8. Serve `champion` through FastAPI.

   Set `MODEL_URI=models:/TumorRiskClassifier@champion` and restart the API:

   ```bash
   docker compose up -d --build api
   curl http://127.0.0.1:8000/model-info
   ```

9. Roll back by reassigning `champion`.

   To roll back, point the `champion` alias at a previous registered version:

   ```bash
   docker compose run --rm trainer python training/train.py --set-candidate-version <version>
   docker compose run --rm trainer python training/train.py --promote-candidate
   docker compose up -d --build api
   ```

   If you need an immediate rollback without candidate validation, use the MLflow
   UI to edit the `champion` alias directly, then restart the API.
