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

## Shadow deployment

The API can serve the current production model while evaluating a second model in
shadow mode:

```text
champion = current production model
candidate = new model under evaluation
```

When `MODEL_URI=models:/TumorRiskClassifier@champion`, `POST /predict` returns
only the `champion` prediction to the caller. Internally, the API also tries to
load `models:/TumorRiskClassifier@candidate`, runs the same request through it,
and logs whether the two models disagree.

Candidate predictions are never included in the HTTP response. Candidate load or
prediction failures are logged and do not fail the user request.

Use the normal training flow to create and promote aliases:

```bash
docker compose up -d mlflow
docker compose --profile training run --rm trainer
docker compose up -d --build api
```

To shadow a different model URI explicitly, set `SHADOW_MODEL_URI` for the API.
If it is not set, the API automatically shadows `candidate` whenever the served
model URI is `models:/<name>@champion`.

Shadow logs are written as JSON to the `tumor_risk.shadow_predictions` logger.
Example disagreement log:

```json
{
  "timestamp": "2026-07-01T06:27:03.961466+00:00",
  "request_id": "shadow-request-123",
  "champion_model_version": "champion",
  "candidate_model_version": "candidate",
  "champion_prediction": 1,
  "candidate_prediction": 0,
  "champion_probability": 0.9,
  "candidate_probability": 0.2,
  "disagreement": true,
  "probability_delta": 0.7,
  "latency_champion": 4.2,
  "latency_candidate": 5.1,
  "candidate_error": null
}
```

Example candidate failure log:

```json
{
  "timestamp": "2026-07-01T06:27:03.961466+00:00",
  "request_id": "shadow-request-124",
  "champion_model_version": "champion",
  "candidate_model_version": null,
  "champion_prediction": 1,
  "candidate_prediction": null,
  "champion_probability": 0.9,
  "candidate_probability": null,
  "disagreement": null,
  "probability_delta": null,
  "latency_champion": 4.2,
  "latency_candidate": 2.3,
  "candidate_error": "candidate registry is unavailable"
}
```

Prometheus exposes shadow metrics at `GET /metrics`:

```text
model_shadow_disagreements_total
model_shadow_probability_delta
```

Use `model_shadow_disagreements_total` together with the
`model_shadow_probability_delta_count` histogram sample count to monitor the
shadow disagreement rate.
