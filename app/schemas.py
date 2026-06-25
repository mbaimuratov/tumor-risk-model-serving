"""API request and response schemas."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float]


class BatchPredictionRequest(BaseModel):
    samples: list[PredictionRequest] = Field(min_length=1, max_length=1_000)


class PredictionResult(BaseModel):
    prediction: int
    risk: str
    malignant_probability: float


class PredictionResponse(PredictionResult):
    model_version: str


class BatchPredictionResponse(BaseModel):
    model_version: str
    predictions: list[PredictionResult]
