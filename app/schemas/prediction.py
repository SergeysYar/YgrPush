from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionValue(BaseModel):
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    confidence: str
    model: str | None = None
    status: str
    training_batches: int = 0
    top_factors: list[str] = Field(default_factory=list)


class PredictionCheckpoint(BaseModel):
    completed_steps: int
    last_measurement_id: int | None = None


class SampleIssueResponse(BaseModel):
    batch_id: int | None = None
    measurement_id: int | None = None
    field_name: str | None = None
    description: str


class DataQualityResponse(BaseModel):
    total_measurements: int
    measurements_with_issues: int
    missing_count: int = 0
    invalid_count: int = 0
    out_of_range_count: int = 0
    issues_by_field: dict[str, int] = Field(default_factory=dict)
    sample_issues: list[SampleIssueResponse] = Field(default_factory=list)


class SimilarBatchResponse(BaseModel):
    batch_id: int
    batch_number: str | None = None
    product_name: str | None = None
    production_date: str | None = None
    distance: float
    ph: float | None = None
    viscosity: float | None = None
    chlorides: float | None = None


class BatchPredictionResponse(BaseModel):
    batch_id: int | None = None
    checkpoint: PredictionCheckpoint
    data_quality: DataQualityResponse
    predictions: dict[str, PredictionValue]
    similar_batches: list[SimilarBatchResponse] = Field(default_factory=list)


class CustomPredictionRequest(BaseModel):
    measurements: list[dict[str, Any]]
