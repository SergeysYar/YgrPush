from __future__ import annotations

from pydantic import BaseModel, Field


class StoredPredictionResponse(BaseModel):
    prediction_id: int
    batch_id: int
    checkpoint_order: int
    created_at: str
    model_version: str
    target: str
    predicted_value: float
    lower_bound: float
    upper_bound: float
    status: str
    confidence: str
    actual_value: float | None = None


class StoredPredictionListResponse(BaseModel):
    count: int
    items: list[StoredPredictionResponse] = Field(default_factory=list)
