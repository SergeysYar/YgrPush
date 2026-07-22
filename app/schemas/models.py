from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StoredModelResponse(BaseModel):
    model_id: str
    model_type: str
    target: str
    created_at: str
    status: str
    version: int | None = None
    artifact_path: str
    metrics_json: dict[str, Any] = Field(default_factory=dict)
    features_json: dict[str, Any] = Field(default_factory=dict)


class ModelListResponse(BaseModel):
    count: int
    items: list[StoredModelResponse]


class PromoteModelResponse(BaseModel):
    status: str
    model_id: str


class TrainingRunResponse(BaseModel):
    run_id: str
    created_at: str
    model_ids: list[str] = Field(default_factory=list)
    batch_ids: list[int] = Field(default_factory=list)
    settings_json: dict[str, Any] = Field(default_factory=dict)


class TrainingRunListResponse(BaseModel):
    count: int
    items: list[TrainingRunResponse]


class TargetModelSummary(BaseModel):
    champion: StoredModelResponse | None = None
    challengers: list[StoredModelResponse] = Field(default_factory=list)
    archived: list[StoredModelResponse] = Field(default_factory=list)


class ModelSummaryResponse(BaseModel):
    total_models: int
    targets: dict[str, TargetModelSummary] = Field(default_factory=dict)
    training_runs: list[TrainingRunResponse] = Field(default_factory=list)
