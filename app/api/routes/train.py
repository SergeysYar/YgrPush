from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from app.api.dependencies import get_settings
from app.ml.service import TrainingPipeline

router = APIRouter()


class TrainRequest(BaseModel):
    model_types: list[str] = Field(
        default_factory=lambda: ["baseline", "ridge", "pls", "bayesian_ridge"]
    )
    protocol_policy: str = "latest"
    snapshot_mode: bool = True

    @field_validator("model_types")
    @classmethod
    def validate_model_types(cls, value: list[str]) -> list[str]:
        allowed = {"baseline", "ridge", "pls", "bayesian_ridge"}
        invalid = [item for item in value if item not in allowed]
        if invalid:
            raise ValueError(f"Unsupported model types: {', '.join(invalid)}")
        return value


@router.post("/train", summary="Train models")
def train_models(
    payload: TrainRequest,
    settings = Depends(get_settings),
) -> dict[str, Any]:
    pipeline = TrainingPipeline(settings.db_path, settings.ml_storage_path)
    results = pipeline.train_all(
        snapshot_mode=payload.snapshot_mode,
        protocol_policy=payload.protocol_policy,
        model_types=payload.model_types,
    )
    pipeline.save_cv_report(results)

    model_ids: list[str] = []
    metrics: dict[str, Any] = {}
    model_mapping = {
        "baseline": "baseline",
        "ridge": "ridge",
        "pls": "pls",
        "bayesian_ridge": "bayesian",
    }
    for model_name in payload.model_types:
        model_result = results.get(model_mapping[model_name], {})
        if not model_result:
            metrics[model_name] = None
            continue

        metrics[model_name] = {}
        for target_name, target_result in model_result.items():
            if not target_result:
                metrics[model_name][target_name] = None
                continue

            model_id = target_result.get("model_id")
            if model_id:
                model_ids.append(model_id)

            if "cv_result" in target_result:
                metrics[model_name][target_name] = target_result["cv_result"]["cv_metrics"]
            else:
                metrics[model_name][target_name] = {
                    key: value
                    for key, value in target_result.items()
                    if key in {"status", "n_samples", "targets"}
                }

    return {
        "status": "ok",
        "requested": payload.model_types,
        "protocol_policy": payload.protocol_policy,
        "snapshot_mode": payload.snapshot_mode,
        "model_ids": model_ids,
        "metrics": metrics,
    }
