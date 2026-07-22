from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_settings
from app.ml.service import TrainingPipeline

router = APIRouter()


class TrainRequest(BaseModel):
    model_types: list[str] = Field(
        default_factory=lambda: ["baseline", "ridge", "pls", "bayesian_ridge"]
    )
    protocol_policy: str = "latest"
    snapshot_mode: bool = True


@router.post("/train", summary="Train models")
def train_models(
    payload: TrainRequest,
    settings = Depends(get_settings),
) -> dict[str, Any]:
    pipeline = TrainingPipeline(settings.db_path, settings.ml_storage_path)
    results = pipeline.train_all()
    pipeline.save_cv_report(results)

    selected = set(payload.model_types)
    model_mapping = {
        "baseline": "baseline",
        "ridge": "ridge",
        "pls": "pls",
        "bayesian_ridge": "bayesian",
    }
    filtered_results = {
        public_name: results.get(internal_name)
        for public_name, internal_name in model_mapping.items()
        if public_name in selected
    }

    model_ids: list[str] = []
    metrics: dict[str, Any] = {}
    for model_name, model_result in filtered_results.items():
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
