from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_settings
from app.database.ml_storage import ModelRegistry
from app.schemas.models import (
    ModelListResponse,
    ModelSummaryResponse,
    PromoteModelResponse,
    TrainingRunListResponse,
)

router = APIRouter()


@router.get("", summary="List trained models", response_model=ModelListResponse)
def list_models(settings = Depends(get_settings)) -> ModelListResponse:
    registry = ModelRegistry(settings.ml_storage_path)
    models = registry.list_models()
    return ModelListResponse(count=len(models), items=models)


@router.get("/summary", summary="Summarize champion/challenger models", response_model=ModelSummaryResponse)
def summarize_models(settings = Depends(get_settings)) -> ModelSummaryResponse:
    registry = ModelRegistry(settings.ml_storage_path)
    return ModelSummaryResponse(**registry.summarize_models())


@router.get("/runs", summary="List training runs", response_model=TrainingRunListResponse)
def list_training_runs(limit: int = 20, settings = Depends(get_settings)) -> TrainingRunListResponse:
    registry = ModelRegistry(settings.ml_storage_path)
    items = registry.list_training_runs(limit=limit)
    return TrainingRunListResponse(count=len(items), items=items)


@router.post("/{model_id}/promote", summary="Promote model to champion", response_model=PromoteModelResponse)
def promote_model(model_id: str, settings = Depends(get_settings)) -> PromoteModelResponse:
    registry = ModelRegistry(settings.ml_storage_path)
    try:
        registry.promote_model(model_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    return PromoteModelResponse(status="promoted", model_id=model_id)
