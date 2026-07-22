from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_settings
from app.database.ml_storage import ModelRegistry

router = APIRouter()


@router.get("", summary="List trained models")
def list_models(settings = Depends(get_settings)) -> dict[str, object]:
    registry = ModelRegistry(settings.ml_storage_path)
    models = registry.list_models()
    return {"count": len(models), "items": models}


@router.get("/summary", summary="Summarize champion/challenger models")
def summarize_models(settings = Depends(get_settings)) -> dict[str, object]:
    registry = ModelRegistry(settings.ml_storage_path)
    return registry.summarize_models()


@router.get("/runs", summary="List training runs")
def list_training_runs(limit: int = 20, settings = Depends(get_settings)) -> dict[str, object]:
    registry = ModelRegistry(settings.ml_storage_path)
    items = registry.list_training_runs(limit=limit)
    return {"count": len(items), "items": items}


@router.post("/{model_id}/promote", summary="Promote model to champion")
def promote_model(model_id: str, settings = Depends(get_settings)) -> dict[str, object]:
    registry = ModelRegistry(settings.ml_storage_path)
    try:
        registry.promote_model(model_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    return {"status": "promoted", "model_id": model_id}
