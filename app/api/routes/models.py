from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.database.ml_storage import ModelRegistry

router = APIRouter()


@router.get("", summary="List trained models")
def list_models() -> dict[str, object]:
    registry = ModelRegistry()
    models = registry.list_models()
    return {"count": len(models), "items": models}


@router.post("/{model_id}/promote", summary="Promote model to champion")
def promote_model(model_id: str) -> dict[str, object]:
    registry = ModelRegistry()
    registry.promote_model(model_id)
    return {"status": "promoted", "model_id": model_id}
