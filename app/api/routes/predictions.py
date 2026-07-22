from __future__ import annotations

from fastapi import APIRouter, Query

from app.database.ml_storage import PredictionRepository

router = APIRouter()

@router.get("", summary="List stored batch predictions")
def list_predictions(
    batch_id: int | None = Query(default=None, ge=1),
    target: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, object]:
    repository = PredictionRepository()
    items = repository.list_predictions(batch_id=batch_id, target=target, limit=limit)
    return {"count": len(items), "items": items}
