from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_settings
from app.database.ml_storage import PredictionRepository
from app.schemas.storage import StoredPredictionListResponse

router = APIRouter()

@router.get("", summary="List stored batch predictions", response_model=StoredPredictionListResponse)
def list_predictions(
    batch_id: int | None = Query(default=None, ge=1),
    target: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    settings = Depends(get_settings),
) -> StoredPredictionListResponse:
    repository = PredictionRepository(settings.ml_storage_path)
    items = repository.list_predictions(batch_id=batch_id, target=target, limit=limit)
    return StoredPredictionListResponse(count=len(items), items=items)
