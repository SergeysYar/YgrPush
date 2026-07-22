from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_settings
from app.database.repositories import BatchRepository
from app.database.source import DatabaseInspector

router = APIRouter()


@router.get("", summary="List batches")
def list_batches(
    status: str | None = Query(default=None),
    product_id: int | None = Query(default=None),
    has_protocol: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    settings = Depends(get_settings),
) -> dict[str, object]:
    inspector = DatabaseInspector(settings.db_path)
    if not inspector.has_table("Batches"):
        raise HTTPException(status_code=404, detail="Batches table not found")
    batches = inspector.list_batches(status=status, product_id=product_id, has_protocol=has_protocol, limit=limit, offset=offset)
    return {"count": len(batches), "items": batches}


@router.get("/{batch_id}", summary="Get batch details")
def get_batch_details(
    batch_id: int,
    settings = Depends(get_settings),
) -> dict[str, object]:
    repository = BatchRepository(settings.db_path)
    payload = repository.get_batch_detail(batch_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    return payload
