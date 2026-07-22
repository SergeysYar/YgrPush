from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_settings
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
