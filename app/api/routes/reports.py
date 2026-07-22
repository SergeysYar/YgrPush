from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/data-quality", summary="Get data quality report")
def get_data_quality() -> dict[str, object]:
    return {"report": "not implemented", "issues": []}
