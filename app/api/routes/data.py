from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings
from app.data.dataset_builder import DatasetBuilder

router = APIRouter()


@router.get("/summary", summary="Get dataset summary")
def get_data_summary(settings = Depends(get_settings)) -> dict[str, object]:
    builder = DatasetBuilder(settings.db_path)
    return builder.get_api_summary()
