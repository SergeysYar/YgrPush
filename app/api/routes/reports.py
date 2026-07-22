from __future__ import annotations

from fastapi import APIRouter

from app.data.dataset_builder import DatasetBuilder
from app.config import settings

router = APIRouter()


@router.get("/data-quality", summary="Get data quality report")
def get_data_quality() -> dict[str, object]:
    builder = DatasetBuilder(settings.db_path)
    summary = builder.get_data_summary()
    return {
        "total_measurements": summary["total_measurements"],
        "total_batches": summary["total_batches"],
        "batches_with_targets": summary["batches_with_targets"],
        "measurements_with_issues": summary["measurements_with_issues"],
        "missing_count": summary["missing_count"],
        "invalid_count": summary["invalid_count"],
        "out_of_range_count": summary["out_of_range_count"],
        "issues_by_field": summary["issues_by_field"],
    }
