from __future__ import annotations

from datetime import datetime, UTC

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_settings
from app.data.dataset_builder import DatasetBuilder
from app.database.ml_storage import DataQualityRepository

router = APIRouter()


@router.get("/data-quality", summary="Get data quality report")
def get_data_quality(
    store: bool = Query(default=False),
    settings = Depends(get_settings),
) -> dict[str, object]:
    builder = DatasetBuilder(settings.db_path)
    summary = builder.get_data_summary()
    measurements = builder.load_measurements()
    report = builder.quality_inspector.inspect_measurements(measurements)

    stored_issues = 0
    if store:
        repository = DataQualityRepository(settings.ml_storage_path)
        stored_issues = repository.save_issues(
            [
                {
                    "batch_id": issue.batch_id,
                    "measurement_id": issue.measurement_id,
                    "issue_type": issue.issue_type,
                    "field_name": issue.field_name,
                    "value": issue.value,
                    "description": issue.description,
                }
                for issue in report.issues
            ],
            created_at=datetime.now(UTC).isoformat(),
        )

    return {
        "total_measurements": summary["total_measurements"],
        "total_batches": summary["total_batches"],
        "batches_with_targets": summary["batches_with_targets"],
        "measurements_with_issues": summary["measurements_with_issues"],
        "missing_count": summary["missing_count"],
        "invalid_count": summary["invalid_count"],
        "out_of_range_count": summary["out_of_range_count"],
        "issues_by_field": summary["issues_by_field"],
        "sample_issues": [
            {
                "batch_id": issue.batch_id,
                "measurement_id": issue.measurement_id,
                "issue_type": issue.issue_type,
                "field_name": issue.field_name,
                "value": issue.value,
                "description": issue.description,
            }
            for issue in report.sample_issues
        ],
        "stored_issues": stored_issues,
    }


@router.get("/data-quality/issues", summary="List stored data quality issues")
def list_data_quality_issues(
    batch_id: int | None = Query(default=None, ge=1),
    issue_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    settings = Depends(get_settings),
) -> dict[str, object]:
    repository = DataQualityRepository(settings.ml_storage_path)
    items = repository.list_issues(batch_id=batch_id, issue_type=issue_type, limit=limit)
    return {"count": len(items), "items": items}
