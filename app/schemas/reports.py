from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReportSampleIssueResponse(BaseModel):
    batch_id: int | None = None
    measurement_id: int | None = None
    issue_type: str | None = None
    field_name: str | None = None
    value: Any = None
    description: str


class DataQualityReportResponse(BaseModel):
    total_measurements: int
    total_batches: int
    batches_with_targets: int
    measurements_with_issues: int
    missing_count: int
    invalid_count: int
    out_of_range_count: int
    issues_by_field: dict[str, int] = Field(default_factory=dict)
    sample_issues: list[ReportSampleIssueResponse] = Field(default_factory=list)
    stored_issues: int = 0


class StoredDataQualityIssueResponse(BaseModel):
    issue_id: int
    batch_id: int
    issue_type: str
    description: str
    created_at: str


class StoredDataQualityIssueListResponse(BaseModel):
    count: int
    items: list[StoredDataQualityIssueResponse] = Field(default_factory=list)
