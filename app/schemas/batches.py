from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BatchInfoResponse(BaseModel):
    batch_id: int
    product_id: int | None = None
    batch_number: str | None = None
    production_date: str | None = None
    status: str | None = None


class ProductInfoResponse(BaseModel):
    product_id: int
    name: str | None = None
    category: str | None = None
    base: str | None = None
    base_code: str | None = None
    viscosity_thickener: str | None = None
    viscosity_softener: str | None = None
    ph_corrector: str | None = None
    viscosity_adjustment: str | None = None


class TargetInfoResponse(BaseModel):
    protocol_id: int | None = None
    batch_id: int | None = None
    test_date: str | None = None
    ph: float | None = None
    viscosity: float | None = None
    chlorides: float | None = None
    source: str | None = None


class ProtocolInfoResponse(BaseModel):
    protocol_id: int
    product_id: int | None = None
    test_date: str | None = None
    ph: Any = None
    chlorides: Any = None
    viscosity: Any = None
    batch_id: int | None = None
    is_compliant: int | None = None
    compliance_percent: float | None = None
    ph_value: float | None = None
    viscosity_value: float | None = None
    chlorides_value: float | None = None


class ComponentInfoResponse(BaseModel):
    component_id: int
    total_mass: float | None = None
    first_step: int | None = None
    last_step: int | None = None
    component_name: str | None = None
    component_group: str | None = None


class SampleIssueResponse(BaseModel):
    batch_id: int | None = None
    measurement_id: int | None = None
    field_name: str | None = None
    description: str


class BatchDataQualityResponse(BaseModel):
    total_measurements: int
    measurements_with_issues: int
    missing_count: int
    invalid_count: int
    out_of_range_count: int
    issues_by_field: dict[str, int] = Field(default_factory=dict)
    sample_issues: list[SampleIssueResponse] = Field(default_factory=list)


class BatchDetailResponse(BaseModel):
    batch: BatchInfoResponse
    product: ProductInfoResponse | None = None
    targets: list[TargetInfoResponse] = Field(default_factory=list)
    measurements: list[dict[str, Any]] = Field(default_factory=list)
    components: list[ComponentInfoResponse] = Field(default_factory=list)
    protocols: list[ProtocolInfoResponse] = Field(default_factory=list)
    data_quality: BatchDataQualityResponse
