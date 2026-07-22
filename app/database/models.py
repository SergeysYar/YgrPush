from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BatchRecord:
    batch_id: int
    product_id: int
    batch_number: str | None
    production_date: str | None
    status: str | None
    extra: dict[str, Any] = None


@dataclass
class MeasurementRecord:
    id: int
    batch_id: int
    timestamp: str | None
    loading_step_id: int | None
    loading_step_type: str | None
    step_type_id: int | None
    values: dict[str, Any] = None
