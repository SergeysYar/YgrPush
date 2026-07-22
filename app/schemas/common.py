from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    database: dict[str, Any]
    ml_storage: dict[str, Any]


class BatchSummary(BaseModel):
    batch_id: int
    product_id: int | None = None
    batch_number: str | None = None
    production_date: datetime | None = None
    status: str | None = None


class BatchListResponse(BaseModel):
    count: int
    items: list[BatchSummary]
