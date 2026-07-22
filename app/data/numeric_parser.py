from __future__ import annotations

import math
from typing import Any


def parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().replace(",", ".")
        if normalized == "":
            return None
        try:
            return float(normalized)
        except ValueError:
            return None
    return None
