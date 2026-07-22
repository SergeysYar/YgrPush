from __future__ import annotations

import math
from typing import Any

from app.data.numeric_parser import parse_numeric


def is_valid_ph(value: Any) -> bool:
    parsed = parse_numeric(value)
    return parsed is not None and 0.0 <= parsed <= 14.0


def clean_duration(value: Any) -> float | None:
    parsed = parse_numeric(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def normalize_measurement_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {**row}
    for key, value in row.items():
        if key.startswith(("start", "end", "mid", "d")) or key.endswith(("_pH", "_PE", "_freq", "_minutes", "_Temp", "value")):
            normalized[key] = parse_numeric(value)
    normalized["valid_ph"] = is_valid_ph(row.get("startpH")) or is_valid_ph(row.get("endpH"))
    normalized["duration_minutes"] = clean_duration(row.get("duration_minutes"))
    return normalized
