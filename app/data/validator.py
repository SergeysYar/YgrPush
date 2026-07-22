from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.data.numeric_parser import parse_numeric


@dataclass
class ValidationRule:
    field: str
    min_value: float | None = None
    max_value: float | None = None
    allow_negative: bool = False
    allow_zero: bool = True
    description: str = ""


# Validation configuration by field
VALIDATION_RULES = {
    "startpH": ValidationRule("startpH", min_value=0.0, max_value=14.0, description="pH value"),
    "endpH": ValidationRule("endpH", min_value=0.0, max_value=14.0, description="pH value"),
    "midpH": ValidationRule("midpH", min_value=0.0, max_value=14.0, description="pH value"),
    "dpH": ValidationRule("dpH", min_value=-14.0, max_value=14.0, description="pH change"),
    "duration_minutes": ValidationRule("duration_minutes", min_value=0.0, allow_zero=True, description="Duration in minutes"),
    "startTemp": ValidationRule("startTemp", min_value=0.0, max_value=150.0, description="Temperature in Celsius"),
    "endTemp": ValidationRule("endTemp", min_value=0.0, max_value=150.0, description="Temperature in Celsius"),
    "midTemp": ValidationRule("midTemp", min_value=0.0, max_value=150.0, description="Temperature in Celsius"),
    "startPE": ValidationRule("startPE", min_value=0.0, description="PE value"),
    "endPE": ValidationRule("endPE", min_value=0.0, description="PE value"),
    "startfreq": ValidationRule("startfreq", min_value=0.0, description="Frequency"),
    "endfreq": ValidationRule("endfreq", min_value=0.0, description="Frequency"),
}


class DataValidator:
    def __init__(self, rules: dict[str, ValidationRule] | None = None) -> None:
        self.rules = rules or VALIDATION_RULES

    def validate_value(self, field: str, value: Any) -> tuple[bool, str | None]:
        """Validate a single value against its rule.
        
        Returns (is_valid, error_message)
        """
        if field not in self.rules:
            return True, None

        rule = self.rules[field]
        parsed = parse_numeric(value)

        if parsed is None:
            return True, None  # None is acceptable; it's a missing value

        if rule.min_value is not None and parsed < rule.min_value:
            return False, f"{field} ({parsed}) is below minimum {rule.min_value}"

        if rule.max_value is not None and parsed > rule.max_value:
            return False, f"{field} ({parsed}) is above maximum {rule.max_value}"

        if not rule.allow_negative and parsed < 0:
            return False, f"{field} ({parsed}) is negative but not allowed"

        if not rule.allow_zero and parsed == 0:
            return False, f"{field} ({parsed}) is zero but not allowed"

        return True, None

    def validate_row(self, row: dict[str, Any]) -> list[tuple[str, str]]:
        """Validate all fields in a row.
        
        Returns list of (field, error_message) tuples.
        """
        errors = []
        for field, value in row.items():
            is_valid, error = self.validate_value(field, value)
            if not is_valid and error:
                errors.append((field, error))
        return errors
