from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from app.data.numeric_parser import parse_numeric
from app.data.validator import DataValidator


@dataclass
class DataQualityIssue:
    batch_id: int | None
    measurement_id: int | None
    issue_type: str  # 'missing', 'invalid', 'out_of_range', 'inconsistent'
    field_name: str
    value: Any = None
    description: str = ""


@dataclass
class DataQualityReport:
    total_batches: int = 0
    total_measurements: int = 0
    measurements_with_issues: int = 0
    missing_count: int = 0
    invalid_count: int = 0
    out_of_range_count: int = 0
    issues_by_field: dict[str, int] = field(default_factory=dict)
    issues: list[DataQualityIssue] = field(default_factory=list)
    sample_issues: list[DataQualityIssue] = field(default_factory=list)

    def add_issue(self, issue: DataQualityIssue, sample_size: int = 10) -> None:
        """Add issue and track top samples."""
        self.issues.append(issue)
        if issue.issue_type == "missing":
            self.missing_count += 1
        elif issue.issue_type == "invalid":
            self.invalid_count += 1
        elif issue.issue_type == "out_of_range":
            self.out_of_range_count += 1

        self.issues_by_field[issue.field_name] = self.issues_by_field.get(issue.field_name, 0) + 1

        if len(self.sample_issues) < sample_size:
            self.sample_issues.append(issue)


class DataQualityInspector:
    def __init__(self, validator: DataValidator | None = None) -> None:
        self.validator = validator or DataValidator()

    def inspect_measurements(self, df: pd.DataFrame) -> DataQualityReport:
        """Inspect measurements dataframe for quality issues."""
        report = DataQualityReport(
            total_measurements=len(df),
            total_batches=df["batchID"].nunique() if "batchID" in df else 0
        )

        measurements_with_issues = set()

        # Check for missing values in critical fields
        critical_fields = ["batchID", "timestamp", "loading_step_id", "id"]
        for field in critical_fields:
            if field in df.columns:
                missing = df[field].isna().sum()
                if missing > 0:
                    report.add_issue(
                        DataQualityIssue(
                            batch_id=None,
                            measurement_id=None,
                            issue_type="missing",
                            field_name=field,
                            description=f"Missing values in {field}: {missing}"
                        )
                    )

        # Validate numeric fields
        for idx, row in df.iterrows():
            errors = self.validator.validate_row(row.to_dict())
            if errors:
                batch_id = int(row.get("batchID")) if "batchID" in row else None
                measurement_id = int(row.get("id")) if "id" in row else None
                measurements_with_issues.add(measurement_id)
                for field, error in errors[:1]:  # Add first error only to avoid duplication
                    report.add_issue(
                        DataQualityIssue(
                            batch_id=batch_id,
                            measurement_id=measurement_id,
                            issue_type="out_of_range",
                            field_name=field,
                            value=row.get(field),
                            description=error
                        )
                    )

        report.measurements_with_issues = len(measurements_with_issues)
        return report
