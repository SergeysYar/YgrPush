from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any

from app.config import settings
from app.database.source import DatabaseInspector
from app.data.cleaner import normalize_measurement_row
from app.data.target_loader import load_targets, TargetProtocolPolicy
from app.data.component_normalizer import ComponentNormalizer
from app.data.data_quality import DataQualityInspector
from app.features.batch_features import BatchFeatureBuilder
from app.features.snapshot_features import SnapshotBuilder


class DatasetBuilder:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or settings.db_path)
        self.inspector = DatabaseInspector(self.db_path)
        self.normalizer = ComponentNormalizer()
        self.quality_inspector = DataQualityInspector()
        self.snapshot_builder = SnapshotBuilder()

    def load_measurements(self) -> pd.DataFrame:
        """Load all measurements from production database."""
        with self.inspector._connect() as conn:
            df = pd.read_sql_query("SELECT * FROM measurements", conn)
        return df

    def load_batch_data(self, batch_id: int) -> dict[str, Any]:
        """Load all data for a specific batch.
        
        Returns dict with:
        - measurements: DataFrame of measurements for this batch
        - components: DataFrame of normalized components
        - targets: Dict of target values (ph, viscosity, chlorides)
        """
        with self.inspector._connect() as conn:
            measurements = pd.read_sql_query(
                "SELECT * FROM measurements WHERE batchID = ? ORDER BY id",
                conn,
                params=(batch_id,)
            )

        components = pd.DataFrame()
        if not measurements.empty:
            measurements = pd.DataFrame(
                [normalize_measurement_row(row) for row in measurements.to_dict(orient="records")]
            )
            components = self.normalizer.normalize_batch_components(measurements)

        targets = load_targets(batch_id, self.db_path, settings.target_protocol_policy)

        return {
            "batch_id": batch_id,
            "measurements": measurements,
            "components": components,
            "targets": targets,
        }

    def build_measurement_dataset(self, apply_quality_checks: bool = True) -> pd.DataFrame:
        """Build cleaned measurement dataset."""
        df = self.load_measurements()

        if apply_quality_checks:
            report = self.quality_inspector.inspect_measurements(df)
            print(f"Data quality: {report.total_measurements} measurements, "
                  f"{report.measurements_with_issues} with issues")

        # Normalize each row
        normalized = []
        for idx, row in df.iterrows():
            cleaned_row = normalize_measurement_row(row.to_dict())
            normalized.append(cleaned_row)

        return pd.DataFrame(normalized)

    def build_batch_features_dataset(self) -> pd.DataFrame:
        """Build dataset with one row per batch.
        
        This dataset contains aggregated features for each complete batch.
        Returns DataFrame with batch_id, all features, and target values.
        """
        with self.inspector._connect() as conn:
            batch_ids_sql = """
            SELECT DISTINCT batchID FROM measurements ORDER BY batchID
            """
            batch_ids = pd.read_sql_query(batch_ids_sql, conn)["batchID"].unique()

        builder = BatchFeatureBuilder()
        rows = []
        for batch_id in batch_ids:
            batch_data = self.load_batch_data(int(batch_id))
            measurements = batch_data["measurements"]
            targets = batch_data["targets"]

            if measurements.empty:
                continue

            features = builder.build_full_batch_features(measurements)
            batch_row = {
                "batch_id": int(batch_id),
                "has_targets": len(targets) > 0,
                **features,
                "target_ph": float(targets[0]["ph"]) if targets and targets[0].get("ph") else None,
                "target_viscosity": float(targets[0]["viscosity"]) if targets and targets[0].get("viscosity") else None,
                "target_chlorides": float(targets[0]["chlorides"]) if targets and targets[0].get("chlorides") else None,
            }
            rows.append(batch_row)

        return pd.DataFrame(rows)

    def build_snapshot_features_dataset(self) -> pd.DataFrame:
        """Build dataset with one row per snapshot of each batch.

        Each batch produces incremental snapshots from first completed step
        to final step. Targets are repeated per batch, but grouping remains
        by batch_id to avoid leakage in validation.
        """
        with self.inspector._connect() as conn:
            batch_ids_sql = """
            SELECT DISTINCT batchID FROM measurements ORDER BY batchID
            """
            batch_ids = pd.read_sql_query(batch_ids_sql, conn)["batchID"].unique()

        rows = []
        for batch_id in batch_ids:
            batch_data = self.load_batch_data(int(batch_id))
            measurements = batch_data["measurements"]
            targets = batch_data["targets"]

            if measurements.empty:
                continue

            target_values = {}
            if targets:
                target_values = {
                    "ph": float(targets[0]["ph"]) if targets[0].get("ph") is not None else None,
                    "viscosity": float(targets[0]["viscosity"]) if targets[0].get("viscosity") is not None else None,
                    "chlorides": float(targets[0]["chlorides"]) if targets[0].get("chlorides") is not None else None,
                }

            snapshots = self.snapshot_builder.build_batch_snapshots(
                measurements,
                batch_id=int(batch_id),
                target_values=target_values,
            )
            if not snapshots:
                continue

            snapshot_weight = 1.0 / len(snapshots)
            for snapshot in snapshots:
                rows.append(
                    {
                        "batch_id": int(batch_id),
                        "checkpoint_order": int(snapshot["checkpoint_order"]),
                        "completed_steps": int(snapshot["completed_steps"]),
                        "is_final_snapshot": bool(snapshot["is_final_snapshot"]),
                        "snapshot_weight": snapshot_weight,
                        "has_targets": any(value is not None for value in target_values.values()),
                        **snapshot["features"],
                        "target_ph": target_values.get("ph"),
                        "target_viscosity": target_values.get("viscosity"),
                        "target_chlorides": target_values.get("chlorides"),
                    }
                )

        return pd.DataFrame(rows)

    def get_data_summary(self) -> dict[str, Any]:
        """Get summary statistics about the dataset."""
        measurements = self.load_measurements()
        report = self.quality_inspector.inspect_measurements(measurements)

        batch_features = self.build_batch_features_dataset()
        batches_with_targets = batch_features[
            batch_features["has_targets"]
        ].shape[0]

        return {
            "total_measurements": report.total_measurements,
            "total_batches": report.total_batches,
            "batches_with_targets": batches_with_targets,
            "measurements_with_issues": report.measurements_with_issues,
            "missing_count": report.missing_count,
            "invalid_count": report.invalid_count,
            "out_of_range_count": report.out_of_range_count,
            "issues_by_field": report.issues_by_field,
        }

    def get_api_summary(self) -> dict[str, Any]:
        """Build API-oriented summary aligned with the product prompt."""
        measurements = self.load_measurements()
        report = self.quality_inspector.inspect_measurements(measurements)

        if measurements.empty:
            return {
                "total_measurements": 0,
                "total_batches": 0,
                "labeled_batches": {"ph": 0, "viscosity": 0, "chlorides": 0},
                "protocol_count": 0,
                "batches_with_repeated_protocols": 0,
                "missing_count": 0,
                "suspicious_count": 0,
                "step_distribution": {},
            }

        batch_ids = sorted(int(batch_id) for batch_id in measurements["batchID"].dropna().astype(int).unique())
        labeled = {"ph": 0, "viscosity": 0, "chlorides": 0}
        repeated_protocols = 0

        protocol_count = 0
        with self.inspector._connect() as conn:
            if self.inspector.has_table("Testing_Protocols"):
                protocol_count = int(
                    pd.read_sql_query("SELECT COUNT(*) AS cnt FROM Testing_Protocols", conn)["cnt"].iloc[0]
                )
                repeated_protocols = int(
                    pd.read_sql_query(
                        """
                        SELECT COUNT(*) AS cnt
                        FROM (
                            SELECT batch_id
                            FROM Testing_Protocols
                            GROUP BY batch_id
                            HAVING COUNT(*) > 1
                        )
                        """,
                        conn,
                    )["cnt"].iloc[0]
                )

        for batch_id in batch_ids:
            targets = load_targets(batch_id, self.db_path, TargetProtocolPolicy.latest)
            if not targets:
                continue
            target_row = targets[-1]
            if target_row.get("ph") is not None:
                labeled["ph"] += 1
            if target_row.get("viscosity") is not None:
                labeled["viscosity"] += 1
            if target_row.get("chlorides") is not None:
                labeled["chlorides"] += 1

        step_distribution_series = measurements.groupby("batchID").size().sort_index()
        step_distribution = {
            str(int(batch_id)): int(count)
            for batch_id, count in step_distribution_series.items()
        }

        return {
            "total_measurements": report.total_measurements,
            "total_batches": report.total_batches,
            "labeled_batches": labeled,
            "protocol_count": protocol_count,
            "batches_with_repeated_protocols": repeated_protocols,
            "missing_count": report.missing_count,
            "suspicious_count": report.invalid_count + report.out_of_range_count,
            "step_distribution": step_distribution,
        }
