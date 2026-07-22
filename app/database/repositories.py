from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from app.data.component_normalizer import ComponentNormalizer
from app.data.data_quality import DataQualityInspector
from app.data.target_loader import load_targets
from app.database.source import DatabaseInspector


class BatchRepository:
    def __init__(self, db_path: Path | str) -> None:
        self.inspector = DatabaseInspector(db_path)
        self.normalizer = ComponentNormalizer()
        self.quality_inspector = DataQualityInspector()

    def get_batch(self, batch_id: int) -> dict[str, Any] | None:
        if not self.inspector.has_table("Batches"):
            return None
        with self.inspector._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT batch_id, product_id, batch_number, production_date, status FROM Batches WHERE batch_id = ?",
                (batch_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "batch_id": row[0],
            "product_id": row[1],
            "batch_number": row[2],
            "production_date": row[3],
            "status": row[4],
        }

    def list_batches(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.inspector.list_batches(**kwargs)

    def get_batch_measurements(self, batch_id: int) -> pd.DataFrame:
        if not self.inspector.has_table("measurements"):
            return pd.DataFrame()

        query = """
        SELECT
            m.*,
            lp.step_order AS loading_process_step_order,
            lp.stage AS loading_process_stage,
            lp.status AS loading_process_status,
            lp.type_id AS loading_process_type_id
        FROM measurements AS m
        LEFT JOIN Loading_Process AS lp
            ON m.loading_step_id = lp.loading_step_id
        WHERE CAST(m.batchID AS INTEGER) = ?
        ORDER BY
            CASE WHEN lp.step_order IS NULL THEN 1 ELSE 0 END,
            lp.step_order ASC,
            m.timestamp ASC,
            m.id ASC
        """
        with self.inspector._connect() as conn:
            try:
                return pd.read_sql_query(query, conn, params=(batch_id,))
            except sqlite3.OperationalError:
                fallback = """
                SELECT *
                FROM measurements
                WHERE CAST(batchID AS INTEGER) = ?
                ORDER BY timestamp ASC, id ASC
                """
                return pd.read_sql_query(fallback, conn, params=(batch_id,))

    def get_batch_protocols(self, batch_id: int) -> list[dict[str, Any]]:
        if not self.inspector.has_table("Testing_Protocols"):
            return []

        query = """
        SELECT
            tp.protocol_id,
            tp.product_id,
            tp.test_date,
            tp.ph,
            tp.chlorides,
            tp.viscosity,
            tp.batch_id,
            tp.is_compliant,
            tp.compliance_percent,
            tv.ph_value,
            tv.viscosity_value,
            tv.chlorides_value
        FROM Testing_Protocols AS tp
        LEFT JOIN testing_protocol_values AS tv
            ON tp.protocol_id = tv.protocol_id
        WHERE tp.batch_id = ?
        ORDER BY tp.test_date DESC, tp.protocol_id DESC
        """
        with self.inspector._connect() as conn:
            try:
                df = pd.read_sql_query(query, conn, params=(batch_id,))
            except sqlite3.OperationalError:
                df = pd.read_sql_query(
                    "SELECT * FROM Testing_Protocols WHERE batch_id = ? ORDER BY test_date DESC, protocol_id DESC",
                    conn,
                    params=(batch_id,),
                )
        return df.to_dict(orient="records")

    def get_product(self, product_id: int | None) -> dict[str, Any] | None:
        if product_id is None or not self.inspector.has_table("Products"):
            return None
        with self.inspector._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT product_id, name, category, base, base_code,
                       viscosity_thickener, viscosity_softener, ph_corrector,
                       viscosity_adjustment
                FROM Products
                WHERE product_id = ?
                """,
                (product_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "product_id": row[0],
            "name": row[1],
            "category": row[2],
            "base": row[3],
            "base_code": row[4],
            "viscosity_thickener": row[5],
            "viscosity_softener": row[6],
            "ph_corrector": row[7],
            "viscosity_adjustment": row[8],
        }

    def get_batch_detail(self, batch_id: int) -> dict[str, Any] | None:
        batch = self.get_batch(batch_id)
        if batch is None:
            return None

        measurements = self.get_batch_measurements(batch_id)
        components = self.normalizer.normalize_batch_components(measurements)
        protocols = self.get_batch_protocols(batch_id)
        product = self.get_product(batch.get("product_id"))
        targets = load_targets(batch_id, self.inspector.db_path)
        quality_report = self.quality_inspector.inspect_measurements(measurements)

        return {
            "batch": batch,
            "product": product,
            "targets": targets,
            "measurements": measurements.to_dict(orient="records"),
            "components": components.to_dict(orient="records"),
            "protocols": protocols,
            "data_quality": {
                "total_measurements": quality_report.total_measurements,
                "measurements_with_issues": quality_report.measurements_with_issues,
                "missing_count": quality_report.missing_count,
                "invalid_count": quality_report.invalid_count,
                "out_of_range_count": quality_report.out_of_range_count,
                "issues_by_field": quality_report.issues_by_field,
                "sample_issues": [
                    {
                        "batch_id": issue.batch_id,
                        "measurement_id": issue.measurement_id,
                        "field_name": issue.field_name,
                        "description": issue.description,
                    }
                    for issue in quality_report.sample_issues
                ],
            },
        }
