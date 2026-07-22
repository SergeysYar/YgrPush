from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config import settings


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()

CREATE_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS model_registry (
        model_id TEXT PRIMARY KEY,
        model_type TEXT NOT NULL,
        target TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,
        version INTEGER NOT NULL,
        artifact_path TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        features_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        model_ids TEXT NOT NULL,
        batch_ids TEXT NOT NULL,
        settings_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_metrics (
        model_id TEXT NOT NULL,
        target TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        PRIMARY KEY (model_id, target, metric_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS batch_predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER NOT NULL,
        checkpoint_order INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        model_version TEXT NOT NULL,
        target TEXT NOT NULL,
        predicted_value REAL NOT NULL,
        lower_bound REAL NOT NULL,
        upper_bound REAL NOT NULL,
        status TEXT NOT NULL,
        confidence TEXT NOT NULL,
        actual_value REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_items (
        item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER NOT NULL,
        feature_name TEXT NOT NULL,
        feature_value REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_quality_issues (
        issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER NOT NULL,
        issue_type TEXT NOT NULL,
        description TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
]


class ModelRegistry:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or settings.ml_storage_path)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, factory=ClosingConnection)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            for ddl in CREATE_TABLES_SQL:
                cur.execute(ddl)
            conn.commit()

    def list_models(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT model_id, model_type, target, created_at, status, artifact_path, metrics_json, features_json FROM model_registry ORDER BY created_at DESC")
            return [dict(row) for row in cur.fetchall()]

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM model_registry WHERE model_id = ?", (model_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_champion_models(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM model_registry WHERE status = 'champion' ORDER BY created_at DESC")
            return [dict(row) for row in cur.fetchall()]

    def get_latest_model(self, target: str, model_type: str | None = None) -> dict[str, Any] | None:
        with self._connect() as conn:
            cur = conn.cursor()
            if model_type:
                cur.execute(
                    "SELECT * FROM model_registry WHERE target = ? AND model_type = ? ORDER BY created_at DESC LIMIT 1",
                    (target, model_type),
                )
            else:
                cur.execute(
                    "SELECT * FROM model_registry WHERE target = ? ORDER BY created_at DESC LIMIT 1",
                    (target,),
                )
            row = cur.fetchone()
            return dict(row) if row else None

    def promote_model(self, model_id: str) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE model_registry SET status = 'archived' WHERE status = 'champion'")
            cur.execute("UPDATE model_registry SET status = 'champion' WHERE model_id = ?", (model_id,))
            conn.commit()

    def get_predictions_for_batch(self, batch_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM batch_predictions WHERE batch_id = ? ORDER BY created_at DESC",
                (batch_id,),
            )
            return [dict(row) for row in cur.fetchall()]


class PredictionRepository:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or settings.ml_storage_path)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, factory=ClosingConnection)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            for ddl in CREATE_TABLES_SQL:
                cur.execute(ddl)
            conn.commit()

    def save_batch_prediction(
        self,
        batch_id: int,
        checkpoint_order: int,
        created_at: str,
        model_version: str,
        target: str,
        predicted_value: float,
        lower_bound: float,
        upper_bound: float,
        status: str,
        confidence: str,
        actual_value: float | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO batch_predictions (batch_id, checkpoint_order, created_at, model_version, target, predicted_value, lower_bound, upper_bound, status, confidence, actual_value) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    checkpoint_order,
                    created_at,
                    model_version,
                    target,
                    predicted_value,
                    lower_bound,
                    upper_bound,
                    status,
                    confidence,
                    actual_value,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def save_prediction_items(self, prediction_id: int, features: dict[str, Any]) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            for feature_name, feature_value in features.items():
                cur.execute(
                    "INSERT INTO prediction_items (prediction_id, feature_name, feature_value) VALUES (?, ?, ?)",
                    (prediction_id, feature_name, float(feature_value) if feature_value is not None else None),
                )
            conn.commit()

    def list_predictions(
        self,
        batch_id: int | None = None,
        target: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            sql = "SELECT * FROM batch_predictions"
            params: list[Any] = []
            conditions: list[str] = []
            if batch_id is not None:
                conditions.append("batch_id = ?")
                params.append(batch_id)
            if target is not None:
                conditions.append("target = ?")
                params.append(target)
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            cur.execute(sql, tuple(params))
            return [dict(row) for row in cur.fetchall()]

    def list_unmatched_predictions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM batch_predictions WHERE actual_value IS NULL ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    def update_actual_value(self, prediction_id: int, actual_value: float) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE batch_predictions SET actual_value = ? WHERE prediction_id = ?",
                (actual_value, prediction_id),
            )
            conn.commit()


class DataQualityRepository:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or settings.ml_storage_path)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, factory=ClosingConnection)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            for ddl in CREATE_TABLES_SQL:
                cur.execute(ddl)
            conn.commit()

    def save_issues(
        self,
        issues: list[dict[str, Any]],
        created_at: str,
    ) -> int:
        if not issues:
            return 0

        with self._connect() as conn:
            cur = conn.cursor()
            for issue in issues:
                batch_id = int(issue["batch_id"]) if issue.get("batch_id") is not None else 0
                description = issue.get("description") or ""
                field_name = issue.get("field_name")
                measurement_id = issue.get("measurement_id")
                value = issue.get("value")
                if field_name:
                    description = f"[field={field_name}] {description}".strip()
                if measurement_id is not None:
                    description = f"[measurement_id={measurement_id}] {description}".strip()
                if value is not None:
                    description = f"{description} [value={value}]".strip()

                cur.execute(
                    """
                    INSERT INTO data_quality_issues (batch_id, issue_type, description, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        issue["issue_type"],
                        description,
                        created_at,
                    ),
                )
            conn.commit()
            return len(issues)

    def list_issues(
        self,
        batch_id: int | None = None,
        issue_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            sql = "SELECT issue_id, batch_id, issue_type, description, created_at FROM data_quality_issues"
            params: list[Any] = []
            conditions: list[str] = []
            if batch_id is not None:
                conditions.append("batch_id = ?")
                params.append(batch_id)
            if issue_type is not None:
                conditions.append("issue_type = ?")
                params.append(issue_type)
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY created_at DESC, issue_id DESC LIMIT ?"
            params.append(limit)
            cur.execute(sql, tuple(params))
            return [dict(row) for row in cur.fetchall()]


def initialize_ml_storage(db_path: Path | str | None = None, create_tables: bool = True) -> ModelRegistry | None:
    try:
        registry = ModelRegistry(db_path)
        if not create_tables:
            return registry
        return registry
    except sqlite3.Error:
        return None
