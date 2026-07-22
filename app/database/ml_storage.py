from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config import settings

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
        conn = sqlite3.connect(self.db_path)
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


def initialize_ml_storage(db_path: Path | str | None = None, create_tables: bool = True) -> ModelRegistry | None:
    try:
        registry = ModelRegistry(db_path)
        if not create_tables:
            return registry
        return registry
    except sqlite3.Error:
        return None
