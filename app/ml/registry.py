from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.database.ml_storage import ModelRegistry


class ModelManager:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.registry = ModelRegistry(db_path)

    def register_model(self, model_type: str, target: str, artifact_path: str, features: dict[str, Any], metrics: dict[str, float]) -> str:
        model_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        version = self.registry.get_next_version(model_type=model_type, target=target)
        conn = self.registry._connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO model_registry (model_id, model_type, target, created_at, status, version, artifact_path, metrics_json, features_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                model_id,
                model_type,
                target,
                created_at,
                "challenger",
                version,
                artifact_path,
                json.dumps(metrics, ensure_ascii=False),
                json.dumps(features, ensure_ascii=False),
            ),
        )
        conn.commit()
        conn.close()
        return model_id
