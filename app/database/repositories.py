from __future__ import annotations

from pathlib import Path
from typing import Any

from app.database.source import DatabaseInspector


class BatchRepository:
    def __init__(self, db_path: Path | str) -> None:
        self.inspector = DatabaseInspector(db_path)

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
