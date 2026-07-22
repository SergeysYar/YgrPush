from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config import settings


class DatabaseInspector:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or settings.db_path)

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise sqlite3.OperationalError(f"Database not found: {self.db_path}")
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def has_table(self, table_name: str) -> bool:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            return cur.fetchone() is not None

    def inspect_tables(self) -> list[str]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [row[0] for row in cur.fetchall()]

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({table_name});")
            return [
                {
                    "cid": row[0],
                    "name": row[1],
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "default": row[4],
                    "pk": bool(row[5]),
                }
                for row in cur.fetchall()
            ]

    def list_batches(self, status: str | None = None, product_id: int | None = None, has_protocol: bool | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        sql = ["SELECT batch_id, product_id, batch_number, production_date, status FROM Batches"]
        conditions: list[str] = []
        params: list[Any] = []
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if product_id is not None:
            conditions.append("product_id = ?")
            params.append(product_id)
        if has_protocol is not None:
            if has_protocol:
                conditions.append("batch_id IN (SELECT DISTINCT batch_id FROM Testing_Protocols)")
            else:
                conditions.append("batch_id NOT IN (SELECT DISTINCT batch_id FROM Testing_Protocols)")
        if conditions:
            sql.append("WHERE " + " AND ".join(conditions))
        sql.append("ORDER BY production_date DESC LIMIT ? OFFSET ?")
        params.extend([limit, offset])
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(" ".join(sql), tuple(params))
            rows = cur.fetchall()
        return [
            {
                "batch_id": row[0],
                "product_id": row[1],
                "batch_number": row[2],
                "production_date": row[3],
                "status": row[4],
            }
            for row in rows
        ]


def inspect_database(db_path: Path | str | None = None) -> None:
    inspector = DatabaseInspector(db_path)
    if not inspector.db_path.exists():
        print(f"Database file not found: {inspector.db_path}")
        return
    tables = inspector.inspect_tables()
    print("Tables:")
    for table in tables:
        print(f"  - {table}")
    if inspector.has_table("measurements"):
        print("\nSchema measurements:")
        schema = inspector.get_table_schema("measurements")
        for col in schema:
            print(f"  - {col['name']} ({col['type']}) notnull={col['notnull']} pk={col['pk']}")
        with inspector._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM measurements")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT batchID) FROM measurements")
            batches = cur.fetchone()[0]
        print(f"\nMeasurements rows: {total}")
        print(f"Unique batchIDs: {batches}")


def ping_database(db_path: Path | str | None = None) -> bool:
    try:
        inspector = DatabaseInspector(db_path)
        _ = inspector.inspect_tables()
        return True
    except sqlite3.Error:
        return False
