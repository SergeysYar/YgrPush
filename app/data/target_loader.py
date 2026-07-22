from __future__ import annotations

import sqlite3
from enum import Enum
from pathlib import Path
from typing import Any

from app.config import settings
from app.data.numeric_parser import parse_numeric


class TargetProtocolPolicy(str, Enum):
    latest = "latest"
    first = "first"
    all = "all"


def parse_target_text(value: Any) -> float | None:
    numeric = parse_numeric(value)
    if numeric is not None:
        return numeric
    if isinstance(value, str):
        cleaned = value.strip().replace(",", ".")
        if cleaned == "":
            return None
    return None


def load_targets(batch_id: int, db_path: Path | str | None = None, policy: TargetProtocolPolicy | str | None = None) -> list[dict[str, Any]]:
    db_path = Path(db_path or settings.db_path)
    policy = TargetProtocolPolicy(policy or settings.target_protocol_policy)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        query = "SELECT tp.protocol_id, tp.batch_id, tp.test_date, tp.ph, tp.chlorides, tp.viscosity, v.ph_value, v.viscosity_value, v.chlorides_value FROM Testing_Protocols tp LEFT JOIN testing_protocol_values v ON tp.protocol_id = v.protocol_id WHERE tp.batch_id = ? ORDER BY tp.test_date ASC, tp.protocol_id ASC"
        cur.execute(query, (batch_id,))
        rows = cur.fetchall()
    targets: list[dict[str, Any]] = []
    for row in rows:
        targets.append(
            {
                "protocol_id": row[0],
                "batch_id": row[1],
                "test_date": row[2],
                "ph": parse_numeric(row[6]) if row[6] is not None else parse_target_text(row[3]),
                "viscosity": parse_numeric(row[7]) if row[7] is not None else parse_target_text(row[5]),
                "chlorides": parse_numeric(row[8]) if row[8] is not None else parse_target_text(row[4]),
                "source": "testing_protocol_values" if row[6] is not None or row[7] is not None or row[8] is not None else "Testing_Protocols",
            }
        )
    if policy == TargetProtocolPolicy.first:
        return targets[:1]
    if policy == TargetProtocolPolicy.latest and targets:
        return [targets[-1]]
    return targets
