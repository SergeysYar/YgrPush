from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.database.source import inspect_database, ping_database
from app.database.ml_storage import initialize_ml_storage

router = APIRouter()


@router.get("", summary="Health check")
def health() -> dict[str, object]:
    db_available = ping_database(settings.db_path)
    ml_storage_available = initialize_ml_storage(settings.ml_storage_path, create_tables=False) is not None
    return {
        "status": "ok" if db_available and ml_storage_available else "degraded",
        "database": {"available": db_available, "path": str(settings.db_path)},
        "ml_storage": {"available": ml_storage_available, "path": str(settings.ml_storage_path)},
    }
