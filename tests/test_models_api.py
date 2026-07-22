import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_settings
from app.api.main import app
from app.config import Settings
from app.database.ml_storage import ModelRegistry
from app.ml.registry import ModelManager


def test_models_summary_and_runs_routes(tmp_path):
    ml_storage_path = Path(tmp_path / "ml_storage.db")
    manager = ModelManager(ml_storage_path)
    registry = ModelRegistry(ml_storage_path)

    model_id_1 = manager.register_model(
        model_type="ridge",
        target="ph",
        artifact_path="artifacts/ridge_ph_v1.joblib",
        features={"feature_names": ["a"]},
        metrics={"mae": 0.11},
    )
    model_id_2 = manager.register_model(
        model_type="ridge",
        target="ph",
        artifact_path="artifacts/ridge_ph_v2.joblib",
        features={"feature_names": ["a", "b"]},
        metrics={"mae": 0.09},
    )
    registry.promote_model(model_id_2)
    registry.create_training_run(
        run_id="run-1",
        created_at="2026-07-22T10:00:00",
        model_ids=[model_id_1, model_id_2],
        batch_ids=[10, 20],
        settings_payload={"snapshot_mode": True, "protocol_policy": "latest"},
    )

    def override_settings():
        return Settings(db_path=tmp_path / "production.db", ml_storage_path=ml_storage_path)

    app.dependency_overrides[get_settings] = override_settings
    client = TestClient(app)

    list_response = client.get("/api/v1/models")
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 2

    summary_response = client.get("/api/v1/models/summary")
    assert summary_response.status_code == 200
    summary_payload = summary_response.json()
    assert summary_payload["total_models"] == 2
    assert summary_payload["targets"]["ph"]["champion"]["model_id"] == model_id_2
    assert len(summary_payload["training_runs"]) == 1

    runs_response = client.get("/api/v1/models/runs")
    assert runs_response.status_code == 200
    runs_payload = runs_response.json()
    assert runs_payload["count"] == 1
    assert runs_payload["items"][0]["batch_ids"] == [10, 20]

    app.dependency_overrides.clear()
