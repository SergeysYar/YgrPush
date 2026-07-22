from pathlib import Path

from app.ml.prediction_service import PredictionService


def test_train_if_no_model_uses_snapshot_mode(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyRegistry:
        def __init__(self, _path):
            pass

        def list_models(self):
            return []

    class DummyPipeline:
        def __init__(self, db_path, ml_storage_path):
            captured["db_path"] = db_path
            captured["ml_storage_path"] = ml_storage_path

        def train_all(self, snapshot_mode=False, protocol_policy=None, model_types=None):
            captured["snapshot_mode"] = snapshot_mode
            captured["protocol_policy"] = protocol_policy
            captured["model_types"] = model_types
            return {}

    monkeypatch.setattr("app.ml.prediction_service.ModelRegistry", DummyRegistry)
    monkeypatch.setattr("app.ml.prediction_service.TrainingPipeline", DummyPipeline)

    service = PredictionService(
        db_path=Path(tmp_path / "production.db"),
        ml_storage_path=Path(tmp_path / "ml_storage.db"),
    )
    service.train_if_no_model()

    assert captured["snapshot_mode"] is True
    assert captured["protocol_policy"] is not None
    assert captured["model_types"] is None
