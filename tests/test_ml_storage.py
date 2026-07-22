import tempfile
from pathlib import Path

from app.database.ml_storage import DataQualityRepository, PredictionRepository


def test_prediction_repository_save_and_list():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "ml_storage.db"
        repository = PredictionRepository(db_path)

        prediction_id = repository.save_batch_prediction(
            batch_id=101,
            checkpoint_order=3,
            created_at="2026-07-22T00:00:00Z",
            model_version="test-model-1",
            target="ph",
            predicted_value=6.8,
            lower_bound=6.3,
            upper_bound=7.3,
            status="ok",
            confidence="approximate",
            actual_value=None,
        )
        repository.save_prediction_items(prediction_id, {"feature_a": 1.5, "feature_b": 2.4})

        predictions = repository.list_predictions(batch_id=101)
        assert len(predictions) == 1
        assert predictions[0]["prediction_id"] == prediction_id
        assert predictions[0]["model_version"] == "test-model-1"
        assert predictions[0]["target"] == "ph"
        assert predictions[0]["predicted_value"] == 6.8

        unmatched = repository.list_unmatched_predictions()
        assert any(p["prediction_id"] == prediction_id for p in unmatched)


def test_data_quality_repository_save_and_list():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "ml_storage.db"
        repository = DataQualityRepository(db_path)

        saved = repository.save_issues(
            [
                {
                    "batch_id": 101,
                    "measurement_id": 55,
                    "issue_type": "out_of_range",
                    "field_name": "startpH",
                    "value": 19.2,
                    "description": "pH is out of allowed range",
                },
                {
                    "batch_id": None,
                    "measurement_id": None,
                    "issue_type": "missing",
                    "field_name": "timestamp",
                    "value": None,
                    "description": "Missing values in timestamp: 2",
                },
            ],
            created_at="2026-07-22T00:00:00+00:00",
        )

        assert saved == 2

        issues = repository.list_issues(limit=10)
        assert len(issues) == 2
        assert issues[0]["issue_type"] in {"out_of_range", "missing"}

        filtered = repository.list_issues(batch_id=101, issue_type="out_of_range", limit=10)
        assert len(filtered) == 1
        assert filtered[0]["batch_id"] == 101
        assert "startpH" in filtered[0]["description"]
