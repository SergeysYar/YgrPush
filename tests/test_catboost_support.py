import pandas as pd

from app.config import settings
from app.ml.catboost_model import CatBoostModel
from app.ml.service import TrainingPipeline


def _build_training_data() -> dict[str, object]:
    frame = pd.DataFrame(
        {
            "batch_id": [1, 2],
            "has_targets": [True, True],
            "snapshot_weight": [1.0, 1.0],
            "num_steps": [3.0, 4.0],
            "average_ph": [5.5, 5.8],
            "target_ph": [5.9, 6.0],
            "target_viscosity": [3200.0, 3300.0],
            "target_chlorides": [1.1, 1.2],
        }
    )
    return {
        "all_batches": frame,
        "ph_data": frame[frame["target_ph"].notna()].reset_index(drop=True),
        "viscosity_data": frame[frame["target_viscosity"].notna()].reset_index(drop=True),
        "chlorides_data": frame[frame["target_chlorides"].notna()].reset_index(drop=True),
        "complete_data": frame.reset_index(drop=True),
    }


def test_catboost_model_availability_flag_is_boolean():
    assert isinstance(CatBoostModel.is_available(), bool)
    assert isinstance(CatBoostModel.is_shap_available(), bool)


def test_training_pipeline_reports_catboost_unavailable(monkeypatch, tmp_path):
    pipeline = TrainingPipeline(tmp_path / "production.db", tmp_path / "ml_storage.db")
    training_data = _build_training_data()

    monkeypatch.setattr(CatBoostModel, "is_available", classmethod(lambda cls: False))

    results = pipeline.train_and_validate_catboost(training_data)

    assert results["ph"]["status"] == "unavailable"
    assert results["ph"]["reason"] == "catboost_not_installed"
    assert results["viscosity"]["status"] == "unavailable"
    assert results["chlorides"]["status"] == "unavailable"


def test_training_pipeline_skips_catboost_when_batches_are_insufficient(monkeypatch, tmp_path):
    pipeline = TrainingPipeline(tmp_path / "production.db", tmp_path / "ml_storage.db")
    training_data = _build_training_data()

    monkeypatch.setattr(CatBoostModel, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(settings, "catboost_min_labeled_batches", 5)

    results = pipeline.train_and_validate_catboost(training_data)

    assert results["ph"]["status"] == "skipped"
    assert results["ph"]["reason"] == "insufficient_labeled_batches"
    assert results["ph"]["available_batches"] == 2
