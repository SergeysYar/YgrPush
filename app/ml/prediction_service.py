from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from app.config import settings
from app.data.dataset_builder import DatasetBuilder
from app.database.repositories import BatchRepository
from app.features.batch_features import BatchFeatureBuilder
from app.database.ml_storage import ModelRegistry, PredictionRepository
from app.ml.service import TrainingPipeline
from app.ml.baseline import BaselineModel
from app.ml.ridge import RidgeModel
from app.ml.bayesian import BayesianRidgeModel
from app.ml.pls import PLSModel


MODEL_CLASSES = {
    "baseline": BaselineModel,
    "ridge": RidgeModel,
    "bayesian": BayesianRidgeModel,
    "pls_multivariate": PLSModel,
}


class PredictionService:
    def __init__(self, db_path: Path | str | None = None, ml_storage_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or settings.db_path)
        self.ml_storage_path = Path(ml_storage_path or settings.ml_storage_path)
        self.dataset_builder = DatasetBuilder(self.db_path)
        self.feature_builder = BatchFeatureBuilder()
        self.registry = ModelRegistry(self.ml_storage_path)
        self.prediction_repository = PredictionRepository(self.ml_storage_path)
        self.batch_repository = BatchRepository(self.db_path)

    def _resolve_model_meta(self, target: str) -> dict[str, Any] | None:
        champions = self.registry.get_champion_models()
        for model_meta in champions:
            if model_meta["target"] == target:
                return model_meta

        latest = self.registry.get_latest_model(target)
        if latest:
            return latest

        # fallback to PLS multivariate if available
        pls_candidate = self.registry.get_latest_model("all", model_type="pls_multivariate")
        if pls_candidate:
            return pls_candidate

        return None

    def _load_model(self, model_meta: dict[str, Any]) -> Any:
        model_type = model_meta["model_type"]
        model_cls = MODEL_CLASSES.get(model_type)
        if model_cls is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        model = model_cls()
        artifact_path = Path(model_meta["artifact_path"])
        if not artifact_path.is_absolute():
            artifact_path = self.ml_storage_path.parent / artifact_path
        model.load(str(artifact_path))
        return model

    def _build_feature_vector(self, measurements: pd.DataFrame) -> pd.DataFrame:
        features = self.feature_builder.build_full_batch_features(measurements)
        return pd.DataFrame([features])

    def _make_interval(self, model: Any, X: pd.DataFrame, prediction: float) -> tuple[float, float, str]:
        if isinstance(model, BayesianRidgeModel) and getattr(model, "pipeline", None) is not None:
            bayesian = model.pipeline.named_steps.get("bayesian")
            if bayesian is not None and hasattr(bayesian, "predict"):
                y_mean, y_std = bayesian.predict(X, return_std=True)
                std = float(y_std[0])
                lower = float(prediction - 1.96 * std)
                upper = float(prediction + 1.96 * std)
                return lower, upper, "bayesian"

        width = max(abs(prediction) * 0.1, 0.5)
        lower = float(prediction - width)
        upper = float(prediction + width)
        return lower, upper, "approximate"

    def _parse_metrics(self, model_meta: dict[str, Any]) -> dict[str, Any]:
        raw_metrics = model_meta.get("metrics_json")
        if isinstance(raw_metrics, dict):
            return raw_metrics
        if not raw_metrics:
            return {}
        try:
            return ast.literal_eval(raw_metrics)
        except (ValueError, SyntaxError):
            return {}

    def _load_quality_standard(self, batch_id: int, target: str) -> tuple[float | None, float | None]:
        batch = self.batch_repository.get_batch(batch_id)
        if batch is None:
            return None, None

        product = self.batch_repository.get_product(batch.get("product_id"))
        category = product.get("category") if product else None
        if not category:
            return None, None

        try:
            with self.batch_repository.inspector._connect() as conn:
                query = """
                SELECT min_value, max_value
                FROM quality_targets
                WHERE category = ? AND indicator = ?
                LIMIT 1
                """
                row = pd.read_sql_query(query, conn, params=(category, target))
        except Exception:
            return None, None

        if row.empty:
            return None, None
        min_value = row.iloc[0]["min_value"]
        max_value = row.iloc[0]["max_value"]
        return (
            float(min_value) if min_value is not None else None,
            float(max_value) if max_value is not None else None,
        )

    def _resolve_status(
        self,
        batch_id: int | None,
        target: str,
        value: float,
        lower: float,
        upper: float,
    ) -> str:
        if batch_id is None:
            return "no_standard"
        min_value, max_value = self._load_quality_standard(batch_id, target)
        if min_value is None and max_value is None:
            return "no_standard"

        def inside(number: float) -> bool:
            if min_value is not None and number < min_value:
                return False
            if max_value is not None and number > max_value:
                return False
            return True

        if inside(lower) and inside(upper):
            return "normal"
        if inside(value):
            return "risk"
        return "high_risk"

    def _resolve_confidence(
        self,
        training_batches: int,
        lower: float,
        upper: float,
        prediction: float,
        quality: dict[str, Any] | None = None,
    ) -> str:
        quality = quality or {}
        issues = int(quality.get("measurements_with_issues", 0))
        width = abs(upper - lower)
        scale = max(abs(prediction), 1.0)
        relative_width = width / scale

        if (
            training_batches >= settings.confidence_high_min_batches
            and relative_width <= 0.25
            and issues == 0
        ):
            return "high"
        if training_batches >= max(5, settings.confidence_high_min_batches // 2) and relative_width <= 0.6:
            return "medium"
        return "low"

    def _predict_target(
        self,
        target: str,
        X: pd.DataFrame,
        batch_id: int | None = None,
        quality: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model_meta = self._resolve_model_meta(target)
        if model_meta is None:
            return {
                "value": None,
                "lower": None,
                "upper": None,
                "confidence": "low",
                "model": None,
                "status": "no_model_available",
                "training_batches": 0,
            }

        model = self._load_model(model_meta)
        raw_pred = model.predict(X)

        if isinstance(raw_pred, np.ndarray):
            if raw_pred.ndim == 2 and raw_pred.shape[1] == 3 and model_meta.get("model_type") == "pls_multivariate":
                idx = {"ph": 0, "viscosity": 1, "chlorides": 2}[target]
                pred_value = float(raw_pred[0, idx])
            else:
                pred_value = float(raw_pred.ravel()[0])
        else:
            pred_value = float(raw_pred)

        lower, upper, _ = self._make_interval(model, X, pred_value)
        metrics = self._parse_metrics(model_meta)
        training_batches = int(metrics.get("n_samples") or metrics.get("n_training_batches") or 0)
        confidence = self._resolve_confidence(training_batches, lower, upper, pred_value, quality)
        status = self._resolve_status(batch_id, target, pred_value, lower, upper)

        return {
            "value": pred_value,
            "lower": lower,
            "upper": upper,
            "confidence": confidence,
            "model": model_meta["model_id"],
            "status": status,
            "training_batches": training_batches,
            "top_factors": [],
        }

    def _build_similar_batches(self, current_features: dict[str, float], batch_id: int, limit: int = 5) -> list[dict[str, Any]]:
        all_batches = self.dataset_builder.build_batch_features_dataset()
        if all_batches.empty:
            return []

        feature_cols = [col for col in all_batches.columns if col not in {"batch_id", "has_targets", "target_ph", "target_viscosity", "target_chlorides"}]
        if not feature_cols:
            return []

        current_vector = np.array([current_features.get(col, 0.0) for col in feature_cols], dtype=float)
        distances = []
        for _, row in all_batches.iterrows():
            if int(row["batch_id"]) == batch_id:
                continue
            vector = np.array([float(row.get(col, 0.0)) for col in feature_cols], dtype=float)
            distance = float(np.linalg.norm(vector - current_vector))
            distances.append({
                "batch_id": int(row["batch_id"]),
                "distance": distance,
                "target_ph": row.get("target_ph"),
                "target_viscosity": row.get("target_viscosity"),
                "target_chlorides": row.get("target_chlorides"),
            })

        return sorted(distances, key=lambda item: item["distance"])[:limit]

    def _get_data_quality(self, measurements: pd.DataFrame) -> dict[str, Any]:
        report = self.dataset_builder.quality_inspector.inspect_measurements(measurements)
        return {
            "total_measurements": report.total_measurements,
            "measurements_with_issues": report.measurements_with_issues,
            "missing_count": report.missing_count,
            "invalid_count": report.invalid_count,
            "out_of_range_count": report.out_of_range_count,
            "issues_by_field": report.issues_by_field,
            "sample_issues": [
                {
                    "batch_id": issue.batch_id,
                    "measurement_id": issue.measurement_id,
                    "field_name": issue.field_name,
                    "description": issue.description,
                }
                for issue in report.sample_issues
            ],
        }

    def _persist_batch_predictions(
        self,
        batch_id: int,
        checkpoint_order: int,
        model_id: str,
        predictions: dict[str, dict[str, Any]],
        actual_values: dict[str, float | None],
        features: dict[str, Any],
    ) -> None:
        created_at = pd.Timestamp.utcnow().isoformat()
        for target_name, prediction in predictions.items():
            prediction_id = self.prediction_repository.save_batch_prediction(
                batch_id=batch_id,
                checkpoint_order=checkpoint_order,
                created_at=created_at,
                model_version=model_id,
                target=target_name,
                predicted_value=prediction["value"],
                lower_bound=prediction["lower"],
                upper_bound=prediction["upper"],
                status=prediction["status"],
                confidence=prediction["confidence"],
                actual_value=actual_values.get(target_name),
            )
            self.prediction_repository.save_prediction_items(prediction_id, features)

    def _target_actual_values(self, batch_data: dict[str, Any]) -> dict[str, float | None]:
        targets = batch_data.get("targets") or []
        if not targets:
            return {"ph": None, "viscosity": None, "chlorides": None}

        latest = targets[-1]
        return {
            "ph": float(latest.get("ph")) if latest.get("ph") is not None else None,
            "viscosity": float(latest.get("viscosity")) if latest.get("viscosity") is not None else None,
            "chlorides": float(latest.get("chlorides")) if latest.get("chlorides") is not None else None,
        }

    def predict_batch(self, batch_id: int, up_to_step_order: int | None = None, up_to_measurement_id: int | None = None) -> dict[str, Any]:
        batch_data = self.dataset_builder.load_batch_data(batch_id)
        measurements = batch_data["measurements"]

        if measurements.empty:
            raise ValueError(f"Batch {batch_id} not found or contains no measurements")

        if up_to_step_order is not None:
            measurements = measurements.iloc[:up_to_step_order]
        if up_to_measurement_id is not None:
            measurements = measurements[measurements["id"] <= up_to_measurement_id]

        features = self._build_feature_vector(measurements)
        quality = self._get_data_quality(measurements)
        predictions = {
            "ph": self._predict_target("ph", features, batch_id=batch_id, quality=quality),
            "viscosity": self._predict_target("viscosity", features, batch_id=batch_id, quality=quality),
            "chlorides": self._predict_target("chlorides", features, batch_id=batch_id, quality=quality),
        }
        prediction_payload = {
            "batch_id": batch_id,
            "checkpoint": {
                "completed_steps": len(measurements),
                "last_measurement_id": int(measurements["id"].iloc[-1]) if not measurements.empty else None,
            },
            "data_quality": quality,
            "predictions": predictions,
            "similar_batches": self._build_similar_batches(features.iloc[0].to_dict(), batch_id),
        }

        actual_values = self._target_actual_values(batch_data)
        self._persist_batch_predictions(
            batch_id=batch_id,
            checkpoint_order=len(measurements),
            model_id=predictions["ph"]["model"] or predictions["viscosity"]["model"] or predictions["chlorides"]["model"],
            predictions=predictions,
            actual_values=actual_values,
            features=features.iloc[0].to_dict(),
        )

        return prediction_payload

    def predict_custom(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "measurements" not in payload:
            raise ValueError("Payload must contain measurements")

        measurements = pd.DataFrame(payload["measurements"])
        if measurements.empty:
            raise ValueError("Payload must contain at least one measurement")

        for col in measurements.columns:
            if measurements[col].dtype == object:
                measurements[col] = pd.to_numeric(measurements[col], errors="coerce")

        features = self._build_feature_vector(measurements)
        quality = {
            "total_measurements": len(measurements),
            "measurements_with_issues": 0,
            "issues_by_field": {},
        }
        return {
            "batch_id": None,
            "checkpoint": {
                "completed_steps": len(measurements),
                "last_measurement_id": None,
            },
            "data_quality": quality,
            "predictions": {
                "ph": self._predict_target("ph", features, quality=quality),
                "viscosity": self._predict_target("viscosity", features, quality=quality),
                "chlorides": self._predict_target("chlorides", features, quality=quality),
            },
            "similar_batches": [],
        }

    def update_prediction_actuals(self) -> int:
        unmatched = self.prediction_repository.list_unmatched_predictions(limit=1000)
        updates = 0
        for row in unmatched:
            batch_id = int(row["batch_id"])
            target = row["target"]
            batch_data = self.dataset_builder.load_batch_data(batch_id)
            actual_values = self._target_actual_values(batch_data)
            actual_value = actual_values.get(target)
            if actual_value is not None:
                self.prediction_repository.update_actual_value(int(row["prediction_id"]), actual_value)
                updates += 1
        return updates

    def train_if_no_model(self) -> None:
        if self.registry.list_models():
            return

        pipeline = TrainingPipeline(self.db_path, self.ml_storage_path)
        pipeline.train_all()
