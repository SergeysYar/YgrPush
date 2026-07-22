from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from app.config import settings
from app.data.dataset_builder import DatasetBuilder
from app.data.target_loader import TargetProtocolPolicy
from app.ml.baseline import BaselineModel
from app.ml.ridge import RidgeModel
from app.ml.pls import PLSModel
from app.ml.bayesian import BayesianRidgeModel
from app.ml.validation import CVValidator, MetricsCalculator
from app.ml.exporter import CVResultsExporter
from app.ml.registry import ModelManager
from app.database.ml_storage import ModelRegistry
from datetime import datetime


class TrainingPipeline:
    """End-to-end training pipeline with cross-validation."""

    def __init__(self, db_path: Path | str | None = None, ml_storage_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or settings.db_path)
        self.ml_storage_path = Path(ml_storage_path or settings.ml_storage_path)
        self.dataset_builder = DatasetBuilder(self.db_path)
        self.ml_registry = ModelRegistry(str(self.ml_storage_path))
        self.model_manager = ModelManager(str(self.ml_storage_path))
        self._artifact_dir = self.ml_storage_path.parent / "artifacts"
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    def _persist_model(self, model: Any, model_type: str, target: str, features: dict[str, Any], metrics: dict[str, Any]) -> str:
        artifact_name = f"{model_type}_{target}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.joblib"
        artifact_path = self._artifact_dir / artifact_name
        model.save(str(artifact_path))
        return self.model_manager.register_model(
            model_type=model_type,
            target=target,
            artifact_path=str(artifact_path),
            features=features,
            metrics=metrics,
        )

    def prepare_training_data(
        self,
        snapshot_mode: bool = False,
        protocol_policy: TargetProtocolPolicy | str | None = None,
    ) -> dict[str, Any]:
        """Prepare training dataset for all targets.
        
        Returns dict with:
        - all_batches: full DataFrame with all targets
        - ph_data: rows with pH label
        - viscosity_data: rows with viscosity label
        - chlorides_data: rows with chlorides label
        - complete_data: rows with all three targets (for PLS)
        """
        batch_df = (
            self.dataset_builder.build_snapshot_features_dataset(
                protocol_policy=protocol_policy,
            )
            if snapshot_mode
            else self.dataset_builder.build_batch_features_dataset(
                protocol_policy=protocol_policy,
            )
        )

        training_data = {
            "all_batches": batch_df,
            "ph_data": batch_df[batch_df["target_ph"].notna()].reset_index(drop=True),
            "viscosity_data": batch_df[batch_df["target_viscosity"].notna()].reset_index(drop=True),
            "chlorides_data": batch_df[batch_df["target_chlorides"].notna()].reset_index(drop=True),
            "complete_data": batch_df[
                (batch_df["target_ph"].notna()) &
                (batch_df["target_viscosity"].notna()) &
                (batch_df["target_chlorides"].notna())
            ].reset_index(drop=True),
        }

        return training_data

    def _extract_features_and_targets(
        self,
        data: pd.DataFrame,
        target_col: str,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
        """Extract X and y, return with batch_ids for grouping."""
        # Feature columns: all except batch_id, has_targets, and target_* columns
        feature_cols = [col for col in data.columns if not col.startswith("target_")]
        feature_cols = [
            col
            for col in feature_cols
            if col not in {"batch_id", "has_targets", "snapshot_weight", "batch_number", "production_date", "batch_status", "product_name"}
        ]

        X = data[feature_cols].fillna(0)  # Impute missing features with 0
        y = data[target_col].values
        group_ids = data["batch_id"].values
        sample_weights = data["snapshot_weight"].values if "snapshot_weight" in data.columns else None

        return X, y, group_ids, sample_weights

    def train_and_validate_baseline(self, training_data: dict[str, Any], reports_dir: Path | None = None) -> dict[str, Any]:
        """Train Baseline models with cross-validation."""
        results = {}

        for target, data_key in [("ph", "ph_data"), ("viscosity", "viscosity_data"), ("chlorides", "chlorides_data")]:
            if training_data[data_key].empty:
                print(f"Skip Baseline for {target}: no labeled data")
                results[target] = None
                continue

            data = training_data[data_key]
            X, y, group_ids, sample_weights = self._extract_features_and_targets(data, f"target_{target}")

            # Baseline doesn't use features, just global mean
            model = BaselineModel()
            cv_validator = CVValidator(group_ids)

            cv_result = cv_validator.validate_model(
                model,
                X,
                y,
                target_name=f"baseline_{target}",
                sample_weight=sample_weights,
            )

            model_id = self._persist_model(
                model=model,
                model_type="baseline",
                target=target,
                features={"feature_names": X.columns.tolist()},
                metrics=cv_result["cv_metrics"],
            )

            results[target] = {
                "model": model,
                "model_id": model_id,
                "cv_result": cv_result,
                "n_samples": len(data),
                "n_folds": cv_result["cv_metrics"]["n_folds"],
            }

            print(f"Baseline {target} (model_id={model_id}):")
            print(f"  MAE: {cv_result['cv_metrics']['mae']:.4f}")
            print(f"  RMSE: {cv_result['cv_metrics']['rmse']:.4f}")
            print(f"  Median AE: {cv_result['cv_metrics']['median_ae']:.4f}")
            print(f"  R²: {cv_result['cv_metrics']['r2']:.4f}")

        return results

    def train_and_validate_ridge(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Train Ridge models with cross-validation."""
        results = {}

        for target, data_key in [("ph", "ph_data"), ("viscosity", "viscosity_data"), ("chlorides", "chlorides_data")]:
            if training_data[data_key].empty:
                print(f"Skip Ridge for {target}: no labeled data")
                results[target] = None
                continue

            data = training_data[data_key]
            X, y, group_ids, sample_weights = self._extract_features_and_targets(data, f"target_{target}")

            model = RidgeModel(alpha=1.0)
            cv_validator = CVValidator(group_ids)

            cv_result = cv_validator.validate_model(
                model,
                X,
                y,
                target_name=f"ridge_{target}",
                sample_weight=sample_weights,
            )

            model_id = self._persist_model(
                model=model,
                model_type="ridge",
                target=target,
                features={"feature_names": X.columns.tolist()},
                metrics=cv_result["cv_metrics"],
            )

            results[target] = {
                "model": model,
                "model_id": model_id,
                "cv_result": cv_result,
                "n_samples": len(data),
                "n_folds": cv_result["cv_metrics"]["n_folds"],
            }

            print(f"Ridge {target} (model_id={model_id}):")
            print(f"  MAE: {cv_result['cv_metrics']['mae']:.4f}")
            print(f"  RMSE: {cv_result['cv_metrics']['rmse']:.4f}")
            print(f"  Median AE: {cv_result['cv_metrics']['median_ae']:.4f}")
            print(f"  R²: {cv_result['cv_metrics']['r2']:.4f}")

        return results

    def train_and_validate_pls(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Train PLS model (only on complete cases with all three targets)."""
        results = {}

        complete_data = training_data["complete_data"]

        if len(complete_data) < 2:
            print(f"Skip PLS: only {len(complete_data)} complete samples (need >= 2)")
            return results

        print(f"Training PLS on {len(complete_data)} complete samples (all 3 targets)")

        # PLS expects multivariate output
        X, _, group_ids, sample_weights = self._extract_features_and_targets(complete_data, "target_ph")
        y = complete_data[["target_ph", "target_viscosity", "target_chlorides"]].values

        try:
            model = PLSModel()
            cv_validator = CVValidator(group_ids)

            # For multivariate validation, we need custom logic
            # For now, train on full data
            try:
                model.fit(X, y, sample_weight=sample_weights)
            except TypeError:
                model.fit(X, y)

            model_id = self._persist_model(
                model=model,
                model_type="pls_multivariate",
                target="all",
                features={"feature_names": X.columns.tolist()},
                metrics={
                    "n_samples": len(complete_data),
                },
            )

            results["pls_multivariate"] = {
                "model": model,
                "model_id": model_id,
                "n_samples": len(complete_data),
                "targets": ["ph", "viscosity", "chlorides"],
                "status": "trained",
            }

            print(f"PLS: trained on {len(complete_data)} samples (model_id={model_id})")

        except Exception as e:
            print(f"PLS training failed: {e}")
            results["pls_multivariate"] = {"status": "failed", "error": str(e)}

        return results

    def train_and_validate_bayesian(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Train BayesianRidge models with cross-validation."""
        results = {}

        for target, data_key in [("ph", "ph_data"), ("viscosity", "viscosity_data"), ("chlorides", "chlorides_data")]:
            if training_data[data_key].empty:
                print(f"Skip BayesianRidge for {target}: no labeled data")
                results[target] = None
                continue

            data = training_data[data_key]
            X, y, group_ids, sample_weights = self._extract_features_and_targets(data, f"target_{target}")

            model = BayesianRidgeModel()
            cv_validator = CVValidator(group_ids)

            cv_result = cv_validator.validate_model(
                model,
                X,
                y,
                target_name=f"bayesian_{target}",
                sample_weight=sample_weights,
            )

            model_id = self._persist_model(
                model=model,
                model_type="bayesian",
                target=target,
                features={"feature_names": X.columns.tolist()},
                metrics=cv_result["cv_metrics"],
            )

            results[target] = {
                "model": model,
                "model_id": model_id,
                "cv_result": cv_result,
                "n_samples": len(data),
                "n_folds": cv_result["cv_metrics"]["n_folds"],
            }

            print(f"BayesianRidge {target} (model_id={model_id}):")
            print(f"  MAE: {cv_result['cv_metrics']['mae']:.4f}")
            print(f"  RMSE: {cv_result['cv_metrics']['rmse']:.4f}")
            print(f"  Median AE: {cv_result['cv_metrics']['median_ae']:.4f}")
            print(f"  R²: {cv_result['cv_metrics']['r2']:.4f}")

        return results

    def train_all(
        self,
        snapshot_mode: bool = False,
        protocol_policy: TargetProtocolPolicy | str | None = None,
        model_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute full training pipeline with all models."""
        print("=" * 60)
        print("TRAINING PIPELINE - Stage 5")
        print("=" * 60)

        # Prepare data
        print("\n[1/5] Preparing training data...")
        training_data = self.prepare_training_data(
            snapshot_mode=snapshot_mode,
            protocol_policy=protocol_policy,
        )

        print(f"  Total batches: {len(training_data['all_batches'])}")
        print(f"  pH labeled: {len(training_data['ph_data'])}")
        print(f"  Viscosity labeled: {len(training_data['viscosity_data'])}")
        print(f"  Chlorides labeled: {len(training_data['chlorides_data'])}")
        print(f"  Complete (all 3 targets): {len(training_data['complete_data'])}")
        print(f"  Snapshot mode: {snapshot_mode}")
        print(f"  Protocol policy: {protocol_policy or settings.target_protocol_policy}")

        all_results = {}
        selected_model_types = set(model_types or ["baseline", "ridge", "pls", "bayesian_ridge"])

        # Train models
        print("\n[2/5] Training Baseline models...")
        all_results["baseline"] = (
            self.train_and_validate_baseline(training_data)
            if "baseline" in selected_model_types
            else {}
        )

        print("\n[3/5] Training Ridge models...")
        all_results["ridge"] = (
            self.train_and_validate_ridge(training_data)
            if "ridge" in selected_model_types
            else {}
        )

        print("\n[4/5] Training PLS model...")
        all_results["pls"] = (
            self.train_and_validate_pls(training_data)
            if "pls" in selected_model_types
            else {}
        )

        print("\n[5/5] Training BayesianRidge models...")
        all_results["bayesian"] = (
            self.train_and_validate_bayesian(training_data)
            if "bayesian_ridge" in selected_model_types
            else {}
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        # Save reports
        try:
            CVResultsExporter.export_cv_report_json(all_results, "reports/cv_report.json")
            CVResultsExporter.export_predictions_csv(all_results, "reports/cv_predictions.csv")
            CVResultsExporter.export_summary_metrics(all_results, "reports/cv_summary.csv")
        except Exception:
            pass

        return all_results

    def save_cv_report(self, results: dict[str, Any], output_path: Path | str | None = None) -> None:
        """Save cross-validation report to file."""
        if output_path is None:
            output_path = Path("reports/cv_report.json")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {}
        for model_type, model_results in results.items():
            if model_results is None:
                continue
            
            report[model_type] = {}
            for target, target_result in model_results.items():
                if target_result is None:
                    report[model_type][target] = None
                    continue

                if "cv_result" in target_result:
                    cv_data = target_result["cv_result"]
                    report[model_type][target] = {
                        "cv_metrics": cv_data["cv_metrics"],
                        "fold_results": cv_data["fold_results"],
                        "n_samples": target_result["n_samples"],
                        "n_folds": target_result["n_folds"],
                    }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"CV report saved to {output_path}")
