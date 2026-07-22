from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np

from app.config import settings
from app.data.dataset_builder import DatasetBuilder
from app.features.batch_features import BatchFeatureBuilder
from app.ml.baseline import BaselineModel
from app.ml.ridge import RidgeModel
from app.ml.pls import PLSModel
from app.ml.bayesian import BayesianRidgeModel
from app.ml.registry import ModelManager
from app.database.ml_storage import ModelRegistry


class TrainingService:
    """Service for training and managing models."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or settings.db_path)
        self.dataset_builder = DatasetBuilder(self.db_path)
        self.model_manager = ModelManager(settings.ml_storage_path)
        self.feature_builder = BatchFeatureBuilder()

    def prepare_training_data(self) -> dict[str, Any]:
        """Prepare training dataset for all targets.
        
        Returns dict with separate datasets for each target:
        - ph_data
        - viscosity_data
        - chlorides_data
        """
        batch_df = self.dataset_builder.build_batch_features_dataset()

        training_data = {
            "all_batches": batch_df,
            "ph_data": batch_df[batch_df["target_ph"].notna()].reset_index(drop=True),
            "viscosity_data": batch_df[batch_df["target_viscosity"].notna()].reset_index(drop=True),
            "chlorides_data": batch_df[batch_df["target_chlorides"].notna()].reset_index(drop=True),
        }

        return training_data

    def train_baseline_models(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Train baseline models for each target."""
        models = {}
        metrics = {}

        for target, data_key in [("ph", "ph_data"), ("viscosity", "viscosity_data"), ("chlorides", "chlorides_data")]:
            if training_data[data_key].empty:
                print(f"Skip Baseline for {target}: no labeled data")
                continue

            df = training_data[data_key]
            X = df.drop(columns=["batch_id", "has_targets", "target_ph", "target_viscosity", "target_chlorides"], errors="ignore")
            y = df[f"target_{target}"]

            baseline = BaselineModel()
            baseline.fit(X, y)
            models[target] = baseline

            # Simple metrics
            y_pred = baseline.predict(X, category=None)
            mae = float(np.mean(np.abs(np.array(y_pred) - y.values)))
            metrics[target] = {"mae": mae, "n_samples": len(df)}

            print(f"Baseline {target}: {len(df)} samples, MAE={mae:.4f}")

        return {"models": models, "metrics": metrics}

    def train_ridge_models(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Train Ridge models for each target."""
        models = {}
        metrics = {}

        for target, data_key in [("ph", "ph_data"), ("viscosity", "viscosity_data"), ("chlorides", "chlorides_data")]:
            if training_data[data_key].empty:
                print(f"Skip Ridge for {target}: no labeled data")
                continue

            df = training_data[data_key]
            X = df.drop(columns=["batch_id", "has_targets", "target_ph", "target_viscosity", "target_chlorides"], errors="ignore")
            y = df[f"target_{target}"]

            model = RidgeModel(alpha=1.0)
            model.fit(X, y)
            models[target] = model

            # Simple eval
            y_pred = model.predict(X)
            mae = float(np.mean(np.abs(y_pred - y.values)))
            metrics[target] = {"mae": mae, "n_samples": len(df)}

            print(f"Ridge {target}: {len(df)} samples, MAE={mae:.4f}")

        return {"models": models, "metrics": metrics}

    def train_pls_models(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Train PLS models (only on complete cases)."""
        models = {}
        metrics = {}

        # PLS only uses batches with all three targets
        all_data = training_data["all_batches"]
        complete_data = all_data[
            (all_data["target_ph"].notna()) &
            (all_data["target_viscosity"].notna()) &
            (all_data["target_chlorides"].notna())
        ].reset_index(drop=True)

        if len(complete_data) < 2:
            print(f"Skip PLS: only {len(complete_data)} complete samples")
            return {"models": {}, "metrics": {}}

        X = complete_data.drop(columns=["batch_id", "has_targets", "target_ph", "target_viscosity", "target_chlorides"], errors="ignore")
        y = complete_data[["target_ph", "target_viscosity", "target_chlorides"]]

        try:
            model = PLSModel()
            model.fit(X, y)
            models["pls"] = model

            y_pred = model.predict(X)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            mae_ph = float(np.mean(np.abs(y_pred[:, 0] - y.iloc[:, 0].values)))
            mae_viscosity = float(np.mean(np.abs(y_pred[:, 1] - y.iloc[:, 1].values)))
            mae_chlorides = float(np.mean(np.abs(y_pred[:, 2] - y.iloc[:, 2].values)))

            metrics["pls"] = {
                "mae_ph": mae_ph,
                "mae_viscosity": mae_viscosity,
                "mae_chlorides": mae_chlorides,
                "n_samples": len(complete_data)
            }

            print(f"PLS: {len(complete_data)} samples, MAE_ph={mae_ph:.4f}")
        except Exception as e:
            print(f"PLS training failed: {e}")

        return {"models": models, "metrics": metrics}

    def train_bayesian_models(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Train Bayesian models for each target."""
        models = {}
        metrics = {}

        for target, data_key in [("ph", "ph_data"), ("viscosity", "viscosity_data"), ("chlorides", "chlorides_data")]:
            if training_data[data_key].empty:
                print(f"Skip BayesianRidge for {target}: no labeled data")
                continue

            df = training_data[data_key]
            X = df.drop(columns=["batch_id", "has_targets", "target_ph", "target_viscosity", "target_chlorides"], errors="ignore")
            y = df[f"target_{target}"]

            model = BayesianRidgeModel()
            model.fit(X, y)
            models[target] = model

            y_pred = model.predict(X)
            mae = float(np.mean(np.abs(y_pred - y.values)))
            metrics[target] = {"mae": mae, "n_samples": len(df)}

            print(f"BayesianRidge {target}: {len(df)} samples, MAE={mae:.4f}")

        return {"models": models, "metrics": metrics}

    def train_all(self) -> dict[str, Any]:
        """Train all models."""
        print("Preparing training data...")
        training_data = self.prepare_training_data()

        print(f"Dataset summary:")
        print(f"  Total batches: {len(training_data['all_batches'])}")
        print(f"  pH labeled: {len(training_data['ph_data'])}")
        print(f"  Viscosity labeled: {len(training_data['viscosity_data'])}")
        print(f"  Chlorides labeled: {len(training_data['chlorides_data'])}")

        results = {}

        print("\nTraining Baseline models...")
        results["baseline"] = self.train_baseline_models(training_data)

        print("\nTraining Ridge models...")
        results["ridge"] = self.train_ridge_models(training_data)

        print("\nTraining PLS models...")
        results["pls"] = self.train_pls_models(training_data)

        print("\nTraining Bayesian models...")
        results["bayesian"] = self.train_bayesian_models(training_data)

        return results
