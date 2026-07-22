from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class CVFold:
    """One fold of cross-validation."""
    fold_idx: int
    train_indices: list[int]
    test_indices: list[int]
    train_groups: list[int]
    test_groups: list[int]


class LeaveOneGroupOut:
    """Leave-One-Group-Out cross-validator.
    
    Splits data so that each group (batch_id) is tested exactly once,
    and all other groups are in train set. Ensures no data leakage:
    all snapshots from same batch stay together (either in train or test).
    """

    def __init__(self, groups: np.ndarray | list):
        """Initialize with group labels.
        
        Args:
            groups: Array-like of group labels (e.g., batch_ids).
                    Same label in multiple rows = same batch group.
        """
        self.groups = np.asarray(groups)
        self.unique_groups = np.unique(self.groups)
        self.n_splits = len(self.unique_groups)

    def split(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> list[CVFold]:
        """Generate train/test splits."""
        folds = []
        for fold_idx, test_group in enumerate(self.unique_groups):
            # Test indices: rows belonging to test_group
            test_mask = self.groups == test_group
            test_indices = np.where(test_mask)[0].tolist()

            # Train indices: all other rows
            train_mask = ~test_mask
            train_indices = np.where(train_mask)[0].tolist()

            fold = CVFold(
                fold_idx=fold_idx,
                train_indices=train_indices,
                test_indices=test_indices,
                train_groups=self.groups[train_indices].tolist(),
                test_groups=self.groups[test_indices].tolist(),
            )
            folds.append(fold)

        return folds


class MetricsCalculator:
    """Calculate cross-validation metrics."""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Median Absolute Error."""
        return float(np.median(np.abs(y_true - y_pred)))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))

    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate all metrics."""
        return {
            "mae": MetricsCalculator.mae(y_true, y_pred),
            "rmse": MetricsCalculator.rmse(y_true, y_pred),
            "median_ae": MetricsCalculator.median_ae(y_true, y_pred),
            "r2": MetricsCalculator.r2_score(y_true, y_pred),
        }


class CVValidator:
    """Cross-validation validator managing full CV loop."""

    def __init__(self, groups: np.ndarray | list):
        """Initialize with group labels."""
        self.cv_splitter = LeaveOneGroupOut(groups)
        self.groups = np.asarray(groups)

    def validate_model(self, model: Any, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series, 
                       target_name: str = "target") -> dict[str, Any]:
        """Run full cross-validation on model.
        
        Args:
            model: Model with fit(X_train, y_train) and predict(X_test) methods
            X: Features array
            y: Target values
            target_name: Name of target for reporting
            
        Returns:
            dict with cv_metrics, fold_results, predictions
        """
        y_array = np.asarray(y) if not isinstance(y, np.ndarray) else y
        folds = self.cv_splitter.split(X, y_array)

        fold_results = []
        all_y_pred = np.empty_like(y_array, dtype=float)
        all_fold_indices = []

        for fold in folds:
            # Get train/test data
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[fold.train_indices].values
                X_test = X.iloc[fold.test_indices].values
            else:
                X_train = X[fold.train_indices]
                X_test = X[fold.test_indices]

            y_train = y_array[fold.train_indices]
            y_test = y_array[fold.test_indices]

            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store predictions
            all_y_pred[fold.test_indices] = y_pred
            all_fold_indices.extend(fold.test_indices)

            # Calculate fold metrics
            fold_metrics = MetricsCalculator.calculate_all(y_test, y_pred)
            fold_metrics["test_group"] = fold.test_groups[0] if fold.test_groups else None
            fold_metrics["n_test_samples"] = len(fold.test_indices)

            fold_results.append(fold_metrics)

        # Calculate overall metrics
        y_pred_sorted = all_y_pred[np.argsort(all_fold_indices)]
        cv_metrics = MetricsCalculator.calculate_all(y_array, y_pred_sorted)
        cv_metrics["n_folds"] = len(folds)
        cv_metrics["n_samples"] = len(y_array)

        return {
            "target": target_name,
            "cv_metrics": cv_metrics,
            "fold_results": fold_results,
            "predictions": all_y_pred,
        }
