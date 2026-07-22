from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CVFold:
    fold_idx: int
    train_indices: list[int]
    test_indices: list[int]
    train_groups: list[int]
    test_groups: list[int]


class LeaveOneGroupOut:
    def __init__(self, groups: np.ndarray | list):
        self.groups = np.asarray(groups)
        self.unique_groups = np.unique(self.groups)
        self.n_splits = len(self.unique_groups)

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> list[CVFold]:
        folds = []
        for fold_idx, test_group in enumerate(self.unique_groups):
            test_mask = self.groups == test_group
            test_indices = np.where(test_mask)[0].tolist()
            train_mask = ~test_mask
            train_indices = np.where(train_mask)[0].tolist()
            folds.append(
                CVFold(
                    fold_idx=fold_idx,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_groups=self.groups[train_indices].tolist(),
                    test_groups=self.groups[test_indices].tolist(),
                )
            )
        return folds


class MetricsCalculator:
    @staticmethod
    def mae(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        errors = np.abs(y_true - y_pred)
        if sample_weight is None:
            return float(np.mean(errors))
        return float(np.average(errors, weights=sample_weight))

    @staticmethod
    def rmse(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        squared = (y_true - y_pred) ** 2
        if sample_weight is None:
            return float(np.sqrt(np.mean(squared)))
        return float(np.sqrt(np.average(squared, weights=sample_weight)))

    @staticmethod
    def median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.median(np.abs(y_true - y_pred)))

    @staticmethod
    def r2_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        if sample_weight is None:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        else:
            mean_true = np.average(y_true, weights=sample_weight)
            ss_res = np.sum(sample_weight * ((y_true - y_pred) ** 2))
            ss_tot = np.sum(sample_weight * ((y_true - mean_true) ** 2))
        if ss_tot == 0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))

    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, float]:
        return {
            "mae": MetricsCalculator.mae(y_true, y_pred, sample_weight=sample_weight),
            "rmse": MetricsCalculator.rmse(y_true, y_pred, sample_weight=sample_weight),
            "median_ae": MetricsCalculator.median_ae(y_true, y_pred),
            "r2": MetricsCalculator.r2_score(y_true, y_pred, sample_weight=sample_weight),
        }


class CVValidator:
    def __init__(self, groups: np.ndarray | list):
        self.cv_splitter = LeaveOneGroupOut(groups)
        self.groups = np.asarray(groups)

    def validate_model(
        self,
        model: Any,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
        target_name: str = "target",
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> dict[str, Any]:
        y_array = np.asarray(y) if not isinstance(y, np.ndarray) else y
        weight_array = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        folds = self.cv_splitter.split(X, y_array)

        fold_results = []
        all_y_pred = np.empty_like(y_array, dtype=float)
        all_fold_indices: list[int] = []

        for fold in folds:
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[fold.train_indices]
                X_test = X.iloc[fold.test_indices]
            else:
                X_train = X[fold.train_indices]
                X_test = X[fold.test_indices]

            y_train = y_array[fold.train_indices]
            y_test = y_array[fold.test_indices]
            w_train = weight_array[fold.train_indices] if weight_array is not None else None
            w_test = weight_array[fold.test_indices] if weight_array is not None else None

            fit_signature = inspect.signature(model.fit)
            if "sample_weight" in fit_signature.parameters:
                model.fit(X_train, y_train, sample_weight=w_train)
            else:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_y_pred[fold.test_indices] = y_pred
            all_fold_indices.extend(fold.test_indices)

            fold_metrics = MetricsCalculator.calculate_all(y_test, y_pred, sample_weight=w_test)
            fold_metrics["test_group"] = fold.test_groups[0] if fold.test_groups else None
            fold_metrics["n_test_samples"] = len(fold.test_indices)
            fold_results.append(fold_metrics)

        sorted_order = np.argsort(all_fold_indices)
        y_pred_sorted = all_y_pred[sorted_order]
        sorted_weights = weight_array[sorted_order] if weight_array is not None else None
        cv_metrics = MetricsCalculator.calculate_all(
            y_array,
            y_pred_sorted,
            sample_weight=sorted_weights,
        )
        cv_metrics["n_folds"] = len(folds)
        cv_metrics["n_samples"] = len(y_array)

        return {
            "target": target_name,
            "cv_metrics": cv_metrics,
            "fold_results": fold_results,
            "predictions": all_y_pred,
        }
