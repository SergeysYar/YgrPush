"""Tests for cross-validation and metrics."""

import numpy as np
import pandas as pd
import pytest

from app.ml.validation import LeaveOneGroupOut, MetricsCalculator, CVValidator


class TestLeaveOneGroupOut:
    """Test LeaveOneGroupOut splitter."""

    def test_split_basic(self):
        """Test basic split functionality."""
        groups = [1, 1, 2, 2, 3, 3]
        cv = LeaveOneGroupOut(groups)

        folds = cv.split(np.zeros((6, 2)))
        assert len(folds) == 3

        # Fold 0: test_group=1
        assert folds[0].fold_idx == 0
        assert folds[0].test_indices == [0, 1]
        assert folds[0].train_indices == [2, 3, 4, 5]
        assert folds[0].test_groups == [1, 1]
        assert folds[0].train_groups == [2, 2, 3, 3]

    def test_split_uneven_groups(self):
        """Test with uneven group sizes."""
        groups = [1, 2, 2, 3, 3, 3, 4]
        cv = LeaveOneGroupOut(groups)

        folds = cv.split(np.zeros((7, 2)))
        assert len(folds) == 4

        # Group 3 has 3 elements
        fold_for_group_3 = next(f for f in folds if f.test_groups[0] == 3)
        assert len(fold_for_group_3.test_indices) == 3
        assert len(fold_for_group_3.train_indices) == 4

    def test_no_data_leakage(self):
        """Verify no data leakage between train and test."""
        groups = [1, 1, 2, 2, 3]
        cv = LeaveOneGroupOut(groups)

        folds = cv.split(np.zeros((5, 2)))

        for fold in folds:
            test_set = set(fold.test_indices)
            train_set = set(fold.train_indices)
            assert len(test_set & train_set) == 0, "Train and test sets overlap"


class TestMetricsCalculator:
    """Test metrics calculation."""

    def test_mae(self):
        """Test Mean Absolute Error."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.5, 3.0])
        mae = MetricsCalculator.mae(y_true, y_pred)
        assert mae == pytest.approx(0.1667, abs=0.001)

    def test_rmse(self):
        """Test Root Mean Squared Error."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.5, 3.0])
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        assert rmse == pytest.approx(0.2041, abs=0.001)

    def test_median_ae(self):
        """Test Median Absolute Error."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.5, 3.0, 4.5, 5.0])
        median_ae = MetricsCalculator.median_ae(y_true, y_pred)
        assert median_ae == pytest.approx(0.25, abs=0.001)

    def test_r2_score(self):
        """Test R² score."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        r2 = MetricsCalculator.r2_score(y_true, y_pred)
        assert r2 == pytest.approx(0.9486, abs=0.001)

    def test_calculate_all(self):
        """Test calculating all metrics together."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.2, 1.8, 3.1, 3.9])

        metrics = MetricsCalculator.calculate_all(y_true, y_pred)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "median_ae" in metrics
        assert "r2" in metrics
        assert metrics["mae"] > 0


class SimpleModel:
    """Simple test model for CV validation."""

    def __init__(self):
        self.mean = None

    def fit(self, X, y):
        self.mean = float(np.mean(y))

    def predict(self, X):
        if self.mean is None:
            return np.zeros(len(X))
        return np.array([self.mean] * len(X))


class TestCVValidator:
    """Test cross-validation validator."""

    def test_cv_validation(self):
        """Test full CV validation loop."""
        # Create simple dataset with batch grouping
        X = pd.DataFrame({
            "feature_1": [1, 2, 3, 4, 5, 6],
            "feature_2": [10, 20, 30, 40, 50, 60],
        })
        y = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        groups = [1, 1, 2, 2, 3, 3]

        model = SimpleModel()
        validator = CVValidator(groups)

        result = validator.validate_model(model, X, y, "test_target")

        assert result["target"] == "test_target"
        assert result["cv_metrics"]["n_folds"] == 3
        assert result["cv_metrics"]["n_samples"] == 6
        assert "mae" in result["cv_metrics"]
        assert "rmse" in result["cv_metrics"]
        assert len(result["fold_results"]) == 3
        assert len(result["predictions"]) == 6

    def test_cv_validation_metrics_structure(self):
        """Verify fold results structure."""
        X = pd.DataFrame({"f": [1, 2, 3, 4]})
        y = np.array([1.0, 2.0, 3.0, 4.0])
        groups = [1, 1, 2, 2]

        model = SimpleModel()
        validator = CVValidator(groups)

        result = validator.validate_model(model, X, y)

        for fold_result in result["fold_results"]:
            assert "mae" in fold_result
            assert "rmse" in fold_result
            assert "median_ae" in fold_result
            assert "r2" in fold_result
            assert "test_group" in fold_result
            assert "n_test_samples" in fold_result

    def test_cv_validation_preserves_dataframe_and_weights(self):
        class WeightedRecordingModel:
            def __init__(self):
                self.seen_columns = None
                self.seen_weights = None
                self.mean = 0.0

            def fit(self, X, y, sample_weight=None):
                self.seen_columns = list(X.columns)
                self.seen_weights = None if sample_weight is None else list(sample_weight)
                self.mean = (
                    float(np.average(y, weights=sample_weight))
                    if sample_weight is not None
                    else float(np.mean(y))
                )

            def predict(self, X):
                return np.array([self.mean] * len(X))

        X = pd.DataFrame({"feature_a": [1, 2, 3, 4], "feature_b": [10, 20, 30, 40]})
        y = np.array([1.0, 2.0, 3.0, 4.0])
        groups = [1, 1, 2, 2]
        weights = np.array([0.5, 0.5, 0.5, 0.5])

        model = WeightedRecordingModel()
        validator = CVValidator(groups)
        result = validator.validate_model(model, X, y, sample_weight=weights)

        assert model.seen_columns == ["feature_a", "feature_b"]
        assert model.seen_weights is not None
        assert result["cv_metrics"]["n_samples"] == 4
