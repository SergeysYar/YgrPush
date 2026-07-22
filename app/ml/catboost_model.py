from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None  # type: ignore[assignment]

try:
    import shap
except Exception:
    shap = None  # type: ignore[assignment]


class CatBoostModel:
    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_names: list[str] = []
        self.categorical_features: list[str] = []
        self.numeric_fill_values: dict[str, float] = {}

    @classmethod
    def is_available(cls) -> bool:
        return CatBoostRegressor is not None

    @classmethod
    def is_shap_available(cls) -> bool:
        return shap is not None

    def _prepare_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        frame = X.copy()
        self.feature_names = list(frame.columns)
        self.categorical_features = [
            column
            for column in frame.columns
            if frame[column].dtype == object or str(frame[column].dtype).startswith("category")
        ]
        numeric_features = [column for column in frame.columns if column not in self.categorical_features]

        for column in numeric_features:
            numeric_series = pd.to_numeric(frame[column], errors="coerce")
            if fit:
                fill_value = float(numeric_series.median()) if not numeric_series.dropna().empty else 0.0
                self.numeric_fill_values[column] = fill_value
            frame[column] = numeric_series.fillna(self.numeric_fill_values.get(column, 0.0))

        for column in self.categorical_features:
            frame[column] = frame[column].fillna("__missing__").astype(str)

        return frame

    def fit(self, X, y, sample_weight=None) -> "CatBoostModel":
        if CatBoostRegressor is None:
            raise ImportError("CatBoost is not installed")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        prepared = self._prepare_features(X, fit=True)
        cat_indices = [prepared.columns.get_loc(column) for column in self.categorical_features]
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_state,
            loss_function="RMSE",
            verbose=False,
        )
        self.model.fit(
            prepared,
            y,
            cat_features=cat_indices,
            sample_weight=sample_weight,
        )
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("CatBoostModel is not trained")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names or None)
        prepared = self._prepare_features(X, fit=False)
        return self.model.predict(prepared)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("CatBoostModel is not trained")
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "categorical_features": self.categorical_features,
                "numeric_fill_values": self.numeric_fill_values,
                "iterations": self.iterations,
                "depth": self.depth,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
            },
            path,
        )

    def load(self, path: str) -> "CatBoostModel":
        payload = joblib.load(path)
        self.model = payload["model"]
        self.feature_names = list(payload.get("feature_names", []))
        self.categorical_features = list(payload.get("categorical_features", []))
        self.numeric_fill_values = dict(payload.get("numeric_fill_values", {}))
        self.iterations = int(payload.get("iterations", self.iterations))
        self.depth = int(payload.get("depth", self.depth))
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.random_state = int(payload.get("random_state", self.random_state))
        return self

    def explain(self, X, top_n: int = 5):
        if self.model is None:
            raise ValueError("CatBoostModel is not trained")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names or None)
        prepared = self._prepare_features(X, fit=False)
        feature_names = list(prepared.columns)

        if shap is not None:
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(prepared.iloc[[0]])
                values = np.asarray(shap_values)[0]
                rows = [
                    {"feature": str(name), "contribution": float(value)}
                    for name, value in zip(feature_names, values)
                ]
                rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
                return {"top_factors": rows[:top_n], "explanation_method": "shap"}
            except Exception:
                pass

        importances = np.asarray(self.model.get_feature_importance())
        rows = [
            {"feature": str(name), "contribution": float(value)}
            for name, value in zip(feature_names, importances)
        ]
        rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return {"top_factors": rows[:top_n], "explanation_method": "feature_importance"}
