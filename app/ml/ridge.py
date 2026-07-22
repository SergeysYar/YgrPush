from __future__ import annotations

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.base import QualityModel


class RidgeModel:
    def __init__(self, alpha: float = 1.0) -> None:
        self.pipeline: Pipeline | None = None
        self.alpha = alpha
        self.feature_names: list[str] = []

    def fit(self, X, y, categorical_features: list[str] | None = None) -> "RidgeModel":
        categorical_features = categorical_features or []
        numeric_features = [col for col in X.columns if col not in (categorical_features or [])]
        self.feature_names = list(X.columns)
        transformer = ColumnTransformer(
            [
                ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ],
            remainder="drop",
        )
        self.pipeline = Pipeline(
            [("transformer", transformer), ("ridge", Ridge(alpha=self.alpha))]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise ValueError("RidgeModel is not trained")
        return self.pipeline.predict(X)

    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise ValueError("RidgeModel is not trained")
        joblib.dump(self.pipeline, path)

    def load(self, path: str) -> "RidgeModel":
        self.pipeline = joblib.load(path)
        return self

    def explain(self, X):
        return {"model": "ridge", "features": self.feature_names}
