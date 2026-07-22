from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
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
        self.transformed_feature_names: list[str] = []

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
        self._refresh_feature_names()
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
        self._refresh_feature_names()
        return self

    def _refresh_feature_names(self) -> None:
        if self.pipeline is None:
            self.transformed_feature_names = []
            return
        transformer = self.pipeline.named_steps.get("transformer")
        if transformer is None:
            self.transformed_feature_names = []
            return
        try:
            names = transformer.get_feature_names_out()
            self.transformed_feature_names = [str(name) for name in names]
        except Exception:
            self.transformed_feature_names = list(self.feature_names)

    def explain(self, X, top_n: int = 5):
        if self.pipeline is None:
            raise ValueError("RidgeModel is not trained")
        transformer = self.pipeline.named_steps.get("transformer")
        ridge = self.pipeline.named_steps.get("ridge")
        if transformer is None or ridge is None:
            return {"top_factors": []}

        if isinstance(X, pd.DataFrame):
            transformed = transformer.transform(X.iloc[[0]])
        else:
            transformed = transformer.transform(X[:1])

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed_row = np.asarray(transformed)[0]
        coefficients = np.asarray(ridge.coef_).ravel()
        feature_names = self.transformed_feature_names or [f"feature_{index}" for index in range(len(coefficients))]

        factor_rows = []
        for name, value, coefficient in zip(feature_names, transformed_row, coefficients):
            contribution = float(value * coefficient)
            factor_rows.append(
                {
                    "feature": self._humanize_feature_name(name),
                    "contribution": contribution,
                }
            )

        factor_rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return {"top_factors": factor_rows[:top_n]}

    def _humanize_feature_name(self, feature_name: str) -> str:
        cleaned = feature_name.replace("numeric__", "").replace("categorical__", "")
        if cleaned.startswith("product_category_"):
            return f"Категория продукта: {cleaned.removeprefix('product_category_')}"
        if cleaned.startswith("product_base_"):
            return f"База продукта: {cleaned.removeprefix('product_base_')}"
        return cleaned
